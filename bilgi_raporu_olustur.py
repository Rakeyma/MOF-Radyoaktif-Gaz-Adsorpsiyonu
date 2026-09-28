"""
bilgi_raporu_olustur.py
=========================
MOF Radyoaktif Gaz Adsorpsiyonu — Bilgi Raporu (teknik metodoloji +
hiperparametre dokümantasyonu) üreticisi.

BİÇİM: danışmanın ONAYLADIĞI iki bilgi raporuyla (Termal Bariyer Kaplamalar
TBC, Katı Elektrolitler) BİREBİR aynı iskelet ve aynı dil (Türkçe):
    1. Veri Seti (1.1 Genel İstatistikler, 1.2 Kapsam Notu, 1.3 Sütun
       Grupları, 1.4 Çapraz Doğrulama Stratejisi)
    2. Ortak Eğitim Çerçevesi (2.1 Mimari Şablonu, 2.2 Kayıp ve Optimizasyon)
    3. Model Teknik Detayları (her model için 2 sütunlu tablo)
    4. Graf İnşa Stratejisi
    5. XAI Teknik Detayları
Onaylı raporlardaki gibi tablolar 2 sütunlu ANAHTAR–DEĞER biçimindedir ve
model tabloları "Pooled OOF sonucu" satırını içerir — yani onaylı raporlarda
olduğu gibi bu rapor da gerçek koşum çıktılarını okur, tamamen statik
DEĞİLDİR (önceki sürüm statikti; onaylı biçime uyacak şekilde değiştirildi).

Çalıştırma:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python bilgi_raporu_olustur.py
Çıktı: MOF_Radyoaktif_Gaz_Adsorpsiyonu_Bilgi_RAPORU.docx
"""
from __future__ import annotations

import json

import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor

from egitim_ortak import (
    TARGET_COLUMNS, PRETRAIN_MAX_EPOCHS, PRETRAIN_PATIENCE, PROXY_TARGET_COLUMN,
)
from graf_ozellik_ortak import CUTOFF, EMB_DIM
from ortak_ozellikler import AUX_FEATURE_COLUMNS, AUX_DIM, AUX_GRUPLARI
from paths import PROJECT_ROOT, MODEL_KLASORLERI, XAI_KLASORLERI, GITHUB_REPO_URL, GAS_FINETUNE_CSV
from veri_indirici_1_jarvis_core_mof import MAX_MATERIALS, MAX_MATERIALS_PRETRAIN, ARKETIPLER
from veri_artirma_3_augmentasyon import N_AUGMENT_PER_BASE


def h(doc, text, level=1):
    """TÜM başlıklar SİYAH kalın metin - rapor_olustur.py ile TUTARLI."""
    sizes = {0: 18, 1: 15, 2: 13, 3: 12}
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = True; r.font.size = Pt(sizes.get(level, 12))
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def para(doc, text, size=10, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text); r.font.size = Pt(size); r.bold = bold
    return p


def bullet(doc, text, size=9.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.5)
    r = p.add_run(text); r.font.size = Pt(size)
    return p


def kv_tablo(doc, satirlar, size=9):
    """Onaylı bilgi raporlarındaki 2 sütunlu ANAHTAR–DEĞER tablosu."""
    tablo = doc.add_table(rows=len(satirlar), cols=2)
    tablo.style = "Table Grid"
    for i, (anahtar, deger) in enumerate(satirlar):
        for j, metin in enumerate((anahtar, str(deger))):
            hucre = tablo.rows[i].cells[j]
            hucre.text = ""
            p = hucre.paragraphs[0]
            r = p.add_run(metin); r.font.size = Pt(size); r.bold = (j == 0)
    doc.add_paragraph()
    return tablo


# ---------------------------------------------------------------------------
# GERÇEK KOŞUM ÇIKTILARINI OKUMA
# ---------------------------------------------------------------------------
def _metrikler(model_adi: str) -> dict | None:
    f = PROJECT_ROOT / model_adi / "sonuclar" / "metrikler.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _egitilmis_modeller() -> list[str]:
    return [m for m in MODEL_KLASORLERI if _metrikler(m) is not None]


def _etiket_kaynak_dagilimi() -> dict[str, dict[str, int]]:
    if not GAS_FINETUNE_CSV.exists():
        return {}
    df = pd.read_csv(GAS_FINETUNE_CSV, low_memory=False)
    return {kol: df[f"label_source_{kol}"].value_counts().to_dict()
            for kol in TARGET_COLUMNS if f"label_source_{kol}" in df.columns}


MIMARI_TURU = {
    "GraphGPS": ("Sıfırdan eğitim; yerel mesaj iletimi (GINEConv) + küresel çok-başlı dikkat",
                 "Yerel komşu küresi (periyodik yarıçap grafiği)"),
    "PNA_GNN": ("Sıfırdan eğitim; çoklu toplayıcı (mean/min/max/std) × ölçekleyici "
                "(identity/amplification/attenuation)", "Yerel komşu küresi"),
    "GIN": ("Sıfırdan eğitim; kenar-özelliği duyarlı Graph Isomorphism Network (GINEConv)",
            "Yerel komşu küresi"),
    "GAT": ("Sıfırdan eğitim; GATv2 — kenar-mesafesi duyarlı dikkat", "Yerel komşu küresi"),
    "GatedGCN": ("Sıfırdan eğitim; öğrenilen kenar kapılama (ResGatedGraphConv)",
                 "Yerel komşu küresi"),
    "DeeperGCN": ("Sıfırdan eğitim; GENConv + DeepGCNLayer 'res+' blokları (en derin model)",
                  "Yerel komşu küresi"),
    "ECC": ("Sıfırdan eğitim; kenar-koşullu dinamik filtre üretimi (NNConv)", "Yerel komşu küresi"),
    "TFN": ("Sıfırdan eğitim; Tensor Field Network, l≤1 Clebsch-Gordan tensör çarpımı (e3nn YOK)",
            "Yerel komşu küresi"),
    "EGNN": ("Sıfırdan eğitim; E(n)-ekvaryant, yalnızca mesafe-değişmez mesaj iletimi — "
             "en hafif model, tüm XAI yöntemlerinin baz modeli", "Yerel komşu küresi"),
    "SE3_Transformer": ("Sıfırdan eğitim; SE(3)-ekvaryant çok-başlı dikkat, l≤1 CG eşleşmesi",
                        "Yerel komşu küresi"),
    "DimeNetPP": ("Sıfırdan eğitim; yönlü/açısal (k,j,i) üçlü mesaj iletimi, Fourier açısal tabanı",
                  "Yerel komşu küresi + üçlü (triplet) indeksleme"),
}


# ---------------------------------------------------------------------------
# §1 VERİ SETİ
# ---------------------------------------------------------------------------
def bolum_veri_seti(doc):
    h(doc, "1. Veri Seti")
    para(doc,
         f"Çalışmada kullanılan MOF kristal yapıları, Hugging Face üzerinde "
         f"barındırılan 'jablonkagroup/core_mof_no_topo' veri setinden (CC BY 4.0) "
         f"indirilmiştir; bu veri seti CoRE-MOF (Computation-Ready, Experimental "
         f"Metal-Organic Frameworks) veritabanının bir türevidir ve CIF gövdesini "
         f"doğrudan kayıt içinde taşır. CIF metni kaynakta '[CIF] ... [/CIF]' "
         f"etiketleriyle sarılı geldiğinden, pymatgen ayrıştırıcısına verilmeden "
         f"önce bu etiketler temizlenmektedir. Kaynak veri setinde DFT oluşum "
         f"enerjisi BULUNMADIĞINDAN, ön-eğitim hedefi olarak onun GCMC-simüle "
         f"edilmiş CO₂ adsorpsiyon ısısı (Widom ekleme yöntemi, kJ/mol → eV) "
         f"enerjisel bir VEKİL olarak kullanılmış ve sütun adı bunu açıkça "
         f"belirtecek şekilde '{PROXY_TARGET_COLUMN}' konmuştur. Kaynak veri "
         f"setinde Xe, Kr veya I₂ adsorpsiyon değeri YOKTUR — asıl hedef "
         f"etiketlerin nasıl üretildiği §1.2'de açıklanmıştır.", size=9.5)
    doc.add_paragraph()

    h(doc, "1.1 Genel İstatistikler", level=2)
    ham_bilgi = PROJECT_ROOT / "data" / "raw" / "kaynak_bilgisi.json"
    kaynak = json.loads(ham_bilgi.read_text(encoding="utf-8")) if ham_bilgi.exists() else {}
    eslesme_yol = PROJECT_ROOT / "data" / "processed" / "eslesme_bilgisi.json"
    eslesme = json.loads(eslesme_yol.read_text(encoding="utf-8")) if eslesme_yol.exists() else {}

    satirlar = [
        ("Kaynak servis", "Hugging Face Datasets — jablonkagroup/core_mof_no_topo (CC BY 4.0)"),
        ("Kaynak veritabanı", "CoRE-MOF türevi (deneysel olarak bilinen MOF yapılarından türetilmiş)"),
        ("Erişim yöntemi", "Anahtarsız — `datasets` kütüphanesi (config='raw_data')"),
        ("İkincil kaynak", "CoRE-MOF açık özet CSV — yapı metni İÇERMEZ, yalnızca "
                            "hesaplanmış gözeneklilik sütunları"),
        ("Üçüncül yedek", f"{len(ARKETIPLER)} bilinen MOF ailesinden (HKUST-1, MOF-5, ZIF-8, "
                           f"UiO-66, MIL-101, ...) prosedürel yapı üretimi — boru hattını ASLA "
                           f"bloklamaz, ancak kristalografik olarak rafine yapılar DEĞİLDİR"),
    ]
    if kaynak:
        satirlar += [
            ("İnce-ayar ham kayıt", kaynak.get("finetune_satir_sayisi", "—")),
            ("Ön-eğitim ham kayıt", kaynak.get("pretrain_satir_sayisi", "—")),
            ("İnce-ayar kaynak dağılımı",
             ", ".join(f"{k}: {v}" for k, v in kaynak.get("finetune_kaynak_dagilimi", {}).items()) or "—"),
            ("Ön-eğitim kaynak dağılımı",
             ", ".join(f"{k}: {v}" for k, v in kaynak.get("pretrain_kaynak_dagilimi", {}).items()) or "—"),
        ]
    if eslesme:
        satirlar += [
            ("Veri artırma sonrası örnek", eslesme.get("finetune_satir", "—")),
            ("Benzersiz temel MOF", eslesme.get("finetune_benzersiz_temel_mof", "—")),
            ("Eşleşme durumu",
             f"{eslesme.get('finetune_tam', '—')}/{eslesme.get('finetune_satir', '—')} TAM "
             f"(1:1 base_mof_id eşleşmesi)"),
        ]
    satirlar += [
        ("Veri boyutu üst sınırları",
         f"MAX_MATERIALS={MAX_MATERIALS} (ince-ayar temel MOF), "
         f"MAX_MATERIALS_PRETRAIN={MAX_MATERIALS_PRETRAIN}, "
         f"N_AUGMENT_PER_BASE={N_AUGMENT_PER_BASE} — ortam değişkeniyle değiştirilebilir; "
         f"bunlar ÜST SINIRDIR, kaynak daha az kayıt döndürürse koşum daha küçük kalır"),
        ("Hedef değişkenler", ", ".join(TARGET_COLUMNS)),
    ]
    kv_tablo(doc, satirlar)

    _bolum_kapsam_notu(doc)
    _bolum_sutun_gruplari(doc)
    _bolum_capraz_dogrulama(doc)


def _bolum_kapsam_notu(doc):
    h(doc, "1.2 Kapsam Notu: Hedef Etiketlerin Kökeni (KRİTİK BULGU)", level=2)
    dagilim = _etiket_kaynak_dagilimi()
    if not dagilim:
        para(doc, "[Henüz üretilmedi: nihai veri seti okunamadı.]", size=9)
        doc.add_paragraph()
        return

    satirlar = []
    for kol, sayim in dagilim.items():
        toplam = sum(sayim.values())
        satirlar.append((kol, ", ".join(f"{k}: {v} (%{100 * v / toplam:.0f})"
                                        for k, v in sorted(sayim.items(), key=lambda kv: -kv[1]))))
    kv_tablo(doc, satirlar)

    if {k for s in dagilim.values() for k in s} != {"PROXY_PORE_CORRELATION"}:
        para(doc, "Etiketler karma kaynaklıdır; yukarıdaki dağılıma bakınız. Vekil "
                   "kaynaklı satırlar deneysel/simülasyon verisi DEĞİLDİR.", size=9.5)
        doc.add_paragraph()
        return

    para(doc,
         "Bu koşumda hedef etiketlerin TAMAMI (%100) 'PROXY_PORE_CORRELATION' "
         "kaynaklıdır. Etiketler deneysel ölçüm DEĞİLDİR, GCMC simülasyonu "
         "DEĞİLDİR, literatürden alınmış DEĞİLDİR: gözeneklilik "
         "tanımlayıcılarından kapalı-form bir formülle ÜRETİLMİŞTİR. Formülün "
         "fiziksel motivasyonu gerçektir (Sikora et al. 2012'nin boyut-eleme "
         "ilkesi: gözenek-sınırlayıcı çap hedef gazın kinetik çapına yaklaştıkça "
         "seçicilik artar), ancak ürettiği SAYILAR gerçek değildir ve üzerlerine "
         "lognormal çarpımsal gürültü eklenmiştir.", size=9.5)
    doc.add_paragraph()
    para(doc, "Etiket üretim formülü (eslesme_4_dataset_birlestirici.py):", size=9.5, bold=True)
    for satir in [
        "boyut_uyum(d) = exp(−(PLD − d)² / (2 × 1.5²)) — d: kinetik çap "
        "(Xe 4.10 Å, Kr 3.69 Å, I₂ 5.00 Å)",
        "xe  = pore_volume × (0.8 + 1.5 × boyut_uyum(Xe)) × (1 + 0.4 × open_metal) × lognormal(0.15)",
        "kr  = pore_volume × (0.5 + 1.0 × boyut_uyum(Kr)) × (1 + 0.2 × open_metal) × lognormal(0.15)",
        "sel = clip(1 + 7 × boyut_uyum(Xe)/(boyut_uyum(Kr)+0.15) × (1 + 0.3 × open_metal), 1, 30)",
        "i2  = pore_volume × (0.3 + 1.0 × boyut_uyum(I₂)) × (1 + 1.2 × open_metal) "
        "× (1 + 0.5 × func) × lognormal(0.18)",
    ]:
        bullet(doc, satir, size=8.5)
    doc.add_paragraph()
    para(doc,
         "DÖNGÜSELLİK: bu formülün girdileri olan pld_A, pore_volume_cm3_g, "
         "open_metal_site ve has_functional_group değişkenlerinin DÖRDÜ DE modele "
         "yardımcı (aux) GİRDİ özelliği olarak verilmektedir (bkz. §1.3). "
         "Dolayısıyla model, kendi girdilerinden hesaplanan bir formülü geri "
         "çözmeyi öğrenmektedir; R² tavanı fiziksel öğrenme kapasitesiyle değil, "
         "etikete enjekte edilen gürültüyle belirlenir. Bunun sonuçlardaki izleri "
         "Sonuç Raporu'nda §2.6'da (gözeneklilik grubunun permütasyon öneminin 3B "
         "yapıdan onlarca kat büyük çıkması) ve §2.4'te (modelin ürettiği "
         "korelasyonun gürültülü gerçek etiketlerinkinden daha güçlü olması) "
         "görülebilir. Bu nedenle raporlanan başarım değerleri boru hattının "
         "teknik doğrulamasıdır; gerçek Xe/Kr/I₂ adsorpsiyon tahmin yeteneği "
         "olarak sunulamaz.", size=9.5)
    doc.add_paragraph()
    para(doc,
         "NLP literatür madenciliği bu koşumda kullanılabilir hiçbir etiket "
         "sağlayamamıştır: CrossRef/arXiv'den yalnızca 2 aday değer "
         "çıkarılabilmiş, ikisi de veri setindeki hiçbir yapıyla eşleşmemiştir. "
         "Dahası bu iki satır hatalı ayrıştırılmıştır — kaynak cümlede 3.46 "
         "mmol/g değeri Xe kapasitesine, Kr ise 350 cm³/g'a aittir; regex aynı "
         "değeri her iki hedefe birden atamıştır. Bu bileşenin gerçek etiket "
         "üretebilmesi için çıkarım mantığının düzeltilmesi gerekmektedir.",
         size=9.5)
    doc.add_paragraph()
    para(doc,
         "Gerçek bir çalışmaya dönüştürmek için gereken adımlar: (1) yayımlanmış "
         "GCMC Xe/Kr izoterm veri setlerinin bağlanması veya bu yapılar için "
         "RASPA ile doğrudan GCMC simülasyonu yapılması, vekil korelasyonun "
         "tamamen devre dışı bırakılması; (2) gözeneklilik tanımlayıcılarının "
         "geometrik yaklaşım yerine Zeo++ ile hesaplanması; (3) etiket üretiminde "
         "kullanılan değişkenlerin model girdisinden çıkarılarak döngüselliğin "
         "kırılması; (4) NLP çıkarım hattının düzeltilmesi; (5) eğitilmemiş "
         "mimarinin tamamlanarak karşılaştırmaya dahil edilmesi. Ayrıntı: depo "
         "kökündeki VERI_KAYNAGI_VE_SINIRLAMALAR.md.", size=9.5)
    doc.add_paragraph()


def _bolum_sutun_gruplari(doc):
    h(doc, "1.3 Sütun Grupları", level=2)
    para(doc, "Veri setindeki sütunlar işlevlerine göre şu gruplara ayrılmaktadır:", size=9.5)
    grup_metni = "; ".join(f"{ad} → {', '.join(kolonlar)}" for ad, kolonlar in AUX_GRUPLARI.items())
    for metin in [
        "Kimliksel: base_mof_id (fold gruplama anahtarı), sample_id, mof_name, graf_cif_path",
        f"Hedef: {', '.join(TARGET_COLUMNS)} — her biri için ayrıca label_source_<hedef> "
        f"sütunu, etiketin kökenini şeffaf biçimde işaretler",
        f"Yardımcı (aux) özellikler ({AUX_DIM} boyut): {', '.join(AUX_FEATURE_COLUMNS)}",
        f"Aux özellik grupları (permütasyon önemi ve XAI analizlerinde kullanılır): {grup_metni}",
        "Veri artırma: is_augmented, augmentation_ops (defect / substitution / functional / noise)",
        "Eşleşme durumu: eslesme_durumu (TAM/EKSIK) — 1:1 base_mof_id eşleşmesi",
    ]:
        bullet(doc, metin)
    doc.add_paragraph()
    para(doc,
         "ÖNEMLİ: gözeneklilik tanımlayıcıları (pore_volume, void_fraction, LCD, "
         "PLD, yüzey alanları) Zeo++ ile veya deneysel olarak ÖLÇÜLMEMİŞTİR — bu "
         "ortamda Zeo++ mevcut olmadığından yapıdan kaba bir geometrik yaklaşımla "
         "kestirilmişlerdir: void_fraction ≈ 1 − ΣV_vdW/V_hücre, "
         "pore_volume = void_fraction × V_hücre / kütle, LCD ≈ en kısa kafes "
         "vektörü × √void_fraction, PLD ≈ 0.55 × LCD "
         "(veri_indirici_1_jarvis_core_mof.estimate_pore_proxies). Bu değerler "
         "ölçülmüş BET yüzey alanı veya gözenek çapı olarak alıntılanamaz.", size=9)
    doc.add_paragraph()


def _bolum_capraz_dogrulama(doc):
    h(doc, "1.4 Çapraz Doğrulama Stratejisi", level=2)
    para(doc,
         "Veri artırma her temel MOF'tan birden fazla varyant ürettiğinden "
         "(defect / substitution / functional-group / gaussian-noise), bir MOF'un "
         "augment kopyasının eğitim setinde, orijinalinin test setinde yer alması "
         "ciddi bir sızıntı riski oluştururdu. Bu nedenle fold bölünmesi örnek "
         "bazında değil, TEMEL MOF (base_mof_id) bazında gruplanarak yapılmıştır.",
         size=9.5)
    doc.add_paragraph()
    egitilmis = _egitilmis_modeller()
    meta = _metrikler(egitilmis[0]) if egitilmis else None
    hp = meta["hiperparametreler"] if meta else {}
    kv_tablo(doc, [
        ("K-Fold sayısı", f"{meta['k_folds'] if meta else '—'} — kod varsayılanı 5'tir; "
                           f"'KFOLD_OVERRIDE' ortam değişkeniyle ezilebilir ve bu koşumda "
                           f"ezilmiştir"),
        ("Fold bölme birimi", "base_mof_id (gruplanmış K-Fold — sızıntı yok)"),
        ("Validasyon seti", "Eğitim fold'undan ayrılan alt küme (erken durdurma için)"),
        ("Validasyon amacı", "Erken durdurma (val MAE izlenir) ve en iyi checkpoint seçimi"),
        ("Rastgelelik tohumu", f"SEED = {hp.get('seed', '—')} (tüm modellerde sabit)"),
        ("Metrik hesaplama", "Havuzlanmış (pooled) OOF — tüm foldların test tahminleri "
                              "birleştirilerek tek bir R²/MAE hesaplanır"),
        ("Seyrek etiket toleransı", "Maskeli MSE — bir örnekte eksik olan hedef kayıp hesabına "
                                     "KATILMAZ (4 hedefin hepsi her satırda dolu olmayabilir)"),
    ])


# ---------------------------------------------------------------------------
# §2 ORTAK EĞİTİM ÇERÇEVESİ
# ---------------------------------------------------------------------------
def bolum_egitim_cercevesi(doc):
    h(doc, "2. Ortak Eğitim Çerçevesi")
    para(doc,
         f"Her model TEK bir MOF kristal yapısını kodlar ve bu gömmeyi, bu projeye "
         f"özgü {AUX_DIM} boyutlu bir yardımcı (aux) özellik vektörüyle birleştirir "
         f"(ortak_ozellikler.py — tüm modeller tarafından paylaşılan merkezi modül). "
         f"Aux vektörü gözeneklilik geometrisi (gözenek hacmi, boşluk oranı, "
         f"LCD/PLD, yüzey alanları), yapısal büyüklükler ve kompozisyon/kimyasal "
         f"değişiklik bilgilerinden (açık metal bölgesi, fonksiyonel grup, "
         f"elektronegatiflik farkı, metal oranı) oluşur.", size=9.5)
    doc.add_paragraph()

    h(doc, "2.1 Mimari Şablonu (GNN Modeller)", level=2)
    para(doc, "Tüm GNN modeller aşağıdaki ortak giriş/çıkış şablonunu paylaşır:", size=9.5)
    for metin in [
        f"Kristal kodlayıcı (modele bağlı, ortak arayüz: encoder.forward(g: dict)) → "
        f"gömme [EMB_DIM = {EMB_DIM}]",
        f"Aux özellik vektörü: {AUX_DIM} boyutlu — train-fold istatistiğiyle standardize "
        f"edilir, eksik değerler train ortalamasıyla doldurulur (sızıntı yok)",
        f"Birleştirme: concat([kristal_gömme, aux]) → [{EMB_DIM} + {AUX_DIM}]",
        "Regresyon başı: Linear → SiLU → Dropout → Linear → SiLU → Linear(→4) — TEK ortak "
        "baş, 4 hedefi BİRLİKTE tahmin eder (çok-görevli öğrenme)",
    ]:
        bullet(doc, metin)
    doc.add_paragraph()

    h(doc, "2.2 Kayıp ve Optimizasyon", level=2)
    egitilmis = _egitilmis_modeller()
    meta = _metrikler(egitilmis[0]) if egitilmis else None
    hp = meta["hiperparametreler"] if meta else {}
    kv_tablo(doc, [
        ("Kayıp fonksiyonu", "Maskeli MSE (eksik hedefler kayba katılmaz); val/test için MAE ve R²"),
        ("Optimizer", f"AdamW (weight_decay={hp.get('weight_decay', '—')})"),
        ("Öğrenme oranı", hp.get("lr", "—")),
        ("LR scheduler", "CosineAnnealingLR — LR'yi MAX_EPOCHS boyunca kademeli düşürür"),
        ("Max epoch / erken durdurma",
         f"{hp.get('max_epochs', '—')} / patience={hp.get('early_stop_patience', '—')}"),
        ("En iyi model", "Val MAE'nin en düşük olduğu epoch'taki ağırlıklar checkpoint'e kaydedilir"),
        ("Gizli katman / dropout",
         f"hidden_dim={hp.get('hidden_dim', '—')}, dropout={hp.get('dropout', '—')}"),
        ("Kesim mesafesi (cutoff)",
         f"{hp.get('cutoff', CUTOFF)} Å — gözenekli MOF'lar için yoğun kristallere kıyasla "
         f"kasıtlı olarak geniş tutulmuştur"),
        ("Veri ön işleme", "Hedef z-skoru normalizasyonu + aux standardizasyonu — HER FOLD için "
                            "TRAIN'den ayrı hesaplanır (sızıntı yok)"),
    ])

    h(doc, "2.3 Transfer Öğrenme (Ön-Eğitim → İnce-Ayar)", level=2)
    para(doc,
         f"Aşama A: her kodlayıcı, gaz adsorpsiyon hedeflerinden bağımsız bir vekil "
         f"hedef ('{PROXY_TARGET_COLUMN}') üzerinde ön-eğitilir ve SADECE kodlayıcı "
         f"ağırlıkları diske kaydedilir (regresyon başı atılır). Bu aşamada K-Fold "
         f"YAPILMAZ — amaç nihai değerlendirme değil, genellenebilir bir yapı temsili "
         f"öğrenmektir; basit tek train/val bölmesi ve erken durdurma yeterlidir. "
         f"Aşama B: bu ağırlıklar yüklenir, ilk N epoch boyunca kodlayıcı DONDURULUR "
         f"(yalnızca regresyon başı eğitilir — doğrusal sondalama), ardından çözülerek "
         f"tüm ağ gaz adsorpsiyon veri setinde ince-ayar edilir.", size=9.5)
    doc.add_paragraph()
    kv_tablo(doc, [
        ("Ön-eğitim hedefi", f"{PROXY_TARGET_COLUMN} — HF veri setinin GCMC-simüle CO₂ "
                              f"adsorpsiyon ısısından türetilen enerjisel vekil; gerçek DFT "
                              f"oluşum enerjisi DEĞİLDİR (kayıtların bir kısmı için ayrıca "
                              f"gözeneklilik/kompozisyondan türetilmiştir)"),
        ("Ön-eğitim max epoch / sabır",
         f"{PRETRAIN_MAX_EPOCHS} / {PRETRAIN_PATIENCE} — kod varsayılanı; bu aşamanın "
         f"hiperparametreleri JSON'a kaydedilmediğinden koşumda ezilmiş olup olmadığı "
         f"doğrulanamaz, şeffaflık için belirtilmiştir"),
        ("Encoder dondurma süresi", f"{hp.get('freeze_encoder_epochs', '—')} epoch"),
        ("Kaydedilen", "Yalnızca encoder.state_dict() — regresyon başı ince-ayarda gaz "
                        "hedeflerine özel olarak sıfırdan oluşturulur"),
    ])


# ---------------------------------------------------------------------------
# §3 MODEL TEKNİK DETAYLARI
# ---------------------------------------------------------------------------
def bolum_model_detaylari(doc):
    h(doc, "3. Model Teknik Detayları")
    df_yol = PROJECT_ROOT / "model_karsilastirma_sonuclari.csv"
    sira = {}
    if df_yol.exists():
        df = pd.read_csv(df_yol)
        ov = df[df["target_column"] == "overall"].sort_values("R2", ascending=False)
        sira = {row["model"]: (i, row["R2"], row["MAE"]) for i, (_, row) in enumerate(ov.iterrows(), 1)}

    egitilmis = _egitilmis_modeller()
    para(doc,
         f"Bu bölümde, bu koşumda EĞİTİLEN {len(egitilmis)} modelin teknik "
         f"ayrıntıları verilmektedir. Tüm modeller aynı ortak çerçeveyi (§2) "
         f"paylaştığından, tablolarda yalnızca modele özgü farklar ve o modelin "
         f"gerçek koşum sonucu yer almaktadır.", size=9.5)
    doc.add_paragraph()

    for i, model_adi in enumerate(egitilmis, 1):
        meta = _metrikler(model_adi)
        hp = meta["hiperparametreler"]
        h(doc, f"3.{i} {model_adi}", level=2)
        tur, graf = MIMARI_TURU.get(model_adi, ("—", "—"))
        mimari_ozgu = ", ".join(f"{k}={v}" for k, v in hp.items()
                                if k in ("n_layers", "heads", "l_max", "max_degree", "hidden"))
        satirlar = [
            ("Mimari türü", tur),
            ("Graf inşası", f"{graf}, cutoff={hp.get('cutoff')} Å"),
            ("Mimariye özgü parametreler", mimari_ozgu or "—"),
            ("Gömme boyutu", f"EMB_DIM = {hp.get('emb_dim')}"),
            ("Batch boyutu", hp.get("batch_size")),
            ("Max epoch / erken durdurma",
             f"{hp.get('max_epochs')} / patience={hp.get('early_stop_patience')}"),
            ("Ön-eğitim kullanıldı mı", "Evet" if meta.get("on_egitimli_kullanildi") else "Hayır"),
        ]
        if model_adi in sira:
            s, r2, mae = sira[model_adi]
            satirlar.append(("Pooled OOF sonucu",
                             f"R²={r2:.4f}, MAE={mae:.4f} — {s}. sıra / {len(sira)}"))
        kv_tablo(doc, satirlar)

    egitilmemis = [m for m in MODEL_KLASORLERI if m not in egitilmis]
    if egitilmemis:
        h(doc, f"3.{len(egitilmis) + 1} Eğitilmemiş Mimariler", level=2)
        para(doc,
             f"{', '.join(egitilmemis)} depoda tam olarak implemente edilmiştir "
             f"(run_*.py + grafik.py mevcuttur) ancak bu koşumda eğitilmemiştir; "
             f"dolayısıyla hiçbir sonuç/grafik üretmemiş ve Sonuç Raporu'ndaki "
             f"hiçbir tabloda yer almamıştır. Eğitilmesi için ilgili run_*.py'nin "
             f"çalıştırılması yeterlidir, kod değişikliği gerekmez.", size=9.5)
        doc.add_paragraph()


# ---------------------------------------------------------------------------
# §4 GRAF İNŞA STRATEJİSİ
# ---------------------------------------------------------------------------
def bolum_graf_insa(doc):
    h(doc, "4. Graf İnşa Stratejisi")
    para(doc,
         f"Tüm modeller ortak bir graf inşa modülünü (graf_ozellik_ortak.py) "
         f"paylaşır: CIF dosyasından periyodik 3B yarıçap grafiği kurulur — her atom "
         f"bir düğüm, kesim yarıçapı ({CUTOFF} Å) içindeki her atom çifti bir "
         f"kenardır. Kenar öznitelikleri Gauss radyal taban fonksiyonlarıyla (RBF) "
         f"kodlanmış atomlar arası mesafedir.", size=9.5)
    doc.add_paragraph()
    kv_tablo(doc, [
        ("Kesim yarıçapı", f"{CUTOFF} Å — gözenekli MOF'larda komşuluk küresinin gözenek "
                            f"boşluğunu da kapsaması için yoğun kristallere kıyasla geniş tutulmuştur"),
        ("Düğüm öznitelikleri", "Atom numarası (Z) gömmesi"),
        ("Kenar öznitelikleri", "Gauss RBF ile kodlanmış atomlar arası mesafe"),
        ("Periyodiklik", "Periyodik görüntüler dikkate alınır"),
        ("Kodlayıcı arayüzü", "encoder.forward(g: dict) — SÖZLÜK tabanlı. DimeNet++ gibi açısal "
                               "modeller ek alanlara (idx_kj / idx_ji / theta) ihtiyaç duyduğundan "
                               "pozisyonel arayüz yerine bu tercih edilmiştir"),
        ("Havuzlama", "scatter-mean (atom → MOF) — graf düzeyinde tek bir gömme vektörü"),
        ("Grafik boyutu", "Bu veri setinde MOF başına yaklaşık 72–172 atom — GraphLIME gibi "
                           "maskeleme tabanlı XAI yöntemlerinin ölçek duyarlılığı bakımından "
                           "belirleyici bir büyüklüktür (bkz. §5.1)"),
    ])


# ---------------------------------------------------------------------------
# §5 XAI TEKNİK DETAYLARI
# ---------------------------------------------------------------------------
XAI_DETAY = {
    "GraphLIME": [
        ("Yöntem", "Yerel vekil model (Huang et al., 2020)"),
        ("Uygulandığı model", "EGNN"),
        ("Çalışma ilkesi", "Tek bir MOF için atomların rastgele alt kümeleri Bernoulli(0.7) "
                            "maskesiyle kapatılır, maskeli tahminler toplanır ve maske→tahmin "
                            "ilişkisine seyrek doğrusal bir model oturtulur"),
        ("Örnekleme", "K=100 maske örneği; z=1 (tüm atomlar açık) daima örneklerden biridir"),
        ("Yerel ağırlık çekirdeği", "exp(−(kapalı_atom_oranı)² / 0.25²)"),
        ("Regresyon", "Çapraz doğrulamalı Lasso (LassoCV), alpha ızgarası logspace(−6, −1, 25)"),
        ("Neden LassoCV", "Sabit alpha (0.01) ile bu boyuttaki MOF'larda (72–172 atom) TÜM "
                           "katsayılar tam olarak sıfıra çöküyor ve açıklamalar BOŞ çıkıyordu: "
                           "maskelemenin tahmin üzerindeki etkisi graf büyüdükçe seyrelmekte "
                           "(4 mesaj-iletim katmanı + LayerNorm) ve sabit ceza terimi sinyali "
                           "tamamen bastırmaktadır. Alpha artık örnek başına veriden seçilir"),
        ("Çıktı", "Atom bazında katsayı (uzun format CSV) + element bazında özet"),
    ],
    "Edge_Attribution": [
        ("Yöntem", "Gradyan tabanlı kenar atfı (saliency + Integrated Gradients)"),
        ("Uygulandığı model", "EGNN"),
        ("Çalışma ilkesi", "Atıf, atomlar arası KENARLARIN (RBF çıktısı) üzerine uygulanır; "
                            "hangi atom–atom etkileşiminin tahmini ne kadar taşıdığı ölçülür"),
        ("Çıktı", "Element-çifti (bağ türü) bazında ve bağ uzunluğu aralığı bazında ortalama önem"),
        ("Yorumlama değeri", "Bağ-uzunluğu profili, modelin hangi mesafe ölçeğindeki komşuluklara "
                              "dayandığını (kısa kimyasal bağ mı, uzun gözenek-boşluğu teması mı) "
                              "ortaya koyar"),
    ],
    "SubgraphX": [
        ("Yöntem", "Monte Carlo Ağaç Araması ile açıklayıcı alt-graf arama (Yuan et al., 2021)"),
        ("Uygulandığı model", "EGNN"),
        ("Çalışma ilkesi", "Tahmini en iyi açıklayan BAĞLANTILI atom alt kümesi ('çekirdek "
                            "alt-graf') MCTS + UCT seçimiyle aranır"),
        ("Ödül", "Çok-hedefli, standardize uzayda hesaplanan ödül"),
        ("Çıktı", "Çekirdek alt-graf boyut dağılımı + elementlerin çekirdeğe girme oranı"),
    ],
    "IntegratedGradients": [
        ("Yöntem", "Integrated Gradients (Sundararajan et al., 2017)"),
        ("Uygulandığı model", "EGNN"),
        ("Çalışma ilkesi", "Maske KULLANMAZ: girdinin kendisi (atomların 3B koordinatları ve "
                            "gözeneklilik/kompozisyon özellikleri) bir taban çizgisinden gerçek "
                            "değere doğru kademeli değiştirilirken gradyanlar integre edilir"),
        ("Teorik garanti", "Completeness aksiyomu — katkıların toplamı tahmin farkına EŞİTTİR"),
        ("Çıktı", "Atom-konumu önemi (element bazında) + yardımcı özellik önemi"),
    ],
}


def bolum_xai_detaylari(doc):
    h(doc, "5. XAI (Açıklanabilir Yapay Zekâ) Teknik Detayları")
    egitilmis = _egitilmis_modeller()
    para(doc,
         f"Dört XAI yönteminin tamamı, eğitilmiş {len(egitilmis)} model arasında en "
         f"hafif ileri-geçişli (en hızlı) mimari olan EGNN üzerine uygulanmıştır. Bu "
         f"bir kısıtlama değil bilinçli bir tercihtir: XAI yöntemleri model başına "
         f"binlerce ileri-geçiş gerektirdiğinden, tek ve tutarlı bir hedef model "
         f"seçilerek yöntemler arası karşılaştırma anlamlı hale getirilmiştir.",
         size=9.5)
    doc.add_paragraph()
    for i, xai_adi in enumerate(XAI_KLASORLERI, 1):
        h(doc, f"5.{i} {xai_adi}", level=2)
        kv_tablo(doc, XAI_DETAY.get(xai_adi, [("—", "—")]))


# ---------------------------------------------------------------------------
# §6 KOD VE VERİ ERİŞİLEBİLİRLİĞİ
# ---------------------------------------------------------------------------
def bolum_erisim(doc):
    h(doc, "6. Kod ve Veri Erişilebilirliği")
    kv_tablo(doc, [
        ("Depo", GITHUB_REPO_URL),
        ("Dahil olanlar", "Tüm .py scriptleri, data/ altındaki nihai veri setleri (ham + işlenmiş, "
                           "CIF yapıları dahil), model checkpoint'leri (.pt), sonuç CSV/JSON "
                           "dosyaları (metrikler.json, test_tahminleri_oof.csv, "
                           "kfold_metrikleri.csv), XAI çıktıları ve tüm grafiklerin .png sürümleri"),
        ("Hariç tutulanlar (.gitignore)",
         "600 dpi .tif grafikler (dosya başına 40–90 MB; .png ikizleri depoda MEVCUTTUR), bu "
         "grafikleri gömen sonuç .docx dosyası ve ham per-kenar XAI dökümü "
         "(edge_attribution_kenarlar.csv, 79 MB) — hepsi ilgili script yeniden "
         "çalıştırıldığında birebir yeniden üretilir"),
        ("Veri kökeni belgesi", "VERI_KAYNAGI_VE_SINIRLAMALAR.md (depo kökü) — her veri "
                                 "parçasının kaynağı ve sentetiklik durumu ayrıntılı olarak"),
    ])


def main() -> None:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10)

    h(doc, "MOF Radyoaktif Gaz Adsorpsiyonu Bilgi Raporu", level=0)
    para(doc,
         "Metal-Organik Çerçevelerde Xe/Kr/I₂ Adsorpsiyon Kapasitesi ve Xe/Kr "
         "Seçicilik Tahmini — Teknik Metodoloji ve Hiperparametre Dokümantasyonu",
         size=12)
    doc.add_paragraph()

    bolum_veri_seti(doc)
    bolum_egitim_cercevesi(doc)
    bolum_model_detaylari(doc)
    bolum_graf_insa(doc)
    bolum_xai_detaylari(doc)
    bolum_erisim(doc)

    out_path = PROJECT_ROOT / "MOF_Radyoaktif_Gaz_Adsorpsiyonu_Bilgi_RAPORU.docx"
    doc.save(str(out_path))
    print(f"Bilgi raporu kaydedildi -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
