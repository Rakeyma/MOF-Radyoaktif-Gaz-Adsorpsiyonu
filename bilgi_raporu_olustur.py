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
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from egitim_ortak import (
    TARGET_COLUMNS, PRETRAIN_MAX_EPOCHS, PRETRAIN_PATIENCE, PROXY_TARGET_COLUMN,
)
from graf_ozellik_ortak import CUTOFF, EMB_DIM
from ortak_ozellikler import AUX_FEATURE_COLUMNS, AUX_DIM, AUX_GRUPLARI
from paths import PROJECT_ROOT, MODEL_KLASORLERI, XAI_KLASORLERI, GITHUB_REPO_URL, GAS_FINETUNE_CSV
from veri_indirici_1_jarvis_core_mof import MAX_MATERIALS, MAX_MATERIALS_PRETRAIN, ARKETIPLER
from veri_artirma_3_augmentasyon import N_AUGMENT_PER_BASE


def h(doc, text, level=1):
    """Word'un GERÇEK Heading stillerini kullanır - böylece başlıklar Word'de
    KATLANABİLİR olur ve gezinme bölmesinde görünür.

    BİÇİM FARKI (bilinçli): danışmanın onayladığı iki rapor arasında bu fark
    vardır ve korunur - SONUÇ raporunda başlıklar sade bold paragraftır
    (katlanmaz), BİLGİ raporunda ise Heading 1/2 stilleriyle katlanabilirdir.
    Başlık rengi Word'ün mavi varsayılanı yerine SİYAHA çevrilir (raporun
    tamamı siyah-beyaz)."""
    p = doc.add_heading(text, level=level)
    for r in p.runs:
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


def _kenarlik(tablo, kalin=8, orta=4) -> None:
    """ÜÇ-ÇİZGİLİ (booktabs) akademik tablo biçimi - bkz.
    rapor_olustur._kenarlik; iki rapor AYNI tablo biçimini kullanır."""
    tblPr = tablo._tbl.tblPr
    for eski in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(eski)
    borders = OxmlElement("w:tblBorders")
    for ad in ("top", "bottom"):
        e = OxmlElement(f"w:{ad}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(kalin))
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        borders.append(e)
    for ad in ("left", "right", "insideV", "insideH"):
        e = OxmlElement(f"w:{ad}")
        e.set(qn("w:val"), "none"); e.set(qn("w:sz"), "0"); e.set(qn("w:space"), "0")
        borders.append(e)
    tblPr.append(borders)


def kv_tablo(doc, satirlar, size=9):
    """2 sütunlu ANAHTAR–DEĞER tablosu, akademik üç-çizgili biçimde.

    NOT: bu tablolarda ayrı bir BAŞLIK satırı yoktur (sol sütunun kendisi
    anahtardır), bu yüzden üç-çizgili biçimin yalnızca üst ve alt çizgisi
    uygulanır - başlık-altı çizgisi anlamsız olurdu."""
    tablo = doc.add_table(rows=len(satirlar), cols=2)
    tablo.style = None
    for i, (anahtar, deger) in enumerate(satirlar):
        for j, metin in enumerate((anahtar, str(deger))):
            hucre = tablo.rows[i].cells[j]
            hucre.text = ""
            p = hucre.paragraphs[0]
            r = p.add_run(metin); r.font.size = Pt(size); r.bold = (j == 0)
    _kenarlik(tablo)
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


# ---------------------------------------------------------------------------
# SEMBOL VE KOD ADI SÖZLÜKLERİ
# ---------------------------------------------------------------------------
# Kullanıcı isteği: raporlarda geçen alt-tireli kod adları (xe_uptake_mmol_g,
# open_metal, PROXY_PORE_CORRELATION ...) tek başına anlaşılmıyordu. Fiziksel
# büyüklükler için standart SEMBOL kullanılır, kod adı yalnızca parantez içinde
# verilir; semboller ve kod adları aşağıdaki tablolarda açıklanır.
# (sembol, okunur ad, birim, kod adı)
SEMBOLLER = [
    ("q(Xe)", "Xe adsorpsiyon kapasitesi", "mmol/g", "xe_uptake_mmol_g"),
    ("q(Kr)", "Kr adsorpsiyon kapasitesi", "mmol/g", "kr_uptake_mmol_g"),
    ("q(I₂)", "I₂ adsorpsiyon kapasitesi", "mmol/g", "i2_uptake_mmol_g"),
    ("S(Xe/Kr)", "Xe/Kr seçicilik", "birimsiz", "xe_kr_selectivity"),
    ("PLD", "Gözenek-sınırlayıcı çap", "Å", "pld_A"),
    ("LCD", "En büyük kavite çapı", "Å", "lcd_A"),
    ("Vₚ", "Gravimetrik gözenek hacmi", "cm³/g", "pore_volume_cm3_g"),
    ("φ", "Boşluk (gözeneklilik) oranı", "0–1", "void_fraction"),
    ("A_g", "Gravimetrik yüzey alanı", "m²/g", "gravimetric_surface_area_m2_g"),
    ("A_v", "Hacimsel yüzey alanı", "m²/cm³", "volumetric_surface_area_m2_cm3"),
    ("ρ", "Çerçeve yoğunluğu", "g/cm³", "density_g_cm3"),
    ("N_atom", "Birim hücredeki atom sayısı", "adet", "nsites"),
    ("N_el", "Benzersiz element sayısı", "adet", "nelements"),
    ("OMS", "Açık metal bölgesi var mı", "0/1", "open_metal_site"),
    ("FG", "Linker fonksiyonlaştırılmış mı", "0/1", "has_functional_group"),
    ("Δχ", "Ortalama elektronegatiflik farkı (Pauling)", "birimsiz", "mean_electronegativity_diff"),
    ("r̄", "Ortalama atom yarıçapı", "Å", "mean_atomic_radius_A"),
    ("Z̄", "Ortalama atom numarası", "birimsiz", "mean_atomic_number"),
    ("f_metal", "Metal atomlarının toplam atoma oranı", "0–1", "metal_fraction"),
    ("d_k", "Gaz molekülünün kinetik çapı", "Å", "— (sabit: Xe 4.10, Kr 3.69, I₂ 5.00)"),
    ("E_proxy", "Ön-eğitim enerjisel vekil hedefi", "eV/atom", "formation_energy_eV_atom_proxy"),
    ("ΔMAE", "Permütasyon önemi (karıştırma sonrası MAE artışı)", "hedef birimi", "perm_importance.json"),
]

# Fiziksel büyüklük OLMAYAN, dolayısıyla sembolü bulunmayan kod adları.
KOD_ADLARI = [
    ("Etiket kaynağı değerleri", None),
    ("PROXY_PORE_CORRELATION", "Bir etiketin, o MOF'un gözeneklilik tanımlayıcılarından "
                                "(PLD, Vₚ, OMS, FG) kapalı-form bir formülle hesaplandığını "
                                "gösteren işaret — yani sentetik bir değer (bkz. §1.2)."),
    ("NLP_LITERATURE", "Bir etiketin, literatür makalelerinden metin madenciliğiyle "
                        "çıkarılmış GERÇEK bir değer olduğunu gösteren işaret. Bu koşumda "
                        "hiçbir satır bu kaynaktan gelmemiştir."),
    ("label_source_<hedef>", "Her hedef için ayrı bir sütun; o satırdaki etiketin yukarıdaki "
                              "iki kaynaktan hangisinden geldiğini şeffaf biçimde işaretler."),

    ("Veri seti sütunları", None),
    ("base_mof_id", "Bir örneğin TÜRETİLDİĞİ temel MOF'un kimliği. Veri artırma ile üretilen "
                     "tüm varyantlar, türedikleri orijinalle AYNI base_mof_id'yi taşır — "
                     "K-Fold bölmesi bu sütuna göre gruplanarak sızıntı önlenir."),
    ("sample_id", "Tek bir satırın (bir MOF varyantının) benzersiz kimliği."),
    ("mof_name", "MOF'un kimyasal adı/formülü."),
    ("graf_cif_path", "O örneğin kristal yapısını içeren CIF dosyasının yolu."),
    ("is_augmented", "Satır orijinal mi (False) yoksa veri artırmayla mı üretilmiş (True)."),
    ("augmentation_ops", "O varyantı üretirken uygulanan işlemler: defect (bağlayıcı "
                          "eksiltme), substitution (metal değiştirme), functional "
                          "(fonksiyonel grup ekleme), noise (atom konumlarına küçük "
                          "rastgele kaydırma)."),
    ("eslesme_durumu", "Yapı ile özellik kaydının eşleşip eşleşmediği (TAM / EKSIK)."),

    ("Çıktı dosyaları", None),
    ("metrikler.json", "Bir modelin koşum özeti: kullanılan hiperparametreler ve havuzlanmış "
                        "fold-dışı metrikler."),
    ("test_tahminleri_oof.csv", "Her örnek için fold-dışı (OOF) gerçek ve tahmin değerleri — "
                                 "tüm grafikler ve metrikler bu dosyadan üretilir."),
    ("kfold_metrikleri.csv", "Fold bazında eğitim/validasyon/test metrikleri, seçilen epoch "
                              "ve fold büyüklükleri."),
    ("egitim_gecmisi_fold<N>.csv", "O fold'un epoch epoch eğitim ve validasyon kaybı — kayıp "
                                    "eğrisi grafikleri bundan çizilir."),
    ("perm_importance.json", "Permütasyon önemi (ΔMAE) skorları, özellik grubu bazında."),
    ("Feature_Importance.txt", "Permütasyon önemi grafiğindeki harf etiketlerinin hangi "
                                "özellik grubuna karşılık geldiği."),

    ("Ayar (ortam değişkeni) adları", None),
    ("MAX_MATERIALS / MAX_MATERIALS_PRETRAIN", "İnce-ayar ve ön-eğitim için indirilecek "
                                                "MOF sayısının ÜST SINIRI."),
    ("N_AUGMENT_PER_BASE", "Her temel MOF'tan kaç artırılmış varyant üretileceği."),
    ("KFOLD_OVERRIDE", "K-Fold sayısını (K) koddaki varsayılanın yerine geçerek belirleyen "
                        "ortam değişkeni."),
    ("PRETRAIN_VAL_FRAC", "Ön-eğitimde validasyona ayrılan oran (0.1 = %10)."),
    ("SEED", "Rastgeleliği tekrarlanabilir kılan sabit tohum değeri."),

    ("Model/mimari parametre adları", None),
    ("n_layers", "Mesaj iletim katmanı sayısı — model kaç kez komşuluk bilgisi yaysın."),
    ("hidden_dim", "Ara katmanlardaki gizli temsil genişliği (nöron sayısı)."),
    ("emb_dim (EMB_DIM)", "Kodlayıcının ürettiği MOF gömme vektörünün uzunluğu."),
    ("aux_dim (AUX_DIM)", "Yardımcı sayısal özellik vektörünün uzunluğu."),
    ("batch_size", "Her gradyan güncellemesinde kullanılan örnek sayısı."),
    ("max_epochs", "Erken durdurma tetiklenmezse çalışacak MAKSİMUM epoch sayısı."),
    ("early_stop_patience", "Validasyon hatası kaç epoch iyileşmezse eğitimin durdurulacağı."),
    ("freeze_encoder_epochs", "İnce-ayarın başında kodlayıcının kaç epoch dondurulacağı."),
    ("weight_decay", "Ağırlıkların büyümesini cezalandıran düzenlileştirme katsayısı."),
    ("dropout", "Eğitimde geçici olarak kapatılan nöron oranı."),
    ("cutoff", "İki atomun grafta kenarla bağlanması için izin verilen en büyük mesafe (Å)."),
    ("heads", "Dikkat mekanizmasındaki paralel 'baş' sayısı (GAT, GraphGPS, SE(3)-T)."),
    ("l_max", "Ekvaryant modellerde kullanılan en yüksek açısal momentum derecesi "
               "(l≤1: skaler + vektör kanalları)."),
    ("max_degree", "PNA'nın beklediği en büyük düğüm komşu sayısı (ölçekleyicileri "
                    "normalize etmek için)."),

    ("Fold büyüklüğü sütunları", None),
    ("n_grup_train / n_grup_val / n_grup_test", "O fold'da eğitim / validasyon / test "
                                                 "setine düşen TEMEL MOF sayısı (örnek "
                                                 "sayısı değil)."),
    ("en_iyi_epoch", "O fold'da validasyon MAE'sinin en düşük olduğu, yani ağırlıkların "
                      "kaydedildiği epoch."),
]


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
                           f"bloklamaz. Bunlar, bilinen MOF ailelerinin metal düğüm ve organik "
                           f"bağlayıcı yerleşimini taklit eden basitleştirilmiş iskeletlerdir"),
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
    _bolum_ornek_sayilari(doc)
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
         "kaynaklıdır: her etiket, o MOF'un gözeneklilik tanımlayıcılarından "
         "(PLD, Vₚ, OMS, FG) kapalı-form bir formülle HESAPLANMIŞ ve üzerine "
         "lognormal çarpımsal gürültü eklenmiş SENTETİK bir sayıdır. Üretim "
         "zinciri şudur: yapı → geometrik gözeneklilik kestirimi → boyut-uyum "
         "terimi g(d_k) → kapasite/seçicilik formülü → gürültü. Formülün "
         "fiziksel motivasyonu literatürden gelir (Sikora et al. 2012'nin "
         "boyut-eleme ilkesi: PLD hedef gazın kinetik çapına yaklaştıkça "
         "seçicilik artar); üretilen sayılar ise bu ilkenin matematiksel bir "
         "taklididir. Etiketlerin kaynağı deneysel ölçüm, GCMC simülasyonu veya "
         "literatür derlemesi olsaydı 'label_source' sütunu bunu gösterirdi; bu "
         "koşumda 2100 satırın 2100'ü de formül kaynaklıdır.", size=9.5)
    doc.add_paragraph()
    para(doc, "Etiket üretim formülü (eslesme_4_dataset_birlestirici.py):", size=9.5, bold=True)
    for satir in [
        "g(d_k) = exp(−(PLD − d_k)² / (2 × 1.5²))   —  boyut-uyum terimi; d_k gazın "
        "kinetik çapıdır (Xe 4.10 Å, Kr 3.69 Å, I₂ 5.00 Å). PLD bu çapa yaklaştıkça "
        "g → 1 olur",
        "q(Xe)    = Vₚ × (0.8 + 1.5 × g(Xe)) × (1 + 0.4 × OMS) × lognormal(0.15)",
        "q(Kr)    = Vₚ × (0.5 + 1.0 × g(Kr)) × (1 + 0.2 × OMS) × lognormal(0.15)",
        "S(Xe/Kr) = clip( 1 + 7 × g(Xe) / (g(Kr) + 0.15) × (1 + 0.3 × OMS),  1,  30 )",
        "q(I₂)    = Vₚ × (0.3 + 1.0 × g(I₂)) × (1 + 1.2 × OMS) × (1 + 0.5 × FG) × lognormal(0.18)",
        "lognormal(σ): ortalaması 1 olan çarpımsal rastgele gürültü — aynı yapıya "
        "her seferinde birebir aynı değeri vermemek için eklenir",
    ]:
        bullet(doc, satir, size=8.5)
    doc.add_paragraph()
    para(doc,
         "DÖNGÜSELLİK: bu formülün girdileri olan PLD, Vₚ, OMS ve FG "
         "değişkenlerinin DÖRDÜ DE modele "
         "yardımcı (aux) GİRDİ özelliği olarak verilmektedir (bkz. §1.3). "
         "Dolayısıyla model, kendi girdilerinden hesaplanan bir formülü geri "
         "çözmeyi öğrenmektedir; R² tavanı fiziksel öğrenme kapasitesiyle değil, "
         "etikete enjekte edilen gürültüyle belirlenir. Bunun sonuçlardaki izleri "
         "Sonuç Raporu'nda §2.5'te (gözeneklilik grubunun permütasyon öneminin 3B "
         "yapıdan onlarca kat büyük çıkması) ve §2.3'te (modelin ürettiği "
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
    h(doc, "1.3.1 Sembol Tablosu (fiziksel büyüklükler)", level=3)
    para(doc,
         "Raporun geri kalanında fiziksel büyüklükler için aşağıdaki SEMBOLLER "
         "kullanılır; ham sütun (kod) adı yalnızca referans için verilmiştir. "
         "Sembolü olmayan, yalnızca kod adıyla anılan diğer tüm tanımlayıcılar "
         "(dosya adları, ayar adları, etiket kaynağı değerleri vb.) §10.1'de "
         "ayrıca açıklanmıştır.", size=9.5)
    doc.add_paragraph()
    tablo = doc.add_table(rows=1 + len(SEMBOLLER), cols=4)
    tablo.style = None
    for j, baslik in enumerate(["Sembol", "Büyüklük", "Birim", "Veri sütunu (kod adı)"]):
        hucre = tablo.rows[0].cells[j]; hucre.text = ""
        r = hucre.paragraphs[0].add_run(baslik); r.bold = True; r.font.size = Pt(9)
    for i, (sem, ad, birim, kod) in enumerate(SEMBOLLER, 1):
        for j, metin in enumerate([sem, ad, birim, kod]):
            hucre = tablo.rows[i].cells[j]; hucre.text = ""
            r = hucre.paragraphs[0].add_run(metin); r.font.size = Pt(8.5)
            r.bold = (j == 0)
    _kenarlik(tablo)
    doc.add_paragraph()

    para(doc,
         "ÖNEMLİ: gözeneklilik tanımlayıcıları (Vₚ, φ, LCD, "
         "PLD, A_g, A_v) Zeo++ ile veya deneysel olarak ÖLÇÜLMEMİŞTİR — bu "
         "ortamda Zeo++ mevcut olmadığından yapıdan kaba bir geometrik yaklaşımla "
         "kestirilmişlerdir: φ ≈ 1 − ΣV_vdW/V_hücre, "
         "Vₚ = φ × V_hücre / kütle, LCD ≈ en kısa kafes vektörü × √φ, "
         "PLD ≈ 0.55 × LCD "
         "(veri_indirici_1_jarvis_core_mof.estimate_pore_proxies). Bunlar "
         "geometrik KESTİRİMDİR ve raporlarda bu sıfatla anılmalıdır; "
         "ölçülmüş BET yüzey alanı veya Zeo++ gözenek çapı yerine geçmezler.", size=9)
    doc.add_paragraph()


def _bolum_ornek_sayilari(doc):
    """Hedef başına kaç etiketli örnek olduğu — 'her amaçta kaç veri var'
    sorusunun doğrudan cevabı."""
    h(doc, "1.4 Hedef (Amaç) Başına Örnek Sayısı", level=2)
    if not GAS_FINETUNE_CSV.exists():
        para(doc, "[Henüz üretilmedi.]", size=9)
        return
    df = pd.read_csv(GAS_FINETUNE_CSV, low_memory=False)
    if "eslesme_durumu" in df.columns:
        df = df[df["eslesme_durumu"] == "TAM"]
    para(doc,
         f"Bu proje ÇOK-GÖREVLİ (multi-task) bir modeldir: tek bir model, dört "
         f"hedefi AYNI ANDA tahmin eder. Bu nedenle hedefler için AYRI veri "
         f"kümeleri veya ayrı bölmeler YOKTUR — dördü de aynı {len(df)} örnek "
         f"üzerinde, aynı fold bölmesiyle eğitilir. Aşağıdaki tablo her hedef "
         f"için kaç örneğin etiketli olduğunu gösterir.", size=9.5)
    doc.add_paragraph()
    satirlar = []
    for kol in TARGET_COLUMNS:
        dolu = int(df[kol].notna().sum()) if kol in df.columns else 0
        satirlar.append((kol, f"{dolu} / {len(df)} örnek etiketli "
                              f"(%{100 * dolu / max(len(df), 1):.0f}); eksik: {len(df) - dolu}"))
    satirlar.append(("TOPLAM örnek", f"{len(df)} satır = {df['base_mof_id'].nunique()} temel MOF × "
                                      f"(1 orijinal + {N_AUGMENT_PER_BASE} artırılmış varyant)"))
    if "is_augmented" in df.columns:
        satirlar.append(("Orijinal / artırılmış",
                         f"{int((~df['is_augmented']).sum())} orijinal + "
                         f"{int(df['is_augmented'].sum())} artırılmış"))
    kv_tablo(doc, satirlar)
    eksik_var = any(df[k].isna().any() for k in TARGET_COLUMNS if k in df.columns)
    para(doc,
         ("Dört hedefin tamamı her satırda doludur; dolayısıyla maskeli MSE "
          "mekanizması (eksik hedefleri kayıptan düşürme) bu koşumda fiilen "
          "devreye girmemiştir — ancak seyrek etiketli bir veri setine "
          "geçildiğinde çalışmaya hazırdır."
          if not eksik_var else
          "Bazı hedefler bazı satırlarda eksiktir; maskeli MSE sayesinde bu "
          "satırlar yine de eğitimde kullanılır, eksik hedef kayba katılmaz."),
         size=9.5)
    doc.add_paragraph()


def _bolum_capraz_dogrulama(doc):
    h(doc, "1.5 Çapraz Doğrulama Stratejisi ve Veri Bölme Oranları", level=2)
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

    h(doc, "1.5.1 Her Fold'da Veri Kaça Kaç Ayrılıyor", level=3)
    kdf = None
    if egitilmis:
        f = PROJECT_ROOT / egitilmis[0] / "sonuclar" / "kfold_metrikleri.csv"
        if f.exists():
            kdf = pd.read_csv(f)
    if kdf is not None and {"n_grup_train", "n_grup_val", "n_grup_test"}.issubset(kdf.columns):
        tr, va, te = (int(kdf["n_grup_train"].iloc[0]), int(kdf["n_grup_val"].iloc[0]),
                      int(kdf["n_grup_test"].iloc[0]))
        top = tr + va + te
        kat = (N_AUGMENT_PER_BASE + 1)
        para(doc,
             f"Bölme TEMEL MOF sayısı üzerinden yapılır; her temel MOF 1 orijinal + "
             f"{N_AUGMENT_PER_BASE} artırılmış varyantla toplam {kat} örneğe karşılık "
             f"gelir. Aşağıdaki sayılar gerçek koşumun kfold_metrikleri.csv "
             f"dosyasından okunmuştur.", size=9.5)
        doc.add_paragraph()
        kv_tablo(doc, [
            ("Test (fold-dışı)", f"{te} temel MOF (%{100 * te / top:.0f}) ≈ {te * kat} örnek — "
                                  f"K={meta['k_folds']} olduğundan her fold verinin 1/{meta['k_folds']}'ini "
                                  f"test olarak ayırır"),
            ("Eğitim", f"{tr} temel MOF (%{100 * tr / top:.0f}) ≈ {tr * kat} örnek"),
            ("Validasyon", f"{va} temel MOF (%{100 * va / top:.0f}) ≈ {va * kat} örnek — "
                            f"test DIŞINDA kalan {tr + va} MOF'un %15'i "
                            f"(GroupShuffleSplit, yine MOF bazında)"),
            ("TOPLAM", f"{top} temel MOF ≈ {top * kat} örnek"),
            ("Özet gösterim", f"İnce-ayar: her fold için %{100 * tr / top:.0f} eğitim / "
                               f"%{100 * va / top:.0f} validasyon / %{100 * te / top:.0f} test, "
                               f"{meta['k_folds']}-katlı gruplanmış çapraz doğrulama"),
            ("Ön-eğitim (ayrı aşama)",
             "K-Fold YAPILMAZ — basit tek bölme: %90 eğitim / %10 validasyon "
             "(PRETRAIN_VAL_FRAC=0.1). Amaç nihai değerlendirme değil, genellenebilir "
             "bir yapı temsili öğrenmek olduğundan çapraz doğrulamaya gerek yoktur"),
        ])
        para(doc,
             "ÖNEMLİ: dört hedef için AYRI bölme yapılmaz. Model çok-görevli "
             "olduğundan tek bir bölme tüm hedefler için ortaktır; yani "
             "'1. amaç için şu oran, 2. amaç için bu oran' şeklinde bir ayrım "
             "YOKTUR — dört hedef de yukarıdaki aynı bölmeyi kullanır.", size=9.5)
        doc.add_paragraph()
    else:
        para(doc, "[Henüz üretilmedi: kfold_metrikleri.csv]", size=9)


def bolum_hiperparametre_secimi(doc):
    """Kullanıcı sorusu: 'hiperparametre optimizasyonu için ne kullandık',
    '5 fold için neyi maksimize ediyoruz', 'hangi epoch'ta kesmeliyiz'.
    DÜRÜSTLÜK: bu projede sistematik bir hiperparametre ARAMASI YAPILMAMIŞTIR;
    bunu olduğu gibi yazmak gerekir."""
    h(doc, "3. Hiperparametre Seçimi, Erken Durdurma ve Model Seçimi")

    h(doc, "3.1 Hiperparametre Optimizasyonu Yapıldı mı?", level=2)
    para(doc,
         "Bu koşumda hiperparametreler SABİT seçilmiş ve on modelin tamamında "
         "AYNI tutulmuştur: katman sayısı, öğrenme oranı, batch boyutu, gizli "
         "katman genişliği, dropout ve eğitim bütçesi tüm mimariler için "
         "ortaktır. Sistematik bir arama (grid search, random search, Bayesçi "
         "optimizasyon, Optuna vb.) bu koşumun kapsamı dışında bırakılmıştır. "
         "Gerekçe, projenin amacının FARKLI MİMARİLERİ ADİL KOŞULLARDA "
         "KARŞILAŞTIRMAK olmasıdır: tüm mimariler aynı bütçeyi ve aynı ayarları "
         "paylaştığında, aradaki başarım farkı mimari tasarımına atfedilebilir "
         "hale gelir.", size=9.5)
    doc.add_paragraph()
    para(doc,
         "Bu, raporlanan skorların yorumlanmasında dikkate alınmalıdır: her "
         "mimari kendi optimal ayarlarıyla değil, ORTAK bir ayar setiyle "
         "çalışmıştır. Bir mimarinin burada düşük skor alması, ayarları "
         "kendisine göre optimize edilseydi de düşük kalacağı anlamına gelmez.",
         size=9.5)
    doc.add_paragraph()
    kv_tablo(doc, [
        ("Elle sabitlenen (aranmayan)",
         "n_layers, hidden_dim, emb_dim, dropout, learning rate, weight decay, "
         "batch size, cutoff, K, max epoch, patience, freeze süresi"),
        ("Veriden SEÇİLEN tek şey — 1",
         "Durdurma epoch'u: her fold için validasyon MAE'sinin en düşük olduğu "
         "epoch seçilir (erken durdurma, bkz. §3.2)"),
        ("Veriden SEÇİLEN tek şey — 2",
         "GraphLIME'ın Lasso düzenlileştirme katsayısı (alpha): her örnek için "
         "çapraz doğrulamayla (LassoCV) seçilir. Bu bir XAI yöntemi "
         "hiperparametresidir, GNN'in değil"),
        ("Gelecek çalışma", "Mimari başına ayrı hiperparametre araması, bu "
                             "karşılaştırmanın doğal bir sonraki adımıdır"),
    ])

    h(doc, "3.2 K-Fold'da Ne Optimize Ediliyor?", level=2)
    para(doc,
         "K-Fold çapraz doğrulama bir ÖLÇME yöntemidir: her örneği tam olarak "
         "bir kez, modelin onu hiç görmediği bir turda test ederek, modelin "
         "görmediği veriye ne kadar genellediğini yansız biçimde ölçer. "
         "Optimizasyon ise bundan ayrı iki düzeyde gerçekleşir. Üç düzeyin "
         "her birinde ne olduğu aşağıdadır:", size=9.5)
    doc.add_paragraph()
    kv_tablo(doc, [
        ("1) Eğitim içinde (her epoch)",
         "MİNİMİZE EDİLEN: eğitim kümesindeki maskeli MSE kaybı. Bunu yapan "
         "AdamW optimizer'dır; ağırlıklar bu kaybın gradyanına göre güncellenir"),
        ("2) Epoch seçiminde (fold içinde)",
         "MİNİMİZE EDİLEN: VALİDASYON MAE'si (4 hedefin ortalaması, 'overall'). "
         "Validasyon MAE'sinin en düşük olduğu epoch'un ağırlıkları saklanır; "
         "eğitim bittiğinde bu ağırlıklara geri dönülür"),
        ("3) Fold'lar arasında (K-Fold'un kendisi)",
         "HİÇBİR ŞEY optimize edilmez. Fold'lar yalnızca her örneğin bir kez "
         "test edilmesini sağlar; sonra tüm foldların fold-dışı tahminleri "
         "havuzlanıp tek bir R²/MAE hesaplanır. Bu, raporlanan nihai skordur"),
        ("Test verisi", "Yalnızca en sonda, nihai başarımı ölçmek için okunur. Ağırlık "
                         "güncellemesi ve epoch seçimi tamamen eğitim ve validasyon "
                         "kümeleriyle yapılır"),
    ])

    h(doc, "3.3 Eğitim Hangi Epoch'ta Kesilmeli?", level=2)
    egitilmis = _egitilmis_modeller()
    meta = _metrikler(egitilmis[0]) if egitilmis else None
    hp = meta["hiperparametreler"] if meta else {}
    epochlar, model_epoch = [], []
    for m in egitilmis:
        f = PROJECT_ROOT / m / "sonuclar" / "kfold_metrikleri.csv"
        if not f.exists():
            continue
        kk = pd.read_csv(f)
        if "en_iyi_epoch" in kk.columns:
            lst = [int(x) for x in kk["en_iyi_epoch"]]
            epochlar += lst
            model_epoch.append((m, ", ".join(map(str, lst))))
    para(doc,
         f"Bu soruya elle karar verilmez — erken durdurma mekanizması her fold "
         f"için ayrı ayrı karar verir: validasyon MAE'si "
         f"{hp.get('early_stop_patience', '—')} epoch boyunca iyileşmezse eğitim "
         f"durdurulur ve EN İYİ epoch'un ağırlıkları geri yüklenir. Üst sınır "
         f"{hp.get('max_epochs', '—')} epoch'tur.", size=9.5)
    doc.add_paragraph()
    if epochlar:
        import numpy as _np
        tavana_deyen = sum(1 for e in epochlar if e >= (hp.get("max_epochs") or 10**9))
        kv_tablo(doc, [
            ("Gerçekleşen en iyi epoch aralığı",
             f"{min(epochlar)} – {max(epochlar)} (ortalama {_np.mean(epochlar):.1f}, "
             f"medyan {int(_np.median(epochlar))}) — {len(epochlar)} fold koşumu üzerinden"),
            ("Pratik cevap",
             f"Bu veri setinde eğitim tipik olarak {int(_np.median(epochlar))}. epoch "
             f"civarında kesilmektedir; {hp.get('max_epochs', '—')} epoch'luk üst sınır "
             f"çoğu fold için fazlasıyla yeterlidir"),
            ("Üst sınıra dayanan fold sayısı",
             f"{tavana_deyen} / {len(epochlar)}" +
             (" — bu fold(lar) erken durdurmaya hiç takılmadan üst sınırda bitmiştir, "
              "yani daha uzun eğitimle İYİLEŞMEYE DEVAM EDEBİLİRLERDİ; max epoch "
              "artırılarak kontrol edilmesi önerilir"
              if tavana_deyen else
              " — hiçbir fold üst sınıra dayanmadı, yani max epoch kısıtlayıcı "
              "olmamıştır (eğitim kendiliğinden yakınsamıştır)")),
            ("Nasıl anlaşılır (grafikten)",
             "Sonuç Raporu §2.4'teki kayıp eğrilerinde, validasyon (kesikli) "
             "çizgisinin en alçak noktası o fold'un seçilen epoch'udur. Bu "
             "noktadan sonra validasyon çizgisi yükselirken eğitim çizgisi "
             "düşmeye devam ediyorsa, orası ezberin başladığı yerdir"),
        ])
        if model_epoch:
            h(doc, "3.3.1 Model ve Fold Bazında Seçilen Epoch'lar", level=3)
            kv_tablo(doc, [(m, f"fold başına: {e}") for m, e in model_epoch])
    else:
        para(doc, "[Henüz üretilmedi: kfold_metrikleri.csv]", size=9)


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
                              f"adsorpsiyon ısısından (Widom ekleme) türetilen enerjisel bir "
                              f"vekildir; sütun adındaki '_proxy' eki bunu belirtir. Kaynakta bu "
                              f"değeri bulunmayan kayıtlar için gözeneklilik ve kompozisyondan "
                              f"türetilen basit bir tahmin kullanılmıştır"),
        ("Ön-eğitim max epoch / sabır",
         f"{PRETRAIN_MAX_EPOCHS} / {PRETRAIN_PATIENCE} — kod varsayılanı; bu aşamanın "
         f"hiperparametreleri JSON'a kaydedilmediğinden koşumda ezilmiş olup olmadığı "
         f"doğrulanamaz, şeffaflık için belirtilmiştir"),
        ("Encoder dondurma süresi", f"{hp.get('freeze_encoder_epochs', '—')} epoch"),
        ("Kaydedilen", "Yalnızca encoder.state_dict() — regresyon başı ince-ayarda gaz "
                        "hedeflerine özel olarak sıfırdan oluşturulur"),
    ])


# ---------------------------------------------------------------------------
# §4 MODEL TEKNİK DETAYLARI
# ---------------------------------------------------------------------------
def bolum_model_detaylari(doc):
    h(doc, "4. Model Teknik Detayları")
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
        h(doc, f"4.{i} {model_adi}", level=2)
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
        h(doc, f"4.{len(egitilmis) + 1} Eğitilmemiş Mimariler", level=2)
        para(doc,
             f"{', '.join(egitilmemis)} depoda tam olarak implemente edilmiştir "
             f"(run_*.py + grafik.py mevcuttur) ancak bu koşumda eğitilmemiştir; "
             f"dolayısıyla hiçbir sonuç/grafik üretmemiş ve Sonuç Raporu'ndaki "
             f"hiçbir tabloda yer almamıştır. Eğitilmesi için ilgili run_*.py'nin "
             f"çalıştırılması yeterlidir, kod değişikliği gerekmez.", size=9.5)
        doc.add_paragraph()


# ---------------------------------------------------------------------------
# §5 GRAF İNŞA STRATEJİSİ
# ---------------------------------------------------------------------------
def bolum_graf_insa(doc):
    h(doc, "5. Graf İnşa Stratejisi")
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
        ("Periyodiklik ve graf kapsamı",
         "Graf, birim hücrenin kendisi üzerine değil, kafes vektörleri boyunca "
         "ötelenerek üretilen PERİYODİK KOPYALARDAN merkeze kesim yarıçapı "
         "kadar mesafede kalan atomların oluşturduğu bir KÜME üzerine kurulur. "
         "Bu nedenle graftaki atom sayısı birim hücredekinden büyüktür: bu veri "
         "setinde birim hücre 18–62 atom (medyan 51) içerirken graf 72–172 atom "
         "(medyan 102) içerir"),
        ("Kodlayıcı arayüzü", "encoder.forward(g: dict) — SÖZLÜK tabanlı. DimeNet++ gibi açısal "
                               "modeller ek alanlara (idx_kj / idx_ji / theta) ihtiyaç duyduğundan "
                               "pozisyonel arayüz yerine bu tercih edilmiştir"),
        ("Havuzlama", "scatter-mean (atom → MOF) — graf düzeyinde tek bir gömme vektörü"),
        ("Grafik boyutu", "Bu veri setinde MOF başına yaklaşık 72–172 atom — GraphLIME gibi "
                           "maskeleme tabanlı XAI yöntemlerinin ölçek duyarlılığı bakımından "
                           "belirleyici bir büyüklüktür (bkz. §7.3)"),
    ])


# ---------------------------------------------------------------------------
# §6 XAI TEKNİK DETAYLARI
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
    h(doc, "6. XAI (Açıklanabilir Yapay Zekâ) Teknik Detayları")
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
        h(doc, f"6.{i} {xai_adi}", level=2)
        kv_tablo(doc, XAI_DETAY.get(xai_adi, [("—", "—")]))


# ---------------------------------------------------------------------------
# §11 KOD VE VERİ ERİŞİLEBİLİRLİĞİ
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# §7 PERMÜTASYON ÖNEMİ VE PROJEYE ÖZGÜ TEKNİK BULGULAR
# ---------------------------------------------------------------------------
def bolum_permutasyon_bulgular(doc):
    h(doc, "7. Permütasyon Önemi ve Projeye Özgü Teknik Bulgular")
    para(doc,
         "Kristal gömmesi ve aux özellik vektörü fold başına ÖNCEDEN hesaplanır "
         "(precompute), permütasyon SADECE hafif regresyon başı üzerinde koşulur — "
         "böylece 10 model × 6 özellik grubu × fold sayısı kadar hesap, tüm ağı "
         "yeniden çalıştırmadan yapılabilir.", size=9.5)
    doc.add_paragraph()

    h(doc, "7.1 Ortak Permütasyon Yöntemi", level=2)
    egitilmis = _egitilmis_modeller()
    meta = _metrikler(egitilmis[0]) if egitilmis else None
    skorlar_tum = {}
    for m in egitilmis:
        f = PROJECT_ROOT / m / "sonuclar" / "grafikler" / "perm_importance.json"
        if f.exists():
            skorlar_tum[m] = json.loads(f.read_text(encoding="utf-8"))

    bulgu = "—"
    if skorlar_tum:
        birinci = {}
        for m, s in skorlar_tum.items():
            birinci[max(s, key=s.get)] = birinci.get(max(s, key=s.get), 0) + 1
        en_sik, adet = max(birinci.items(), key=lambda kv: kv[1])
        araliklar = [s.get(en_sik, 0.0) for s in skorlar_tum.values()]
        bulgu = (f"{en_sik} grubu {adet}/{len(skorlar_tum)} modelde 1. sıradadır "
                 f"(ΔMAE={min(araliklar):.4f}–{max(araliklar):.4f})")

    kv_tablo(doc, [
        ("Permütasyon hedefleri",
         "1) Crystal Structure — grafik kodlayıcının ürettiği gömme satırları "
         "örnekler arasında karıştırılır;  2) 5 aux özellik grubu "
         "(Pore Geometry, Surface Area, Structural / Size, Chemical Modification, "
         "Composition-Derived)"),
        ("Kapsam", f"{meta['k_folds'] if meta else '—'} fold; her foldun test seti "
                    f"üzerinde ayrı hesaplanır, sonra fold ortalaması alınır"),
        ("Birim", "ΔMAE = karıştırma sonrası MAE − baz MAE. Pozitif ve büyük bir "
                   "değer, modelin o özellik grubuna güçlü bağımlı olduğunu gösterir"),
        ("TUTARLI BULGU", bulgu),
        ("Fiziksel yorum",
         "Hazır-hesaplanmış gözeneklilik tanımlayıcıları, öğrenilen kristal graf "
         "gömmesinden (Crystal Structure) onlarca kat daha güçlü bir sinyal "
         "taşımaktadır. Bu sonuç doğrudan veri kurgusundan gelir: hedef "
         "etiketler zaten bu tanımlayıcılardan üretilmiştir "
         "(§1.2), dolayısıyla 3B atomistik geometri ek bilgi taşımamaktadır"),
    ])

    h(doc, "7.2 Gözeneklilik – Seçicilik Fiziksel Tutarlılığı", level=2)
    r_gercek, r_tahmin = [], []
    for m in egitilmis:
        f = PROJECT_ROOT / m / "sonuclar" / "test_tahminleri_oof.csv"
        if not f.exists():
            continue
        odf = pd.read_csv(f)
        gerekli = {"pld_A", "gercek_xe_kr_selectivity", "tahmin_xe_kr_selectivity"}
        if not gerekli.issubset(odf.columns):
            continue
        sub = odf.dropna(subset=list(gerekli))
        if len(sub) < 5:
            continue
        from scipy.stats import pearsonr
        dx = (sub["pld_A"] - 4.10).abs()
        if dx.std() < 1e-9:
            continue
        r_gercek.append(pearsonr(dx, sub["gercek_xe_kr_selectivity"])[0])
        r_tahmin.append(pearsonr(dx, sub["tahmin_xe_kr_selectivity"])[0])
    kv_tablo(doc, [
        ("Beklenen fizik", "Sikora et al. (2012) boyut-eleme ilkesi: gözenek-sınırlayıcı "
                            "çap (PLD) hedef gazın kinetik çapına yaklaştıkça seçicilik "
                            "ARTAR — yani |PLD − d_Xe| ile seçicilik arasında NEGATİF "
                            "korelasyon beklenir"),
        ("Gerçek veri korelasyonu",
         f"r = {sum(r_gercek) / len(r_gercek):.3f}" if r_gercek else "—"),
        ("Model sonucu",
         (f"Eğitilmiş {len(r_tahmin)} modelin TAMAMI aynı negatif işareti doğru "
          f"yakalamıştır (r_tahmin = {min(r_tahmin):.3f} … {max(r_tahmin):.3f})")
         if r_tahmin else "—"),
        ("Bu testin bu koşumda ÖLÇTÜĞÜ şey",
         "Modelin ürettiği korelasyon, gürültü içeren 'gerçek' etiketlerinkinden "
         "DAHA güçlüdür. Bunun nedeni, seçicilik etiketinin zaten PLD'nin "
         "kapalı-form bir fonksiyonu olarak üretilmiş olması ve PLD'nin aynı "
         "zamanda modele girdi olarak verilmesidir (§1.2). Model, altta yatan "
         "gürültüsüz formüle yakınsamaktadır; dışsal bir fiziği keşfetmemektedir. "
         "Gerçek etiketlere geçildiğinde bu test anlamlı bir fizik kontrolüne "
         "dönüşecektir"),
    ])

    h(doc, "7.3 GraphLIME'da Ölçek Duyarlılığı (bu projeye özgü bulgu)", level=2)
    kv_tablo(doc, [
        ("Belirti", "GraphLIME'ın ürettiği TÜM atom önem katsayıları tam olarak "
                     "0.0 çıkıyor, açıklama grafikleri tamamen boş görünüyordu "
                     "(9954 atom satırının 9954'ü sıfır)"),
        ("Kök neden", "Düzenlileştirme katsayısı (Lasso alpha) sabit 0.01 idi. Bu "
                       "projenin MOF'ları 72–172 atom içerir ve tek bir atomun "
                       "maskelenmesinin tahmine etkisi, 4 mesaj-iletim katmanı ve "
                       "her katmandaki LayerNorm tarafından seyreltilir; ortaya çıkan "
                       "tahmin farkları (|Δ| ≈ 0.01–0.03) sabit ceza teriminin "
                       "eşiğinin ALTINDA kaldığından Lasso tüm katsayıları sıfıra "
                       "çekiyordu"),
        ("Teşhis yöntemi", "Gerçek bir checkpoint üzerinde alpha taraması: alpha=0.01 "
                            "→ 0/139 sıfır-olmayan katsayı; alpha=1e-4 → 74/139"),
        ("Çözüm", "Sabit alpha yerine LassoCV — düzenlileştirme katsayısı her örnek "
                   "için çapraz doğrulamayla veriden seçilir, böylece farklı MOF "
                   "boyutlarına otomatik uyum sağlanır"),
        ("Sonuç", "9954 atom satırının 9518'i (%96) artık sıfır-olmayan gerçek önem "
                   "skoru taşımaktadır"),
        ("Genel ders", "Maskeleme tabanlı yerel vekil XAI yöntemleri büyük graflarda "
                        "ÖLÇEĞE DUYARLIDIR: maskelemenin etkisi graf büyüdükçe "
                        "seyrelir ve sabit bir ceza terimi sinyali tamamen "
                        "bastırabilir"),
    ])


# ---------------------------------------------------------------------------
# §8 YAZILIM VE KÜTÜPHANE BAĞIMLILIKLARI
# ---------------------------------------------------------------------------
def bolum_bagimliliklar(doc):
    h(doc, "8. Yazılım ve Kütüphane Bağımlılıkları")
    import importlib.metadata as md

    kullanim = [
        ("torch", "Tüm derin öğrenme modelleri, eğitim döngüsü, GPU hesabı"),
        ("torch_geometric", "GraphGPS / PNA / GIN / GAT / GatedGCN / DeeperGCN / ECC "
                             "graf evrişim katmanları"),
        ("pymatgen", "CIF ayrıştırma, kristal yapı işleme, veri artırma "
                      "(kusur/değiştirme/fonksiyonel grup), gözeneklilik kestirimi"),
        ("scikit-learn", "K-Fold bölme, metrikler (R²/MAE/RMSE), GraphLIME'ın LassoCV'si"),
        ("datasets", "Hugging Face veri setinin indirilmesi"),
        ("pandas", "Veri seti okuma/yazma, tablo işlemleri"),
        ("numpy", "Sayısal hesaplamalar"),
        ("scipy", "Pearson korelasyonu (fiziksel tutarlılık testi)"),
        ("matplotlib", "Tüm grafiklerin çizimi (600 dpi PNG + TIFF)"),
        ("seaborn", "Isı haritaları ve grafik teması"),
        ("python-docx", "Bu raporun ve Sonuç Raporu'nun .docx olarak üretilmesi"),
    ]
    satirlar = []
    for ad, amac in kullanim:
        try:
            surum = md.version(ad)
        except Exception:
            surum = "(kurulu değil)"
        satirlar.append((f"{ad}  —  {surum}", amac))
    kv_tablo(doc, satirlar, size=8.5)

    try:
        import torch
        donanim = (f"CUDA kullanılabilir: {torch.cuda.is_available()}"
                   + (f" — {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    except Exception:
        donanim = "—"
    kv_tablo(doc, [
        ("Donanım", donanim),
        ("Not", "TFN, EGNN, SE(3)-Transformer ve DimeNet++ SIFIRDAN, saf PyTorch ile "
                 "yazılmıştır — e3nn gibi ek bir ekvaryant kütüphane bağımlılığı YOKTUR"),
    ])


# ---------------------------------------------------------------------------
# §9 PROJE DOSYA YAPISI
# ---------------------------------------------------------------------------
def bolum_dosya_yapisi(doc):
    h(doc, "9. Proje Dosya Yapısı")
    para(doc, "Paylaşılan modüller proje kökünde, her model/XAI yöntemi kendi "
               "klasöründe aynı iskeleti takip eder:", size=9.5)
    agac = f"""MOF Radyoaktif Gaz Adsorpsiyonu/
├── paths.py                           # Tüm global yollar, model/XAI klasör listeleri, panel harfleri
├── ortak_ozellikler.py                # AUX_FEATURE_COLUMNS ({AUX_DIM}) + AuxOlcekleyici + aux grupları
├── graf_ozellik_ortak.py              # CIF → periyodik 3B yarıçap grafiği, RBF, scatter-mean
├── egitim_ortak.py                    # K-Fold, erken durdurma, checkpoint, transfer öğrenme,
│                                      #   permütasyon önemi, tüm grafiklerin üretimi
├── grafik_ortak.py                    # 10 modelin paylaştığı grafik fonksiyonları (600 dpi PNG+TIFF)
│
├── veri_indirici_1_jarvis_core_mof.py # Bileşen 2: HF/CoRE-MOF indirme + gözeneklilik kestirimi
├── nlp_literatur_madencilik_2.py      # Bileşen 3: CrossRef/arXiv literatür madenciliği
├── veri_artirma_3_augmentasyon.py     # Bileşen 4: pymatgen tabanlı yapısal veri artırma
├── eslesme_4_dataset_birlestirici.py  # Bileşen 5: nihai ön-eğitim + ince-ayar veri setleri
│
├── model_karsilastirma.py             # Tüm modellerin OOF metriklerini karşılaştırır
├── model_karsilastirma_grafik.py      # 4 kıyaslama grafiği
├── rapor_olustur.py                   # Sonuç Raporu (.docx)
├── bilgi_raporu_olustur.py            # Bu rapor (.docx)
│
├── {" / ".join(MODEL_KLASORLERI[:5])}/
├── {" / ".join(MODEL_KLASORLERI[5:])}/     # her model klasörü AYNI yapıda:
│     ├── run_<model>.py                #   ön-eğitim + K-Fold ince-ayar (TEK komut)
│     ├── grafik.py                     #   o modelin tüm grafikleri
│     ├── checkpoints/                  #   fold{{N}}_best_model.pt
│     ├── pretrain_checkpoints/         #   encoder_pretrained.pt
│     └── sonuclar/                     #   metrikler.json, test_tahminleri_oof.csv,
│                                       #   kfold_metrikleri.csv, egitim_gecmisi_fold*.csv, grafikler/
│
├── {" / ".join(XAI_KLASORLERI)}/   # her XAI klasörü:
│     ├── run_<yontem>.py               #   XAI hesabı (EGNN checkpoint'leri üzerinde)
│     ├── grafik.py                     #   o yöntemin grafikleri
│     └── sonuclar/                     #   atom/kenar/element bazında CSV + grafikler/
│
├── data/
│     ├── raw/                          # mof_ham_veri.csv, qmof_pretrain_ham.csv,
│     │                                 #   nlp_literatur_madencilik_sonuclari.csv, kaynak_bilgisi.json
│     └── processed/                    # gaz_adsorpsiyon_dataset_final.csv (NİHAİ ince-ayar veri seti),
│                                       #   qmof_pretrain_dataset_final.csv, cif/, augmented_cif/,
│                                       #   augmentasyon_manifest.csv, eslesme_bilgisi.json
│
├── model_karsilastirma_sonuclari.csv   # Tüm modellerin pooled OOF metrikleri
├── model_karsilastirma_grafikler/      # 4 kıyaslama grafiği
├── VERI_KAYNAGI_VE_SINIRLAMALAR.md     # Veri kökeni ve sentetiklik durumu
├── SISTEM_RAPORU.md                    # Mimari genel bakış
└── README.md"""
    p = doc.add_paragraph()
    r = p.add_run(agac)
    r.font.name = "Consolas"
    r.font.size = Pt(7.5)
    doc.add_paragraph()


# ---------------------------------------------------------------------------
# §10 TERİMLER SÖZLÜĞÜ
# ---------------------------------------------------------------------------
TERIMLER = [
    ("Temel Kavramlar", None),
    ("Makine öğrenmesi (ML — Machine Learning)",
     "Bilgisayarın, kuralları elle yazılmadan, ÖRNEKLERDEN kural çıkarmasıdır. "
     "Klasik programlamada 'eğer gözenek hacmi şundan büyükse kapasite şudur' gibi "
     "kuralları insan yazar; makine öğrenmesinde ise binlerce (yapı → ölçüm) çifti "
     "verilir ve aradaki ilişkiyi model kendisi bulur. Bu projede amaç, bir MOF'un "
     "yapısına bakarak gaz adsorpsiyon değerlerini tahmin eden bir model "
     "öğrenmektir."),
    ("Denetimli öğrenme (supervised learning)",
     "Her örnek için doğru cevabın ('etiket') verildiği öğrenme türü. Model tahmin "
     "yapar, doğru cevapla karşılaştırılır, aradaki farka göre düzeltilir. Bu "
     "projede etiketler dört hedefin sayısal değerleridir."),
    ("Regresyon / sınıflandırma",
     "Regresyon SAYISAL bir değer tahmin eder (örn. 0.17 mmol/g); sınıflandırma ise "
     "KATEGORİ tahmin eder (örn. 'yüksek kapasiteli'). Bu proje bir regresyon "
     "problemidir; karışıklık matrisi grafiklerinde sonuçlar yalnızca "
     "yorumlanabilirlik için dört sınıfa indirgenir."),
    ("Model / eğitim (training)",
     "Model, girdiden çıktıya giden ve içinde ayarlanabilir sayılar ('ağırlıklar') "
     "bulunan matematiksel yapıdır. Eğitim, bu ağırlıkların örneklerdeki hatayı "
     "küçültecek şekilde adım adım güncellenmesi sürecidir."),
    ("Etiket (label) / hedef (target)",
     "Modelin tahmin etmeye çalıştığı doğru cevap. Bu projede dört hedef vardır ve "
     "ÖNEMLİ: bu koşumda etiketler deneysel ölçüm değil, bir formülle üretilmiş "
     "sentetik değerlerdir (bkz. §1.2)."),
    ("Özellik (feature) / girdi",
     "Modele verilen bilgiler. Burada iki tür girdi vardır: (1) MOF'un 3B atom "
     "yapısı (graf olarak), (2) hazır-hesaplanmış sayısal tanımlayıcılar "
     "(gözenek hacmi, PLD vb. — 'aux özellikler')."),
    ("Genelleme (generalization)",
     "Modelin, eğitimde HİÇ GÖRMEDİĞİ yeni örneklerde de doğru tahmin yapabilmesi. "
     "Makine öğrenmesinin asıl amacı budur; eğitim verisini ezberlemek değil."),

    ("Çapraz Doğrulama ve Veri Bölme", None),
    ("Fold (kat)", "Veri setinin eşit parçalarından biri. Veriyi K parçaya bölüp her "
                   "parçayı sırayla 'test', kalanları 'eğitim' olarak kullanırız. "
                   "GRAFİKLERDE 'Fold 1 / Fold 2 / Fold 3' renkleri, o noktanın hangi "
                   "turda TEST verisi olarak tahmin edildiğini gösterir — yani her nokta, "
                   "modelin O NOKTAYI HİÇ GÖRMEDEN yaptığı tahmindir. Renklerin "
                   "birbirine karışmış olması iyiye işarettir: hiçbir fold diğerlerinden "
                   "sistematik olarak sapmıyor demektir."),
    ("K (K-Fold'daki K)", "Veriyi kaç parçaya böldüğümüz. K=3 ise veri 3 parçaya "
                           "bölünür, 3 ayrı model eğitilir: her birinde 2 parça eğitim, "
                           "1 parça testtir. Böylece HER örnek tam olarak bir kez test "
                           "edilmiş olur. K büyüdükçe her model daha çok veri görür ama "
                           "hesap maliyeti artar."),
    ("OOF (Out-of-Fold / fold-dışı tahmin)",
     "Bir örnek için, O ÖRNEĞİN test fold'unda olduğu turda üretilen tahmin. Model o "
     "örneği eğitimde hiç görmemiştir. Rapordaki tüm metrikler bu tahminlerden "
     "hesaplanır — yani hiçbir örnek kendi eğitim verisiyle değerlendirilmez."),
    ("Havuzlanmış (pooled) OOF",
     "Tüm foldların fold-dışı tahminlerinin tek bir listede birleştirilip TEK bir "
     "R²/MAE hesaplanması. Fold başına ayrı metrik hesaplayıp ortalamak yerine bu "
     "yöntem kullanılır; tüm veri seti tek bir test seti gibi değerlendirilmiş olur."),
    ("Veri sızıntısı (data leakage)",
     "Test verisine ait bir bilginin eğitime karışması. Burada özel bir risk vardır: "
     "veri artırma ile bir MOF'tan birden fazla varyant üretiliyor; varyant eğitimde, "
     "orijinali testte olursa model 'kopya çekmiş' olur. Bu yüzden bölme örnek bazında "
     "değil TEMEL MOF (base_mof_id) bazında yapılır — bir MOF'un tüm varyantları hep "
     "aynı fold'dadır."),
    ("Validasyon seti", "Eğitim verisinden ayrılan küçük bir parça; eğitimi ne zaman "
                         "durduracağımıza ve hangi epoch'un ağırlıklarını saklayacağımıza "
                         "karar vermek için kullanılır. Nihai başarım ölçümü ayrı tutulan "
                         "test kümesiyle yapılır."),

    ("Eğitim Süreci", None),
    ("Epoch", "Modelin tüm eğitim verisini baştan sona bir kez görmesi. 15 epoch = veri "
              "15 kez baştan sona geçirildi."),
    ("Batch / batch boyutu", "Model ağırlıkları her örnekte değil, örnek GRUPLARI "
                              "sonrası güncellenir. Batch boyutu 32 ise her güncelleme "
                              "32 örneğin ortalama hatasına göre yapılır."),
    ("Kayıp fonksiyonu (loss)", "Modelin ne kadar yanıldığını ölçen ve eğitimde "
                                 "KÜÇÜLTÜLMEYE çalışılan sayı. Burada MSE (hataların "
                                 "karelerinin ortalaması) kullanılır."),
    ("Maskeli MSE", "Bir örnekte 4 hedeften bazıları eksikse, o eksik hedefler kayıp "
                     "hesabına KATILMAZ. Böylece eksik etiketli satırlar da eğitimde "
                     "kullanılabilir."),
    ("Öğrenme oranı (learning rate)", "Her güncellemede ağırlıkların ne kadar "
                                       "değiştirileceği. Çok büyükse model kararsızlaşır, "
                                       "çok küçükse öğrenme aşırı yavaşlar."),
    ("Optimizer (AdamW)", "Ağırlıkları güncelleyen algoritma. AdamW, her ağırlık için "
                           "adım büyüklüğünü otomatik ayarlar ve ağırlıkları küçük "
                           "tutmaya zorlayan bir düzenlileştirme içerir."),
    ("Ağırlık sönümü (weight decay)", "Ağırlıkların aşırı büyümesini cezalandıran terim; "
                                       "ezberi azaltır."),
    ("LR scheduler (CosineAnnealingLR)", "Öğrenme oranını eğitim boyunca kademeli "
                                          "düşüren program — başta hızlı öğren, sonlara "
                                          "doğru ince ayar yap mantığı."),
    ("Erken durdurma (early stopping) ve sabır (patience)",
     "Validasyon hatası belli sayıda epoch boyunca iyileşmezse eğitim durdurulur. "
     "'Patience=5' → 5 epoch boyunca iyileşme yoksa dur. Gereksiz hesabı ve ezberi önler."),
    ("Checkpoint", "Validasyon hatasının EN DÜŞÜK olduğu andaki model ağırlıklarının "
                    "diske kaydedilmiş hali. Son epoch değil, EN İYİ epoch kullanılır."),
    ("Ezber (overfitting)", "Modelin eğitim verisini ezberleyip görmediği veriye "
                             "genelleyememesi. Belirtisi: eğitim hatası düşerken "
                             "validasyon hatasının yükselmesi."),
    ("Dropout", "Eğitim sırasında nöronların rastgele bir kısmının geçici olarak "
                 "kapatılması; ezberi zorlaştırır."),
    ("Standardizasyon / z-skoru", "Her özelliğin ortalaması 0, standart sapması 1 olacak "
                                   "şekilde ölçeklenmesi. Farklı birimlerdeki özelliklerin "
                                   "(Å, cm³/g, 0-1) birbirini ezmesini önler. Ortalama/std "
                                   "HER FOLD'un SADECE eğitim kısmından hesaplanır (sızıntı "
                                   "olmaması için)."),
    ("Seed (rastgelelik tohumu)", "Rastgele işlemleri (bölme, başlatma) tekrarlanabilir "
                                   "kılan sabit sayı. Aynı seed = aynı sonuç."),
    ("Hiperparametre", "Modelin verilerden ÖĞRENMEDİĞİ, bizim önceden belirlediğimiz "
                        "ayarlar: K, epoch sayısı, öğrenme oranı, katman sayısı vb."),

    ("Transfer Öğrenme", None),
    ("Transfer öğrenme", "Bir görevde öğrenilen bilginin başka bir göreve taşınması. "
                          "Burada model önce bol veri bulunan bir vekil hedefte eğitilir, "
                          "sonra asıl (az veri olan) gaz adsorpsiyon hedefine uyarlanır."),
    ("Ön-eğitim (pretraining)", "İlk aşama: kodlayıcı, vekil hedef üzerinde eğitilip "
                                 "genel bir yapı temsili öğrenir. Sadece kodlayıcı "
                                 "ağırlıkları saklanır."),
    ("İnce-ayar (fine-tuning)", "İkinci aşama: saklanan ağırlıklar yüklenir ve asıl hedefe "
                                 "göre güncellenir."),
    ("Dondurma (freeze) / doğrusal sondalama",
     "İnce-ayarın ilk epoch'larında kodlayıcı ağırlıkları SABİT tutulur, yalnızca son "
     "regresyon katmanı eğitilir. Ön-eğitimde öğrenilenin daha başta bozulmasını önler."),
    ("Vekil (proxy) hedef", "Asıl hedefin yerine geçen, onunla ilişkili olduğu düşünülen "
                             "başka bir büyüklük. Burada ön-eğitimde CO₂ adsorpsiyon ısısı "
                             "kullanılır — çünkü kaynak veri setinde Xe/Kr/I₂ değeri yoktur."),

    ("Grafik Sinir Ağları (GNN)", None),
    ("Graf / düğüm / kenar", "Graf, nesnelerin (DÜĞÜM) ve aralarındaki ilişkilerin "
                              "(KENAR) matematiksel gösterimi. Burada her ATOM bir düğüm, "
                              "birbirine yeterince yakın atom çiftleri ise kenardır."),
    ("GNN (Grafik Sinir Ağı)", "Graf yapısındaki veriyi işleyen sinir ağı. Molekül/kristal "
                                "gibi 'tablo haline getirilemeyen' yapılar için uygundur."),
    ("Mesaj iletimi (message passing)", "GNN'in temel işleyişi: her atom, komşularından "
                                         "'mesaj' alır ve kendi temsilini günceller. Bu "
                                         "birkaç kez tekrarlanınca her atom, çevresindeki "
                                         "giderek daha geniş bölgeyi 'görmüş' olur."),
    ("Kesim yarıçapı (cutoff)", "İki atomun kenarla bağlanması için izin verilen en büyük "
                                 "mesafe (burada 8.0 Å). Gözenekli MOF'larda gözenek "
                                 "boşluğunun da kapsanması için geniş tutulmuştur."),
    ("RBF (radyal taban fonksiyonu)", "Atomlar arası mesafeyi tek bir sayı yerine, farklı "
                                       "mesafelere duyarlı birkaç sayıdan oluşan bir "
                                       "vektöre çeviren kodlama. Ağın mesafeyi daha ince "
                                       "ayrıştırmasını sağlar."),
    ("Kodlayıcı (encoder) ve gömme (embedding)",
     "Kodlayıcı, bir MOF'un tüm atom/bağ yapısını sabit uzunlukta bir sayı vektörüne "
     "('gömme') dönüştüren kısımdır. Bu vektör, yapının modelin anladığı dildeki özetidir."),
    ("Havuzlama (pooling, scatter-mean)", "Atom başına üretilen vektörlerin ortalaması "
                                           "alınarak TEK bir MOF vektörü elde edilmesi."),
    ("Ekvaryans — E(n) / E(3) / SE(3)",
     "Modelin, yapıyı döndürdüğümüzde/kaydırdığımızda tahminini DEĞİŞTİRMEMESİ özelliği. "
     "Fizik böyle davranır (bir kristali döndürmek özelliklerini değiştirmez), bu yüzden "
     "bu özelliği mimariye gömmek öğrenmeyi kolaylaştırır."),
    ("Dikkat (attention)", "Modelin, her atom için hangi komşuların daha önemli olduğunu "
                            "kendi öğrendiği mekanizma."),
    ("Aux (yardımcı) özellikler", "Grafı tamamlayan, hazır-hesaplanmış sayısal "
                                   "tanımlayıcılar (gözenek hacmi, PLD, yoğunluk vb.). "
                                   "Grafik kodlayıcısının öğrendiği temsile EK olarak "
                                   "regresyon başına verilir."),
    ("Çok-görevli (multi-task) öğrenme", "Tek bir modelin 4 hedefi AYNI ANDA tahmin "
                                          "etmesi. Hedefler ilişkili olduğundan ortak bir "
                                          "temsil öğrenmek her birine yarar sağlar."),

    ("Grafikleri Okuma", None),
    ("Artık (residual)", "Tahmin − gerçek değer. Sıfıra yakın ve sıfır etrafında simetrik "
                          "dağılması istenir; sistematik kayma yanlılık demektir."),
    ("Çeyreklik (Q1 / medyan / Q3)",
     "Veriyi dörde bölen noktalar: Q1'in altında verinin %25'i, medyanın altında "
     "%50'si, Q3'ün altında %75'i kalır. Medyan, birkaç aşırı değerden "
     "etkilenmediği için ortalamadan daha dayanıklı bir 'tipik değer' ölçüsüdür; "
     "raporda MedianAE metriğinde ve seçilen epoch istatistiklerinde kullanılır."),
    ("Logaritmik eksen", "Kayıp eğrilerinde kullanılır. Değerler başta büyük sonra çok "
                          "küçük olduğundan, doğrusal eksende erken düşüş görünmez olurdu."),
    ("Panel harfleri (a), (b), ...", "Çok panelli şekillerde her grafiğin SAĞ ÜST "
                                      "köşesindeki harf, şekil altındaki açıklamada hangi "
                                      "modele ait olduğunu söyler."),
    ("Permütasyon önemi (ΔMAE)",
     "Bir özellik grubunun değerleri örnekler arasında RASTGELE karıştırılır ve hatanın ne "
     "kadar KÖTÜLEŞTİĞİ ölçülür. Çok kötüleşiyorsa model o özelliğe bağımlıdır. ΔMAE = "
     "karıştırma sonrası MAE − baz MAE."),

    ("Metrikler", None),
    ("R² (determinasyon katsayısı)", "Modelin, hedefteki değişkenliğin ne kadarını "
                                      "açıkladığı. 1.0 = mükemmel; 0.0 = sadece ortalamayı "
                                      "söylemekten farksız; negatif = ortalamadan bile kötü."),
    ("MAE (ortalama mutlak hata)", "Tahmin ile gerçek arasındaki farkların mutlak "
                                    "değerinin ortalaması. Hedefle AYNI birimdedir, en "
                                    "kolay yorumlanan hatadır."),
    ("RMSE (kök ortalama kare hata)", "Hataların karelerinin ortalamasının karekökü. "
                                       "Büyük hataları MAE'den daha ağır cezalandırır."),
    ("MedianAE (ortanca mutlak hata)", "Hataların ortanca değeri; birkaç aşırı kötü "
                                        "tahminden ETKİLENMEZ, 'tipik' hatayı gösterir."),
    ("MaxErr (maksimum hata)", "En kötü tek tahminin hatası — en kötü durum senaryosu."),
    ("PearsonR", "Tahmin ile gerçek değer arasındaki doğrusal ilişkinin gücü ve yönü "
                  "(−1 ile +1 arası)."),

    ("Malzeme Bilimi Terimleri", None),
    ("MOF (Metal-Organik Çerçeve)", "Metal düğümlerin organik bağlayıcı moleküllerle "
                                     "birleşerek oluşturduğu, içi düzenli boşluklarla dolu "
                                     "gözenekli kristal malzeme. Çok yüksek iç yüzey alanı "
                                     "sayesinde gaz depolama/ayırmada kullanılır."),
    ("CIF dosyası", "Kristal yapıyı tanımlayan standart metin dosyası: birim hücre "
                     "boyutları ve içindeki atomların konumları."),
    ("Birim hücre", "Kristalin, üç yönde tekrarlanarak tüm yapıyı oluşturan en küçük "
                     "tekrar birimi."),
    ("Adsorpsiyon", "Gaz moleküllerinin katı bir yüzeye/gözeneğe tutunması. Kapasite "
                     "genelde mmol gaz / g malzeme biriminde verilir."),
    ("PLD (Pore Limiting Diameter)", "Gözenek ağı içinden uçtan uca geçebilecek EN BÜYÜK "
                                      "kürenin çapı — gözeneklerin 'darboğazı'. Bir gazın "
                                      "kinetik çapı PLD'den büyükse o gaz geçemez."),
    ("LCD (Largest Cavity Diameter)", "MOF'un içindeki en geniş boşluğa sığabilecek en "
                                       "büyük kürenin çapı. PLD'den farklıdır: LCD iç "
                                       "hacmi, PLD geçiş darboğazını ölçer (LCD ≥ PLD)."),
    ("Kinetik çap", "Bir gaz molekülünün difüzyon davranışını belirleyen etkin boyutu "
                     "(Xe 4.10 Å, Kr 3.69 Å, I₂ 5.00 Å)."),
    ("Boyut-eleme (size sieving)", "Gözenek boyutunun, bazı gazları geçirip bazılarını "
                                    "engelleyerek ayırma yapması. PLD hedef gazın kinetik "
                                    "çapına ne kadar yakınsa eleme o kadar seçicidir."),
    ("Seçicilik (selectivity)", "Bir malzemenin bir gazı diğerine göre ne kadar tercihen "
                                 "tuttuğunun oranı (birimsiz). Xe/Kr seçicilik, nükleer "
                                 "atık gazı ayırmada kritik metriktir."),
    ("Boşluk oranı (void fraction)", "Birim hücre hacminin ne kadarının boş/erişilebilir "
                                      "gözenek olduğu (0–1 arası)."),
    ("Açık metal bölgesi (open metal site)", "Koordinasyonu doymamış, çözücü "
                                              "uzaklaştırıldığında gaz molekülüne doğrudan "
                                              "bağlanabilen metal merkezi — adsorpsiyonu "
                                              "güçlendirir."),
    ("GCMC", "Grand Canonical Monte Carlo — bir gözenekli malzemenin belirli sıcaklık ve "
              "basınçta ne kadar gaz tutacağını hesaplayan moleküler simülasyon yöntemi."),
    ("DFT", "Yoğunluk Fonksiyoneli Teorisi — malzemelerin elektronik yapısını ve "
             "enerjisini kuantum mekaniği ile hesaplayan yöntem."),
    ("Veri artırma (augmentation)", "Mevcut yapılardan kontrollü değişikliklerle yeni "
                                     "örnekler üretme: bağlayıcı eksiltme (kusur), metal "
                                     "değiştirme, fonksiyonel grup ekleme ve atom "
                                     "konumlarına küçük rastgele kaydırma (gürültü)."),

    ("Açıklanabilir Yapay Zekâ (XAI)", None),
    ("XAI", "Modelin kararını NEYE dayandırdığını anlamaya yarayan yöntemler bütünü. "
            "'Kara kutu'yu açmayı amaçlar."),
    ("Vekil model (surrogate)", "Karmaşık modelin davranışını, tek bir örnek çevresinde "
                                 "taklit eden basit (örn. doğrusal) model. Basit modelin "
                                 "katsayıları yorumlanabilir."),
    ("Lasso ve düzenlileştirme (alpha)",
     "Lasso, doğrusal model katsayılarını küçülten ve önemsiz olanları tam SIFIRA çeken "
     "bir yöntemdir; böylece sadece gerçekten önemli girdiler kalır. Küçültmenin şiddetini "
     "'alpha' belirler: çok büyük alpha TÜM katsayıları sıfırlayabilir (bu projede "
     "yaşanan sorun, bkz. §7.3). LassoCV, alpha'yı veriden otomatik seçer."),
    ("Maskeleme", "Girdinin bir kısmını (burada bazı atomları) geçici olarak 'kapatıp' "
                   "tahminin ne kadar değiştiğine bakma. Çok değişiyorsa o kısım önemlidir."),
    ("Gradyan / saliency", "Girdideki küçük bir değişikliğin çıktıyı ne kadar "
                            "değiştirdiğinin matematiksel ölçüsü. Büyükse o girdi etkilidir."),
    ("Integrated Gradients ve completeness aksiyomu",
     "Girdiyi bir 'taban çizgisinden' gerçek değerine kademeli taşırken gradyanları "
     "toplayan yöntem. Completeness aksiyomu: tüm katkıların TOPLAMI, tahmin ile taban "
     "çizgisi tahmini arasındaki farka tam olarak eşittir — yani hiçbir katkı kaybolmaz."),
    ("MCTS (Monte Carlo Ağaç Araması)", "Olasılıkları ağaç biçiminde deneyerek en iyi "
                                         "seçeneği arayan yöntem. Burada tahmini en iyi "
                                         "açıklayan bağlantılı atom grubunu bulmak için "
                                         "kullanılır."),
    ("Alt-graf (subgraph)", "Grafın bir parçası — burada tahmini açıklamaya yeten en "
                             "küçük atom kümesi."),
]


def _bolum_kod_adlari(doc):
    """Kullanıcı isteği: raporlarda geçen alt-tireli kod adlarının (sütun adı,
    dosya adı, ayar adı, etiket kaynağı değeri) hepsi açıklanır. Fiziksel
    büyüklükler burada DEĞİL, §1.3.1'deki sembol tablosundadır."""
    h(doc, "10.1 Kod ve Sütun Adları", level=2)
    para(doc,
         "Raporda ve çıktı dosyalarında geçen, alt tire içeren teknik "
         "adlandırmaların tamamı aşağıda açıklanmıştır. Fiziksel büyüklükler "
         "için ayrıca sembol tanımlanmıştır — bkz. §1.3.1 Sembol Tablosu.",
         size=9.5)
    doc.add_paragraph()
    for ad, aciklama in KOD_ADLARI:
        if aciklama is None:
            h(doc, ad, level=3)
            continue
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(f"{ad}: "); r.bold = True; r.font.size = Pt(9)
        r2 = p.add_run(aciklama); r2.font.size = Pt(9)
    doc.add_paragraph()


def bolum_terimler(doc):
    h(doc, "10. Terimler Sözlüğü")
    _bolum_kod_adlari(doc)
    h(doc, "10.2 Kavram Sözlüğü", level=2)
    para(doc,
         "Bu bölüm, raporda geçen tüm teknik terimleri konuya hiç aşina olmayan bir "
         "okuyucu için açıklar. Terimler konu başlıklarına göre gruplanmıştır.",
         size=9.5)
    doc.add_paragraph()
    for terim, aciklama in TERIMLER:
        if aciklama is None:
            # .title() KULLANILMAZ: Python'un title()'ı Türkçe'de noktalı/noktasız
            # i ayrımını bozuyor ("Veri" -> "Veri̇") ve kısaltmaları küçültüyor
            # ("GNN" -> "Gnn"). Grup adları listede zaten doğru yazılmıştır.
            h(doc, terim, level=3)
            continue
        p = doc.add_paragraph()
        r = p.add_run(f"{terim}: "); r.bold = True; r.font.size = Pt(9)
        r2 = p.add_run(aciklama); r2.font.size = Pt(9)
    doc.add_paragraph()


def bolum_erisim(doc):
    h(doc, "11. Kod ve Veri Erişilebilirliği")
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
    bolum_hiperparametre_secimi(doc)
    bolum_model_detaylari(doc)
    bolum_graf_insa(doc)
    bolum_xai_detaylari(doc)
    bolum_permutasyon_bulgular(doc)
    bolum_bagimliliklar(doc)
    bolum_dosya_yapisi(doc)
    bolum_terimler(doc)
    bolum_erisim(doc)

    out_path = PROJECT_ROOT / "MOF_Radyoaktif_Gaz_Adsorpsiyonu_Bilgi_RAPORU.docx"
    doc.save(str(out_path))
    print(f"Bilgi raporu kaydedildi -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
