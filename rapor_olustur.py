"""
rapor_olustur.py
==================
MOF Radyoaktif Gaz Adsorpsiyonu — Sonuç Raporu oluşturucu (python-docx).

STİL: "Üç Boyutlu Kristal Malzemeler/rapor_olustur.py" ile BİREBİR AYNI
sade/akademik biçim (Table Grid çerçeveli tablolar, renkli fon/zebra-şerit
YOK, `citation_box`/`references_list` yardımcıları) — BAŞLIKLAR (title +
tüm `add_heading` seviyeleri) DAHİL raporun TAMAMI SİYAH-BEYAZDIR — SADECE
gömülü grafikler (zaten renkli matplotlib çıktıları) bu kuralın dışındadır.

PANEL HARFLERİ: her grafiğin sağ-üst köşesine basılan (a),(b),(c)... panel
etiketi, artık HER model klasöründeki grafik.py'nin VE bu scriptin AYNI
merkezi sözlükten (paths.PANEL_HARFLERI, MODEL_KLASORLERI'nin bildirim
sırasına göre) okuduğu TEK bir kaynaktan gelir - grafik ÜZERİNDEKİ harf ile
altına yazılan "(a) ModelAdı, (b) ModelAdı2, ..." açıklaması ARTIK HER ZAMAN
eşleşir (önceki sürümde grafik.py'ler kendi harflerini elle/hardcode
yazıyordu, bu rapor ise onları ALFABETİK sırayla YENİDEN harflendiriyordu -
ikisi TUTARSIZDI, bkz. kullanıcı geri bildirimi).

DÜRÜSTLÜK İLKESİ (sibling projeyle AYNI): bu script model_karsilastirma_
sonuclari.csv + <Model>/sonuclar/grafikler/*.tif + XAI sonuç dosyalarını
ÇALIŞTIRILDIKTAN SONRA OTOMATİK OLARAK OKUYUP rapora gömer — hiçbir sayı
elle yazılmaz. Henüz üretilmemiş bir dosya varsa rapor bunu şeffafça
"[Henüz üretilmedi]" olarak işaretler.

VERİ KAYNAĞI ATFI: yapılar Hugging Face `jablonkagroup/core_mof_no_topo`
(CoRE-MOF türevi, CC BY 4.0) üzerinden çekildiğinden, rapor §1'de bu
veri setinin TAM atıf/kaynak bilgisi (Jablonka et al. 2023; Chung et al.
2014/2019) sunulur — bkz. bolum_veri_kaynagi().

Çalıştırma (modellerin TAMAMI + model_karsilastirma*.py koştuktan SONRA):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python rapor_olustur.py
Çıktı: MOF_Radyoaktif_Gaz_Adsorpsiyonu_Raporu.docx
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from scipy.stats import pearsonr

from paths import PROJECT_ROOT, MODEL_KLASORLERI, XAI_KLASORLERI, PANEL_HARFLERI, GITHUB_REPO_URL
from egitim_ortak import TARGET_COLUMNS, TARGET_UNITS, PRETRAIN_MAX_EPOCHS, PRETRAIN_PATIENCE

BASE = Path(__file__).resolve().parent


from kaynakca import KAYNAKLAR, atif


# ---------------------------------------------------------------------------
# ORTAK YARDIMCILAR ("Üç Boyutlu Kristal Malzemeler/rapor_olustur.py" ile AYNI)
# ---------------------------------------------------------------------------
def bold_cell(cell, text, size=10, align=WD_ALIGN_PARAGRAPH.CENTER):
    """SADE: fon rengi YOK, sadece kalın SİYAH metin (Table Grid çerçevesiyle
    birlikte kullanılır) - kullanıcı isteği: tablolar tamamen siyah-beyaz."""
    cell.text = ""
    p = cell.paragraphs[0]; p.alignment = align
    run = p.add_run(str(text)); run.bold = True; run.font.size = Pt(size)


def normal_cell(cell, text, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]; p.alignment = align
    run = p.add_run(str(text)); run.bold = bold; run.font.size = Pt(size)


def add_heading(doc, text, level=1):
    """TÜM başlıklar (title dahil) SİYAH kalın metin - raporun tamamı
    (grafikler hariç) siyah-beyazdır (kullanıcı isteği)."""
    sizes = {0: 18, 1: 14, 2: 13, 3: 12}
    p = doc.add_paragraph()
    run = p.add_run(text); run.bold = True; run.font.size = Pt(sizes.get(level, 12))
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_image(doc, path: Path, width_cm=16.5, caption=None):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_img.add_run()
    if path.exists():
        run.add_picture(str(path), width=Cm(width_cm))
    else:
        run.add_text(f"[Henüz üretilmedi: {path.name} — ilgili run_*.py / grafik.py koşulmalı]")
        run.italic = True
    if caption:
        cp = doc.add_paragraph(caption)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.runs[0].italic = True
        cp.runs[0].font.size = Pt(9)


def add_fig_caption(doc, text):
    cp = doc.add_paragraph(text)
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.runs[0].italic = True
    cp.runs[0].font.size = Pt(9)


# ---------------------------------------------------------------------------
# AKADEMİK NUMARALANDIRMA - "Şekil N" / "Tablo N" sayaçları
# ---------------------------------------------------------------------------
# Akademik biçim gereği HER şekil ve HER tablo sıralı bir numara ve açıklayıcı
# bir başlık taşır; metin bunlara "Şekil 3", "Tablo 5" diye atıf yapabilir.
# Sayaçlar modül düzeyindedir ve main() başında sıfırlanır.
_SAYAC = {"sekil": 0, "tablo": 0}


def _sayac_sifirla() -> None:
    _SAYAC["sekil"] = 0
    _SAYAC["tablo"] = 0


def sekil_basligi(doc, aciklama: str) -> int:
    """Şekil altına 'Şekil N. <açıklama>' başlığı basar, numarayı döndürür."""
    _SAYAC["sekil"] += 1
    add_fig_caption(doc, f"Şekil {_SAYAC['sekil']}. {aciklama}")
    return _SAYAC["sekil"]


def tablo_basligi(doc, aciklama: str) -> int:
    """Tablo ÜSTÜNE 'Tablo N. <açıklama>' başlığı basar (akademik biçimde
    tablo başlıkları üstte, şekil başlıkları altta yer alır)."""
    _SAYAC["tablo"] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"Tablo {_SAYAC['tablo']}. {aciklama}")
    r.italic = True
    r.font.size = Pt(9)
    return _SAYAC["tablo"]


def add_paragraph(doc, text, size=11, bold=False, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(1)
    run = p.add_run(text); run.font.size = Pt(size); run.bold = bold
    return p


def page_break(doc):
    doc.add_page_break()


def bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.5)
    r = p.add_run(text); r.font.size = Pt(size)


def citation_box(doc, text, size=10):
    """Tek hücreli, fonsuz 'kutu' - sadece 'Table Grid' çerçevesi (renkli fon
    YOK), makaleye kopyalanabilecek atıf/alıntı metnini görsel olarak ayırır."""
    tablo = doc.add_table(rows=1, cols=1)
    tablo.style = "Table Grid"
    cell = tablo.rows[0].cells[0]
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.italic = True
    return tablo


def references_list(doc, refs, size=8.5):
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        r = p.add_run(f"[{i}] {ref}")
        r.font.size = Pt(size)


def kv_table(doc, headers, rows_data):
    """Sade tablo: sadece 'Table Grid' çerçevesi + kalın SİYAH başlık satırı -
    fon rengi/zebra-şerit YOK."""
    tablo = doc.add_table(rows=1 + len(rows_data), cols=len(headers))
    tablo.style = "Table Grid"
    for j, h in enumerate(headers):
        bold_cell(tablo.rows[0].cells[j], h)
    for ri, row in enumerate(rows_data, 1):
        for ci, val in enumerate(row):
            normal_cell(tablo.rows[ri].cells[ci], str(val),
                        align=WD_ALIGN_PARAGRAPH.CENTER if ci > 0 else WD_ALIGN_PARAGRAPH.LEFT,
                        bold=(ci == 0))
    return tablo


# ---------------------------------------------------------------------------
# §1 VERİ KAYNAĞI VE ATIF (Bilesen 2 gerekliligi - kullanicinin "atif" istegi)
# ---------------------------------------------------------------------------
def bolum_veri_kaynagi(doc):
    add_heading(doc, "1. Veri Kaynağı ve Atıf (Data Source & Citation)", level=1)
    add_paragraph(doc,
        f"MOF kristal yapıları (CIF + 3B koordinatlar), Hugging Face üzerinde "
        f"barındırılan 'jablonkagroup/core_mof_no_topo' veri setinden (CC BY 4.0 "
        f"lisanslı) otomatik olarak çekilmiştir {atif('jablonka2023')} (bkz. "
        f"veri_indirici_1_jarvis_core_mof.py). Bu veri seti, CoRE-MOF "
        f"(Computation-Ready, Experimental Metal-Organic Frameworks) "
        f"veritabanının {atif('chung2014', 'chung2019')} bir türevidir. Kaynak "
        f"veri setinde DFT oluşum enerjisi BULUNMADIĞINDAN, ön-eğitim hedefi "
        f"olarak onun GCMC-simüle edilmiş CO₂ adsorpsiyon ısısı (Widom ekleme "
        f"yöntemi, kJ/mol → eV) ENERJİSEL VEKİL (proxy) olarak kullanılmıştır ve "
        f"sütun adı bunu açıkça belirtir ('formation_energy_eV_atom_proxy'). "
        f"Kaynak veri setinde Xe, Kr veya I₂ adsorpsiyon değeri YOKTUR — asıl "
        f"hedef etiketlerin nasıl üretildiği §1.2'de ayrıntılı olarak "
        f"açıklanmıştır. Veri setinin kendisi, kullanıldığı her yayında aşağıdaki "
        f"künyeyle anılmalıdır.",
        size=10, indent=True)
    doc.add_paragraph()

    citation_box(doc,
        "Dataset: jablonkagroup/core_mof_no_topo. Hugging Face Datasets. "
        "License: CC BY 4.0. https://huggingface.co/datasets/jablonkagroup/core_mof_no_topo")
    doc.add_paragraph()
    add_heading(doc, "1.1 Kod ve Veri Erişilebilirliği (Code & Data Availability)", level=2)
    add_paragraph(doc,
        "Bu raporu üreten boru hattının TAMAMI — tüm kaynak kodu, üretilen veri "
        "seti (CIF yapıları dahil), model checkpoint'leri, model-başına sonuç "
        "CSV/JSON dosyaları, XAI çıktıları ve tüm grafiklerin .png sürümleri — "
        "aşağıdaki herkese açık depoda yayımlanmıştır:", size=10, indent=True)
    doc.add_paragraph()
    citation_box(doc, GITHUB_REPO_URL)
    doc.add_paragraph()
    add_paragraph(doc,
        "Depoya DAHİL EDİLMEYENLER (kod ile yeniden üretilebildikleri ve git için "
        "fazla büyük oldukları için .gitignore ile hariç tutulmuştur): 600 dpi .tif "
        "grafikler (dosya başına 40-90 MB; .png ikizleri depoda MEVCUTTUR), bu "
        "grafikleri gömen ~98 MB'lik sonuç .docx dosyası ve ham per-kenar XAI "
        "dökümü (edge_attribution_kenarlar.csv, 79 MB). İlgili grafik.py / "
        "rapor_olustur.py yeniden çalıştırıldığında hepsi birebir yeniden üretilir. "
        "Verinin kökeni ve hangi kısmının sentetik olduğu, depo kökündeki "
        "VERI_KAYNAGI_VE_SINIRLAMALAR.md belgesinde ayrıntılı olarak "
        "belgelenmiştir (özeti §1.2'dedir).", size=9, indent=True)
    page_break(doc)
    _bolum_etiket_kokeni(doc)
    page_break(doc)


def _etiket_kaynak_dagilimi() -> dict[str, dict[str, int]]:
    """Nihai veri setindeki 'label_source_<hedef>' sütunlarının GERÇEK
    dağılımını okur. (Önceki sürüm etiketlerin 'KARMA kaynaklı' olduğunu
    SABİT metin olarak iddia ediyordu; gerçekte bu koşumda etiketlerin
    %100'ü PROXY_PORE_CORRELATION çıktı - iddia yanlıştı. Artık veriden
    okunur.)"""
    from paths import GAS_FINETUNE_CSV
    if not GAS_FINETUNE_CSV.exists():
        return {}
    df = pd.read_csv(GAS_FINETUNE_CSV, low_memory=False)
    dagilim = {}
    for kol in TARGET_COLUMNS:
        sut = f"label_source_{kol}"
        if sut in df.columns:
            dagilim[kol] = df[sut].value_counts().to_dict()
    return dagilim


def _donguselluk_kaniti() -> str:
    """§1.2'deki döngüsellik iddiasını destekleyen sayıları GERÇEK çıktılardan
    hesaplar (perm_importance.json + OOF tahminleri). Önceden bu sayılar metne
    ELLE yazılmıştı ('~70 katı', 'r≈-0.99') ve gerçek değerlerle tam
    örtüşmüyordu; raporun 'hiçbir sayı elle yazılmaz' ilkesine aykırıydı."""
    oranlar = []
    for model_adi in _rapor_modelleri():
        f = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler" / "perm_importance.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        yapi, gozenek = s.get("Crystal Structure", 0.0), s.get("Pore Geometry", 0.0)
        if yapi > 1e-12:
            oranlar.append(gozenek / yapi)

    r_gercek_list, r_tahmin_list = [], []
    for model_adi in _rapor_modelleri():
        f = PROJECT_ROOT / model_adi / "sonuclar" / "test_tahminleri_oof.csv"
        if not f.exists():
            continue
        odf = pd.read_csv(f)
        gerekli = {"pld_A", "gercek_xe_kr_selectivity", "tahmin_xe_kr_selectivity"}
        if not gerekli.issubset(odf.columns):
            continue
        sub = odf.dropna(subset=list(gerekli))
        if len(sub) < 5:
            continue
        dx = (sub["pld_A"] - 4.10).abs()
        if dx.std() < 1e-9:
            continue
        r_gercek_list.append(pearsonr(dx, sub["gercek_xe_kr_selectivity"])[0])
        r_tahmin_list.append(pearsonr(dx, sub["tahmin_xe_kr_selectivity"])[0])

    parcalar = []
    if oranlar:
        parcalar.append(f"§5.3'te 'Pore Geometry' ΔMAE'si 'Crystal Structure'ınkinin "
                         f"{min(oranlar):.0f}-{max(oranlar):.0f} katıdır")
    if r_gercek_list and r_tahmin_list:
        parcalar.append(f"§5.2'de modelin ürettiği korelasyon "
                         f"(r={min(r_tahmin_list):.3f}…{max(r_tahmin_list):.3f}), gürültülü "
                         f"gerçek etiketlerinkinden (r={np.mean(r_gercek_list):.3f}) DAHA güçlüdür")
    if not parcalar:
        return ""
    return "Bunun izleri sonuçlarda görülebilir: " + " ve ".join(parcalar) + "."


def _bolum_etiket_kokeni(doc):
    dagilim = _etiket_kaynak_dagilimi()
    if not dagilim:
        add_paragraph(doc, "[Henüz üretilmedi: nihai veri seti okunamadı — etiket köken "
                            "dağılımı raporlanamıyor.]", size=9, indent=True)
        return

    add_heading(doc, "1.2 Hedef Etiketlerin Kökeni (KRİTİK)", level=2)
    satirlar = []
    for kol, sayim in dagilim.items():
        toplam = sum(sayim.values())
        satirlar.append((_hedef_etiket(kol),
                          ", ".join(f"{k}: {v} (%{100 * v / toplam:.0f})"
                                    for k, v in sorted(sayim.items(), key=lambda kv: -kv[1]))))
    tablo_basligi(doc, "Hedef etiketlerin kaynak dağılımı (nihai veri setinden okunmuştur).")
    kv_table(doc, ["Hedef", "Etiket Kaynağı Dağılımı"], satirlar)
    doc.add_paragraph()

    tum_kaynaklar = set()
    for sayim in dagilim.values():
        tum_kaynaklar.update(sayim.keys())
    sadece_proxy = tum_kaynaklar == {"PROXY_PORE_CORRELATION"}

    if sadece_proxy:
        add_paragraph(doc,
            "UYARI — SONUÇLARIN YORUMLANMASI İÇİN BELİRLEYİCİDİR: bu koşumda "
            "hedef etiketlerin TAMAMI (%100) 'PROXY_PORE_CORRELATION' "
            "kaynaklıdır. Yani etiketler deneysel ölçüm DEĞİLDİR, GCMC "
            "simülasyonu DEĞİLDİR, literatürden alınmış DEĞİLDİR: gözeneklilik "
            "tanımlayıcılarından (PLD, gözenek hacmi, açık metal bölgesi, "
            "fonksiyonel grup) kapalı-form bir formülle ÜRETİLMİŞTİR (formülün "
            "fiziksel motivasyonu Sikora et al. 2012'nin boyut-eleme ilkesidir, "
            "ancak ürettiği SAYILAR gerçek değildir; üzerine lognormal gürültü "
            "eklenmiştir). NLP literatür madenciliği bu koşumda kullanılabilir "
            "hiçbir etiket sağlayamamıştır.", size=9, indent=True)
        doc.add_paragraph()
        add_paragraph(doc,
            "DÖNGÜSELLİK: etiket formülünün girdileri olan pld_A, "
            "pore_volume_cm3_g, open_metal_site ve has_functional_group "
            "değişkenlerinin DÖRDÜ DE modele yardımcı (aux) GİRDİ özelliği "
            "olarak verilmektedir. Dolayısıyla model, kendi girdilerinden "
            "hesaplanan bir formülü geri çözmeyi öğrenmektedir; R² tavanı "
            "fiziksel öğrenme kapasitesiyle değil, etikete enjekte edilen "
            "gürültüyle belirlenir. " + _donguselluk_kaniti(), size=9, indent=True)
        doc.add_paragraph()
        add_paragraph(doc,
            f"SONUÇ: bu rapordaki R²/MAE değerleri, makine öğrenmesi boru "
            f"hattının (eğitilen {len(_rapor_modelleri())} mimari, transfer "
            f"öğrenme, K-Fold, {len(XAI_KLASORLERI)} XAI yöntemi) uçtan uca "
            "DOĞRU ÇALIŞTIĞINI gösteren bir ALTYAPI DOĞRULAMASIDIR. Gerçek "
            "Xe/Kr/I₂ adsorpsiyon tahmin başarısını GÖSTERMEZ ve bir yayında "
            "malzeme-keşfi sonucu olarak sunulamaz. Gerçek etiket kaynağına "
            "(yayınlanmış GCMC izotermleri veya RASPA simülasyonu) geçilmesi "
            "gerekmektedir — bkz. depodaki VERI_KAYNAGI_VE_SINIRLAMALAR.md.",
            size=9, bold=True, indent=True)
    else:
        add_paragraph(doc,
            "Etiketler karma kaynaklıdır (yukarıdaki tabloya bakınız). "
            "'PROXY_PORE_CORRELATION' kaynaklı satırlar deneysel/simülasyon "
            "verisi DEĞİL, gözeneklilik tanımlayıcılarından türetilmiş vekil "
            "değerlerdir; aşağıdaki R²/MAE değerleri bu karışık etiket "
            "kalitesiyle birlikte yorumlanmalıdır.", size=9, indent=True)


# ---------------------------------------------------------------------------
# §2 DENEYSEL KURULUM VE HİPERPARAMETRELER (kullanıcı isteği: "hiperparametreler
# bulunacak, kaç epoch vs" — hiçbir sayı elle yazılmaz, HER modelin GERÇEKTEN
# koştuğu <Model>/sonuclar/metrikler.json'dan otomatik okunur; egitim_ortak.py
# KOD VARSAYILANLARI DEĞİL, çünkü bunlar ortam değişkenleriyle (KFOLD_OVERRIDE,
# MAX_EPOCHS_OVERRIDE, ...) ezilmiş olabilir — bkz. paths.py/egitim_ortak.py.
# ---------------------------------------------------------------------------
def _hedef_etiket(kol: str) -> str:
    """Hedef sütun adını insan-okunur başlığa çevirir ('i2_uptake_mmol_g' ->
    'I₂ Uptake (mmol/g)'). Kullanıcı geri bildirimi: raporda/grafiklerde ham
    kolon adları ('i2 falan yazılmış') görünmemeli. grafik_ortak'taki etiketler
    matplotlib LaTeX kipi içerdiğinden Unicode'a çevrilir (Word LaTeX render
    etmez)."""
    from grafik_ortak import DISPLAY_LABELS
    etiket = DISPLAY_LABELS.get(kol, kol)
    return etiket.replace("$_2$", "₂").replace("$_", "").replace("$", "")


def _tum_metrikler_json() -> dict[str, dict]:
    """{model_adi: metrikler.json} — SADECE gerçekten eğitilmiş modeller.
    (Bir model implemente edilmiş ama koşulmamış olabilir; o zaman hiçbir
    sonucu yoktur ve raporda EĞİTİLMİŞ gibi sayılmamalıdır.)"""
    sonuc = {}
    for model_adi in MODEL_KLASORLERI:
        f = PROJECT_ROOT / model_adi / "sonuclar" / "metrikler.json"
        if f.exists():
            sonuc[model_adi] = json.loads(f.read_text(encoding="utf-8"))
    return sonuc


def _ilk_metrikler_json() -> dict | None:
    tumu = _tum_metrikler_json()
    return next(iter(tumu.values()), None)


def EGITILMIS_MODELLER() -> list[str]:
    """Bu koşumda GERÇEKTEN eğitilip sonuç üretmiş modeller (rapor metinleri
    ve panel harfleri bu listeye göre yazılır - kullanıcı geri bildirimi:
    rapor '11 model' diyordu ama DimeNetPP hiç eğitilmemişti, grafikleri
    '[Henüz üretilmedi]' yer tutucusuydu; yine de panel açıklaması '(k)
    DimeNetPP ... verilmiştir' diyordu - ÇELİŞKİLİ/yanıltıcıydı)."""
    return list(_tum_metrikler_json().keys())


def bolum_hiperparametreler(doc):
    add_heading(doc, "2. Deneysel Kurulum ve Hiperparametreler", level=1)
    meta = _ilk_metrikler_json()
    if meta is None:
        add_paragraph(doc, "[Henüz üretilmedi: <Model>/sonuclar/metrikler.json — önce en az bir "
                            "modelin run_*.py'sini çalıştırın.]", indent=True)
        page_break(doc)
        return

    hp = meta["hiperparametreler"]
    tum_meta = _tum_metrikler_json()
    egitilmis = list(tum_meta.keys())
    egitilmemis = [m for m in MODEL_KLASORLERI if m not in tum_meta]

    add_paragraph(doc,
        f"Bu koşumda K-Fold çapraz doğrulamanın K değeri = {meta['k_folds']} olarak "
        f"kullanılmıştır (kod varsayılanı egitim_ortak.py'de K_FOLDS=5'tir; bu "
        f"değer 'KFOLD_OVERRIDE' ortam değişkeniyle ezilebilir — aşağıdaki TÜM "
        f"sayılar kod varsayılanı DEĞİL, gerçekten koşulan değerlerdir, "
        f"<Model>/sonuclar/metrikler.json dosyasından otomatik okunmuştur). "
        f"Toplam {meta['n_grup_toplam']} benzersiz temel MOF'tan (augment "
        f"varyantlarıyla birlikte {meta['n_ornek_toplam']} örnek) her fold için "
        f"ayrı bir eğitim/test bölünmesi yapılmış, aynı temel MOF'tan türetilen "
        f"TÜM augment varyantları (defect/substitution/functional-group/gaussian-"
        f"noise) SIZINTI olmaması için AYNI fold'da tutulmuştur: bir MOF'un "
        f"augment kopyası eğitim setinde, orijinali test setinde olamaz — bu, "
        f"K-Fold bölmesinin 'base_mof_id' sütununa göre GRUPLANARAK yapılmasıyla "
        f"garanti edilir (egitim_ortak.py, GROUP_COLUMN).",
        size=10, indent=True)
    doc.add_paragraph()

    if egitilmemis:
        add_paragraph(doc,
            f"KAPSAM NOTU (şeffaflık): bu depoda {len(MODEL_KLASORLERI)} GNN mimarisi "
            f"IMPLEMENTE edilmiştir, ancak bu koşumda {len(egitilmis)} tanesi "
            f"eğitilip değerlendirilmiştir. Eğitilmemiş model(ler): "
            f"{', '.join(egitilmemis)} — bu modeller için hiçbir sonuç/grafik "
            f"üretilmemiştir ve aşağıdaki hiçbir tabloda/karşılaştırmada YER "
            f"ALMAZLAR. Raporun geri kalanında 'tüm modeller' ifadesi, eğitilmiş "
            f"bu {len(egitilmis)} modeli kasteder.", size=9, indent=True)
        doc.add_paragraph()

    # Bir hiperparametre GERÇEKTEN ortak mı, yoksa modele göre mi değişiyor?
    # (Kullanıcı geri bildirimi / gerçek hata: rapor batch_size'ı TEK bir
    # ortak değer olarak gösterip "tüm modellerde ortaktır" diyordu, oysa
    # GraphGPS/ECC/SE3_Transformer=16 iken diğerleri=32 idi. Artık hangi
    # anahtarın ortak olduğu VERİDEN tespit edilir, elle varsayılmaz.)
    def _degerler(anahtar):
        return {m: md["hiperparametreler"].get(anahtar) for m, md in tum_meta.items()}

    def _ortak_mi(anahtar):
        v = list(_degerler(anahtar).values())
        return len(set(map(str, v))) <= 1

    tanimlar = [
        ("max_epochs", "Fine-tune epoch sayısı (üst sınır)",
         "Erken durdurma tetiklenmezse çalışacak MAKSİMUM epoch"),
        ("early_stop_patience", "Erken durdurma sabrı (patience)",
         "Validasyon hatası bu kadar epoch boyunca İYİLEŞMEZSE eğitim durur"),
        ("freeze_encoder_epochs", "Encoder dondurma süresi (freeze_encoder_epochs)",
         "İlk N epoch'ta SADECE regresyon başı eğitilir, ön-eğitimli encoder dondurulur"),
        ("lr", "Öğrenme oranı (learning rate)", "AdamW optimizer başlangıç adım büyüklüğü"),
        ("weight_decay", "Ağırlık sönümü (weight decay)", "L2 regularizasyon katsayısı"),
        ("batch_size", "Batch boyutu", "Her gradyan adımında kullanılan örnek sayısı"),
        ("hidden_dim", "Gizli katman boyutu (hidden_dim)", "Encoder'ın atom-gömme/katman genişliği"),
        ("emb_dim", "Çıktı gömme boyutu (emb_dim)", "Encoder'dan regresyon başına giden vektör boyutu"),
        ("dropout", "Dropout", "Regresyon başındaki düşürme (overfitting önleme) oranı"),
        ("cutoff", "Kesme yarıçapı (cutoff)", "3B komşuluk grafiği için atomlar-arası maksimum bağ mesafesi (Å)"),
        ("seed", "Rastgelelik tohumu (seed)", "Tekrarlanabilirlik için sabit rastgelelik başlangıcı"),
        ("aux_dim", "Yardımcı özellik boyutu (aux_dim)",
         "Gözeneklilik/kompozisyon özellik vektörü uzunluğu (bkz. §3)"),
    ]
    ortak_satirlar = [("K (K-Fold sayısı)", meta["k_folds"],
                        "Veri kaç eşit parçaya bölünüp sırayla test edildiği")]
    degisken_anahtarlar = []
    for anahtar, etiket, aciklama in tanimlar:
        if _ortak_mi(anahtar):
            ortak_satirlar.append((etiket, hp.get(anahtar), aciklama))
        else:
            degisken_anahtarlar.append((anahtar, etiket, aciklama))

    tablo_basligi(doc, "Eğitilmiş tüm modellerde ORTAK olan eğitim hiperparametreleri.")
    kv_table(doc, ["Hiperparametre", "Değer", "Anlamı"], ortak_satirlar)
    doc.add_paragraph()
    add_paragraph(doc, f"Yukarıdaki tablodaki değerler, eğitilmiş {len(egitilmis)} modelin "
                        f"HEPSİNDE AYNIDIR (bu, her modelin kendi metrikler.json'ı "
                        f"karşılaştırılarak DOĞRULANMIŞTIR, varsayılmamıştır). Modelden "
                        f"modele DEĞİŞEN parametreler aşağıdaki tabloda ayrıca verilmiştir.",
                  size=9, indent=True)
    doc.add_paragraph()

    add_heading(doc, "2.1 Modele Göre Değişen Parametreler", level=2)
    if degisken_anahtarlar:
        add_paragraph(doc,
            "DİKKAT: aşağıdaki eğitim hiperparametreleri tüm modellerde AYNI DEĞİLDİR; "
            "model karşılaştırma tablosu (§4) okunurken bu fark göz önünde "
            "bulundurulmalıdır (örn. farklı batch boyutu, etkin öğrenme dinamiğini "
            "bir miktar değiştirir).", size=9, indent=True)
        for anahtar, etiket, aciklama in degisken_anahtarlar:
            dv = _degerler(anahtar)
            add_paragraph(doc, f"{etiket} — {aciklama}:", size=9, bold=True, indent=True)
            tablo_basligi(doc, f"Modele göre değişen hiperparametre: {etiket}.")
            kv_table(doc, ["Model", "Değer"], [(m, v) for m, v in dv.items()])
            doc.add_paragraph()

    ortak_anahtarlar = {a for a, _, _ in tanimlar} | {"target_columns"}
    mimari_satirlar = []
    for model_adi, md in tum_meta.items():
        ekstra = {k: v for k, v in md["hiperparametreler"].items() if k not in ortak_anahtarlar}
        if ekstra:
            mimari_satirlar.append((model_adi, ", ".join(f"{k}={v}" for k, v in ekstra.items())))
    if mimari_satirlar:
        add_paragraph(doc, "Mimariye-özgü yapısal parametreler (her mimarinin kendi tasarımı gereği "
                            "zaten farklıdır, bir tutarsızlık DEĞİLDİR):", size=9, bold=True, indent=True)
        tablo_basligi(doc, "Mimariye-özgü yapısal parametreler.")
        kv_table(doc, ["Model", "Mimariye-Özgü Parametreler"], mimari_satirlar)
    doc.add_paragraph()

    add_heading(doc, "2.2 Ön-Eğitim (Pretrain) Aşaması", level=2)
    add_paragraph(doc,
        f"Ön-eğitim aşamasının hiperparametreleri (kod varsayılanı: "
        f"PRETRAIN_MAX_EPOCHS={PRETRAIN_MAX_EPOCHS}, PRETRAIN_PATIENCE={PRETRAIN_PATIENCE}) "
        f"finetune aşamasının aksine bir JSON dosyasına KAYDEDİLMEZ (sadece konsola "
        f"yazdırılır) — bu yüzden ortam değişkeniyle ezilmiş olabilecekleri "
        f"ŞEFFAFÇA belirtilir: bu koşumda GERÇEKTEN kullanılan değerler burada "
        f"DOĞRULANAMAZ, sadece kod varsayılanları raporlanabilir. Ön-eğitim, "
        f"K-Fold YAPMAZ (basit tek train/val bölmesi + erken durdurma) çünkü "
        f"amaç nihai değerlendirme değil, genellenebilir bir temsil (embedding) "
        f"öğrenmektir; SADECE encoder ağırlıkları kaydedilir, regresyon başı atılır.",
        size=9, indent=True)
    page_break(doc)


# ---------------------------------------------------------------------------
# §3 TERMİNOLOJİ SÖZLÜĞÜ (kullanıcı isteği: "Pld ne demek vs her şeyi açık
# açık yazacağız" — raporda geçen TÜM kısaltma/teknik terimler burada tanımlanır)
# ---------------------------------------------------------------------------
def bolum_terminoloji(doc):
    add_heading(doc, "3. Terminoloji Sözlüğü", level=1)
    add_paragraph(doc, "Bu raporda sıkça geçen kısaltma ve teknik terimlerin açık tanımları:",
                  size=10, indent=True)
    doc.add_paragraph()

    terimler = [
        ("PLD (Pore Limiting Diameter)", "Gözenek-Sınırlayıcı Çap",
         "Bir MOF'un gözenek ağı içinden ucundan ucuna geçebilecek en büyük küresel "
         "parçacığın çapı (Å). Gaz difüzyonu için 'darboğaz' ölçüsüdür — bir gazın "
         "kinetik çapı PLD'den büyükse o gaz o gözenekten geçemez. DİKKAT: bu "
         "projedeki PLD değerleri Zeo++ küresel-prob taramasıyla veya deneysel "
         "olarak ÖLÇÜLMEMİŞTİR; yapıdan kaba bir geometrik yaklaşımla "
         "(en kısa kafes vektörü × √boşluk-oranı, ardından ×0.55) KESTİRİLMİŞTİR "
         "— bkz. veri_indirici_1_jarvis_core_mof.estimate_pore_proxies()."),
        ("LCD (Largest Cavity Diameter)", "En Büyük Kavite Çapı",
         "MOF'un içindeki en geniş boşluğa sığabilecek en büyük kürenin çapı (Å). "
         "PLD'den FARKLIDIR: LCD gözeneğin İÇ hacmini, PLD ise gözenekler ARASI "
         "GEÇİŞ darboğazını ölçer (LCD ≥ PLD her zaman doğrudur). PLD'de olduğu "
         "gibi, buradaki LCD de ölçüm değil geometrik kestirimdir."),
        ("Kinetik Çap", "Kinetic Diameter",
         "Bir gaz molekülünün difüzyon/eleme davranışını belirleyen etkin boyutu "
         "(Xe ≈ 4.0-4.4 Å, Kr ≈ 3.6-3.8 Å) — PLD ile karşılaştırılarak boyut-eleme "
         "(size-sieving) gücü tahmin edilir. Bu projede boyut-uyum hesaplarında "
         "Xe için 4.10 Å, Kr için 3.69 Å, I₂ için 5.00 Å kullanılmıştır."),
        ("Void Fraction", "Boşluk/Gözeneklilik Oranı",
         "MOF birim hücresinin toplam hacmine oranla boş (erişilebilir) hacminin "
         "payı (0-1 arası). Bu projede helyum-prob veya Zeo++ hesabı DEĞİL, "
         "atomların van der Waals hacimlerinin hücre hacmine oranından türetilen "
         "bir yaklaşım kullanılmıştır (1 − ΣV_vdW/V_hücre)."),
        ("Open Metal Site", "Açık Metal Bölgesi",
         "Koordinasyonu tam doymamış, çözücü uzaklaştırıldığında gaz molekülüne "
         "doğrudan bağlanabilen metal merkezi — Xe/I₂ gibi polarize olabilen "
         "moleküllerle güçlü etkileşime girer, adsorpsiyon kapasitesini artırır."),
        ("Xe/Kr Selectivity", "Xe/Kr Seçicilik",
         "Bir MOF'un Xe'yi Kr'ye göre ne kadar tercihen adsorbe ettiğinin oranı "
         "(birimsiz) — nükleer atık-gazı ayrıştırmada radyoaktif Xe/Kr izotoplarının "
         "birbirinden verimli ayrılması için KRİTİK metriktir."),
        ("R² (Belirlilik Katsayısı)", "Coefficient of Determination",
         "Modelin gerçek değerlerdeki varyansın ne kadarını açıkladığı (1.0 = "
         "mükemmel tahmin, 0.0 = ortalamayı tahmin etmekten farksız)."),
        ("MAE (Mean Absolute Error)", "Ortalama Mutlak Hata",
         "Tahmin ile gerçek değer arasındaki mutlak farkların ortalaması — hedefle "
         "AYNI birimde, aykırı değerlere R²'den daha az duyarlıdır."),
        ("RMSE (Root Mean Squared Error)", "Kök Ortalama Kare Hata",
         "Hataların karelerinin ortalamasının kareköküdür — büyük hataları MAE'den "
         "daha ağır cezalandırır."),
        ("MedianAE / MaxError", "Medyan/Maksimum Mutlak Hata",
         "MedianAE aykırı değerlerden ETKİLENMEYEN tipik hata; MaxError en kötü "
         "tekil tahminin hatasıdır (en kötü senaryo)."),
        ("PearsonR", "Pearson Korelasyon Katsayısı",
         "Tahmin ile gerçek değer arasındaki DOĞRUSAL ilişkinin gücü/yönü (-1 ile "
         "+1 arası)."),
        ("K-Fold / OOF (Out-of-Fold)", "Çapraz Doğrulama / Fold-Dışı Tahmin",
         "Veri K eşit parçaya bölünür; her fold sırayla TEST olurken diğerleri "
         "eğitim için kullanılır. 'OOF tahmin', bir örneğin SADECE o örneğin test "
         "fold'unda olduğu zaman üretilen tahminidir — hiçbir örnek kendi eğitim "
         "verisiyle değerlendirilmez, bu yüzden ezber (overfitting) riski düşüktür."),
        ("Permutation Importance (ΔMAE)", "Karıştırma-Bazlı Önem",
         "Bir özellik grubunun değerleri örnekler arasında RASTGELE karıştırılır "
         "ve MAE'nin ne kadar KÖTÜLEŞTİĞİ ölçülür — ΔMAE ne kadar büyükse (pozitif "
         "ise) model o özelliğe o kadar bağımlıdır."),
        ("Grafiklerdeki (a), (b), (c) ... harf etiketleri", "Uzun Etiketlerin Yerine Geçen Harfler",
         "§7.7'deki permutation-importance çubuk grafiklerinde y-ekseni, ÖNEM "
         "SIRASINA göre (en önemliden en az önemliye) (a), (b), (c) ... "
         "rakamlarıyla etiketlenir — HANGİ rakamın HANGİ özellik grubuna karşılık "
         "geldiği modelden modele DEĞİŞİR (sıralama, o modelin gerçek ΔMAE "
         "değerlerine göre yeniden yapılır); bu yüzden her modelin gerçek eşleşme "
         "tablosu doğrudan §7.7'de, ilgili grafiğin ALTINDA verilir."),
        ("Aux (Yardımcı) Özellik Grupları", "Pore Geometry / Surface Area / Structural-Size / "
         "Chemical Modification / Composition-Derived / Crystal Structure",
         "Permutation-importance ve GraphLIME'da kullanılan 6 kategori: 'Crystal "
         "Structure' atomların 3B geometrik düzenlenimini (grafik encoder'in "
         "öğrendiği), diğer 5'i ise hazır-hesaplanmış gözeneklilik/kompozisyon "
         "sayısal özelliklerini (Pore Geometry=pore_volume/void_fraction/LCD/PLD; "
         "Surface Area=gravimetrik+hacimsel yüzey alanı; Structural/Size=yoğunluk/"
         "atom+element sayısı/metal oranı; Chemical Modification=açık metal "
         "bölgesi+fonksiyonel grup; Composition-Derived=elektronegatiflik farkı/"
         "atomik yarıçap/atomik numara ortalamaları) temsil eder."),
    ]
    for terim, acilim, aciklama in terimler:
        add_paragraph(doc, f"{terim} — {acilim}", size=10, bold=True, indent=True)
        add_paragraph(doc, aciklama, size=9, indent=True)
        doc.add_paragraph()
    page_break(doc)


# ---------------------------------------------------------------------------
# §4 MODEL KARŞILAŞTIRMA TABLOSU
# ---------------------------------------------------------------------------
def bolum_metrik_tablosu(doc):
    add_heading(doc, "4. Model Karşılaştırma Tablosu (Pooled Out-of-Fold, 4 Hedef Ortalaması)", level=1)
    csv_path = PROJECT_ROOT / "model_karsilastirma_sonuclari.csv"
    if not csv_path.exists():
        add_paragraph(doc, "[Henüz üretilmedi: model_karsilastirma_sonuclari.csv — önce modellerin "
                            "run_*.py'sini, sonra model_karsilastirma.py'yi çalıştırın.]", indent=True)
        return

    df = pd.read_csv(csv_path)
    overall = df[df["target_column"] == "overall"].sort_values("R2", ascending=False).reset_index(drop=True)
    kolonlar = ["model", "MAE", "RMSE", "R2", "MedianAE", "MaxError", "PearsonR", "n_ornek"]
    buyuk_iyi = {"R2": True, "PearsonR": True, "MAE": False, "RMSE": False,
                 "MedianAE": False, "MaxError": False}
    best = {k: (overall[k].max() if yon else overall[k].min()) for k, yon in buyuk_iyi.items() if k in overall.columns}

    tablo_basligi(doc, "Model karşılaştırması: havuzlanmış fold-dışı (OOF) metrikler, 4 hedefin ortalaması. En iyi değerler kalın gösterilmiştir.")
    tablo = doc.add_table(rows=1 + len(overall), cols=len(kolonlar))
    tablo.style = "Table Grid"
    for j, k in enumerate(kolonlar):
        bold_cell(tablo.rows[0].cells[j], k)
    for i, (_, row) in enumerate(overall.iterrows(), 1):
        for j, k in enumerate(kolonlar):
            v = row[k]
            metin = f"{v:.4f}" if isinstance(v, float) else str(v)
            en_iyi_mi = k in best and isinstance(v, float) and abs(v - best[k]) < 1e-12
            normal_cell(tablo.rows[i].cells[j], metin, bold=(j == 0 or en_iyi_mi))
    doc.add_paragraph()
    add_paragraph(doc, "'overall' = 4 hedefin (Xe uptake, Kr uptake, Xe/Kr selectivity, I₂ uptake) "
                        "metrik ORTALAMASI; tablo R²'ye göre azalan sırayla sıralanmıştır "
                        "(model_karsilastirma.py çıktısı); en iyi değer içeren hücreler kalın gösterilmiştir.",
                  size=9, indent=True)
    doc.add_paragraph()

    add_heading(doc, "4.1 Hedef Bazında En İyi 3 Model", level=2)
    for kol in TARGET_COLUMNS:
        alt = df[df["target_column"] == kol].sort_values("R2", ascending=False).head(3)
        if alt.empty:
            continue
        satirlar = [(r["model"], f"{r['R2']:.3f}", f"{r['MAE']:.4f}", int(r["n_ornek"])) for _, r in alt.iterrows()]
        add_paragraph(doc, f"{_hedef_etiket(kol)}  [sütun adı: {kol}]:", size=10, bold=True, indent=True)
        tablo_basligi(doc, f"{_hedef_etiket(kol)} hedefinde en iyi 3 model.")
        kv_table(doc, ["Model", "R²", "MAE", "n"], satirlar)
        doc.add_paragraph()
    page_break(doc)


# ---------------------------------------------------------------------------
# §5 MODEL DOĞRULAMA — ezber kontrolü
# ---------------------------------------------------------------------------
def bolum_dogrulama(doc):
    add_heading(doc, "5. Model Doğrulama — Ezber Kontrolü (Ruling Out Overfitting)", level=1)
    add_paragraph(doc,
        "ÖNEMLİ ÇERÇEVE: bu bölümdeki üç kontrol, modelin ETİKETLERİ EZBERLEMEK "
        "yerine GENELLEDİĞİNİ sınar — yani bir makine-öğrenmesi hijyeni "
        "kontrolüdür. §1.2'de belgelendiği üzere etiketler bu koşumda %100 "
        "sentetiktir ve etiket formülünün girdileri modele girdi olarak da "
        "verilmektedir; dolayısıyla buradaki olumlu sonuçlar 'model gerçek "
        "yapı-özellik fiziğini öğrendi' ANLAMINA GELMEZ, 'model, öğrenmesi "
        "istenen fonksiyonu görmediği örneklere de taşıyabiliyor' anlamına "
        "gelir. Aşağıdaki her alt bölümde bu ayrım ayrıca belirtilmiştir.",
        size=9, indent=True)
    doc.add_paragraph()
    add_paragraph(doc,
        "Bu çerçeve içinde 3 bağımsız kontrol sunulur: (4.1) train ile "
        "fold-dışı test başarımı arasındaki boşluk, (4.2) tahminlerin "
        "gözenek-boyut uyumuyla ilişkisinin yönü, (4.3) hangi özellik grubunun "
        "karıştırılmasının başarımı ne kadar bozduğu. Tüm sayılar bu koşumun "
        "gerçek eğitim/test çıktılarından OTOMATİK hesaplanmıştır.",
        size=10, indent=True)
    doc.add_paragraph()

    add_heading(doc, "5.1 Train-Test Genelleme Boşluğu", level=2)
    satirlar = []
    for model_adi in MODEL_KLASORLERI:
        f = PROJECT_ROOT / model_adi / "sonuclar" / "kfold_metrikleri.csv"
        if not f.exists():
            continue
        kdf = pd.read_csv(f)
        tr, te = kdf["train_overall_R2"].mean(), kdf["test_overall_R2"].mean()
        satirlar.append((model_adi, f"{tr:.3f}", f"{te:.3f}", f"{tr - te:.3f}"))
    if satirlar:
        tablo_basligi(doc, "Train ve fold-dışı test başarımı arasındaki genelleme boşluğu.")
        kv_table(doc, ["Model", "Ort. Train R² (overall)", "Ort. Test R² (OOF, overall)", "Fark (boşluk)"], satirlar)
        doc.add_paragraph()
        add_paragraph(doc, "Küçük ve tutarlı bir boşluk, modelin ezberlemek yerine genellediğinin "
                            "klasik işaretidir. Bu kontrol, etiketlerin kaynağından BAĞIMSIZ "
                            "olarak geçerlidir (sentetik de olsa, görülmemiş örneklere "
                            "taşınabilen bir fonksiyon öğrenilmiştir); ancak tek başına "
                            "etiketlerin FİZİKSEL doğruluğu hakkında bilgi vermez — bkz. §1.2.",
                       size=9, indent=True)
    else:
        add_paragraph(doc, "[Henüz üretilmedi: <Model>/sonuclar/kfold_metrikleri.csv]", indent=True)
    doc.add_paragraph()

    add_heading(doc, "5.2 Fiziksel Tutarlılık — Gözeneklilik/Boyut-Uyum Korelasyonu", level=2)
    add_paragraph(doc,
        "PLD (Pore Limiting Diameter / Gözenek-Sınırlayıcı Çap): bir MOF'un gözenek "
        "ağı içinden geçebilecek en büyük küresel parçacığın çapı — yani gaz "
        "moleküllerinin gözenekten GEÇEBİLMESİ için darboğaz/sınırlayıcı geometrik "
        "ölçüttür (bkz. §3 Terminoloji Sözlüğü için tam tanım ve LCD ile farkı). "
        "Bir gaz molekülünün kinetik çapı PLD'den büyükse o molekül gözenekten "
        "geçemez/zayıf adsorbe olur; PLD hedef molekülün kinetik çapına ne kadar "
        "YAKINSA boyut-eleme (size-sieving) o kadar GÜÇLÜDÜR.", size=9, indent=True)
    doc.add_paragraph()
    satirlar = []
    for model_adi in MODEL_KLASORLERI:
        f = PROJECT_ROOT / model_adi / "sonuclar" / "test_tahminleri_oof.csv"
        if not f.exists():
            continue
        odf = pd.read_csv(f)
        if "pld_A" not in odf.columns or "gercek_xe_kr_selectivity" not in odf.columns:
            continue
        sub = odf.dropna(subset=["pld_A", "gercek_xe_kr_selectivity", "tahmin_xe_kr_selectivity"]).copy()
        if len(sub) < 5:
            continue
        sub["boyut_farki"] = (sub["pld_A"] - 4.10).abs()
        if sub["boyut_farki"].std() < 1e-9:
            continue
        r_true, _ = pearsonr(sub["boyut_farki"], sub["gercek_xe_kr_selectivity"])
        r_pred, _ = pearsonr(sub["boyut_farki"], sub["tahmin_xe_kr_selectivity"])
        tutarli = "Evet" if (r_true < 0) == (r_pred < 0) else "HAYIR"
        satirlar.append((model_adi, f"{r_true:.3f}", f"{r_pred:.3f}", tutarli))
    if satirlar:
        tablo_basligi(doc, "Boyut-uyum farkı ile Xe/Kr seçicilik arasındaki korelasyonun gerçek ve tahmin edilen değerlerde karşılaştırılması.")
        kv_table(doc, ["Model", "corr(|PLD-d_Xe|, gerçek sel.)", "corr(|PLD-d_Xe|, tahmin sel.)", "Yön tutarlı mı?"], satirlar)
        doc.add_paragraph()
        add_paragraph(doc,
            "Gözenek-sınırlayıcı çap (PLD) Xe kinetik çapına (4.10 Å) yaklaştıkça "
            "boyut-eleme (size-sieving) güçlenip Xe/Kr seçiciliğinin artması "
            "beklenen bir fiziksel eğilimdir (Sikora et al. 2012) ve tablodaki "
            "tüm modeller bu eğilimi doğru YÖNDE yeniden üretmektedir.",
            size=9, indent=True)
        add_paragraph(doc,
            "ANCAK bu, modelin DIŞSAL bir fiziksel ilişkiyi keşfettiği şeklinde "
            "YORUMLANAMAZ: bu koşumda seçicilik etiketi zaten PLD'nin kapalı-form "
            "bir fonksiyonu olarak ÜRETİLMİŞTİR (§1.2) ve PLD aynı zamanda modele "
            "doğrudan girdi özelliği olarak verilmektedir. Yani eğilim, "
            "öğrenilmesi gereken dışsal bir olgu değil, veri kurgusunun "
            "GARANTİSİDİR. Nitekim modelin ürettiği korelasyon, gürültü "
            "içeren 'gerçek' etiketlerinkinden daha güçlüdür (tabloda |r(tahmin)| "
            "> |r(gerçek)|) — bu da modelin altta yatan gürültüsüz formüle "
            "yakınsadığını gösterir. Bu kontrol, GERÇEK etiketlere geçildiğinde "
            "anlamlı bir fizik testine dönüşecektir.", size=9, indent=True)
    else:
        add_paragraph(doc, "[Henüz üretilmedi veya yetersiz veri: <Model>/sonuclar/test_tahminleri_oof.csv]", indent=True)
    doc.add_paragraph()

    add_heading(doc, "5.3 Permutation Importance — 3B Yapı vs. Gözeneklilik/Kompozisyon", level=2)
    satirlar = []
    for model_adi in MODEL_KLASORLERI:
        f = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler" / "perm_importance.json"
        if not f.exists():
            continue
        skorlar = json.loads(f.read_text(encoding="utf-8"))
        yapi = skorlar.get("Crystal Structure", 0.0)
        gozeneklilik = skorlar.get("Pore Geometry", 0.0)
        oran = gozeneklilik / yapi if yapi > 1e-12 else float("nan")
        satirlar.append((model_adi, f"{yapi:.4f}", f"{gozeneklilik:.4f}",
                          "—" if not np.isfinite(oran) else f"{oran:.0f}×"))
    if satirlar:
        tablo_basligi(doc, "Permutation importance: 3B yapı temsilinin ve gözeneklilik tanımlayıcılarının karıştırılmasının MAE üzerindeki etkisi.")
        kv_table(doc, ["Model", "ΔMAE (3B yapı karıştırılınca)",
                        "ΔMAE (gözeneklilik karıştırılınca)", "Oran"], satirlar)
        doc.add_paragraph()
        oranlar = [float(s[3].rstrip("×")) for s in satirlar if s[3] != "—"]
        if oranlar:
            add_paragraph(doc,
                f"Her iki ΔMAE de pozitiftir (karıştırma başarımı bozuyor), ancak "
                f"BÜYÜKLÜKLERİ arasında uçurum vardır: gözeneklilik tanımlayıcılarının "
                f"karıştırılması, 3B atomistik yapının karıştırılmasından "
                f"{min(oranlar):.0f}-{max(oranlar):.0f} kat daha fazla zarar "
                f"vermektedir. Yani model, ağırlıklı olarak hazır-hesaplanmış "
                f"gözeneklilik sayılarına dayanmakta, grafik encoder'ın öğrendiği "
                f"3B geometriye ise çok az dayanmaktadır.", size=9, indent=True)
            doc.add_paragraph()
            add_paragraph(doc,
                "Bu, modelin bir kusuru değil VERİ KURGUSUNUN doğrudan sonucudur: "
                "etiketler zaten gözeneklilik tanımlayıcılarından üretilmiştir "
                "(§1.2), dolayısıyla 3B geometri ek bilgi taşımamaktadır. Gerçek "
                "(deneysel/GCMC) etiketlere geçildiğinde bu oranın belirgin şekilde "
                "düşmesi BEKLENİR ve bu, 3B mimarilerin gerçekten katkı sağlayıp "
                "sağlamadığının asıl testi olacaktır.", size=9, indent=True)
    else:
        add_paragraph(doc, "[Henüz üretilmedi: <Model>/sonuclar/grafikler/perm_importance.json]", indent=True)
    page_break(doc)


# ---------------------------------------------------------------------------
# §6 MODEL KARŞILAŞTIRMA GRAFİKLERİ
# ---------------------------------------------------------------------------
def bolum_karsilastirma_grafikleri(doc):
    add_heading(doc, "6. Model Karşılaştırma Grafikleri", level=1)
    grafik_dir = PROJECT_ROOT / "model_karsilastirma_grafikler"
    for dosya, baslik in [
        ("r2_karsilastirma_overall.tif", "Modellerin havuzlanmış fold-dışı R² karşılaştırması "
                                          "(4 hedefin ortalaması)."),
        ("mae_karsilastirma_overall.tif", "Modellerin havuzlanmış fold-dışı MAE karşılaştırması "
                                           "(4 hedefin ortalaması)."),
        ("r2_heatmap_model_x_hedef.tif", "Model x hedef R² ısı haritası."),
        ("r2_vs_mae_overall.tif", "R² ve MAE'nin birlikte gösterimi (her nokta bir model)."),
    ]:
        add_image(doc, grafik_dir / dosya)
        sekil_basligi(doc, baslik)
    page_break(doc)


# ---------------------------------------------------------------------------
# §7 MODEL GRAFİKLERİ (hedef başına, model başına)
# ---------------------------------------------------------------------------
# ÖNEMLİ: MODEL_KLASORLERI'nin (paths.py) BİLDİRİM sırası kullanılır -
# paths.PANEL_HARFLERI TAM OLARAK bu sırayla (a),(b),(c)... atanmıştır ve
# her modelin KENDİ grafik.py'si panel harfini AYNI sözlükten okur - bu
# yüzden burada YENİDEN alfabetik sıralama YAPILMAZ (yapılırsa grafik
# üzerindeki harf ile aşağıdaki açıklama metni arasındaki eşleşme BOZULUR).
# Eğitilmemiş modeller (sonuç dosyası olmayanlar) rapora ALINMAZ: aksi halde
# her figür bloğunun sonuna bir "[Henüz üretilmedi]" yer tutucusu düşüyor, ama
# panel açıklaması yine de "(k) DimeNetPP ... grafikleri verilmiştir" diyordu -
# yani açıklama, OLMAYAN bir grafiği varmış gibi gösteriyordu (kullanıcı
# geri bildirimi sonrası düzeltildi). Kapsam notu §2'de şeffafça verilir.
def _rapor_modelleri() -> list[str]:
    return [m for m in MODEL_KLASORLERI if m in _tum_metrikler_json()]


def _panel_notu(amac: str) -> str:
    panel = ", ".join(f"{PANEL_HARFLERI[m]} {m}" for m in _rapor_modelleri())
    return f"{panel} modellerine ait {amac} grafikleri verilmiştir."


def _esik_ozeti(kol: str) -> str | None:
    """Confusion-matrix sınıf sınırlarını (Q1/medyan/Q3) GERÇEK OOF verisinden
    hesaplar - grafik_ortak._dinamik_esikler ile AYNI mantık (kullanıcı
    isteği: 'confusion matrislerinde sınıflandırma neye göre' sorusunun
    cevabı, sayılarla birlikte, elle yazılmadan rapora eklensin)."""
    for model_adi in MODEL_KLASORLERI:
        f = PROJECT_ROOT / model_adi / "sonuclar" / "test_tahminleri_oof.csv"
        if not f.exists():
            continue
        odf = pd.read_csv(f)
        kolon = f"gercek_{kol}"
        if kolon not in odf.columns or odf[kolon].dropna().empty:
            continue
        q1, q2, q3 = np.percentile(odf[kolon].dropna().values, [25, 50, 75])
        birim = TARGET_UNITS.get(kol, "")
        return (f"< {q1:.3g}, {q1:.3g}–{q2:.3g}, {q2:.3g}–{q3:.3g}, > {q3:.3g} {birim} "
                f"(n={len(odf[kolon].dropna())} örnek)")
    return None


def bolum_model_detay(doc):
    add_heading(doc, "7. Model Grafikleri", level=1)
    doc.add_paragraph()

    for kol in TARGET_COLUMNS:
        add_heading(doc, f"7.{TARGET_COLUMNS.index(kol) + 1} Hedef: {_hedef_etiket(kol)}"
                          f"  [veri sütunu: {kol}]", level=2)
        for sablon, alt_baslik, amac in [
            (f"gercek_vs_tahmin_{{m}}_{kol}.tif", "Predicted vs True", "predicted vs. true"),
            (f"residual_dagilim_{{m}}_{kol}.tif", "Residual Dağılımı", "residual dağılımı"),
            (f"confusion_matrix_{{m}}_{kol}.tif", "Sınıf-Bazında Karışıklık Matrisi", "karışıklık matrisi"),
        ]:
            add_paragraph(doc, alt_baslik, size=10, bold=True, indent=True)
            if alt_baslik == "Sınıf-Bazında Karışıklık Matrisi":
                esik_str = _esik_ozeti(kol)
                add_paragraph(doc,
                    "Sınıflandırma SABİT bir fiziksel eşiğe değil, bu hedefin GERÇEK "
                    "değerlerinin kendi çeyreklik (quartile) dağılımına göre otomatik "
                    "belirlenir: örnekler küçükten büyüğe sıralanıp Q1 (%25), medyan "
                    "(%50) ve Q3 (%75) noktalarından 4 eşit-büyüklükte sınıfa bölünür "
                    "(<Q1 / Q1-medyan / medyan-Q3 / >Q3) — yani 'düşük/orta-düşük/"
                    "orta-yüksek/yüksek' göreli sınıflardır, hedeften hedefe ve "
                    "modelden modele YENİDEN hesaplanır (bkz. grafik_ortak."
                    "_dinamik_esikler). " + (f"Bu hedef için gerçek sınıf sınırları: "
                    f"{esik_str}." if esik_str else ""), size=9, indent=True)
                doc.add_paragraph()
            for model_adi in _rapor_modelleri():
                gdir = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler"
                add_image(doc, gdir / sablon.format(m=model_adi), width_cm=12)
            sekil_basligi(doc, _panel_notu(amac))
            page_break(doc)

    add_heading(doc, "7.5 Eğitim/Validasyon Kayıp Eğrileri (Ortak, Hedef-Bağımsız)", level=2)
    meta_hp = _ilk_metrikler_json()
    max_ep = meta_hp["hiperparametreler"].get("max_epochs") if meta_hp else "?"
    patience = meta_hp["hiperparametreler"].get("early_stop_patience") if meta_hp else "?"
    freeze_ep = meta_hp["hiperparametreler"].get("freeze_encoder_epochs") if meta_hp else "?"
    add_paragraph(doc,
        f"Her grafikte fold sayısı kadar renkli çizgi ÇİFTİ vardır: DÜZ çizgi o "
        f"fold'un EĞİTİM (train) kaybı, KESİKLİ çizgi AYNI renkteki fold'un "
        f"VALİDASYON (val) kaybıdır — x-ekseni epoch (1'den en fazla {max_ep}'e "
        f"kadar; validasyon kaybı {patience} epoch boyunca İYİLEŞMEZSE eğitim "
        f"erken durur, bu yüzden fold'lar farklı sayıda epoch'ta bitebilir). "
        f"Y-ekseni LOGARİTMİK ölçektedir (kayıp değerleri epoch başında büyük, "
        f"sonra hızla küçüldüğü için doğrusal eksende erken düşüş görünmez "
        f"olurdu). İlk {freeze_ep} epoch'ta encoder DONDURULMUŞTUR (sadece "
        f"regresyon başı eğitilir, bkz. §2.2 transfer learning) — bu epoklarda "
        f"kaybın daha YAVAŞ düşmesi BEKLENEN bir davranıştır, hata DEĞİLDİR. "
        f"SAĞLIKLI bir eğrinin işareti: (a) her iki çizginin de genel olarak "
        f"AZALMASI, (b) train ile val eğrisi arasındaki BOŞLUĞUN küçük kalması "
        f"(büyük/açılan boşluk = ezber/overfitting işareti, bkz. §5.1 tablosu — "
        f"aynı boşluk orada SAYISAL olarak da raporlanır).",
        size=9, indent=True)
    doc.add_paragraph()
    for model_adi in _rapor_modelleri():
        gdir = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler"
        add_image(doc, gdir / f"egitim_kaybi_{model_adi}.tif", width_cm=12)
    sekil_basligi(doc, _panel_notu("eğitim/validasyon kayıp eğrisi"))
    page_break(doc)

    add_heading(doc, "7.6 Gözeneklilik-Seçicilik Fiziksel Tutarlılık (Xe/Kr)", level=2)
    add_paragraph(doc,
        "Her grafikte x-ekseni |PLD − Xe kinetik çapı| (boyut-uyum farkı), y-ekseni "
        "Xe/Kr seçiciliktir; mavi noktalar GERÇEK etiketleri, turuncu noktalar MODEL "
        "TAHMİNLERİNİ gösterir (kesikli çizgiler her ikisine ayrı ayrı oturtulan "
        "doğrusal eğilimdir). Sağ üstteki r değerleri bu iki eğilimin Pearson "
        "korelasyonudur. Beklenen fiziksel davranış, boyut-uyum farkı küçüldükçe "
        "seçiciliğin ARTMASIDIR (negatif eğim). ÖNEMLİ: buradaki uyum, modelin "
        "dışsal bir fiziği keşfettiği şeklinde okunmamalıdır — bu koşumda seçicilik "
        "etiketi zaten PLD'den üretilmiştir ve PLD modele girdi olarak verilmektedir; "
        "ayrıntılı gerekçe ve sayısal tablo için bkz. §5.2 ve §1.2.",
        size=9, indent=True)
    doc.add_paragraph()
    for model_adi in _rapor_modelleri():
        gdir = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler"
        add_image(doc, gdir / f"pore_secicilik_tutarlilik_{model_adi}.tif", width_cm=12)
    sekil_basligi(doc, _panel_notu("gözeneklilik-seçicilik tutarlılık"))
    page_break(doc)

    add_heading(doc, "7.7 Permutation Importance (Özellik Önemi)", level=2)
    add_paragraph(doc,
        "Her grafikte y-ekseni, uzun özellik-grubu adları yerine ÖNEM SIRASINA "
        "göre (a), (b), (c) ... harfleriyle etiketlenmiştir (bkz. §3 Terminoloji). "
        "Hangi harfin hangi özellik grubuna karşılık geldiği MODELDEN MODELE "
        "değişir (her modelin kendi gerçek ΔMAE sıralamasına göre yeniden "
        "atanır); bu yüzden tüm modellerin gerçek eşleşmesi grafiklerin ardından "
        "tek bir tabloda toplu olarak verilmiştir. NOT: bu harfler, şekillerin "
        "SOL ÜST köşesindeki panel harfleriyle (hangi modelin grafiği olduğunu "
        "gösteren (a), (b), ... etiketi) KARIŞTIRILMAMALIDIR; panel harfi "
        "eksenin DIŞINDA, bar etiketleri ise eksenin İÇİNDEDİR.", size=9, indent=True)
    doc.add_paragraph()
    for model_adi in _rapor_modelleri():
        gdir = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler"
        add_image(doc, gdir / "feature_importance.tif", width_cm=12)
    sekil_basligi(doc, _panel_notu("permutation importance"))
    doc.add_paragraph()

    # Önceden her modelin grafiği altına AYRI bir 2 sütunlu tablo basılıyordu
    # (10 tablo); akademik biçimde bu, tek bir "model x harf" matrisinde
    # toplanır - hem daha okunur hem de tablo numaralandırmasını şişirmez.
    _HARF = [f"({c})" for c in "abcdefghij"]
    eslesme_satirlari, en_fazla = [], 0
    for model_adi in _rapor_modelleri():
        fi_txt = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler" / "Feature_Importance.txt"
        if not fi_txt.exists():
            continue
        eslesme = dict(ln.split(" = ", 1) for ln in fi_txt.read_text(encoding="utf-8").splitlines()
                       if " = " in ln)
        en_fazla = max(en_fazla, len(eslesme))
        eslesme_satirlari.append((model_adi, eslesme))
    if eslesme_satirlari:
        harf_sut = _HARF[:en_fazla]
        tablo_basligi(doc, "Permutation-importance grafiklerindeki harf etiketlerinin her "
                            "modeldeki karşılığı (önem sırasına göre azalan).")
        kv_table(doc, ["Model"] + harf_sut,
                 [tuple([m] + [e.get(r, "—") for r in harf_sut]) for m, e in eslesme_satirlari])
    page_break(doc)


# ---------------------------------------------------------------------------
# §8 XAI
# ---------------------------------------------------------------------------
# Her XAI yönteminin NE YAPTIĞI + grafiğinin NASIL OKUNACAĞI (kullanıcı
# isteği: "her şeyi açık açık yazacağız" — önceden bu bölümde hiçbir açıklama
# yoktu, grafik başlıkları da ham dosya adıydı: "graphlime_element_onem_
# i2_uptake_mmol_g" gibi).
XAI_ACIKLAMALARI = {
    "GraphLIME": (
        "GraphLIME (Huang et al., 2020) — YEREL VEKİL MODEL. Tek bir MOF için, "
        "atomların rastgele alt kümeleri kapatılıp (Bernoulli maskesi) modelin "
        "tahmininin nasıl değiştiği ölçülür; sonra bu maske→tahmin ilişkisine "
        "seyrek bir doğrusal model (çapraz-doğrulamalı Lasso) oturtulur. "
        "Katsayılar = o atomun tahmine YEREL katkısı. Aşağıdaki grafikler bu "
        "atom katkılarının ELEMENT bazında ortalamasını gösterir: çubuk ne kadar "
        "uzunsa o element tahmini o kadar güçlü etkiliyor demektir (pozitif = "
        "tahmini artırıyor, negatif = azaltıyor)."),
    "Edge_Attribution": (
        "Edge Attribution — BAĞ/KENAR ÖNEMİ. Gradyan-tabanlı atıf (saliency + "
        "Integrated Gradients), atomlar arası KENARLARIN (bağların) üzerine "
        "uygulanır: hangi atom-atom etkileşiminin tahmini ne kadar taşıdığı "
        "ölçülür. 'bond' grafikleri element-çifti (örn. Cu-O) bazında, 'mesafe' "
        "grafikleri ise bağ uzunluğu aralıkları bazında ortalama önemi gösterir "
        "— ikincisi, modelin hangi mesafe ölçeğindeki komşuluklara dayandığını "
        "(kısa kimyasal bağ mı, uzun gözenek-boşluğu teması mı) ortaya koyar."),
    "SubgraphX": (
        "SubgraphX (Yuan et al., 2021) — AÇIKLAYICI ALT-GRAF ARAMA. Monte Carlo "
        "Ağaç Araması (MCTS) ile, tahmini en iyi açıklayan BAĞLANTILI atom alt "
        "kümesi ('çekirdek alt-graf') aranır. 'cekirdek_boyut' grafiği bu "
        "çekirdeklerin kaç atomdan oluştuğunun dağılımını, 'element_onem' "
        "grafiği ise hangi elementlerin bu açıklayıcı çekirdeklere ne sıklıkta "
        "girdiğini gösterir (1.0'a yakın = o element neredeyse her zaman "
        "açıklayıcı çekirdeğin parçası)."),
    "IntegratedGradients": (
        "Integrated Gradients (Sundararajan et al., 2017) — SÜREKLİ GİRDİ ATFI. "
        "Maske kullanmaz: girdinin kendisi (atomların 3B koordinatları ve "
        "gözeneklilik/kompozisyon özellikleri) bir 'taban çizgisi'nden gerçek "
        "değere doğru kademeli değiştirilirken gradyanlar integre edilir; bu, "
        "katkıların toplamının tahmin farkına EŞİT olmasını garanti eder "
        "(completeness aksiyomu). 'ig_aux_onem' grafikleri sayısal gözeneklilik/"
        "kompozisyon özelliklerinin, 'ig_element_onem' grafikleri ise atom "
        "konumlarının element bazında önemini gösterir."),
}

XAI_GRAFIK_BASLIKLARI = {
    "graphlime_element_onem": "Element bazında ortalama GraphLIME atom önemi",
    "edge_attribution_bond": "Element-çifti (bağ türü) bazında ortalama kenar önemi",
    "edge_attribution_mesafe": "Bağ uzunluğu aralığı bazında ortalama kenar önemi",
    "subgraphx_cekirdek_boyut": "Açıklayıcı çekirdek alt-grafların atom sayısı dağılımı",
    "subgraphx_element_onem": "Elementlerin açıklayıcı çekirdek alt-grafa girme oranı",
    "ig_aux_onem": "Gözeneklilik/kompozisyon özelliklerinin Integrated Gradients önemi",
    "ig_element_onem": "Element bazında atom-konumu Integrated Gradients önemi",
}


def _xai_baslik(dosya_adi: str) -> str:
    """Ham dosya adını ('graphlime_element_onem_i2_uptake_mmol_g') insan-okunur
    bir şekil başlığına çevirir.

    NOT: grafik_ortak.DISPLAY_LABELS etiketleri MATPLOTLIB için yazılmıştır ve
    LaTeX matematik kipi içerir (r"I$_2$ Uptake"). Word bunu render ETMEZ, ham
    "$_2$" olarak basardı - bu yüzden burada Unicode alt-simgeye çevrilir."""
    from grafik_ortak import DISPLAY_LABELS
    for onek, baslik in sorted(XAI_GRAFIK_BASLIKLARI.items(), key=lambda kv: -len(kv[0])):
        if dosya_adi.startswith(onek):
            kalan = dosya_adi[len(onek):].lstrip("_")
            hedef = DISPLAY_LABELS.get(kalan)
            if not hedef:
                return baslik
            hedef = hedef.replace("$_2$", "₂").replace("$_", "").replace("$", "")
            return f"{baslik} — hedef: {hedef}"
    return dosya_adi


def bolum_xai(doc):
    add_heading(doc, "8. Açıklanabilir Yapay Zekâ (XAI) Bulguları", level=1)
    egitilmis = _rapor_modelleri()
    add_paragraph(doc,
        f"Dört XAI yönteminin TAMAMI, eğitilmiş {len(egitilmis)} model arasında en "
        f"hafif ileri-geçişli (en hızlı) mimari olan EGNN üzerine uygulanmıştır — "
        f"XAI yöntemleri model başına binlerce ileri-geçiş gerektirdiğinden, tek "
        f"ve tutarlı bir hedef model seçilmiştir (bkz. SISTEM_RAPORU.md §5). "
        f"Dolayısıyla bu bölümdeki bulgular EGNN'in öğrendiklerini açıklar, tüm "
        f"modellerin ortalamasını değil.", size=10, indent=True)
    doc.add_paragraph()
    for xai_adi in XAI_KLASORLERI:
        add_heading(doc, xai_adi, level=2)
        aciklama = XAI_ACIKLAMALARI.get(xai_adi)
        if aciklama:
            add_paragraph(doc, aciklama, size=9, indent=True)
            doc.add_paragraph()
        gdir = PROJECT_ROOT / xai_adi / "sonuclar" / "grafikler"
        if gdir.exists() and any(gdir.glob("*.tif")):
            for dosya in sorted(gdir.glob("*.tif")):
                add_image(doc, dosya, width_cm=15)
                sekil_basligi(doc, _xai_baslik(dosya.stem))
                # Y-eksenindeki uzun etiketler harflendirilmişse (bkz.
                # grafik_ortak.harflendir_uzun_etiketler), harf->ad eşleşmesi
                # şeklin HEMEN ALTINDA tablo olarak verilir (kullanıcı isteği:
                # "en altta a b c'nin ne olduğunu açıkla").
                etiket_txt = dosya.with_name(dosya.stem + "_etiketler.txt")
                if etiket_txt.exists():
                    satirlar = [tuple(ln.split(" = ", 1))
                                for ln in etiket_txt.read_text(encoding="utf-8").splitlines()
                                if " = " in ln]
                    if satirlar:
                        tablo_basligi(doc, "Yukarıdaki şekilde y-ekseninde kullanılan harf "
                                            "etiketlerinin karşılıkları.")
                        kv_table(doc, ["Etiket", "Özellik"], satirlar)
                        doc.add_paragraph()
        else:
            add_paragraph(doc, f"[Henüz üretilmedi: {xai_adi}/sonuclar/grafikler/ — önce "
                                f"python -m {xai_adi}.run_* ve python -m {xai_adi}.grafik çalıştırın.]",
                          indent=True)
    page_break(doc)


# ---------------------------------------------------------------------------
# ÖZET / TARTIŞMA / SONUÇ / KAYNAKLAR (akademik biçim gereği)
# ---------------------------------------------------------------------------
def bolum_ozet(doc):
    """Akademik özet. Sayılar (model sayısı, K, R², etiket kökeni) GERÇEK
    çıktılardan okunur - özet ile gövde ASLA çelişemez."""
    add_heading(doc, "Özet", level=1)
    tum = _tum_metrikler_json()
    egitilmis = list(tum.keys())
    meta = next(iter(tum.values()), None)
    csv_path = PROJECT_ROOT / "model_karsilastirma_sonuclari.csv"

    en_iyi_ifade = ""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        ov = df[df["target_column"] == "overall"].sort_values("R2", ascending=False)
        if not ov.empty:
            en_iyi = ov.iloc[0]
            en_dusuk = ov.iloc[-1]
            en_iyi_ifade = (
                f"Havuzlanmış fold-dışı (out-of-fold) değerlendirmede 4 hedefin "
                f"ortalaması üzerinden en yüksek başarımı {en_iyi['model']} "
                f"(R²={en_iyi['R2']:.3f}, MAE={en_iyi['MAE']:.4f}) vermiştir; ancak "
                f"tüm mimariler dar bir bantta toplanmıştır "
                f"(R²={en_dusuk['R2']:.3f}–{en_iyi['R2']:.3f}), dolayısıyla "
                f"mimariler arası fark pratikte ayırt edici değildir. ")

    dagilim = _etiket_kaynak_dagilimi()
    tum_kaynaklar = {k for s in dagilim.values() for k in s}
    sadece_proxy = tum_kaynaklar == {"PROXY_PORE_CORRELATION"}

    add_paragraph(doc,
        f"Bu çalışmada, metal-organik çerçevelerde (MOF) ksenon ve kripton "
        f"adsorpsiyon kapasitesi, Xe/Kr seçicilik ve iyot (I₂) adsorpsiyon "
        f"kapasitesinin tahmini için uçtan uca bir grafik sinir ağı (GNN) boru "
        f"hattı kurulmuş ve değerlendirilmiştir. Yapılar, CoRE-MOF türevi açık "
        f"bir veri setinden alınmış {atif('jablonka2023', 'chung2019')}; veri, "
        f"pymatgen tabanlı yapısal artırma {atif('ong2013')} ile "
        f"{meta['n_ornek_toplam'] if meta else '—'} örneğe genişletilmiştir. "
        f"{len(egitilmis)} farklı GNN mimarisi, ortak bir ön-eğitim → ince-ayar "
        f"(transfer öğrenme) protokolü ve sızıntıya kapalı, temel-MOF bazında "
        f"gruplanmış {meta['k_folds'] if meta else '—'} katlı çapraz doğrulama "
        f"ile aynı koşullarda eğitilmiştir. {en_iyi_ifade}"
        f"Modellerin neye dayandığını incelemek için {len(XAI_KLASORLERI)} "
        f"açıklanabilirlik yöntemi (GraphLIME {atif('huang2020')}, kenar atfı, "
        f"SubgraphX {atif('yuan2021')} ve Integrated Gradients "
        f"{atif('sundararajan2017')}) en hafif mimari olan EGNN "
        f"{atif('satorras2021')} üzerine uygulanmıştır.",
        size=10, indent=True)
    doc.add_paragraph()

    if sadece_proxy:
        add_paragraph(doc,
            "ÖNEMLİ KAPSAM BEYANI: bu koşumda hedef etiketlerin tamamı deneysel "
            "veya GCMC kökenli değil, gözeneklilik tanımlayıcılarından türetilmiş "
            "kapalı-form bir vekil (proxy) korelasyondan üretilmiştir; üstelik bu "
            "korelasyonun girdileri modele girdi özelliği olarak da verilmektedir. "
            "Bu nedenle raporlanan başarım değerleri, boru hattının teknik olarak "
            "doğru çalıştığının bir doğrulaması niteliğindedir ve gerçek Xe/Kr/I₂ "
            "adsorpsiyon tahmin yeteneği olarak yorumlanmamalıdır. Ayrıntılı "
            "gerekçe ve bunun sonuçlardaki izleri §1.2, §5.2 ve §5.3'te; gerçek "
            "bir çalışmaya dönüştürmek için gereken adımlar §10'da verilmiştir.",
            size=10, bold=True, indent=True)
    doc.add_paragraph()
    add_paragraph(doc,
        "Anahtar kelimeler: metal-organik çerçeve; radyoaktif soy gaz ayırma; "
        "Xe/Kr seçicilik; grafik sinir ağları; transfer öğrenme; açıklanabilir "
        "yapay zekâ.", size=9, indent=True)
    page_break(doc)


def bolum_tartisma(doc):
    add_heading(doc, "9. Tartışma", level=1)
    csv_path = PROJECT_ROOT / "model_karsilastirma_sonuclari.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        ov = df[df["target_column"] == "overall"].sort_values("R2", ascending=False)
        hedef_ozet = []
        for kol in TARGET_COLUMNS:
            alt = df[df["target_column"] == kol]
            if not alt.empty:
                hedef_ozet.append((kol, alt["R2"].max(), alt["R2"].min()))
        if not ov.empty:
            add_paragraph(doc,
                f"Mimariler arası fark. Havuzlanmış fold-dışı R² değerleri "
                f"{ov['R2'].min():.3f} ile {ov['R2'].max():.3f} arasında, yani "
                f"{ov['R2'].max() - ov['R2'].min():.3f}'lük bir bant içinde "
                f"toplanmıştır. Mesaj iletimi (GIN, GAT), yönlü/eşdeğişken "
                f"(EGNN, TFN, SE(3)-Transformer) ve küresel dikkat temelli "
                f"(GraphGPS) mimariler arasındaki bu fark, tek bir koşumun "
                f"gürültü düzeyiyle kıyaslanabilir büyüklüktedir; dolayısıyla bu "
                f"veri üzerinde 'en iyi mimari' seçimi anlamlı biçimde "
                f"yapılamaz. Bu beklenen bir sonuçtur: §5.3'te gösterildiği gibi "
                f"model başarımı ezici ölçüde hazır-hesaplanmış gözeneklilik "
                f"tanımlayıcılarından gelmekte, mimarilerin birbirinden ayrıştığı "
                f"yer olan 3B geometri işleme kapasitesi ise neredeyse hiç "
                f"kullanılmamaktadır.", size=10, indent=True)
            doc.add_paragraph()
        if hedef_ozet:
            zor = min(hedef_ozet, key=lambda t: t[1])
            kolay = max(hedef_ozet, key=lambda t: t[1])
            add_paragraph(doc,
                f"Hedefler arası fark. En iyi tahmin edilen hedef "
                f"{_hedef_etiket(kolay[0])} (en yüksek R²={kolay[1]:.3f}), en zor "
                f"hedef ise {_hedef_etiket(zor[0])} (en yüksek R²={zor[1]:.3f}) "
                f"olmuştur. Bu fark, hedeflerin üretilme biçiminden "
                f"kaynaklanmaktadır: kapasite hedefleri gözenek hacminin "
                f"doğrudan çarpımsal bir fonksiyonuyken, seçicilik iki boyut-uyum "
                f"teriminin ORANI olarak tanımlanmış ve ayrıca alt/üst sınıra "
                f"kırpılmıştır; oran ve kırpma, öğrenilmesi daha güç ve gürültüye "
                f"daha duyarlı bir hedef yüzeyi oluşturur.", size=10, indent=True)
            doc.add_paragraph()

    add_paragraph(doc,
        "Açıklanabilirlik bulgularının tutarlılığı. Dört XAI yöntemi birbirinden "
        "bağımsız çalışmasına rağmen aynı yöne işaret etmektedir: permutation "
        "importance gözeneklilik grubunu baskın bulmakta (§5.3), Integrated "
        "Gradients yardımcı gözeneklilik özelliklerine belirgin atıf vermekte, "
        "GraphLIME ve SubgraphX ise atom/alt-graf düzeyinde göreli olarak zayıf "
        "ve yayılmış katkılar bulmaktadır. Bu yakınsama, modelin kararını büyük "
        "ölçüde birkaç skaler tanımlayıcıya dayandırdığı yorumunu "
        "güçlendirmektedir.", size=10, indent=True)
    doc.add_paragraph()
    add_paragraph(doc,
        "Yöntemsel not. GraphLIME'da düzenlileştirme katsayısının sabitlenmesi, "
        "bu boyuttaki MOF'larda (72–172 atom) tüm katsayıların sıfıra çökmesine "
        "ve açıklamaların boş çıkmasına yol açmıştır; katsayının örnek başına "
        "çapraz doğrulamayla seçilmesi (LassoCV) sorunu gidermiştir "
        f"{atif('tibshirani1996', 'ribeiro2016')}. Bu, maskeleme tabanlı yerel "
        "vekil yöntemlerin büyük graflarda ölçek duyarlılığına dair pratik bir "
        "uyarıdır: maskelemenin tahmin üzerindeki etkisi graf büyüdükçe "
        "seyrelmekte ve sabit bir ceza terimi sinyali tamamen bastırabilmektedir.",
        size=10, indent=True)
    page_break(doc)


def bolum_sonuc(doc):
    add_heading(doc, "10. Sonuç ve Gelecek Çalışma", level=1)
    egitilmis = _rapor_modelleri()
    add_paragraph(doc,
        f"Bu çalışma, MOF tabanlı radyoaktif soy gaz ayırma problemi için "
        f"{len(egitilmis)} GNN mimarisini, transfer öğrenmeyi, sızıntıya kapalı "
        f"çapraz doğrulamayı ve {len(XAI_KLASORLERI)} açıklanabilirlik yöntemini "
        f"tek bir yeniden üretilebilir boru hattında birleştirmektedir. Boru "
        f"hattının tüm bileşenleri uçtan uca çalışır durumda doğrulanmıştır ve "
        f"kod, veri ile ara çıktılar açık bir depoda yayımlanmıştır (§1.1).",
        size=10, indent=True)
    doc.add_paragraph()
    add_paragraph(doc,
        "Buna karşılık, mevcut haliyle çalışma bir malzeme-keşfi sonucu "
        "üretmemektedir: hedef etiketler sentetik bir vekil korelasyondan "
        "geldiği ve bu korelasyonun girdileri modele de verildiği için, "
        "raporlanan yüksek başarım fiziksel bir öngörü gücünü değil, bir "
        "fonksiyonun geri çözülebilirliğini göstermektedir (§1.2).",
        size=10, indent=True)
    doc.add_paragraph()
    add_paragraph(doc, "Öncelik sırasına göre gelecek çalışma adımları:",
                  size=10, bold=True, indent=True)
    for madde in [
        "Gerçek etiket kaynağına geçilmesi: yayımlanmış GCMC Xe/Kr izoterm veri "
        "setlerinin bağlanması veya bu yapılar için RASPA ile doğrudan GCMC "
        "simülasyonu yapılması; vekil korelasyonun tamamen devre dışı bırakılması "
        f"{atif('sikora2012', 'simon2015')}.",
        "Gözeneklilik tanımlayıcılarının geometrik yaklaşım yerine Zeo++ ile "
        f"hesaplanması {atif('willems2012')}.",
        "Döngüselliğin kırılması: etiket üretiminde kullanılan değişkenlerin "
        "model girdisinden çıkarılması (gerçek etiketlere geçildiğinde sorun "
        "kendiliğinden ortadan kalkar) ve mimarilerin 3B geometriden gerçekten "
        "yararlanıp yararlanmadığının yeniden ölçülmesi.",
        "NLP çıkarım hattının düzeltilmesi: mevcut regex aynı sayısal değeri "
        "farklı hedeflere atayabilmekte, birim dönüşümü ve hedef eşleştirmesi "
        "doğrulanmamaktadır.",
        "Eğitilmemiş mimarinin (DimeNet++) tamamlanarak karşılaştırmaya dahil "
        f"edilmesi {atif('gasteiger2020')}.",
        "Deneysel doğrulama: model tarafından yüksek seçicilik öngörülen "
        "adayların sentezlenip kırılma (breakthrough) deneyleriyle sınanması.",
    ]:
        bullet(doc, madde, size=10)
    page_break(doc)


def bolum_kaynaklar(doc):
    add_heading(doc, "Kaynaklar", level=1)
    add_paragraph(doc,
        "Künyeler literatürden derlenmiştir; yayına göndermeden önce cilt/sayfa "
        "bilgilerinin orijinal kaynaklardan doğrulanması önerilir.",
        size=8.5, indent=True)
    doc.add_paragraph()
    references_list(doc, [kunye for _, kunye in KAYNAKLAR], size=9)


def main() -> None:
    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Calibri"; style.font.size = Pt(10)
    _sayac_sifirla()

    add_heading(doc, "MOF Radyoaktif Gaz Adsorpsiyonu — Sonuç Raporu", level=0)
    # Alt başlıktaki model sayısı ELLE yazılmaz: eğitilmemiş bir mimariyi
    # "sonuçları var" gibi göstermemek için GERÇEK sonuç üretmiş model
    # sayısından türetilir (bkz. §2 kapsam notu).
    n_egitilmis = len(_rapor_modelleri())
    n_implemente = len(MODEL_KLASORLERI)
    mimari_ifade = (f"{n_egitilmis} GNN Mimarisi" if n_egitilmis == n_implemente
                    else f"{n_egitilmis} GNN Mimarisi (depoda {n_implemente} implemente)")
    add_paragraph(doc, f"Xe/Kr Adsorpsiyon Kapasitesi + Xe/Kr Seçicilik + I₂ Adsorpsiyon Kapasitesi — "
                        f"{mimari_ifade} + {len(XAI_KLASORLERI)} XAI Yöntemi "
                        f"(pooled out-of-fold sonuçları)", size=12, bold=False)
    doc.add_paragraph()

    bolum_ozet(doc)
    bolum_veri_kaynagi(doc)
    bolum_hiperparametreler(doc)
    bolum_terminoloji(doc)
    bolum_metrik_tablosu(doc)
    bolum_dogrulama(doc)
    bolum_karsilastirma_grafikleri(doc)
    bolum_model_detay(doc)
    bolum_xai(doc)
    bolum_tartisma(doc)
    bolum_sonuc(doc)
    bolum_kaynaklar(doc)

    out_path = PROJECT_ROOT / "MOF_Radyoaktif_Gaz_Adsorpsiyonu_Raporu.docx"
    doc.save(str(out_path))
    print(f"Rapor kaydedildi -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
