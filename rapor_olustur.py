"""
rapor_olustur.py
==================
MOF Radyoaktif Gaz Adsorpsiyonu — Sonuç Raporu oluşturucu (python-docx).

STİL: "Üç Boyutlu Kristal Malzemeler/rapor_olustur.py" ile BİREBİR AYNI
sade/akademik biçim (ÜÇ-ÇİZGİLİ dergi tabloları, renkli fon/zebra-şerit
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

BİÇİM: danışmanın ONAYLADIĞI iki rapor (Termal Bariyer Kaplamalar TBC,
Katı Elektrolitler) ile BİREBİR aynı iskelet kullanılır:
    1. Model Karşılaştırma Tablosu (+ "Metrik Tanımları" + "Tablo Yorumu")
    2. Model Grafikleri (2.1 gerçek-tahmin, 2.2 hata dağılımı, 2.3 karışıklık
       matrisi, 2.4 fiziksel tutarlılık, 2.5 kayıp eğrileri, 2.6 özellik önemi,
       2.7 Model Yorumları) — 2.5 onaylı raporlarda YOKTUR, kullanıcı bu
       eğrilerin açıklanmasını ayrıca istediği için EK olarak korunmuştur
    3. XAI Grafikleri (her yöntem için grafikler + "Yorum")
    4. Kıyaslama Grafikleri (+ "Yorum")
Onaylı raporlarda Özet/Tartışma/Sonuç/Kaynakça bölümleri YOKTUR; yorum yükü
"Tablo Yorumu", "2.7 Model Yorumları" ve XAI "Yorum" paragraflarındadır.

DÜRÜSTLÜK İLKESİ (sibling projeyle AYNI): bu script model_karsilastirma_
sonuclari.csv + <Model>/sonuclar/grafikler/*.tif + XAI sonuç dosyalarını
ÇALIŞTIRILDIKTAN SONRA OTOMATİK OLARAK OKUYUP rapora gömer — hiçbir sayı
elle yazılmaz. Henüz üretilmemiş bir dosya varsa rapor bunu şeffafça
"[Henüz üretilmedi]" olarak işaretler.

VERİ KAYNAĞI: yapıların kökeni, etiketlerin nasıl üretildiği ve hangi
kısmının sentetik olduğu BİLGİ RAPORU §1'de belgelenir (onaylı raporlarda da
veri seti ayrıntısı bilgi raporundadır); bu rapor yalnızca "Tablo Yorumu"
içinde sonuçların yorumlanması için gereken kritik uyarıyı taşır.

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


# ---------------------------------------------------------------------------
# ORTAK YARDIMCILAR ("Üç Boyutlu Kristal Malzemeler/rapor_olustur.py" ile AYNI)
# ---------------------------------------------------------------------------
def bold_cell(cell, text, size=10, align=WD_ALIGN_PARAGRAPH.CENTER):
    """SADE: fon rengi YOK, sadece kalın SİYAH metin (üç-çizgili tablo
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
    """Şekil altına 'Fig N: <açıklama>' başlığı basar, numarayı döndürür.

    BİÇİM: danışmanın ONAYLADIĞI raporlarla (Termal Bariyer Kaplamalar, Katı
    Elektrolitler) BİREBİR aynı - orada da şekil açıklamaları
    'Fig 1: (a) CHGNet, (b) M3GNet, ... grafikleri verilmiştir.' biçimindedir."""
    _SAYAC["sekil"] += 1
    add_fig_caption(doc, f"Fig {_SAYAC['sekil']}: {aciklama}")
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
    """Tek hücreli, fonsuz 'kutu' - sadece ince çerçeve (renkli fon
    YOK), makaleye kopyalanabilecek atıf/alıntı metnini görsel olarak ayırır."""
    tablo = doc.add_table(rows=1, cols=1)
    tablo.style = None
    _kenarlik(tablo, kalin_ust=4, kalin_alt=4)
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


def _kenarlik(tablo, kalin_ust=8, orta=4, kalin_alt=8) -> None:
    """ÜÇ-ÇİZGİLİ (booktabs) akademik tablo biçimi: yalnızca (1) tablonun
    üstünde kalın bir çizgi, (2) başlık satırının altında ince bir çizgi,
    (3) tablonun altında kalın bir çizgi bulunur. DİKEY çizgi ve satır-arası
    yatay çizgi YOKTUR.

    Word'ün 'Table Grid' stili tüm hücreleri kutular; bu, kelime-işlemci
    varsayılanıdır ve akademik dergilerde (ACS, Elsevier, Nature) kullanılmaz.
    Bu fonksiyon bunun yerine dergi biçimini uygular."""
    tblPr = tablo._tbl.tblPr
    for eski in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(eski)
    borders = OxmlElement("w:tblBorders")
    for ad, sz in [("top", kalin_ust), ("bottom", kalin_alt)]:
        e = OxmlElement(f"w:{ad}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        borders.append(e)
    for ad in ("left", "right", "insideV", "insideH"):
        e = OxmlElement(f"w:{ad}")
        e.set(qn("w:val"), "none"); e.set(qn("w:sz"), "0"); e.set(qn("w:space"), "0")
        borders.append(e)
    tblPr.append(borders)

    # başlık satırının ALTINA tek ince çizgi
    for hucre in tablo.rows[0].cells:
        tcPr = hucre._tc.get_or_add_tcPr()
        tcB = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(orta))
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        tcB.append(e)
        tcPr.append(tcB)


def _baslik_satirini_tekrarla(tablo) -> None:
    """Tablo sayfa sınırını aşarsa başlık satırı her sayfada TEKRARLANIR -
    akademik biçimde uzun tablolar için standarttır."""
    trPr = tablo.rows[0]._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader")
    e.set(qn("w:val"), "true")
    trPr.append(e)


def kv_table(doc, headers, rows_data):
    """Akademik ÜÇ-ÇİZGİLİ tablo: kalın başlık satırı, dikey çizgi yok,
    satır-arası çizgi yok, fon rengi/zebra-şerit yok."""
    tablo = doc.add_table(rows=1 + len(rows_data), cols=len(headers))
    tablo.style = None
    for j, h in enumerate(headers):
        bold_cell(tablo.rows[0].cells[j], h)
    for ri, row in enumerate(rows_data, 1):
        for ci, val in enumerate(row):
            normal_cell(tablo.rows[ri].cells[ci], str(val),
                        align=WD_ALIGN_PARAGRAPH.CENTER if ci > 0 else WD_ALIGN_PARAGRAPH.LEFT,
                        bold=(ci == 0))
    _kenarlik(tablo)
    _baslik_satirini_tekrarla(tablo)
    return tablo


# ---------------------------------------------------------------------------
# §1 VERİ KAYNAĞI VE ATIF (Bilesen 2 gerekliligi - kullanicinin "atif" istegi)
# ---------------------------------------------------------------------------
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
        parcalar.append(f"§2.6'da 'Pore Geometry' ΔMAE'si 'Crystal Structure'ınkinin "
                         f"{min(oranlar):.0f}-{max(oranlar):.0f} katıdır")
    if r_gercek_list and r_tahmin_list:
        parcalar.append(f"§2.4'te modelin ürettiği korelasyon "
                         f"(r={min(r_tahmin_list):.3f}…{max(r_tahmin_list):.3f}), gürültülü "
                         f"gerçek etiketlerinkinden (r={np.mean(r_gercek_list):.3f}) DAHA güçlüdür")
    if not parcalar:
        return ""
    return "Bunun izleri sonuçlarda görülebilir: " + " ve ".join(parcalar) + "."


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


def _rapor_modelleri() -> list[str]:
    return [m for m in MODEL_KLASORLERI if m in _tum_metrikler_json()]


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


# ---------------------------------------------------------------------------
# MODELE ÖZGÜ MİMARİ NOTLARI (onaylı raporlardaki "Mimari Notu" sütunu)
# ---------------------------------------------------------------------------
MIMARI_NOTU = {
    "GraphGPS": "Yerel mesaj iletimi (GINEConv) + küresel çok-başlı dikkat",
    "PNA_GNN": "Çoklu toplayıcı (mean/min/max/std) × ölçekleyici kombinasyonu",
    "GIN": "Kenar-özelliği duyarlı Graph Isomorphism Network",
    "GAT": "GATv2 — kenar-mesafesi duyarlı dikkat",
    "GatedGCN": "Öğrenilen kenar kapılama (residual gated)",
    "DeeperGCN": "GENConv + 'res+' blokları, 8 katman (en derin model)",
    "ECC": "Kenar-koşullu dinamik filtre üretimi (NNConv)",
    "TFN": "Tensor Field Network — sıfırdan, l≤1 Clebsch-Gordan eşleşmesi",
    "EGNN": "E(n)-ekvaryant, mesafe-değişmez mesaj iletimi — en hafif model, XAI tabanı",
    "SE3_Transformer": "SE(3)-ekvaryant çok-başlı dikkat, l≤1 CG eşleşmesi",
    "DimeNetPP": "Yönlü/açısal (k,j,i) üçlü mesaj iletimi",
}


def _karsilastirma_df() -> pd.DataFrame | None:
    yol = PROJECT_ROOT / "model_karsilastirma_sonuclari.csv"
    if not yol.exists():
        return None
    return pd.read_csv(yol)


# ---------------------------------------------------------------------------
# §1 MODEL KARŞILAŞTIRMA TABLOSU
# ---------------------------------------------------------------------------
def bolum_karsilastirma_tablosu(doc):
    add_heading(doc, "1. Model Karşılaştırma Tablosu", level=1)
    df = _karsilastirma_df()
    tum_meta = _tum_metrikler_json()
    if df is None or not tum_meta:
        add_paragraph(doc, "[Henüz üretilmedi: model_karsilastirma_sonuclari.csv — önce "
                            "modellerin run_*.py'sini, sonra model_karsilastirma.py'yi "
                            "çalıştırın.]", indent=True)
        page_break(doc)
        return

    meta = next(iter(tum_meta.values()))
    overall = df[df["target_column"] == "overall"].sort_values("R2", ascending=False).reset_index(drop=True)
    egitilmis = _rapor_modelleri()
    egitilmemis = [m for m in MODEL_KLASORLERI if m not in tum_meta]

    add_paragraph(doc,
        f"Aşağıdaki tablo, {len(egitilmis)} farklı GNN mimarisini "
        f"{meta['k_folds']}-katlı çapraz doğrulama (K-Fold, temel-MOF bazlı, veri "
        f"sızıntısı yok) sonuçlarına göre karşılaştırmaktadır. Metrikler, tüm "
        f"foldların havuzlanmış (pooled) fold-dışı (OOF) tahminleri üzerinden "
        f"hesaplanmıştır — yani {meta['n_ornek_toplam']} kaydın tamamı tam bir "
        f"test seti gibi değerlendirilmiştir. Modeller R² skoruna göre azalan "
        f"sırada sıralanmıştır. Tabloda raporlanan değerler 4 hedefin "
        f"(Xe kapasitesi, Kr kapasitesi, Xe/Kr seçicilik, I₂ kapasitesi) metrik "
        f"ORTALAMASIDIR; hedef bazında ayrıntı §2'deki grafiklerde verilmektedir.",
        size=10, indent=True)
    if egitilmemis:
        add_paragraph(doc,
            f"Kapsam notu: depoda {len(MODEL_KLASORLERI)} mimari implemente "
            f"edilmiştir, ancak bu koşumda {len(egitilmis)} tanesi eğitilmiştir; "
            f"{', '.join(egitilmemis)} eğitilmediği için hiçbir tabloda/grafikte "
            f"yer almamaktadır.", size=9, indent=True)
    doc.add_paragraph()

    kolonlar = ["Sıra", "Model", "R²", "MAE", "RMSE", "MedianAE", "MaxErr", "PearsonR", "Mimari Notu"]
    tablo_basligi(doc, "Havuzlanmış fold-dışı (OOF) model karşılaştırması, 4 hedefin ortalaması.")
    tablo = doc.add_table(rows=1 + len(overall), cols=len(kolonlar))
    tablo.style = None
    for j, k in enumerate(kolonlar):
        bold_cell(tablo.rows[0].cells[j], k, size=8.5)
    en_iyi = {"R2": overall["R2"].max(), "PearsonR": overall["PearsonR"].max(),
              "MAE": overall["MAE"].min(), "RMSE": overall["RMSE"].min(),
              "MedianAE": overall["MedianAE"].min(), "MaxError": overall["MaxError"].min()}
    for i, (_, row) in enumerate(overall.iterrows(), 1):
        hucreler = [
            (str(i), False),
            (row["model"], True),
            (f"{row['R2']:.4f}", abs(row["R2"] - en_iyi["R2"]) < 1e-12),
            (f"{row['MAE']:.4f}", abs(row["MAE"] - en_iyi["MAE"]) < 1e-12),
            (f"{row['RMSE']:.4f}", abs(row["RMSE"] - en_iyi["RMSE"]) < 1e-12),
            (f"{row['MedianAE']:.4f}", abs(row["MedianAE"] - en_iyi["MedianAE"]) < 1e-12),
            (f"{row['MaxError']:.4f}", abs(row["MaxError"] - en_iyi["MaxError"]) < 1e-12),
            (f"{row['PearsonR']:.4f}", abs(row["PearsonR"] - en_iyi["PearsonR"]) < 1e-12),
            (MIMARI_NOTU.get(row["model"], "—"), False),
        ]
        for j, (metin, kalin) in enumerate(hucreler):
            normal_cell(tablo.rows[i].cells[j], metin,
                        size=8.5,
                        align=WD_ALIGN_PARAGRAPH.LEFT if j in (1, 8) else WD_ALIGN_PARAGRAPH.CENTER,
                        bold=kalin)
    _kenarlik(tablo)
    _baslik_satirini_tekrarla(tablo)
    doc.add_paragraph()

    add_paragraph(doc,
        "Metrik Tanımları: R² (Determinasyon Katsayısı) — Modelin hedef "
        "değişkenin varyansını açıklama oranı. MAE (Ortalama Mutlak Hata) — "
        "hedefle aynı birimde, doğrudan yorumlanabilir hata. RMSE (Kök Ortalama "
        "Kare Hata) — büyük hataları MAE'ye kıyasla daha fazla cezalandırır. "
        "MedianAE (Ortanca Mutlak Hata) — aykırı değerlere karşı dayanıklı, "
        "tipik tahmin kalitesi. MaxErr (Maksimum Hata) — en kötü-durum "
        "senaryosu. PearsonR — tahmin ve gerçek değer arasındaki doğrusal "
        "korelasyon. Not: 4 hedefin birimleri farklı olduğundan (mmol/g ve "
        "birimsiz seçicilik) ortalama metrikler birimsiz kabul edilmelidir; "
        "birim taşıyan hedef-bazında değerler §2'de sunulmaktadır.",
        size=9, indent=True)
    doc.add_paragraph()
    _tablo_yorumu(doc, overall, df)
    doc.add_paragraph()
    _bolum_terimler(doc, meta)
    page_break(doc)


def _bolum_terimler(doc, meta) -> None:
    """Bu raporu ve grafiklerini okumak için gereken terimler, konuya aşina
    OLMAYAN bir okuyucu için açıklanır. Tam sözlük (eğitim süreci, GNN
    mimarisi, malzeme bilimi, XAI terimleri dahil) Bilgi Raporu §10'dadır."""
    k = meta["k_folds"] if meta else "K"
    add_paragraph(doc, "Terim Açıklamaları (bu raporu okumak için):", size=10, bold=True, indent=True)
    for terim, aciklama in [
        ("K-Fold çapraz doğrulama",
         f"Veri {k} eşit parçaya ('fold') bölünür. {k} ayrı model eğitilir; her birinde "
         f"parçalardan biri TEST, kalanları EĞİTİM olarak kullanılır. Böylece HER örnek "
         f"tam olarak bir kez, modelin onu hiç görmediği bir turda test edilmiş olur. "
         f"Buradaki K, kaç parçaya bölündüğümüzdür — bu koşumda K={k}."),
        ("Fold (kat)",
         "Bu parçalardan biri. GRAFİKLERDEKİ 'Fold 1 / Fold 2 / Fold 3' renkleri, o "
         "noktanın hangi turda TEST verisi olarak tahmin edildiğini gösterir; yani her "
         "nokta, modelin o MOF'u hiç görmeden yaptığı tahmindir. Renklerin birbirine "
         "karışmış olması, foldların birbiriyle tutarlı sonuç verdiğini gösterir."),
        ("Fold-dışı (OOF) tahmin ve 'havuzlanmış' metrik",
         "Bir örnek için, o örneğin TEST fold'unda olduğu turda üretilen tahmin. "
         "'Havuzlanmış (pooled)' metrik, tüm foldların bu tahminlerinin tek listede "
         "birleştirilip tek bir R²/MAE hesaplanmasıdır — yani tüm veri seti tek bir test "
         "seti gibi değerlendirilir ve hiçbir örnek kendi eğitim verisiyle ölçülmez."),
        ("Sızıntısız (temel-MOF bazlı) bölme",
         "Veri artırma ile bir MOF'tan birden fazla varyant üretildiğinden, varyantın "
         "eğitimde orijinalinin testte olması 'kopya çekme' anlamına gelirdi. Bu yüzden "
         "bölme örnek bazında değil TEMEL MOF bazında yapılır: bir MOF'un tüm varyantları "
         "hep aynı fold'dadır."),
        ("Artık (residual)",
         "Tahmin − gerçek değer. Hata dağılımı grafiklerinde bu değerin histogramı "
         "gösterilir; sıfır etrafında dar ve simetrik olması istenir."),
        ("Permütasyon önemi (ΔMAE)",
         "Bir özellik grubunun değerleri örnekler arasında rastgele karıştırılır ve "
         "hatanın ne kadar KÖTÜLEŞTİĞİ ölçülür. Çok kötüleşiyorsa model o özelliğe "
         "bağımlıdır. ΔMAE = karıştırma sonrası MAE − baz MAE."),
        ("Panel harfleri (a), (b), ...",
         "Çok panelli şekillerde her grafiğin SAĞ ÜST köşesindeki harf, şekil altındaki "
         "açıklamada hangi modele ait olduğunu söyler."),
        ("Hedefler",
         "Xe/Kr/I₂ kapasitesi: 1 gram MOF'un tutabildiği gaz miktarı (mmol/g). Xe/Kr "
         "seçicilik: MOF'un Xe'yi Kr'ye göre ne kadar tercihen tuttuğu (birimsiz) — "
         "nükleer atık gazı ayırmada kritik metrik."),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        r = p.add_run(f"{terim}: "); r.bold = True; r.font.size = Pt(9)
        r2 = p.add_run(aciklama); r2.font.size = Pt(9)
    doc.add_paragraph()
    add_paragraph(doc,
        "Eğitim süreci (epoch, batch, erken durdurma, transfer öğrenme), GNN mimarisi "
        "(mesaj iletimi, gömme, ekvaryans), malzeme bilimi (MOF, PLD, LCD, boyut-eleme, "
        "GCMC) ve XAI (Lasso, maskeleme, Integrated Gradients) terimlerinin tam "
        "açıklamaları Bilgi Raporu §10 'Terimler Sözlüğü'ndedir.", size=9, indent=True)


def _tablo_yorumu(doc, overall: pd.DataFrame, df: pd.DataFrame) -> None:
    """Onaylı raporlardaki 'Tablo Yorumu' paragrafı - TÜM sayılar gerçek
    çıktılardan okunur, elle yazılmaz."""
    en_iyi = overall.iloc[0]
    en_dusuk = overall.iloc[-1]
    bant = en_iyi["R2"] - en_dusuk["R2"]

    hedef_satir = []
    for kol in TARGET_COLUMNS:
        alt = df[df["target_column"] == kol]
        if not alt.empty:
            hedef_satir.append((kol, alt["R2"].max()))
    kolay = max(hedef_satir, key=lambda t: t[1]) if hedef_satir else None
    zor = min(hedef_satir, key=lambda t: t[1]) if hedef_satir else None

    dagilim = _etiket_kaynak_dagilimi()
    tum_kaynaklar = {k for s in dagilim.values() for k in s}
    sadece_proxy = tum_kaynaklar == {"PROXY_PORE_CORRELATION"}

    metin = (
        f"Tablo Yorumu: En yüksek R² skoruna {en_iyi['model']} ({en_iyi['R2']:.4f}) "
        f"ulaşmıştır; en düşük skor {en_dusuk['model']}'e ({en_dusuk['R2']:.4f}) "
        f"aittir. Aradaki fark yalnızca {bant:.4f}'tür — yani "
        f"{len(overall)} mimarinin tamamı son derece dar bir performans bandında "
        f"toplanmıştır. Mesaj iletimi (GIN, GAT, GatedGCN), çoklu-toplayıcı "
        f"(PNA-GNN), küresel dikkat (GraphGPS) ve E(3)-ekvaryant (EGNN, TFN, "
        f"SE(3)-Transformer) gibi birbirinden tasarım olarak çok farklı aileler "
        f"arasındaki bu fark, tek bir koşumun gürültü düzeyiyle kıyaslanabilir "
        f"büyüklüktedir; dolayısıyla bu veri seti üzerinde mimariler arasında "
        f"anlamlı bir üstünlük sıralaması yapılamaz. ")
    if kolay and zor:
        metin += (
            f"Hedefler arasında ise belirgin bir fark vardır: en iyi öğrenilen "
            f"hedef {_hedef_etiket(kolay[0])} (en yüksek R²={kolay[1]:.3f}), en zor "
            f"hedef {_hedef_etiket(zor[0])} (en yüksek R²={zor[1]:.3f}) olmuştur. "
            f"Bu fark hedeflerin tanımından kaynaklanmaktadır: kapasite hedefleri "
            f"gözenek hacminin doğrudan çarpımsal bir fonksiyonuyken, seçicilik "
            f"iki boyut-uyum teriminin ORANI olarak tanımlanmış ve alt/üst sınıra "
            f"kırpılmıştır; oran ve kırpma, öğrenilmesi daha güç ve gürültüye daha "
            f"duyarlı bir hedef yüzeyi oluşturur. ")
    if sadece_proxy:
        metin += (
            "KRİTİK UYARI — bu skorların yorumlanması için belirleyicidir: bu "
            "koşumda hedef etiketlerin TAMAMI (%100) 'PROXY_PORE_CORRELATION' "
            "kaynaklıdır: her etiket, o MOF'un gözeneklilik tanımlayıcılarından "
            "(PLD, gözenek hacmi, açık metal bölgesi, fonksiyonel grup) "
            "kapalı-form bir formülle hesaplanmış ve üzerine lognormal gürültü "
            "eklenmiş sentetik bir sayıdır. Dahası bu formülün dört girdisinin "
            "dördü de modele "
            "yardımcı (aux) GİRDİ özelliği olarak verilmektedir — yani model, "
            "kendi girdilerinden hesaplanan bir formülü geri çözmeyi "
            "öğrenmektedir ve R² tavanı fiziksel öğrenme kapasitesiyle değil, "
            "etikete enjekte edilen gürültüyle belirlenir. " + _donguselluk_kaniti() +
            " Dolayısıyla yukarıdaki değerler, boru hattının teknik olarak doğru "
            "çalıştığını gösteren bir ALTYAPI DOĞRULAMASIDIR; gerçek Xe/Kr/I₂ "
            "adsorpsiyon tahmin başarısı olarak sunulamaz. Ayrıntılı köken "
            "dökümü ve gerçek bir çalışmaya dönüştürme adımları Bilgi Raporu "
            "§1.2'de ve depodaki VERI_KAYNAGI_VE_SINIRLAMALAR.md belgesinde "
            "verilmiştir.")
    add_paragraph(doc, metin, size=9, indent=True)


# ---------------------------------------------------------------------------
# §2 MODEL GRAFİKLERİ
# ---------------------------------------------------------------------------
def _panel_notu(amac: str) -> str:
    panel = ", ".join(f"{PANEL_HARFLERI[m]} {m}" for m in _rapor_modelleri())
    return f"{panel} {amac} grafikleri verilmiştir."


def _grafik_blogu(doc, sablon: str, amac: str) -> None:
    for model_adi in _rapor_modelleri():
        gdir = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler"
        add_image(doc, gdir / sablon.format(m=model_adi), width_cm=12)
    sekil_basligi(doc, _panel_notu(amac))
    page_break(doc)


def bolum_model_grafikleri(doc):
    add_heading(doc, "2. Model Grafikleri", level=1)
    add_paragraph(doc,
        "Bu bölümdeki her şekil, eğitilmiş tüm modellerin aynı grafik türünü "
        "yan yana gösterir; panel harfleri (şekillerin sağ üst köşesinde) şekil "
        "açıklamasındaki model adlarıyla eşleşir. Bu projede dört hedef "
        "bulunduğundan (Xe kapasitesi, Kr kapasitesi, Xe/Kr seçicilik, I₂ "
        "kapasitesi), grafik türlerinin her biri dört hedef için ayrı ayrı "
        "verilmektedir.", size=10, indent=True)
    doc.add_paragraph()

    add_heading(doc, "2.1 Gerçek Değer – Tahmin Grafikleri", level=2)
    add_paragraph(doc,
        "Her panelde x-ekseni gerçek, y-ekseni tahmin edilen değerdir; kesikli "
        "köşegen mükemmel tahmin çizgisidir (y = x). Noktalar fold'lara göre "
        "renklendirilmiştir ve sol üstteki kutuda o hedefe ait havuzlanmış R² ve "
        "MAE değerleri yer alır. Köşegenden sistematik sapma yanlılığı, köşegen "
        "etrafındaki yayılma ise rastgele hatayı gösterir.", size=9, indent=True)
    doc.add_paragraph()
    for kol in TARGET_COLUMNS:
        _grafik_blogu(doc, f"gercek_vs_tahmin_{{m}}_{kol}.tif",
                      f"modellerine ait {_hedef_etiket(kol)} hedefi için gerçek değer – tahmin")

    add_heading(doc, "2.2 Hata Dağılımı Grafikleri", level=2)
    add_paragraph(doc,
        "Artık (residual = tahmin − gerçek) dağılımının histogramıdır; kırmızı "
        "kesikli çizgi sıfır hatayı, sol üstteki kutu ortalama ve standart "
        "sapmayı gösterir. En yüksek sütunun üzerindeki sayı o aralığa düşen "
        "örnek adedidir. Sıfır etrafında simetrik ve dar bir dağılım, "
        "yanlılığı düşük ve tutarlı bir modeli işaret eder.", size=9, indent=True)
    doc.add_paragraph()
    for kol in TARGET_COLUMNS:
        _grafik_blogu(doc, f"residual_dagilim_{{m}}_{kol}.tif",
                      f"modellerine ait {_hedef_etiket(kol)} hedefi için hata dağılımı")

    add_heading(doc, "2.3 Sınıflandırma Karışıklık Matrisi Grafikleri", level=2)
    add_paragraph(doc,
        "Regresyon çıktısı, yorumlanabilirlik için dört sınıfa indirgenmiştir. "
        "Sınıflandırma SABİT bir fiziksel eşiğe değil, her hedefin GERÇEK "
        "değerlerinin kendi çeyreklik dağılımına göre belirlenir: örnekler "
        "küçükten büyüğe sıralanıp Q1 (%25), medyan (%50) ve Q3 (%75) "
        "noktalarından dört eşit-büyüklükte sınıfa bölünür (<Q1 / Q1–medyan / "
        "medyan–Q3 / >Q3), yani 'düşük / orta-düşük / orta-yüksek / yüksek' "
        "göreli sınıflardır ve her hedef için yeniden hesaplanır. Köşegen "
        "üzerindeki hücreler doğru sınıflandırmaları gösterir.", size=9, indent=True)
    doc.add_paragraph()
    for kol in TARGET_COLUMNS:
        esik = _esik_ozeti(kol)
        if esik:
            add_paragraph(doc, f"{_hedef_etiket(kol)} için sınıf sınırları: {esik}",
                          size=9, bold=True, indent=True)
        _grafik_blogu(doc, f"confusion_matrix_{{m}}_{kol}.tif",
                      f"modellerine ait {_hedef_etiket(kol)} hedefi için karışıklık matrisi")

    add_heading(doc, "2.4 Gözeneklilik – Seçicilik Fiziksel Tutarlılık Grafikleri", level=2)
    add_paragraph(doc,
        "Gözeneklilik literatürünün temel bulgusunun (Sikora et al. 2012) bu "
        "projedeki karşılığı: Gözenek-Sınırlayıcı Çap (PLD) hedef gazın kinetik "
        "çapına ne kadar yakınsa boyut-eleme (size-sieving) o kadar güçlenir ve "
        "Xe/Kr seçicilik o kadar yükselir — yani |PLD − d_kinetik(Xe)| ile "
        "seçicilik arasında NEGATİF bir eğilim beklenir. Her panelde mavi "
        "noktalar gerçek, turuncu noktalar tahmin edilen seçiciliği, kesikli "
        "çizgiler her ikisine ayrı ayrı oturtulan doğrusal eğilimi gösterir; "
        "sağ üstteki kutuda iki eğilimin Pearson korelasyonu verilmiştir.",
        size=9, indent=True)
    doc.add_paragraph()
    add_paragraph(doc,
        "ÖNEMLİ: buradaki uyum, veri kurgusunun bir sonucudur. Bu koşumda "
        "seçicilik etiketi zaten PLD'nin kapalı-form "
        "bir fonksiyonu olarak üretilmiştir ve PLD aynı zamanda modele girdi "
        "özelliği olarak verilmektedir; dolayısıyla eğilim, öğrenilmesi gereken "
        "veri kurgusunun garantisidir. Nitekim modelin "
        "ürettiği korelasyon, gürültü içeren 'gerçek' etiketlerinkinden daha "
        "güçlü çıkmaktadır — bu da modelin altta yatan gürültüsüz formüle "
        "yakınsadığını gösterir. Gerçek etiketlere geçildiğinde bu kontrol "
        "anlamlı bir fizik testine dönüşecektir.", size=9, indent=True)
    doc.add_paragraph()
    _grafik_blogu(doc, "pore_secicilik_tutarlilik_{m}.tif",
                  "modellerine ait gözeneklilik – seçicilik fiziksel tutarlılık")

    # NOT: danışmanın onayladığı raporlarda eğitim/validasyon kayıp eğrisi
    # bölümü YOKTUR; ancak kullanıcı bu eğrilerin açıkça anlatılmasını ayrıca
    # istediğinden, onaylı iskeleti bozmayan bir EK alt bölüm olarak korunmuştur.
    add_heading(doc, "2.5 Eğitim/Validasyon Kayıp Eğrileri", level=2)
    _bolum_kayip_egrileri(doc)

    add_heading(doc, "2.6 Özellik Önemi Analizi", level=2)
    _bolum_ozellik_onemi(doc)

    add_heading(doc, "2.7 Model Yorumları", level=2)
    _bolum_model_yorumlari(doc)


def _bolum_kayip_egrileri(doc):
    meta = _ilk_metrikler_json()
    hp = meta["hiperparametreler"] if meta else {}
    add_paragraph(doc,
        f"Her panelde fold sayısı kadar renkli çizgi ÇİFTİ vardır: DÜZ çizgi o "
        f"fold'un EĞİTİM (train) kaybı, KESİKLİ çizgi AYNI renkteki fold'un "
        f"VALİDASYON (val) kaybıdır. X-ekseni epoch'tur (1'den en fazla "
        f"{hp.get('max_epochs', '—')}'e kadar; validasyon kaybı "
        f"{hp.get('early_stop_patience', '—')} epoch boyunca iyileşmezse eğitim "
        f"erken durur, bu yüzden fold'lar farklı sayıda epoch'ta bitebilir). "
        f"Y-ekseni LOGARİTMİK ölçektedir — kayıp değerleri epoch başında büyük, "
        f"sonra hızla küçüldüğünden doğrusal eksende erken düşüş görünmez olurdu. "
        f"İlk {hp.get('freeze_encoder_epochs', '—')} epoch'ta kodlayıcı "
        f"DONDURULMUŞTUR (yalnızca regresyon başı eğitilir — transfer öğrenmenin "
        f"doğrusal sondalama aşaması); bu epoklarda kaybın daha yavaş düşmesi "
        f"BEKLENEN bir davranıştır, hata değildir. Sağlıklı bir eğrinin işareti: "
        f"(a) her iki çizginin de genel olarak azalması, (b) train ile val "
        f"eğrisi arasındaki boşluğun küçük kalması — büyüyen/açılan bir boşluk "
        f"ezber (overfitting) işareti olurdu.", size=9, indent=True)
    doc.add_paragraph()
    _grafik_blogu(doc, "egitim_kaybi_{m}.tif",
                  "modellerine ait eğitim/validasyon kayıp eğrisi")


def _bolum_ozellik_onemi(doc):
    add_paragraph(doc,
        "Permütasyon önemi analizi kapsamında, her model için kristal yapı "
        "gömmesi (Crystal Structure) ve 5 gözeneklilik/kompozisyon aux özellik "
        "grubu (Pore Geometry, Surface Area, Structural / Size, Chemical "
        "Modification, Composition-Derived) sırayla karıştırılmış ve ortaya "
        "çıkan MAE artışı (ΔMAE) ölçülmüştür. Değerler çapraz doğrulama "
        "boyunca fold ortalaması olarak raporlanmıştır. Grafiklerde y-ekseni, "
        "uzun grup adları yerine önem sırasına göre (a), (b), (c) ... "
        "harfleriyle etiketlenmiştir; harflerin her modeldeki karşılığı "
        "grafiklerin ardından tablo olarak verilmiştir.", size=9, indent=True)
    doc.add_paragraph()
    for model_adi in _rapor_modelleri():
        gdir = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler"
        add_image(doc, gdir / "feature_importance.tif", width_cm=12)
    sekil_basligi(doc, _panel_notu("modellerine ait permütasyon önemi (ΔMAE)"))
    doc.add_paragraph()

    _HARF = [f"({c})" for c in "abcdefghij"]
    satirlar, en_fazla = [], 0
    for model_adi in _rapor_modelleri():
        fi = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler" / "Feature_Importance.txt"
        if not fi.exists():
            continue
        eslesme = dict(ln.split(" = ", 1) for ln in fi.read_text(encoding="utf-8").splitlines()
                       if " = " in ln)
        en_fazla = max(en_fazla, len(eslesme))
        satirlar.append((model_adi, eslesme))
    if satirlar:
        harf_sut = _HARF[:en_fazla]
        tablo_basligi(doc, "Permütasyon önemi grafiklerindeki harf etiketlerinin her modeldeki "
                            "karşılığı (önem sırasına göre azalan).")
        kv_table(doc, ["Model"] + harf_sut,
                 [tuple([m] + [e.get(h, "—") for h in harf_sut]) for m, e in satirlar])
        doc.add_paragraph()

    # Sayısal özet tablosu — Crystal Structure vs Pore Geometry
    satirlar2 = []
    for model_adi in _rapor_modelleri():
        f = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler" / "perm_importance.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        yapi, gozenek = s.get("Crystal Structure", 0.0), s.get("Pore Geometry", 0.0)
        oran = gozenek / yapi if yapi > 1e-12 else float("nan")
        satirlar2.append((model_adi, f"{yapi:.4f}", f"{gozenek:.4f}",
                          "—" if not np.isfinite(oran) else f"{oran:.0f}×"))
    if satirlar2:
        tablo_basligi(doc, "3B yapı gömmesi ile gözeneklilik tanımlayıcılarının "
                            "karıştırılmasının MAE üzerindeki etkisi (ΔMAE).")
        kv_table(doc, ["Model", "ΔMAE (3B yapı)", "ΔMAE (gözeneklilik)", "Oran"], satirlar2)
        doc.add_paragraph()
        oranlar = [float(s[3].rstrip("×")) for s in satirlar2 if s[3] != "—"]
        if oranlar:
            add_paragraph(doc,
                f"Her iki ΔMAE de pozitiftir (karıştırma başarımı bozuyor), ancak "
                f"büyüklükleri arasında uçurum vardır: gözeneklilik "
                f"tanımlayıcılarının karıştırılması, 3B atomistik yapının "
                f"karıştırılmasından {min(oranlar):.0f}–{max(oranlar):.0f} kat daha "
                f"fazla zarar vermektedir. Yani model ağırlıklı olarak "
                f"hazır-hesaplanmış gözeneklilik sayılarına dayanmakta, grafik "
                f"kodlayıcının öğrendiği 3B geometriye ise çok az dayanmaktadır. "
                f"Bu, modelin bir kusuru değil veri kurgusunun doğrudan sonucudur: "
                f"etiketler zaten gözeneklilik tanımlayıcılarından üretildiğinden "
                f"3B geometri ek bilgi taşımamaktadır. Gerçek etiketlere "
                f"geçildiğinde bu oranın belirgin biçimde düşmesi beklenir ve bu, "
                f"3B mimarilerin gerçekten katkı sağlayıp sağlamadığının asıl testi "
                f"olacaktır.", size=9, indent=True)
    page_break(doc)


def _bolum_model_yorumlari(doc):
    """Onaylı raporlardaki '2.6 Model Yorumları' - her model için ayrı paragraf,
    sayılar gerçek çıktılardan."""
    df = _karsilastirma_df()
    tum_meta = _tum_metrikler_json()
    if df is None or not tum_meta:
        add_paragraph(doc, "[Henüz üretilmedi.]", indent=True)
        return
    overall = df[df["target_column"] == "overall"].sort_values("R2", ascending=False).reset_index(drop=True)
    sira = {row["model"]: i for i, (_, row) in enumerate(overall.iterrows(), 1)}

    for _, row in overall.iterrows():
        model_adi = row["model"]
        add_paragraph(doc, model_adi, size=10, bold=True, indent=True)
        hp = tum_meta.get(model_adi, {}).get("hiperparametreler", {})
        mimari = ", ".join(f"{k}={v}" for k, v in hp.items()
                           if k in ("n_layers", "heads", "l_max", "max_degree", "hidden"))
        perm_yol = PROJECT_ROOT / model_adi / "sonuclar" / "grafikler" / "perm_importance.json"
        onem_cumlesi = ""
        if perm_yol.exists():
            skor = json.loads(perm_yol.read_text(encoding="utf-8"))
            sirali = sorted(skor.items(), key=lambda kv: -kv[1])[:2]
            onem_cumlesi = ("Permütasyon önemi analizinde " +
                            " ve ".join(f"{ad} ({deger:.4f})" for ad, deger in sirali) +
                            " en belirleyici gruplardır.")
        add_paragraph(doc,
            f"{MIMARI_NOTU.get(model_adi, '')}{' (' + mimari + ')' if mimari else ''} "
            f"mimarisiyle R²={row['R2']:.4f}, MAE={row['MAE']:.4f} elde etmiş ve "
            f"{len(overall)} model arasında {sira[model_adi]}. sırada yer almıştır. "
            f"{onem_cumlesi}", size=9, indent=True)
        doc.add_paragraph()
    page_break(doc)


# ---------------------------------------------------------------------------
# §3 XAI GRAFİKLERİ
# ---------------------------------------------------------------------------
XAI_BASLIK_ADI = {
    "GraphLIME": "GraphLIME (EGNN üzerinde)",
    "Edge_Attribution": "Edge Attribution (EGNN üzerinde)",
    "SubgraphX": "SubgraphX (EGNN üzerinde)",
    "IntegratedGradients": "Integrated Gradients (EGNN üzerinde)",
}


def bolum_xai(doc):
    add_heading(doc, "3. XAI (Açıklanabilir Yapay Zekâ) Grafikleri", level=1)
    add_paragraph(doc,
        f"Bu bölümde modelin tahminlerini anlamlandırmak için "
        f"{len(XAI_KLASORLERI)} farklı XAI yöntemi uygulanmıştır: GraphLIME "
        f"(yerel vekil model), Edge Attribution (kenar/bağ atfı), SubgraphX "
        f"(açıklayıcı alt-graf arama) ve Integrated Gradients (sürekli girdi "
        f"atfı). Tüm yöntemler, eğitilmiş {len(_rapor_modelleri())} model "
        f"arasında en hafif ileri-geçişli mimari olan EGNN üzerine "
        f"uygulanmıştır — XAI yöntemleri model başına binlerce ileri-geçiş "
        f"gerektirdiğinden tek ve tutarlı bir hedef model seçilmiştir. "
        f"Dolayısıyla bu bölümdeki bulgular EGNN'in öğrendiklerini açıklar, tüm "
        f"modellerin ortalamasını değil.", size=10, indent=True)
    doc.add_paragraph()

    for i, xai_adi in enumerate(XAI_KLASORLERI, 1):
        add_heading(doc, f"3.{i}. {XAI_BASLIK_ADI.get(xai_adi, xai_adi)}", level=2)
        gdir = PROJECT_ROOT / xai_adi / "sonuclar" / "grafikler"
        if gdir.exists() and any(gdir.glob("*.tif")):
            for dosya in sorted(gdir.glob("*.tif")):
                add_image(doc, dosya, width_cm=15)
                sekil_basligi(doc, _xai_baslik(dosya.stem))
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
            add_paragraph(doc, f"[Henüz üretilmedi: {xai_adi}/sonuclar/grafikler/]", indent=True)
        add_paragraph(doc, "Yorum", size=10, bold=True, indent=True)
        add_paragraph(doc, XAI_ACIKLAMALARI.get(xai_adi, ""), size=9, indent=True)
        doc.add_paragraph()
    page_break(doc)


# ---------------------------------------------------------------------------
# §4 KIYASLAMA GRAFİKLERİ
# ---------------------------------------------------------------------------
def bolum_kiyaslama(doc):
    add_heading(doc, "4. Kıyaslama Grafikleri", level=1)
    grafik_dir = PROJECT_ROOT / "model_karsilastirma_grafikler"
    for dosya, baslik in [
        ("r2_karsilastirma_overall.tif", "R² karşılaştırması — tüm modeller (4 hedefin ortalaması)."),
        ("mae_karsilastirma_overall.tif", "MAE karşılaştırması — tüm modeller (4 hedefin ortalaması)."),
        ("r2_heatmap_model_x_hedef.tif", "Model × hedef R² ısı haritası."),
        ("r2_vs_mae_overall.tif", "R² ve MAE ikili karşılaştırma (her nokta bir model)."),
    ]:
        add_image(doc, grafik_dir / dosya)
        sekil_basligi(doc, baslik)
    doc.add_paragraph()

    add_paragraph(doc, "Yorum", size=10, bold=True, indent=True)
    df = _karsilastirma_df()
    if df is not None:
        overall = df[df["target_column"] == "overall"].sort_values("R2", ascending=False)
        en_iyi, en_dusuk = overall.iloc[0], overall.iloc[-1]
        isi_yorum = ""
        hedef_r2 = {}
        for kol in TARGET_COLUMNS:
            alt = df[df["target_column"] == kol]
            if not alt.empty:
                hedef_r2[kol] = (alt["R2"].min(), alt["R2"].max())
        if hedef_r2:
            zor = min(hedef_r2.items(), key=lambda kv: kv[1][1])
            isi_yorum = (
                f"Isı haritası, farkın modeller arasında değil HEDEFLER arasında "
                f"olduğunu net biçimde göstermektedir: sütunlar (hedefler) "
                f"birbirinden belirgin biçimde ayrışırken, satırlar (modeller) "
                f"neredeyse aynı renktedir. En zor hedef {_hedef_etiket(zor[0])} "
                f"olup tüm modellerde R²={zor[1][0]:.3f}–{zor[1][1]:.3f} bandında "
                f"kalmaktadır. ")
        add_paragraph(doc,
            f"R² ve MAE karşılaştırma grafikleri, {en_iyi['model']} "
            f"({en_iyi['R2']:.4f}) ile {en_dusuk['model']} ({en_dusuk['R2']:.4f}) "
            f"arasındaki farkın yalnızca {en_iyi['R2'] - en_dusuk['R2']:.4f} "
            f"olduğunu, yani tüm mimarilerin tek bir dar bant içinde toplandığını "
            f"göstermektedir. R² – MAE ikili grafiğinde de modeller sıkı bir küme "
            f"oluşturmakta, hiçbir model diğerlerinden belirgin biçimde "
            f"ayrışmamaktadır. {isi_yorum}"
            f"Bu tablo, §2.6'daki permütasyon önemi bulgusuyla tutarlıdır: "
            f"başarım ezici ölçüde hazır-hesaplanmış gözeneklilik "
            f"tanımlayıcılarından gelmekte, mimarilerin birbirinden ayrıştığı yer "
            f"olan 3B geometri işleme kapasitesi ise neredeyse hiç "
            f"kullanılmamaktadır — bu nedenle mimari seçiminin sonucu "
            f"değiştirmemesi beklenen bir durumdur.", size=9, indent=True)


def main() -> None:
    doc = Document()
    style = doc.styles["Normal"]; style.font.name = "Calibri"; style.font.size = Pt(10)
    _sayac_sifirla()

    add_heading(doc, "MOF Radyoaktif Gaz Adsorpsiyonu Raporu", level=0)
    add_paragraph(doc,
        "Metal-Organik Çerçevelerde Xe/Kr/I₂ Adsorpsiyon Kapasitesi ve Xe/Kr "
        "Seçicilik Tahmini İçin Grafik Sinir Ağı Modelleri ve Karşılaştırmalı "
        "Analiz", size=12)
    doc.add_paragraph()

    bolum_karsilastirma_tablosu(doc)
    bolum_model_grafikleri(doc)
    bolum_xai(doc)
    bolum_kiyaslama(doc)

    out_path = PROJECT_ROOT / "MOF_Radyoaktif_Gaz_Adsorpsiyonu_Raporu.docx"
    doc.save(str(out_path))
    print(f"Rapor kaydedildi -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
