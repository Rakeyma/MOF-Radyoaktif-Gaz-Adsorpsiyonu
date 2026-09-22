"""
paths.py
========
AMAC:
    Bu projedeki TUM scriptlerin (veri_indirici_1_..., nlp_literatur_madencilik_2,
    veri_artirma_3_augmentasyon, eslesme_4_dataset_birlestirici, 11 model
    klasorundeki (GraphGPS/, PNA_GNN/, GIN/, GAT/, GatedGCN/, DeeperGCN/,
    ECC/, TFN/, EGNN/, SE3_Transformer/, DimeNetPP/) run_*.py / grafik.py
    dosyalarinin ve 4 XAI klasorunun (GraphLIME/, Edge_Attribution/,
    SubgraphX/, IntegratedGradients/) ortak kullandigi dosya/dizin
    yollarini TEK bir yerden tanimlamak.

    "Üç Boyutlu Kristal Malzemeler" projesindeki paths.py ile AYNI mimari
    ilke: bir script hangi dizinden calistirilirsa calistirilsin, yollar
    HER ZAMAN proje kok dizinine gore cozulur - calisma dizinine (cwd)
    bagimli KALMAZ.

BU PROJENIN "Üç Boyutlu Kristal Malzemeler"DEN FARKI:
    Hedef artik TEK bir skaler (formation energy) DEGIL, RADYOAKTIF SOY
    GAZ (Ksenon, Kripton) VE IYOT adsorpsiyonu icin COK-HEDEFLI (multi-
    target) bir regresyon: Xe kapasitesi, Kr kapasitesi, Xe/Kr secicilik
    (selectivity) ve I2 kapasitesi (bkz. egitim_ortak.TARGET_COLUMNS).
    Ayrica bu proje IKI AYRI veri seti/asama kullanir (Transfer Learning,
    bkz. egitim_ortak.py):
        (A) ON-EGITIM (pretraining) veri seti: devasa olcekli (~100k),
            proxy hedef (formation energy / elektronik ozellik) - QMOF
            benzeri kaynaktan.
        (B) INCE-AYAR (fine-tuning) veri seti: kucuk/orta olcekli, GERCEK
            radyoaktif gaz adsorpsiyon hedefleriyle etiketlenmis, veri
            artirma (augmentation) ile bin(ler)ce ornege genisletilmis.
    Bir API anahtari GEREKMEZ (hem JARVIS-QMOF/CoRE-MOF hem CrossRef/arXiv
    anahtarsiz/genel-erisimli uc noktalar kullanir) - script'ler HICBIR
    ZAMAN kullanicidan manuel veri indirmesini beklemez.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# KOK DIZIN
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# .env YUKLEME (opsiyonel API anahtarlari icin - CROSSREF_MAILTO nazik/kibar
# kullanim icin onerilir ama GEREKMEZ; sessizce gecer, hata vermez)
# --------------------------------------------------------------------------
def _dotenv_yukle(env_path: Path) -> None:
    if not env_path.exists():
        return
    for satir in env_path.read_text(encoding="utf-8").splitlines():
        satir = satir.strip()
        if not satir or satir.startswith("#") or "=" not in satir:
            continue
        anahtar, deger = satir.split("=", 1)
        anahtar, deger = anahtar.strip(), deger.strip().strip('"').strip("'")
        os.environ.setdefault(anahtar, deger)


_dotenv_yukle(PROJECT_ROOT / ".env")

CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", "").strip() or None
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

# --------------------------------------------------------------------------
# VERI DIZINLERI
# --------------------------------------------------------------------------
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# --- Asama 1: veri_indirici_1 (Bilesen 2 - JARVIS-QMOF / CoRE-MOF) ---
RAW_MOF_CSV = DATA_RAW / "mof_ham_veri.csv"
RAW_MOF_META = DATA_RAW / "kaynak_bilgisi.json"
CIF_DIR = DATA_PROCESSED / "cif"                       # ham (augment edilmemis) temel MOF CIF'leri
QMOF_PRETRAIN_RAW_CSV = DATA_RAW / "qmof_pretrain_ham.csv"  # devasa on-egitim (proxy hedef) ham verisi

# --- Asama 2: nlp_literatur_madencilik_2 (Bilesen 3 - CrossRef/arXiv) ---
NLP_LITERATUR_CSV = DATA_RAW / "nlp_literatur_madencilik_sonuclari.csv"

# --- Asama 3: veri_artirma_3_augmentasyon (Bilesen 4 - pymatgen/ase) ---
AUGMENTED_CIF_DIR = DATA_PROCESSED / "augmented_cif"
AUGMENTATION_MANIFEST_CSV = DATA_PROCESSED / "augmentasyon_manifest.csv"

# --- Asama 4: graf uretimi (graf_ozellik_ortak / manifest) ---
GRAPH_DIR = DATA_PROCESSED / "graphs"
MANIFEST_CSV = GRAPH_DIR / "manifest.csv"

# --- Asama 5: eslesme_4_dataset_birlestirici (NIHAI veri setleri) ---
# (A) ON-EGITIM (Transfer Learning asama 1) - buyuk olcekli proxy-hedefli veri seti
QMOF_PRETRAIN_CSV = DATA_PROCESSED / "qmof_pretrain_dataset_final.csv"
# (B) INCE-AYAR (Transfer Learning asama 2) - gaz adsorpsiyon hedefli, artirilmis veri seti
GAS_FINETUNE_CSV = DATA_PROCESSED / "gaz_adsorpsiyon_dataset_final.csv"

# --------------------------------------------------------------------------
# ON-EGITIMLI ENKODER AGIRLIKLARI (her model klasorunun kendi
# pretrain_checkpoints/ alt-dizinine yazdigi, finetune asamasinda
# egitim_ortak.finetune_k_fold_egit tarafindan okunan dosya adi)
# --------------------------------------------------------------------------
PRETRAINED_ENCODER_FILENAME = "encoder_pretrained.pt"

# --------------------------------------------------------------------------
# MODEL / XAI KLASOR LISTELERI (model_karsilastirma*.py ve rapor
# scriptlerinin merkezi olarak kullandigi TEK liste)
# --------------------------------------------------------------------------
# 10 ZORUNLU model + 1 OTONOM EKLENEN model (DimeNet++ - bkz. DimeNetPP/run_dimenetpp.py
# dokstringindeki gerekce: yonlu/acisal mesaj iletimi, gozenek geometrisinin
# (pencere boyutu/kavite sekli) boyut-secici (size-selective) Xe/Kr/I2
# adsorpsiyonunu belirleyen BIRINCIL fiziksel etken olmasi nedeniyle secildi).
MODEL_KLASORLERI = [
    "GraphGPS", "PNA_GNN", "GIN", "GAT", "GatedGCN",
    "DeeperGCN", "ECC", "TFN", "EGNN", "SE3_Transformer",
    "DimeNetPP",
]

# 3 ZORUNLU XAI yontemi + 1 OTONOM EKLENEN yontem (Integrated Gradients -
# bkz. IntegratedGradients/run_integrated_gradients.py dokstringi: SUREKLI
# atom-konumu/ozellik uzayinda gurultusuz, dogrudan turevlenebilir bir
# atfetme saglar - GraphLIME/SubgraphX'in AYRIK maskeleme/alt-graf aramasini
# TAMAMLAR).
XAI_KLASORLERI = ["GraphLIME", "Edge_Attribution", "SubgraphX", "IntegratedGradients"]

# --------------------------------------------------------------------------
# PANEL HARFLERI - (a),(b),(c)... - HER modelin grafik.py'sinin (grafik
# uzerine basilan panel etiketi) VE rapor_olustur.py'nin (grafik altina
# yazilan "hangi harf hangi model" aciklama metni) AYNI TEK sozlukten
# okumasi icin merkezilestirilmistir. ONCEDEN her model kendi harfini
# hardcode ediyordu VE rapor bunlari ALFABETIK sirayla YENIDEN harflendirip
# aciklama yaziyordu - IKISI TUTARSIZDI (grafik uzerindeki harf, altindaki
# aciklamadaki harfle ESLESMIYORDU). Artik TEK dogru kaynak burasi.
# --------------------------------------------------------------------------
PANEL_HARFLERI = {model: f"({chr(97 + i)})" for i, model in enumerate(MODEL_KLASORLERI)}

# NOT: Dizinlerin gercekten diskte olusturulmasi (mkdir) ilgili scriptin
# sorumlulugundadir. paths.py sadece yol TANIMLARINI saglar, yan etki
# (disk islemi, .env disinda) yapmaz.
