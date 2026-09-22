"""
GIN/grafik.py — OOF grafikleri (hedef-basina 4x, egitim-kaybi, permutation importance)
Calistirma: cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m GIN.grafik

TUM 11 model klasorunde AYNI 6 satirlik desen: egitim_ortak.tum_grafikleri_uret
cagrisi - grafik mantiginin TAMAMI grafik_ortak.py + egitim_ortak.py'de.
"""
from __future__ import annotations
import sys
from pathlib import Path

try:
    import grafik_ortak as go  # noqa: F401 (egitim_ortak icinde kullanilir)
    from paths import PANEL_HARFLERI
    from egitim_ortak import EgitimAyarlari, model_factory_olustur, tum_grafikleri_uret
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m GIN.grafik")

from GIN.run_gin import GINEncoder, MODEL_ADI, CHECKPOINT_DIR, SONUC_DIR

GRAFIK_DIR = SONUC_DIR / "grafikler"

if __name__ == "__main__":
    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI)
    model_factory = model_factory_olustur(lambda: GINEncoder())
    tum_grafikleri_uret(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR, GRAFIK_DIR, panel=PANEL_HARFLERI[MODEL_ADI])
