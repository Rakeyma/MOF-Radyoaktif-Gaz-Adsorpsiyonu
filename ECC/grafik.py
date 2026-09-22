"""ECC/grafik.py — cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m ECC.grafik"""
from __future__ import annotations
import sys
try:
    import grafik_ortak as go  # noqa: F401
    from paths import PANEL_HARFLERI
    from egitim_ortak import EgitimAyarlari, model_factory_olustur, tum_grafikleri_uret
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m ECC.grafik")

from ECC.run_ecc import ECCEncoder, MODEL_ADI, CHECKPOINT_DIR, SONUC_DIR

GRAFIK_DIR = SONUC_DIR / "grafikler"

if __name__ == "__main__":
    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI)
    model_factory = model_factory_olustur(lambda: ECCEncoder())
    tum_grafikleri_uret(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR, GRAFIK_DIR, panel=PANEL_HARFLERI[MODEL_ADI])
