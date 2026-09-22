"""
PNA_GNN/grafik.py — OOF grafikleri. PNAConv'un deg-bagimli yapisi nedeniyle
permutation importance/grafik asamasinda encoder YENIDEN insa EDILMEZ
(checkpoint zaten TAM egitilmis agirlik+mimariyi tasir); bu yuzden deg
histogrami, KAYITLI test verisinin kendi graf onbellegi uzerinden checkpoint
ile AYNI sekilde yeniden turetilir (egitim sirasindaki degerle ayni olmasi
GEREKMEZ - deg sadece olcekleyici sabitleri belirler, agirlik sekli sabittir).

Calistirma: cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m PNA_GNN.grafik
"""
from __future__ import annotations
import sys

import pandas as pd

try:
    import grafik_ortak as go  # noqa: F401
    from paths import PANEL_HARFLERI
    from paths import GAS_FINETUNE_CSV
    from egitim_ortak import EgitimAyarlari, model_factory_olustur, tum_grafikleri_uret, grafik_onbellek_olustur
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m PNA_GNN.grafik")

from PNA_GNN.run_pna_gnn import PNAEncoder, derece_histogrami_hesapla, MODEL_ADI, CHECKPOINT_DIR, SONUC_DIR

GRAFIK_DIR = SONUC_DIR / "grafikler"

if __name__ == "__main__":
    df = pd.read_csv(GAS_FINETUNE_CSV, low_memory=False)
    df = df[df["eslesme_durumu"] == "TAM"].reset_index(drop=True)
    cache = grafik_onbellek_olustur(df)
    deg = derece_histogrami_hesapla(cache)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI)
    model_factory = model_factory_olustur(lambda: PNAEncoder(deg))
    tum_grafikleri_uret(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR, GRAFIK_DIR, panel=PANEL_HARFLERI[MODEL_ADI])
