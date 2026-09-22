"""
GraphLIME/grafik.py — element-bazinda |GraphLIME onemi| bar grafikleri (4 hedefin HER BIRI icin)
Calistirma: cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m GraphLIME.grafik
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

try:
    import grafik_ortak as go
    from egitim_ortak import TARGET_COLUMNS
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m GraphLIME.grafik")

SONUC_DIR = Path(__file__).resolve().parent / "sonuclar"
GRAFIK_DIR = SONUC_DIR / "grafikler"


def element_onem_grafigi(element_df: pd.DataFrame, kol: str, cikti_yolu) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    alt = element_df[element_df["target_column"] == kol].sort_values("abs_ortalama", ascending=False).head(12)
    if alt.empty:
        return
    # kullanici geri bildirimi: (1) bos/anlamsiz genis eksen araligi, (2) kucuk
    # degerlerde ic ice giren/cakisan sayilar (degerler ONCEDEN olceklenir),
    # (3) eksen basligindaki hedef-adi PARANTEZI (aciklayici ek) kaldirildi -
    # hangi hedefe ait oldugu makale/figur altyazisinda belirtilecek.
    degerler, x_etiket = go.olcekle_ve_etiketle(alt["ortalama"].values, "Mean GraphLIME Importance")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    renkler = ["firebrick" if v < 0 else "steelblue" for v in degerler]
    ax.barh(alt["element"][::-1], degerler[::-1], color=renkler[::-1])
    ax.axvline(0, color="black", lw=0.8)
    go._sikitir_xlim(ax, degerler)
    ax.set_xlabel(x_etiket)
    ax.set_ylabel("Element")
    fig.tight_layout()
    go.panel_ekle(fig)
    go.kaydet(fig, cikti_yolu)


if __name__ == "__main__":
    atom_csv = SONUC_DIR / "graphlime_element_onem.csv"
    if not atom_csv.exists():
        sys.exit("HATA: once python -m GraphLIME.run_graphlime calistirin.")
    GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
    element_df = pd.read_csv(atom_csv)
    go.set_panel("(l)")
    for kol in TARGET_COLUMNS:
        element_onem_grafigi(element_df, kol, GRAFIK_DIR / f"graphlime_element_onem_{kol}.tif")
    print(f"GraphLIME grafikleri -> {GRAFIK_DIR.resolve()}")
