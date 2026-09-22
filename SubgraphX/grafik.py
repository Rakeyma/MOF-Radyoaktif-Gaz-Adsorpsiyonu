"""
SubgraphX/grafik.py — element-bazinda SubgraphX onem + cekirdek-alt-graf boyut dagilimi grafikleri.
Calistirma: cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m SubgraphX.grafik
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

try:
    import grafik_ortak as go
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m SubgraphX.grafik")

SONUC_DIR = Path(__file__).resolve().parent / "sonuclar"
GRAFIK_DIR = SONUC_DIR / "grafikler"


def element_onem_grafigi(element_df: pd.DataFrame, cikti_yolu) -> None:
    import matplotlib.pyplot as plt

    alt = element_df.sort_values("ortalama", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(alt["element"][::-1], alt["ortalama"][::-1], color="mediumseagreen")
    ax.set_xlabel("Mean SubgraphX Importance")
    ax.set_ylabel("Element")
    fig.tight_layout()
    go.panel_ekle(fig)
    go.kaydet(fig, cikti_yolu)


def cekirdek_boyut_grafigi(cekirdek_df: pd.DataFrame, cikti_yolu) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(cekirdek_df["cekirdek_atom_orani"], bins=15, color="mediumseagreen", edgecolor="white")
    ax.set_xlabel("Core Subgraph Size / Total Atom Count")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    go.panel_ekle(fig)
    go.kaydet(fig, cikti_yolu)


if __name__ == "__main__":
    element_csv = SONUC_DIR / "subgraphx_element_onem.csv"
    cekirdek_csv = SONUC_DIR / "subgraphx_cekirdek_alt_graf.csv"
    if not element_csv.exists() or not cekirdek_csv.exists():
        sys.exit("HATA: once python -m SubgraphX.run_subgraphx calistirin.")
    GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
    go.set_panel("(n)")
    element_onem_grafigi(pd.read_csv(element_csv), GRAFIK_DIR / "subgraphx_element_onem.tif")
    cekirdek_boyut_grafigi(pd.read_csv(cekirdek_csv), GRAFIK_DIR / "subgraphx_cekirdek_boyut.tif")
    print(f"SubgraphX grafikleri -> {GRAFIK_DIR.resolve()}")
