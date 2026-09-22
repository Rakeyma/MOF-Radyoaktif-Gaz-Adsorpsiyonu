"""
IntegratedGradients/grafik.py — element-bazinda pozisyon-IG + gozeneklilik-ozelligi
aux-IG bar grafikleri (4 hedefin HER BIRI icin).
Calistirma: cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m IntegratedGradients.grafik
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

try:
    import grafik_ortak as go
    from egitim_ortak import TARGET_COLUMNS
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m IntegratedGradients.grafik")

SONUC_DIR = Path(__file__).resolve().parent / "sonuclar"
GRAFIK_DIR = SONUC_DIR / "grafikler"


def element_ig_grafigi(element_df: pd.DataFrame, kol: str, cikti_yolu) -> None:
    import matplotlib.pyplot as plt

    alt = element_df[element_df["target_column"] == kol].sort_values("abs_ortalama", ascending=False).head(12)
    if alt.empty:
        return
    # kullanici geri bildirimi: ic ice giren sayilar/bos aralik VE aciklayici
    # hedef-adi parantezi - degerler ONCEDEN olceklenir, olcek DOGRUDAN
    # basligin icine gomulur, hedef-adi eksen basligindan kaldirilir.
    degerler, x_etiket = go.olcekle_ve_etiketle(alt["ortalama"].values, "Mean Position IG Importance")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    renkler = ["firebrick" if v < 0 else "purple" for v in degerler]
    ax.barh(alt["element"][::-1], degerler[::-1], color=renkler[::-1])
    ax.axvline(0, color="black", lw=0.8)
    go._sikitir_xlim(ax, degerler)
    ax.set_xlabel(x_etiket)
    ax.set_ylabel("Element")
    fig.tight_layout()
    go.panel_ekle(fig)
    go.kaydet(fig, cikti_yolu)


def aux_ig_grafigi(aux_df: pd.DataFrame, kol: str, cikti_yolu) -> None:
    import matplotlib.pyplot as plt

    alt = aux_df[aux_df["target_column"] == kol].groupby("aux_feature")["ig_aux_importance"].agg(
        ortalama="mean", abs_ortalama=lambda x: x.abs().mean()).reset_index().sort_values(
        "abs_ortalama", ascending=False).head(12)
    if alt.empty:
        return
    degerler, x_etiket = go.olcekle_ve_etiketle(alt["ortalama"].values, "Mean Aux-Feature IG Importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    renkler = ["firebrick" if v < 0 else "purple" for v in degerler]
    ax.barh(alt["aux_feature"][::-1], degerler[::-1], color=renkler[::-1])
    ax.axvline(0, color="black", lw=0.8)
    go._sikitir_xlim(ax, degerler)
    ax.set_xlabel(x_etiket)
    ax.set_ylabel("Porosity / Composition Feature")
    fig.tight_layout()
    go.panel_ekle(fig)
    go.kaydet(fig, cikti_yolu)


if __name__ == "__main__":
    element_csv = SONUC_DIR / "ig_element_onem.csv"
    aux_csv = SONUC_DIR / "ig_aux_ozellik_onem.csv"
    if not element_csv.exists() or not aux_csv.exists():
        sys.exit("HATA: once python -m IntegratedGradients.run_integrated_gradients calistirin.")
    GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
    element_df, aux_df = pd.read_csv(element_csv), pd.read_csv(aux_csv)
    go.set_panel("(o)")
    for kol in TARGET_COLUMNS:
        element_ig_grafigi(element_df, kol, GRAFIK_DIR / f"ig_element_onem_{kol}.tif")
        aux_ig_grafigi(aux_df, kol, GRAFIK_DIR / f"ig_aux_onem_{kol}.tif")
    print(f"IntegratedGradients grafikleri -> {GRAFIK_DIR.resolve()}")
