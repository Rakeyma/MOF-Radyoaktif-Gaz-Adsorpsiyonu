"""
Edge_Attribution/grafik.py — bag-tipi (element cifti) + mesafe-profili Integrated
Gradients bar/cizgi grafikleri (4 hedefin HER BIRI icin).
Calistirma: cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m Edge_Attribution.grafik
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

try:
    import grafik_ortak as go
    from egitim_ortak import TARGET_COLUMNS
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m Edge_Attribution.grafik")

SONUC_DIR = Path(__file__).resolve().parent / "sonuclar"
GRAFIK_DIR = SONUC_DIR / "grafikler"


def bond_onem_grafigi(bond_df: pd.DataFrame, kol: str, cikti_yolu) -> None:
    import matplotlib.pyplot as plt

    alt = bond_df[bond_df["target_column"] == kol].sort_values("abs_ortalama", ascending=False).head(12)
    if alt.empty:
        return
    # kullanici geri bildirimi: (1) eksen basligindaki birim/sayilar UST USTE
    # BINIP bozuk gorunuyordu, (2) eksen bos araliklara genisliyordu, (3) hedef-
    # adi parantezi (aciklayici ek) kaldirildi - degerler ONCEDEN olceklenir,
    # olcek DOGRUDAN basligin icine gomulur.
    degerler, x_etiket = go.olcekle_ve_etiketle(alt["ortalama"].values, "Mean Integrated Gradients")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    renkler = ["firebrick" if v < 0 else "darkorange" for v in degerler]
    ax.barh(alt["bond_pair"][::-1], degerler[::-1], color=renkler[::-1])
    ax.axvline(0, color="black", lw=0.8)
    go._sikitir_xlim(ax, degerler)
    ax.set_xlabel(x_etiket)
    ax.set_ylabel("Bond (Element Pair)")
    fig.tight_layout()
    go.panel_ekle(fig)
    go.kaydet(fig, cikti_yolu)


def mesafe_profili_grafigi(mesafe_df: pd.DataFrame, kol: str, cikti_yolu) -> None:
    import matplotlib.pyplot as plt

    alt = mesafe_df[mesafe_df["target_column"] == kol].sort_values("mesafe_bin_A")
    if alt.empty:
        return
    # Kullanici geri bildirimi: y ekseni basligi cok uzundu VE aciklayici
    # hedef-adi parantezi iceriyordu - kaldirildi; deger olcegi kucukse
    # olcek basligin icine gomulur (matplotlib'in cakisabilecek otomatik
    # offset-metni KULLANILMAZ).
    y_degerler, y_etiket = go.olcekle_ve_etiketle(alt["abs_ortalama"].values, "Mean |IG|")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(alt["mesafe_bin_A"], y_degerler, marker="o", color="steelblue", lw=1.8)
    ax.set_xlabel("Edge Distance (Å)")
    ax.set_ylabel(y_etiket)
    fig.tight_layout()
    go.panel_ekle(fig)
    go.kaydet(fig, cikti_yolu)


if __name__ == "__main__":
    bond_csv = SONUC_DIR / "edge_attribution_bond_onem.csv"
    mesafe_csv = SONUC_DIR / "edge_attribution_mesafe_onem.csv"
    if not bond_csv.exists() or not mesafe_csv.exists():
        sys.exit("HATA: once python -m Edge_Attribution.run_edge_attribution calistirin.")
    GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
    bond_df, mesafe_df = pd.read_csv(bond_csv), pd.read_csv(mesafe_csv)
    go.set_panel("(m)")
    for kol in TARGET_COLUMNS:
        bond_onem_grafigi(bond_df, kol, GRAFIK_DIR / f"edge_attribution_bond_{kol}.tif")
        mesafe_profili_grafigi(mesafe_df, kol, GRAFIK_DIR / f"edge_attribution_mesafe_{kol}.tif")
    print(f"Edge_Attribution grafikleri -> {GRAFIK_DIR.resolve()}")
