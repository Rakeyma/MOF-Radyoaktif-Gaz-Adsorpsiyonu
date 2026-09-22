"""
model_karsilastirma_grafik.py
==============================
model_karsilastirma_sonuclari.csv'den 11 model x 4 hedef (+overall)
karsilastirma grafikleri uretir - "Üç Boyutlu Kristal Malzemeler/
model_karsilastirma_grafik.py" ile AYNI ilke (600 dpi .tif+.png, bold/
Ingilizce), COK-HEDEFLI heatmap'e genellenmis.

Grafikler (model_karsilastirma_grafikler/ altinda):
    r2_karsilastirma_overall.tif    mae_karsilastirma_overall.tif
    r2_heatmap_model_x_hedef.tif    r2_vs_mae_overall.tif

CALISTIRMA:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python model_karsilastirma_grafik.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'axes.labelsize': 14, 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'font.weight': 'bold',
    'legend.fontsize': 10, 'savefig.dpi': 600,
})

from paths import PROJECT_ROOT, MODEL_KLASORLERI
from egitim_ortak import TARGET_COLUMNS
from grafik_ortak import KISA_ETIKETLER

CSV_YOLU = PROJECT_ROOT / "model_karsilastirma_sonuclari.csv"
GRAFIK_DIR = PROJECT_ROOT / "model_karsilastirma_grafikler"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)


def _kaydet(fig, path: Path) -> None:
    fig.savefig(path, dpi=600, bbox_inches="tight", format="tiff")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _panel_etiket(ax, harf: str, fontsize: float = 16.0) -> None:
    ax.text(0.98, 1.03, f"({harf})", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=fontsize, fontweight="bold")


def _renk_listesi(n: int, cmap: str = "RdYlGn", ters: bool = False) -> list:
    sira = np.linspace(0.9, 0, n) if ters else np.linspace(0, 0.9, n)
    return [plt.get_cmap(cmap)(v) for v in sira]


def r2_karsilastirma_grafigi(df: pd.DataFrame) -> None:
    df = df.sort_values("R2", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["model"], df["R2"], color=_renk_listesi(len(df)))
    ax.set_xlabel("R²")
    for i, v in enumerate(df["R2"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9, fontweight="bold")
    if df["R2"].notna().any():
        ax.set_xlim(right=df["R2"].max() * 1.15 if df["R2"].max() > 0 else 0.1)
    _panel_etiket(ax, "a")
    fig.tight_layout()
    _kaydet(fig, GRAFIK_DIR / "r2_karsilastirma_overall.tif")


def mae_karsilastirma_grafigi(df: pd.DataFrame) -> None:
    df = df.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["model"], df["MAE"], color=_renk_listesi(len(df), ters=True))
    ax.set_xlabel("MAE")
    for i, v in enumerate(df["MAE"]):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9, fontweight="bold")
    if df["MAE"].notna().any():
        ax.set_xlim(right=df["MAE"].max() * 1.20)
    _panel_etiket(ax, "b")
    fig.tight_layout()
    _kaydet(fig, GRAFIK_DIR / "mae_karsilastirma_overall.tif")


def r2_heatmap_grafigi(uzun_df: pd.DataFrame) -> None:
    pivot = uzun_df[uzun_df["target_column"] != "overall"].pivot(
        index="model", columns="target_column", values="R2")
    pivot = pivot.reindex(columns=[c for c in TARGET_COLUMNS if c in pivot.columns])
    if pivot.empty:
        return
    # Kullanici geri bildirimi: sutun basliklarinda ham kolon adlari (orn.
    # "i2_uptake_mmol_g") gorunuyordu - insan-okunur KISA etiketlerle
    # (grafik_ortak.KISA_ETIKETLER) degistirildi.
    pivot = pivot.rename(columns=lambda c: KISA_ETIKETLER.get(c, c))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(pivot) + 2))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0.0,
                cbar_kws={"label": "R²"}, annot_kws={"fontsize": 9, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Target")
    ax.set_ylabel("Model")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _panel_etiket(ax, "c")
    fig.tight_layout()
    _kaydet(fig, GRAFIK_DIR / "r2_heatmap_model_x_hedef.tif")


def r2_vs_mae_grafigi(df: pd.DataFrame) -> None:
    # Kullanici geri bildirimi: noktalarin YANINA yazilan model-adi etiketleri
    # (R2 degerleri birbirine cok yakinken, bkz. gercek koşum: 0.851-0.854)
    # UST USTE BINIYOR/okunamaz hale geliyordu - metin-etiketleri KALDIRILIP
    # renk<->model eslemesini gosteren TEK bir sag-ust LEJANT ile degistirildi.
    fig, ax = plt.subplots(figsize=(7, 6.5))
    renkler = _renk_listesi(len(df))
    for (_, row), renk in zip(df.iterrows(), renkler):
        ax.scatter(row["MAE"], row["R2"], s=90, color=renk, edgecolors="black",
                   linewidths=0.6, label=row["model"])
    # veri noktalarina COK yakin durmamasi icin eksenlere kucuk bir ust-pay
    # birakilir, lejant sag-ust koseye (kullanici istegi) 2 sutunlu/kucuk
    # yazi ile sigdirilir.
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + (y1 - y0) * 0.22)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2, frameon=True,
              title="Model", title_fontsize=8.5, framealpha=0.9)
    ax.set_xlabel("MAE")
    ax.set_ylabel("R²")
    _panel_etiket(ax, "d")
    fig.tight_layout()
    _kaydet(fig, GRAFIK_DIR / "r2_vs_mae_overall.tif")


def main() -> None:
    if not CSV_YOLU.exists():
        sys.exit(f"HATA: '{CSV_YOLU}' bulunamadi. Once model_karsilastirma.py calistirin.")

    uzun_df = pd.read_csv(CSV_YOLU)
    overall_df = uzun_df[uzun_df["target_column"] == "overall"].dropna(subset=["R2"]).reset_index(drop=True)
    if overall_df.empty:
        sys.exit("HATA: hicbir modelde gecerli R2 skoru yok.")

    r2_karsilastirma_grafigi(overall_df)
    mae_karsilastirma_grafigi(overall_df)
    r2_heatmap_grafigi(uzun_df)
    r2_vs_mae_grafigi(overall_df)

    print(f"4 karsilastirma grafigi -> {GRAFIK_DIR.resolve()}")


if __name__ == "__main__":
    main()
