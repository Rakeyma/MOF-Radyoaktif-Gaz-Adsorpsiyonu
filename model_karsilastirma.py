"""
model_karsilastirma.py
======================
AMAC:
    11 modelin (GraphGPS, PNA_GNN, GIN, GAT, GatedGCN, DeeperGCN, ECC, TFN,
    EGNN, SE3_Transformer, DimeNetPP) HER BIRININ kendi
    `<Model>/sonuclar/test_tahminleri_oof.csv` dosyasinda ZATEN kaydettigi
    pooled out-of-fold (gercek, tahmin) ciftlerinden GENISLETILMIS, 4
    HEDEFIN (Xe/Kr/I2 kapasitesi, Xe/Kr secicilik) HER BIRI icin ayri ayri
    VE genel (overall, ortalama) bir metrik tablosu uretmek - "Üç Boyutlu
    Kristal Malzemeler/model_karsilastirma.py" ile AYNI ilke (YENIDEN
    EGITIM GEREKMEZ), COK-HEDEFLI versiyona genellenmis.

GIRDI:
    <Model>/sonuclar/test_tahminleri_oof.csv (henuz calistirilmamis bir
    model varsa bu script onu atlayip uyari basar, hata vermez).

CIKTI (kok dizinde):
    model_karsilastirma_sonuclari.csv   (uzun format: model x hedef x metrik)
    model_karsilastirma_sonuclari.txt

CALISTIRMA:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python model_karsilastirma.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, median_absolute_error, max_error, r2_score,
)

from paths import PROJECT_ROOT, MODEL_KLASORLERI
from egitim_ortak import TARGET_COLUMNS


def metrik_hesapla_genis(gercek: np.ndarray, tahmin: np.ndarray) -> dict:
    gecerli = ~np.isnan(gercek)
    if gecerli.sum() < 2:
        return {k: float("nan") for k in ("MAE", "RMSE", "R2", "MedianAE", "MaxError", "PearsonR")} | {"n_ornek": int(gecerli.sum())}
    g, t = gercek[gecerli], tahmin[gecerli]
    r = pearsonr(g, t)[0] if np.std(g) > 1e-10 else float("nan")
    return {
        "MAE": float(mean_absolute_error(g, t)),
        "RMSE": float(np.sqrt(mean_squared_error(g, t))),
        "R2": float(r2_score(g, t)) if np.std(g) > 1e-10 else float("nan"),
        "MedianAE": float(median_absolute_error(g, t)),
        "MaxError": float(max_error(g, t)),
        "PearsonR": float(r),
        "n_ornek": int(gecerli.sum()),
    }


def main() -> None:
    satirlar = []
    for model_adi in MODEL_KLASORLERI:
        oof_csv = PROJECT_ROOT / model_adi / "sonuclar" / "test_tahminleri_oof.csv"
        if not oof_csv.exists():
            print(f"[ATLANDI] {model_adi}: '{oof_csv}' henuz yok (model daha calistirilmamis).")
            continue

        df = pd.read_csv(oof_csv)
        for kol in TARGET_COLUMNS:
            g_kol, t_kol = f"gercek_{kol}", f"tahmin_{kol}"
            if g_kol not in df.columns or t_kol not in df.columns:
                continue
            m = metrik_hesapla_genis(df[g_kol].values, df[t_kol].values)
            satirlar.append({"model": model_adi, "target_column": kol, **m})

        overall_r2 = [s["R2"] for s in satirlar if s["model"] == model_adi and np.isfinite(s["R2"])]
        overall_mae = [s["MAE"] for s in satirlar if s["model"] == model_adi and np.isfinite(s["MAE"])]
        print(f"[OK] {model_adi}: overall R2={np.mean(overall_r2) if overall_r2 else float('nan'):.3f}  "
              f"overall MAE={np.mean(overall_mae) if overall_mae else float('nan'):.4f}")

    if not satirlar:
        print("\nHic model sonucu bulunamadi - once en az bir '<Model>/run_*.py' calistirilmali.")
        return

    uzun_df = pd.DataFrame(satirlar)

    # "overall" satirini (4 hedefin ortalamasi) HER model icin ekle - siralama/rapor icin.
    ozet_satirlar = []
    for model_adi, grup in uzun_df.groupby("model"):
        ozet = {"model": model_adi, "target_column": "overall"}
        for metrik in ("MAE", "RMSE", "R2", "MedianAE", "MaxError", "PearsonR"):
            gecerli = grup[metrik].dropna()
            ozet[metrik] = float(gecerli.mean()) if len(gecerli) else float("nan")
        ozet["n_ornek"] = int(grup["n_ornek"].sum())
        ozet_satirlar.append(ozet)
    uzun_df = pd.concat([uzun_df, pd.DataFrame(ozet_satirlar)], ignore_index=True)

    csv_path = PROJECT_ROOT / "model_karsilastirma_sonuclari.csv"
    uzun_df.to_csv(csv_path, index=False)

    overall_df = uzun_df[uzun_df["target_column"] == "overall"].sort_values("R2", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("GENEL MODEL KARSILASTIRMASI (4 hedefin ortalamasi, pooled out-of-fold, R2'ye gore siralanmis)")
    print("=" * 100)
    print(overall_df[["model", "MAE", "RMSE", "R2", "MedianAE", "MaxError", "PearsonR", "n_ornek"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    s = ["=" * 100, "MOF RADYOAKTIF GAZ ADSORPSIYON PIPELINE — MODEL KARSILASTIRMASI", "=" * 100, ""]
    s.append(f"{'Model':<18}{'MAE':>10}{'RMSE':>10}{'R2':>8}{'MedianAE':>11}{'MaxErr':>10}{'PearsonR':>10}{'n':>9}")
    s.append("-" * 100)
    for _, row in overall_df.iterrows():
        s.append(
            f"{row['model']:<18}{row['MAE']:>10.4f}{row['RMSE']:>10.4f}{row['R2']:>8.3f}"
            f"{row['MedianAE']:>11.4f}{row['MaxError']:>10.4f}{row['PearsonR']:>10.4f}{int(row['n_ornek']):>9d}"
        )
    s.append("")
    s.append("Hedef-basina detay (asagida her hedef icin en iyi 3 model):")
    for kol in TARGET_COLUMNS:
        alt = uzun_df[uzun_df["target_column"] == kol].sort_values("R2", ascending=False).head(3)
        s.append(f"\n  {kol}:")
        for _, row in alt.iterrows():
            s.append(f"    {row['model']:<18} R2={row['R2']:.3f}  MAE={row['MAE']:.4f}  (n={int(row['n_ornek'])})")
    s.append("")
    s.append("Notlar:")
    s.append("  - 'overall' satiri 4 hedefin (Xe/Kr/I2 kapasitesi, Xe/Kr secicilik) metrik ORTALAMASIDIR.")
    s.append("  - MedianAE: MAE'nin aykiri-deger-dayanikli hali. MaxError: en kotu-durum mutlak hata.")
    s.append("  - Etiketlerin bir kismi NLP-literatur (gercek), bir kismi gozeneklilik-proxy korelasyon")
    s.append("    kaynaklidir (bkz. eslesme_4_dataset_birlestirici.py 'label_source_<hedef>' sutunlari) -")
    s.append("    R2/MAE yorumlanirken bu etiket-kalitesi karisikligi goz onunde bulundurulmalidir.")
    s.append("  - 11 modelin TAMAMI AYNI egitim protokolunu (K-fold, erken durdurma, transfer learning)")
    s.append("    paylasir (egitim_ortak.py) - farklar SADECE encoder mimarisinden kaynaklanir.")
    s.append("=" * 100)

    txt_path = PROJECT_ROOT / "model_karsilastirma_sonuclari.txt"
    txt_path.write_text("\n".join(s), encoding="utf-8")

    print(f"\nTablo (CSV) -> {csv_path.resolve()}")
    print(f"Ozet (TXT)  -> {txt_path.resolve()}")


if __name__ == "__main__":
    main()
