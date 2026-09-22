"""
GraphLIME/run_graphlime.py
=============================
AMAC:
    Huang et al. (2020) "GraphLIME: Local Interpretable Model Explanations
    for Graph Neural Networks" yontemini EGNN modeline (bkz. XAI_HEDEF_MODEL)
    uygular - "Üç Boyutlu Kristal Malzemeler/GraphLIME/run_graphlime.py" ile
    AYNI metodoloji, COK-HEDEFLI (4 gaz-adsorpsiyon hedefi) versiyona
    genellenmis.

    NEDEN EGNN: 11 model arasinda en hafif/hizli ileri-gecisli model
    oldugundan, TUM 4 XAI yontemi icin XAI_HEDEF_MODEL olarak secilmistir.

YONTEM (graf-seviyesi COK-HEDEFLI regresyona uyarlanmis GraphLIME):
    1) K=100 rastgele ikili ATOM maskesi z^(k) ∈ {0,1}^M orneklenir
       (Bernoulli(0.7)); z=1 (tum atomlar acik) DAIMA orneklerden biridir.
    2) Maskeli tahmin f(z^(k)) ∈ R^4 (4 hedef), atom-gomme katmaninin
       (encoder.atom_emb) CIKTISINI maskeleyen bir forward-hook ile TEK
       seferde (tum 4 hedef icin BIRLIKTE) hesaplanir.
    3) LIME'in YEREL AGIRLIK CEKIRDEGI (exponential proximity kernel):
       w^(k) = exp(-(kapali_atom_orani)^2 / kernel_width^2).
    4) HER hedef icin AYRI bir agirlikli LASSO (z -> f_k(z)-f_k(1)) fit
       edilir - katsayilar = o HEDEF icin ATOM ONEM SKORU (ayni K=100
       ornekleme HERKES icin paylasilir, sadece regresyon HEDEFE OZEL).

CIKTI:
    GraphLIME/sonuclar/graphlime_atom_onem.csv     — malzeme+atom+hedef bazinda katsayi (uzun format)
    GraphLIME/sonuclar/graphlime_element_onem.csv  — element sembolu + hedef bazinda ozet

CALISTIRMA (EGNN egitimi bittikten sonra, kok dizinden):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m GraphLIME.run_graphlime
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

try:
    from paths import GAS_FINETUNE_CSV
    from ortak_ozellikler import aux_ham_matris, AuxOlcekleyici
    from graf_ozellik_ortak import batch_graphs, CUTOFF
    from egitim_ortak import (
        RegressionHead, cif_den_3b_graf, K_FOLDS, SEED, TARGET_COLUMNS, N_TARGETS,
        GROUP_COLUMN, ID_COLUMN, NAME_COLUMN,
    )
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m GraphLIME.run_graphlime")

from EGNN.run_egnn import EGNNEncoder, MODEL_ADI as XAI_HEDEF_MODEL
from EGNN.run_egnn import CHECKPOINT_DIR as HEDEF_CKPT_DIR

PROJE_DIZINI = Path(__file__).resolve().parent
SONUC_DIR = PROJE_DIZINI / "sonuclar"
SONUC_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 100
KEEP_PROB = 0.7
KERNEL_WIDTH = 0.25
LASSO_ALPHA_ARANACAK = np.logspace(-6, -1, 25)  # LassoCV bu izgeradan capraz-dogrulamayla secer
MAX_PER_FOLD = 30  # devasa veri setinde XAI orneklemesini pratik tutmak icin


def _atom_maskeli_tahmin(model, g_batch, aux_t, mask, device):
    def emb_hook(module, inp, out):
        return out * mask.to(out.device).unsqueeze(-1)

    handle = model.encoder.atom_emb.register_forward_hook(emb_hook)
    try:
        pred = model(g_batch, aux_t, device)  # [1, N_TARGETS]
    finally:
        handle.remove()
    return pred


def graphlime_hesapla(model, g_raw: dict, aux_vec: np.ndarray, device: str,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Dondurur: (coef [N_TARGETS, M], taban_tahmin [N_TARGETS]).

    NOT: sabit LASSO_ALPHA=0.01 (kardes projeden - kucuk kristal hucreleri
    icin ayarlanmis) bu projede TUM katsayilari SIFIRA cekiyordu, cunku bu
    MOF'lar cok daha fazla atom icerir (~72-172) ve 4 EGNN katmani + her
    katmandaki LayerNorm tek bir atomun maskelenmesinin etkisini agirlik
    ortalamasinda/normalize edilmis temsilde ciddi sekilde soluklastirir ->
    maskeleme-kaynakli tahmin farklari (|delta| ~ 0.01-0.03) sabit alpha'nin
    esiginin ALTINDA kaliyordu (dogrulandi: bu veri uzerinde alpha=0.01 ->
    0/M sifir-olmayan katsayi, alpha=1e-4 -> onlarca sifir-olmayan katsayi).
    LassoCV, alpha'yi HER ornek icin capraz-dogrulamayla veriden secer, bu
    yuzden farkli MOF boyutu/hedef olcegine otomatik uyum saglar."""
    from sklearn.linear_model import LassoCV

    M = g_raw["n_atoms"]
    g_batch = batch_graphs([g_raw])
    aux_t = torch.tensor(aux_vec[None], dtype=torch.float)

    Z = (rng.random((N_SAMPLES, M)) < KEEP_PROB).astype(np.float32)
    Z[0] = 1.0

    preds = np.empty((N_SAMPLES, N_TARGETS), dtype=np.float64)
    with torch.no_grad():
        for k in range(N_SAMPLES):
            mask = torch.from_numpy(Z[k])
            preds[k] = _atom_maskeli_tahmin(model, g_batch, aux_t, mask, device).cpu().numpy()[0]

    kapali_oran = 1.0 - Z.mean(axis=1)
    weights = np.exp(-(kapali_oran ** 2) / (KERNEL_WIDTH ** 2))

    coef_mat = np.zeros((N_TARGETS, M))
    for t in range(N_TARGETS):
        y = preds[:, t] - preds[0, t]
        if np.abs(y).max() < 1e-12:
            continue  # bu hedef icin maskeleme HICBIR farka yol acmadi (sabit tahmin)
        lasso = LassoCV(alphas=LASSO_ALPHA_ARANACAK, cv=min(5, N_SAMPLES // 10),
                         fit_intercept=True, max_iter=5000)
        lasso.fit(Z, y, sample_weight=weights)
        coef_mat[t] = lasso.coef_
    return coef_mat, preds[0]


def _z_sembol(z_tensor: torch.Tensor) -> list[str]:
    from pymatgen.core import Element
    return [Element.from_Z(int(z)).symbol for z in z_tensor.cpu().numpy()]


def main():
    from sklearn.model_selection import KFold

    if not GAS_FINETUNE_CSV.exists():
        sys.exit(f"HATA: '{GAS_FINETUNE_CSV}' bulunamadi.")

    df = pd.read_csv(GAS_FINETUNE_CSV)
    df = df[df["eslesme_durumu"] == "TAM"].dropna(subset=[GROUP_COLUMN]).reset_index(drop=True)
    print(f"Veri: {len(df)} satir, {df[GROUP_COLUMN].nunique()} benzersiz temel MOF")
    print(f"XAI hedef modeli: {XAI_HEDEF_MODEL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    uniq_grp = np.array(sorted(df[GROUP_COLUMN].unique()))
    k_eff = min(K_FOLDS, len(uniq_grp)) if len(uniq_grp) >= 2 else 1
    kf = KFold(k_eff, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)

    atom_kayitlari: list[dict] = []

    for fold_no, (_, te_pos) in enumerate(kf.split(uniq_grp), 1):
        ckpt_path = HEDEF_CKPT_DIR / f"fold{fold_no}_best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold_no}: checkpoint yok ({ckpt_path}), atlaniyor.")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = RegressionHead(EGNNEncoder(), n_outputs=N_TARGETS).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        olcekleyici = AuxOlcekleyici()
        olcekleyici.ortalama = ckpt["aux_ortalama"]
        olcekleyici.std = ckpt["aux_std"]

        test_grp = set(uniq_grp[te_pos])
        df_test = df[df[GROUP_COLUMN].isin(test_grp)].drop_duplicates(GROUP_COLUMN).reset_index(drop=True)
        if MAX_PER_FOLD is not None:
            df_test = df_test.sample(min(MAX_PER_FOLD, len(df_test)), random_state=SEED).reset_index(drop=True)
        aux_matris = olcekleyici.transform(aux_ham_matris(df_test))

        print(f"\nFold {fold_no}: {len(df_test)} ornek uzerinde GraphLIME hesaplaniyor...")
        for i, (_, row) in enumerate(df_test.iterrows()):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(df_test)}] {row[NAME_COLUMN]}", flush=True)

            g = cif_den_3b_graf(row["graf_cif_path"], CUTOFF)
            try:
                coef_mat, taban_tahmin = graphlime_hesapla(model, g, aux_matris[i], device, rng)
            except Exception as e:
                print(f"    UYARI: {row[NAME_COLUMN]} atlandi ({e})")
                continue

            semboller = _z_sembol(g["z"])
            pos_np = g["pos"].numpy()
            for a_idx in range(g["n_atoms"]):
                kayit = {
                    "fold": fold_no, ID_COLUMN: row[ID_COLUMN], GROUP_COLUMN: row[GROUP_COLUMN],
                    NAME_COLUMN: row[NAME_COLUMN], "atom_idx": a_idx, "element": semboller[a_idx],
                    "x": float(pos_np[a_idx, 0]), "y": float(pos_np[a_idx, 1]), "z": float(pos_np[a_idx, 2]),
                }
                for t, kol in enumerate(TARGET_COLUMNS):
                    kayit[f"graphlime_onem_{kol}"] = float(coef_mat[t, a_idx])
                    kayit[f"taban_tahmin_{kol}"] = float(taban_tahmin[t])
                    kayit[f"gercek_{kol}"] = row.get(kol, np.nan)
                atom_kayitlari.append(kayit)

        del model

    if not atom_kayitlari:
        sys.exit(f"HATA: Hicbir checkpoint bulunamadi ({HEDEF_CKPT_DIR}). Once EGNN egitimini tamamlayin.")

    atom_df = pd.DataFrame(atom_kayitlari)
    atom_csv = SONUC_DIR / "graphlime_atom_onem.csv"
    atom_df.to_csv(atom_csv, index=False)
    print(f"\nAtom onem CSV -> {atom_csv} ({len(atom_df)} satir)")

    ozet_satirlari = []
    for kol in TARGET_COLUMNS:
        g = (atom_df.groupby("element")[f"graphlime_onem_{kol}"]
             .agg(ortalama="mean", abs_ortalama=lambda x: x.abs().mean(), std="std", adet="count")
             .reset_index())
        g["target_column"] = kol
        ozet_satirlari.append(g)
    element_df = pd.concat(ozet_satirlari, ignore_index=True).sort_values(
        ["target_column", "abs_ortalama"], ascending=[True, False])
    element_csv = SONUC_DIR / "graphlime_element_onem.csv"
    element_df.to_csv(element_csv, index=False)
    print(f"Element onem CSV -> {element_csv}")
    for kol in TARGET_COLUMNS:
        print(f"\n--- {kol}: Element Basi Ortalama |GraphLIME Onemi| (ilk 10) ---")
        print(element_df[element_df["target_column"] == kol].head(10).to_string(index=False))
    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
