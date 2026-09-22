"""
Edge_Attribution/run_edge_attribution.py
===========================================
AMAC:
    Gradyan-tabanli KENAR (bag) onem atfi - EGNN modeline (XAI_HEDEF_MODEL)
    uygular, COK-HEDEFLI (4 gaz-adsorpsiyon hedefi) versiyona genellenmis.
    Iki TAMAMLAYICI gradyan yontemi hesaplar:

    (a) VANILLA GRADIENT SALIENCY (Simonyan et al. 2014): her kenarin
        mesafe-RBF ciktisini olcekleyen bir maske m_e (baslangicta TUMU
        1.0) tanimlanir, HER hedef icin ∂f_t/∂m_e|_{m=1} DOGRUDAN autograd
        ile alinir (tek ileri+geri gecis, TUM 4 hedef icin BIRLIKTE -
        torch.autograd.grad'a grad_outputs olarak bir birim-vektor
        VERILEREK her hedef ayri ayri geri-yayilir).
    (b) INTEGRATED GRADIENTS (Sundararajan et al. 2017) - kenar maskesi
        UZERINDE: taban (m=0) ile gercek (m=1) arasinda n_ig_steps noktada
        gradyan hesaplanip ORTALAMASI alinir - gradyan doygunlugu
        sorununu hafifletir.

    Her ikisi de EGNNEncoder.rbf'in (GaussianRBF) CIKTISINI kenar
    maskesiyle carpan bir forward-hook kullanir.

CIKTI:
    Edge_Attribution/sonuclar/edge_attribution_kenarlar.csv    — ham kenar skorlari (hedef-basina sutun)
    Edge_Attribution/sonuclar/edge_attribution_bond_onem.csv   — element cifti + hedef bazinda
    Edge_Attribution/sonuclar/edge_attribution_mesafe_onem.csv — mesafe profili + hedef bazinda

CALISTIRMA (EGNN egitimi bittikten sonra, kok dizinden):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m Edge_Attribution.run_edge_attribution
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
    from graf_ozellik_ortak import batch_graphs, edge_geometri, CUTOFF
    from egitim_ortak import (
        RegressionHead, cif_den_3b_graf, K_FOLDS, SEED, TARGET_COLUMNS, N_TARGETS,
        GROUP_COLUMN, ID_COLUMN, NAME_COLUMN,
    )
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m Edge_Attribution.run_edge_attribution")

from EGNN.run_egnn import EGNNEncoder, MODEL_ADI as XAI_HEDEF_MODEL
from EGNN.run_egnn import CHECKPOINT_DIR as HEDEF_CKPT_DIR

PROJE_DIZINI = Path(__file__).resolve().parent
SONUC_DIR = PROJE_DIZINI / "sonuclar"
SONUC_DIR.mkdir(parents=True, exist_ok=True)

N_IG_STEPS = 20
MAX_PER_FOLD = 30


def _kenar_maskeli_ileri(model, g_batch, aux_t, mask, device):
    def rbf_hook(module, inp, out):
        return out * mask.to(out.device).unsqueeze(-1)

    handle = model.encoder.rbf.register_forward_hook(rbf_hook)
    try:
        pred = model(g_batch, aux_t, device)  # [1, N_TARGETS]
    finally:
        handle.remove()
    return pred


def gradyan_atfi_hesapla(model, g_raw: dict, aux_vec: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dondurur: (vanilla_grad [N_TARGETS,E], ig [N_TARGETS,E], taban_tahmin [N_TARGETS])."""
    E = g_raw["edge_index"].size(1)
    g_batch = batch_graphs([g_raw])
    aux_t = torch.tensor(aux_vec[None], dtype=torch.float)

    if E == 0:
        return np.zeros((N_TARGETS, 0)), np.zeros((N_TARGETS, 0)), np.zeros(N_TARGETS)

    # --- (a) vanilla gradient saliency @ m=1 ---
    mask_full = torch.ones(E, device=device, requires_grad=True)
    pred_full = _kenar_maskeli_ileri(model, g_batch, aux_t, mask_full, device)[0]  # [N_TARGETS]
    vanilla = np.zeros((N_TARGETS, E))
    for t in range(N_TARGETS):
        grad = torch.autograd.grad(pred_full[t], mask_full, retain_graph=(t < N_TARGETS - 1))[0]
        vanilla[t] = grad.detach().cpu().numpy()
    taban_tahmin = pred_full.detach().cpu().numpy()

    # --- (b) integrated gradients (baseline m=0 -> gercek m=1) ---
    ig_toplam = np.zeros((N_TARGETS, E))
    for s in range(1, N_IG_STEPS + 1):
        alpha = s / N_IG_STEPS
        mask_s = torch.full((E,), alpha, device=device, requires_grad=True)
        pred_s = _kenar_maskeli_ileri(model, g_batch, aux_t, mask_s, device)[0]
        for t in range(N_TARGETS):
            grad = torch.autograd.grad(pred_s[t], mask_s, retain_graph=(t < N_TARGETS - 1))[0]
            ig_toplam[t] += grad.detach().cpu().numpy()
    ig = ig_toplam / N_IG_STEPS  # (m_gercek - m_taban) = 1 - 0 = 1, carpim gerek yok

    return vanilla, ig, taban_tahmin


def _z_sembol(z_tensor: torch.Tensor) -> list[str]:
    from pymatgen.core import Element
    return [Element.from_Z(int(z)).symbol for z in z_tensor.cpu().numpy()]


def main():
    from sklearn.model_selection import KFold

    if not GAS_FINETUNE_CSV.exists():
        sys.exit(f"HATA: '{GAS_FINETUNE_CSV}' bulunamadi.")

    df = pd.read_csv(GAS_FINETUNE_CSV)
    df = df[df["eslesme_durumu"] == "TAM"].dropna(subset=[GROUP_COLUMN]).reset_index(drop=True)
    print(f"Veri: {len(df)} satir. XAI hedef modeli: {XAI_HEDEF_MODEL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    uniq_grp = np.array(sorted(df[GROUP_COLUMN].unique()))
    k_eff = min(K_FOLDS, len(uniq_grp)) if len(uniq_grp) >= 2 else 1
    kf = KFold(k_eff, shuffle=True, random_state=SEED)

    kenar_kayitlari: list[dict] = []

    for fold_no, (_, te_pos) in enumerate(kf.split(uniq_grp), 1):
        ckpt_path = HEDEF_CKPT_DIR / f"fold{fold_no}_best_model.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold_no}: checkpoint yok, atlaniyor.")
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

        print(f"\nFold {fold_no}: {len(df_test)} ornek uzerinde gradyan atfi hesaplaniyor...")
        for i, (_, row) in enumerate(df_test.iterrows()):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(df_test)}] {row[NAME_COLUMN]}", flush=True)

            g = cif_den_3b_graf(row["graf_cif_path"], CUTOFF)
            try:
                vanilla, ig, taban = gradyan_atfi_hesapla(model, g, aux_matris[i], device)
            except Exception as e:
                print(f"    UYARI: {row[NAME_COLUMN]} atlandi ({e})")
                continue
            if g["edge_index"].size(1) == 0:
                continue

            semboller = _z_sembol(g["z"])
            r, _ = edge_geometri(g["pos"], g["edge_index"])
            src, dst = g["edge_index"]
            for e_idx in range(g["edge_index"].size(1)):
                kayit = {
                    "fold": fold_no, ID_COLUMN: row[ID_COLUMN], GROUP_COLUMN: row[GROUP_COLUMN],
                    NAME_COLUMN: row[NAME_COLUMN], "edge_idx": e_idx,
                    "src_element": semboller[src[e_idx]], "dst_element": semboller[dst[e_idx]],
                    "distance_A": float(r[e_idx]),
                }
                for t, kol in enumerate(TARGET_COLUMNS):
                    kayit[f"vanilla_grad_{kol}"] = float(vanilla[t, e_idx])
                    kayit[f"integrated_gradients_{kol}"] = float(ig[t, e_idx])
                kenar_kayitlari.append(kayit)

        del model

    if not kenar_kayitlari:
        sys.exit(f"HATA: Hicbir checkpoint bulunamadi ({HEDEF_CKPT_DIR}). Once EGNN egitimini tamamlayin.")

    kenar_df = pd.DataFrame(kenar_kayitlari)
    kenar_csv = SONUC_DIR / "edge_attribution_kenarlar.csv"
    kenar_df.to_csv(kenar_csv, index=False)
    print(f"\nKenar skorlari CSV -> {kenar_csv} ({len(kenar_df)} satir)")

    kenar_df["bond_pair"] = kenar_df.apply(
        lambda r: "-".join(sorted([r["src_element"], r["dst_element"]])), axis=1)

    bond_ozet, mesafe_ozet = [], []
    for kol in TARGET_COLUMNS:
        b = (kenar_df.groupby("bond_pair")[f"integrated_gradients_{kol}"]
             .agg(ortalama="mean", abs_ortalama=lambda x: x.abs().mean(), adet="count").reset_index())
        b["target_column"] = kol
        bond_ozet.append(b)

        kenar_df["mesafe_bin_A"] = (kenar_df["distance_A"] // 0.5) * 0.5
        m = (kenar_df.groupby("mesafe_bin_A")[f"integrated_gradients_{kol}"]
             .agg(ortalama="mean", abs_ortalama=lambda x: x.abs().mean(), adet="count").reset_index())
        m["target_column"] = kol
        mesafe_ozet.append(m)

    pd.concat(bond_ozet, ignore_index=True).to_csv(SONUC_DIR / "edge_attribution_bond_onem.csv", index=False)
    pd.concat(mesafe_ozet, ignore_index=True).to_csv(SONUC_DIR / "edge_attribution_mesafe_onem.csv", index=False)
    print(f"Bag/mesafe onem CSV'leri -> {SONUC_DIR.resolve()}")
    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
