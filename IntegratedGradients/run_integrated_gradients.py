"""
IntegratedGradients/run_integrated_gradients.py
===================================================
AMAC (OTONOM EKLENEN 4. XAI YONTEMI - bkz. paths.XAI_KLASORLERI dokstringi):
    Sundararajan, Taly & Yan (2017) "Axiomatic Attribution for Deep
    Networks" (Integrated Gradients, IG) yontemini, EGNN modelinin (XAI_HEDEF_MODEL)
    GERCEKTEN SUREKLI girdilerine - ATOM 3B KARTEZYEN KONUMLARI (pos) VE
    gozeneklilik/kompozisyon ozellik vektoru (aux) - DOGRUDAN uygular.

    NEDEN OTONOM OLARAK EKLENDI (kullanicinin "ek SOTA XAI yontemi"
    gereksinimi) VE DIGER 3 ZORUNLU YONTEMDEN FARKI:
        - GraphLIME: AYRIK/kombinatoryal atom MASKELERI (Bernoulli orneklem)
          uzerinde bir dogrusal SURROGATE modeldir.
        - Edge_Attribution: kenar (RBF cikti) uzerinde bir [0,1] MASKE
          degiskenine gore gradyan alir (IG dahil, ama maske uzerinde -
          GERCEK bir fiziksel buyukluk degil, YAPAY bir 'acik/kapali'
          interpolasyon degiskenidir).
        - BU MODUL: maske YOKTUR. IG DOGRUDAN GERCEK, SUREKLI fiziksel
          buyukluklerin (atom konumlari - 3B gozenek geometrisini TANIMLAYAN
          buyukluk - VE gozeneklilik tanimlayicilari) KENDISI uzerinde
          hesaplanir; baseline'dan (pos: tum atomlarin agirlik-merkezine
          COKERTILMESI = 'hicbir uzamsal/geometrik bilgi yok'; aux: standardize
          uzayda SIFIR vektoru = 'hicbir gozeneklilik bilgisi yok') gercek
          girdiye DOGRU SUREKLI bir yol boyunca gradyanlar ORTALANIR.
        Bu, 3B POROZ MOF yapilari icin ozellikle uygundur: HANGI atomun
        TAM KONUMUNUN (orn. bir gozenek penceresini daraltan/genisleten bir
        atom) tahmini EN COK etkiledigini, AYRIK maskeleme/alt-graf aramanin
        yakalayamayacagi bir HASSASIYETLE (surekli/turevlenebilir) ortaya
        koyar - GraphLIME/SubgraphX'i (ayrik) TAMAMLAR.

    MATEMATIKSEL TAMLIK (completeness) AKSIYOMU: pozisyon-IG (3 eksen
    toplami, atom basina) + aux-IG (tum ozellikler toplami) = f(gercek) -
    f(taban) - HER ornek icin bu ozdesligin YAKLASIK saglandigi (N_IG_STEPS
    yeterince buyukse) rapor scriptinde CAPRAZ KONTROL edilebilir.

CIKTI:
    IntegratedGradients/sonuclar/ig_atom_konum_onem.csv     — atom+eksen+hedef bazinda pozisyon-IG
    IntegratedGradients/sonuclar/ig_aux_ozellik_onem.csv    — gozeneklilik ozelligi+hedef bazinda aux-IG
    IntegratedGradients/sonuclar/ig_element_onem.csv        — element sembolu bazinda ozet (pozisyon-IG)

CALISTIRMA (EGNN egitimi bittikten sonra, kok dizinden):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m IntegratedGradients.run_integrated_gradients
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
    from ortak_ozellikler import aux_ham_matris, AuxOlcekleyici, AUX_FEATURE_COLUMNS
    from graf_ozellik_ortak import batch_graphs, CUTOFF
    from egitim_ortak import (
        RegressionHead, cif_den_3b_graf, K_FOLDS, SEED, TARGET_COLUMNS, N_TARGETS,
        GROUP_COLUMN, ID_COLUMN, NAME_COLUMN,
    )
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m IntegratedGradients.run_integrated_gradients")

from EGNN.run_egnn import EGNNEncoder, MODEL_ADI as XAI_HEDEF_MODEL
from EGNN.run_egnn import CHECKPOINT_DIR as HEDEF_CKPT_DIR

PROJE_DIZINI = Path(__file__).resolve().parent
SONUC_DIR = PROJE_DIZINI / "sonuclar"
SONUC_DIR.mkdir(parents=True, exist_ok=True)

N_IG_STEPS = 20
MAX_PER_FOLD = 30


def integrated_gradients_hesapla(model, g_raw: dict, aux_vec: np.ndarray, device: str):
    """Dondurur: (pos_ig [N_TARGETS, n_atoms, 3], aux_ig [N_TARGETS, AUX_DIM],
    taban_tahmin [N_TARGETS], gercek_tahmin [N_TARGETS]) - kenar TOPOLOJISI
    (edge_index) SABIT tutulur, SADECE konum/ozellik DEGERLERI enterpole edilir
    (graf-IG icin standart pratik)."""
    g_batch0 = batch_graphs([g_raw])
    pos0 = g_batch0["pos"].to(device)
    pos_baseline = pos0.mean(dim=0, keepdim=True).expand_as(pos0)  # tum atomlar agirlik-merkezine cokertilir
    aux0 = torch.tensor(aux_vec[None], dtype=torch.float, device=device)
    aux_baseline = torch.zeros_like(aux0)  # standardize uzayda 'notr/bilgi-yok' girdi

    n_atoms = pos0.size(0)
    grad_pos_toplam = torch.zeros(N_TARGETS, n_atoms, 3)
    grad_aux_toplam = torch.zeros(N_TARGETS, aux0.size(-1))

    for s in range(1, N_IG_STEPS + 1):
        alpha = s / N_IG_STEPS
        pos_alpha = (pos_baseline + alpha * (pos0 - pos_baseline)).clone().detach().requires_grad_(True)
        aux_alpha = (aux_baseline + alpha * (aux0 - aux_baseline)).clone().detach().requires_grad_(True)

        g_batch = dict(g_batch0)
        g_batch["pos"] = pos_alpha
        pred = model(g_batch, aux_alpha, device)[0]  # [N_TARGETS]

        for t in range(N_TARGETS):
            grad_pos, grad_aux = torch.autograd.grad(
                pred[t], [pos_alpha, aux_alpha], retain_graph=(t < N_TARGETS - 1))
            grad_pos_toplam[t] += grad_pos.detach().cpu()
            grad_aux_toplam[t] += grad_aux[0].detach().cpu()

    avg_grad_pos = grad_pos_toplam / N_IG_STEPS
    avg_grad_aux = grad_aux_toplam / N_IG_STEPS

    delta_pos = (pos0 - pos_baseline).detach().cpu()          # [n_atoms, 3]
    delta_aux = (aux0 - aux_baseline).detach().cpu()[0]        # [AUX_DIM]

    pos_ig = avg_grad_pos * delta_pos.unsqueeze(0)             # [N_TARGETS, n_atoms, 3]
    aux_ig = avg_grad_aux * delta_aux.unsqueeze(0)              # [N_TARGETS, AUX_DIM]

    with torch.no_grad():
        taban_tahmin = model(g_batch0 | {"pos": pos_baseline}, aux_baseline, device)[0].cpu().numpy()
        gercek_tahmin = model(g_batch0, aux0, device)[0].cpu().numpy()

    return pos_ig.numpy(), aux_ig.numpy(), taban_tahmin, gercek_tahmin


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

    atom_kayitlari, aux_kayitlari = [], []

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

        print(f"\nFold {fold_no}: {len(df_test)} ornek uzerinde Integrated Gradients hesaplaniyor...")
        for i, (_, row) in enumerate(df_test.iterrows()):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i + 1}/{len(df_test)}] {row[NAME_COLUMN]}", flush=True)

            g = cif_den_3b_graf(row["graf_cif_path"], CUTOFF)
            try:
                pos_ig, aux_ig, taban, gercek = integrated_gradients_hesapla(model, g, aux_matris[i], device)
            except Exception as e:
                print(f"    UYARI: {row[NAME_COLUMN]} atlandi ({e})")
                continue

            semboller = _z_sembol(g["z"])
            pos_np = g["pos"].numpy()
            for t, kol in enumerate(TARGET_COLUMNS):
                for a_idx in range(g["n_atoms"]):
                    atom_kayitlari.append({
                        "fold": fold_no, ID_COLUMN: row[ID_COLUMN], GROUP_COLUMN: row[GROUP_COLUMN],
                        NAME_COLUMN: row[NAME_COLUMN], "target_column": kol,
                        "atom_idx": a_idx, "element": semboller[a_idx],
                        "x": float(pos_np[a_idx, 0]), "y": float(pos_np[a_idx, 1]), "z": float(pos_np[a_idx, 2]),
                        "ig_x": float(pos_ig[t, a_idx, 0]), "ig_y": float(pos_ig[t, a_idx, 1]),
                        "ig_z": float(pos_ig[t, a_idx, 2]),
                        "ig_position_importance": float(pos_ig[t, a_idx].sum()),
                    })
                for j, aux_ad in enumerate(AUX_FEATURE_COLUMNS):
                    aux_kayitlari.append({
                        "fold": fold_no, ID_COLUMN: row[ID_COLUMN], GROUP_COLUMN: row[GROUP_COLUMN],
                        NAME_COLUMN: row[NAME_COLUMN], "target_column": kol, "aux_feature": aux_ad,
                        "ig_aux_importance": float(aux_ig[t, j]),
                    })

        del model

    if not atom_kayitlari:
        sys.exit(f"HATA: Hicbir checkpoint bulunamadi ({HEDEF_CKPT_DIR}). Once EGNN egitimini tamamlayin.")

    atom_df = pd.DataFrame(atom_kayitlari)
    atom_df.to_csv(SONUC_DIR / "ig_atom_konum_onem.csv", index=False)
    aux_df = pd.DataFrame(aux_kayitlari)
    aux_df.to_csv(SONUC_DIR / "ig_aux_ozellik_onem.csv", index=False)

    element_df = (
        atom_df.groupby(["element", "target_column"])["ig_position_importance"]
        .agg(ortalama="mean", abs_ortalama=lambda x: x.abs().mean(), adet="count")
        .reset_index().sort_values(["target_column", "abs_ortalama"], ascending=[True, False])
    )
    element_df.to_csv(SONUC_DIR / "ig_element_onem.csv", index=False)

    print(f"\nAtom-konum IG CSV -> {SONUC_DIR / 'ig_atom_konum_onem.csv'} ({len(atom_df)} satir)")
    print(f"Aux-ozellik IG CSV -> {SONUC_DIR / 'ig_aux_ozellik_onem.csv'} ({len(aux_df)} satir)")
    print(f"Element onem CSV  -> {SONUC_DIR / 'ig_element_onem.csv'}")
    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
