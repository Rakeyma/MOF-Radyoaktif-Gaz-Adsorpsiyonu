"""
SubgraphX/run_subgraphx.py
=============================
AMAC:
    Yuan et al. (2021) "On Explainability of Graph Neural Networks via
    Subgraph Explorations" (SubgraphX) yontemini EGNN modeline (XAI_HEDEF_MODEL)
    uygular - "Üç Boyutlu Kristal Malzemeler/SubgraphX/run_subgraphx.py" ile
    AYNI Monte Carlo Tree Search (MCTS) metodolojisi, COK-HEDEFLI (4 hedef)
    versiyona genellenmis.

    COK-HEDEFLI ODUL TASARIMI (ONEMLI): egitim_ortak.RegressionHead.forward()
    ciktisi, egitim SIRASINDA STANDARDIZE EDILMIS hedeflerle (bkz.
    egitim_ortak.MOFDataset - (deger-ortalama)/std) KARSILASTIRILDIGINDAN,
    modelin HAM ciktisi ZATEN yaklasik BIRIM olcekte VE 4 hedef arasinda
    KARSILASTIRILABILIRDIR - bu yuzden MCTS odulu, 4 hedefin STANDARDIZE
    UZAYDAKI ortalama mutlak degisimi uzerinden TEK bir skalere indirgenir
    (ek bir normalizasyon adimina GEREK KALMAZ).

YONTEM (Monte Carlo Tree Search + Shapley-tarzi odul, EGNN'in ATOM-MASKELEME
hook'uyla - GraphLIME ile AYNI mekanizma, farkli arama stratejisi):
    1) KOK DURUM: TUM atomlarin tutuldugu durum S_0 = {0,...,M-1}.
    2) SECIM: UCT skoruna gore en umut verici COCUK secilir:
           UCT(c) = Q(c) + c_uct * sqrt(ln(N(ebeveyn)) / (1+N(c)))
    3) GENISLEME: secilen dugumun HER atomunu TEK TEK cikaran cocuk
       durumlar uretilir (min_size_frac altina inilmez).
    4) DEGERLENDIRME/ODUL: reward(S) = -mean_t|f(S)[t] - f(S_0)[t]| - lambda*|S|
       (S disindaki atomlar encoder.atom_emb ciktisi SIFIRLANARAK 'kaldirilir').
    5) GERI-YAYILIM: odul, kok'e kadar TUM ata dugumlerin (N, W) istatistiklerine eklenir.
    6) N_MCTS_ITER yineleme sonunda EN YUKSEK odullu durum secilir; atom-bazinda
       SUREKLI onem skoru, TUM ziyaret edilen durumlar uzerinden atomun ne
       siklikla TUTULDUGU (odul-agirlikli retention) olarak turetilir.

CIKTI:
    SubgraphX/sonuclar/subgraphx_atom_onem.csv          — atom-bazinda surekli onem (retention)
    SubgraphX/sonuclar/subgraphx_cekirdek_alt_graf.csv  — ornek basina secilen "en iyi" alt-graf + hedef-basina fidelity
    SubgraphX/sonuclar/subgraphx_element_onem.csv       — element sembolu bazinda ozet

CALISTIRMA (EGNN egitimi bittikten sonra, kok dizinden):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m SubgraphX.run_subgraphx
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
    sys.exit("HATA: kok dizinden calistirin: python -m SubgraphX.run_subgraphx")

from EGNN.run_egnn import EGNNEncoder, MODEL_ADI as XAI_HEDEF_MODEL
from EGNN.run_egnn import CHECKPOINT_DIR as HEDEF_CKPT_DIR

PROJE_DIZINI = Path(__file__).resolve().parent
SONUC_DIR = PROJE_DIZINI / "sonuclar"
SONUC_DIR.mkdir(parents=True, exist_ok=True)

N_MCTS_ITER = 25
C_UCT = 3.0
LAMBDA_BOYUT = 0.02
MIN_SIZE_FRAC = 0.3
MAX_PER_FOLD = 20  # MCTS pahali - GraphLIME/Edge_Attribution'dan daha kucuk ornek boyutu


class _MCTSDugum:
    __slots__ = ("S", "ebeveyn", "cocuklar", "N", "W", "genisletildi")

    def __init__(self, S: frozenset, ebeveyn=None):
        self.S = S
        self.ebeveyn = ebeveyn
        self.cocuklar: list[_MCTSDugum] = []
        self.N = 0
        self.W = 0.0
        self.genisletildi = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def _maskeli_tahmin(model, g_batch, aux_t, tutulan_atomlar: frozenset, n_atom: int, device: str) -> np.ndarray:
    mask = torch.zeros(n_atom)
    mask[list(tutulan_atomlar)] = 1.0

    def emb_hook(module, inp, out):
        return out * mask.to(out.device).unsqueeze(-1)

    handle = model.encoder.atom_emb.register_forward_hook(emb_hook)
    try:
        with torch.no_grad():
            pred = model(g_batch, aux_t, device)[0]  # [N_TARGETS], standardize edilmis olcek
    finally:
        handle.remove()
    return pred.cpu().numpy()


def subgraphx_calistir(model, g_raw: dict, aux_vec: np.ndarray, device: str,
                        rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    M = g_raw["n_atoms"]
    g_batch = batch_graphs([g_raw])
    aux_t = torch.tensor(aux_vec[None], dtype=torch.float)
    min_size = max(1, int(M * MIN_SIZE_FRAC))

    taban_pred = _maskeli_tahmin(model, g_batch, aux_t, frozenset(range(M)), M, device)

    def odul_hesapla(S: frozenset) -> float:
        pred = _maskeli_tahmin(model, g_batch, aux_t, S, M, device)
        fidelity = -float(np.mean(np.abs(pred - taban_pred)))
        return fidelity - LAMBDA_BOYUT * len(S)

    kok = _MCTSDugum(frozenset(range(M)))
    en_iyi = {"S": kok.S, "reward": odul_hesapla(kok.S)}
    ziyaret_agirlik: dict[int, float] = {a: 0.0 for a in range(M)}
    ziyaret_toplam = 0.0

    for _ in range(N_MCTS_ITER):
        dugum = kok
        yol = [dugum]
        while dugum.genisletildi and dugum.cocuklar:
            dugum = max(dugum.cocuklar, key=lambda c: c.Q + C_UCT * np.sqrt(np.log(dugum.N + 1) / (1 + c.N)))
            yol.append(dugum)

        if not dugum.genisletildi and len(dugum.S) > min_size:
            dugum.genisletildi = True
            for atom in dugum.S:
                yeni_S = frozenset(dugum.S - {atom})
                if len(yeni_S) >= 1:
                    dugum.cocuklar.append(_MCTSDugum(yeni_S, ebeveyn=dugum))
            if dugum.cocuklar:
                dugum = rng.choice(dugum.cocuklar)
                yol.append(dugum)

        odul = odul_hesapla(dugum.S)
        if odul > en_iyi["reward"]:
            en_iyi = {"S": dugum.S, "reward": odul}

        for d in yol:
            d.N += 1
            d.W += odul

        agirlik = np.exp(odul)  # pozitif (yuksek-odullu) durumlara daha fazla agirlik
        for atom in dugum.S:
            ziyaret_agirlik[atom] += agirlik
        ziyaret_toplam += agirlik

    onem = np.array([ziyaret_agirlik[a] / max(ziyaret_toplam, 1e-9) for a in range(M)])
    en_iyi_pred = _maskeli_tahmin(model, g_batch, aux_t, en_iyi["S"], M, device)

    return {
        "cekirdek_atomlar": sorted(en_iyi["S"]), "odul": en_iyi["reward"],
        "n_cekirdek": len(en_iyi["S"]), "n_toplam": M,
        "taban_tahmin": taban_pred, "cekirdek_tahmin": en_iyi_pred,
    }, onem


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
    rng = np.random.default_rng(SEED)

    atom_kayitlari, cekirdek_kayitlari = [], []

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

        print(f"\nFold {fold_no}: {len(df_test)} ornek uzerinde SubgraphX (MCTS) calistiriliyor...")
        for i, (_, row) in enumerate(df_test.iterrows()):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i + 1}/{len(df_test)}] {row[NAME_COLUMN]}", flush=True)

            g = cif_den_3b_graf(row["graf_cif_path"], CUTOFF)
            try:
                sonuc, onem = subgraphx_calistir(model, g, aux_matris[i], device, rng)
            except Exception as e:
                print(f"    UYARI: {row[NAME_COLUMN]} atlandi ({e})")
                continue

            semboller = _z_sembol(g["z"])
            pos_np = g["pos"].numpy()
            for a_idx in range(g["n_atoms"]):
                atom_kayitlari.append({
                    "fold": fold_no, ID_COLUMN: row[ID_COLUMN], GROUP_COLUMN: row[GROUP_COLUMN],
                    NAME_COLUMN: row[NAME_COLUMN], "atom_idx": a_idx, "element": semboller[a_idx],
                    "x": float(pos_np[a_idx, 0]), "y": float(pos_np[a_idx, 1]), "z": float(pos_np[a_idx, 2]),
                    "subgraphx_onem": float(onem[a_idx]),
                    "in_cekirdek_alt_graf": a_idx in sonuc["cekirdek_atomlar"],
                })

            kayit = {
                "fold": fold_no, ID_COLUMN: row[ID_COLUMN], GROUP_COLUMN: row[GROUP_COLUMN],
                NAME_COLUMN: row[NAME_COLUMN], "n_cekirdek": sonuc["n_cekirdek"], "n_toplam": sonuc["n_toplam"],
                "cekirdek_atom_orani": sonuc["n_cekirdek"] / max(sonuc["n_toplam"], 1), "odul": sonuc["odul"],
                "cekirdek_atomlar": ",".join(map(str, sonuc["cekirdek_atomlar"])),
            }
            for t, kol in enumerate(TARGET_COLUMNS):
                kayit[f"taban_tahmin_{kol}"] = float(sonuc["taban_tahmin"][t])
                kayit[f"cekirdek_tahmin_{kol}"] = float(sonuc["cekirdek_tahmin"][t])
            cekirdek_kayitlari.append(kayit)

        del model

    if not atom_kayitlari:
        sys.exit(f"HATA: Hicbir checkpoint bulunamadi ({HEDEF_CKPT_DIR}). Once EGNN egitimini tamamlayin.")

    atom_df = pd.DataFrame(atom_kayitlari)
    atom_df.to_csv(SONUC_DIR / "subgraphx_atom_onem.csv", index=False)
    cekirdek_df = pd.DataFrame(cekirdek_kayitlari)
    cekirdek_df.to_csv(SONUC_DIR / "subgraphx_cekirdek_alt_graf.csv", index=False)

    element_df = (
        atom_df.groupby("element")["subgraphx_onem"]
        .agg(ortalama="mean", std="std", adet="count",
             cekirdek_orani=lambda x: atom_df.loc[x.index, "in_cekirdek_alt_graf"].mean())
        .reset_index().sort_values("ortalama", ascending=False)
    )
    element_df.to_csv(SONUC_DIR / "subgraphx_element_onem.csv", index=False)

    print(f"\nAtom onem CSV     -> {SONUC_DIR / 'subgraphx_atom_onem.csv'} ({len(atom_df)} satir)")
    print(f"Cekirdek alt-graf CSV -> {SONUC_DIR / 'subgraphx_cekirdek_alt_graf.csv'} ({len(cekirdek_df)} satir)")
    print(f"Element onem CSV  -> {SONUC_DIR / 'subgraphx_element_onem.csv'}")
    print(f"\nOrtalama cekirdek/toplam atom orani: {cekirdek_df['cekirdek_atom_orani'].mean():.3f}")
    print("\n--- Element Basi Ortalama SubgraphX Onemi (ilk 15) ---")
    print(element_df.head(15).to_string(index=False))
    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
