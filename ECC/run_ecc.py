"""
ECC/run_ecc.py
================
AMAC:
    Simonovsky & Komodakis (2017) "Dynamic Edge-Conditioned Filters in
    Convolutional Neural Networks on Graphs" (ECC) - MOF Xe/Kr/I2
    adsorpsiyon kapasitesi + Xe/Kr secicilik tahmini (Bilesen 5: on-egitim
    + ince-ayar).

    torch_geometric.nn.NNConv resmi katmani kullanilir: HER kenar icin,
    kucuk bir MLP (filtre-ureten ag, 'edge network') mesafe-RBF'ini
    dogrudan bir [HIDDEN x HIDDEN] AGIRLIK MATRISINE cozer (dinamik filtre).

MIMARI:
    Z-gommesi(64) -> 3 x NNConv(edge_nn=MLP(N_RBF->32->HIDDEN*HIDDEN), aggr='mean')
    + LayerNorm + reziduel -> scatter-mean(atom -> MOF) -> Linear ->
    EMB_DIM=64 -> [gomme | aux(15)] -> egitim_ortak 'RegressionHead' basi
    -> [Xe, Kr, Xe/Kr, I2]

    NOT: NNConv'un dinamik agirlik matrisi HIDDEN*HIDDEN buyuklugunde
    oldugundan, diger modellerden FARKLI OLARAK SADECE 3 katman + kucuk
    batch (16) kullanilir - devasa/gozenekli MOF veri setinde bellek
    tasmasini onlemek icin bilincli bir tasarim tercihi (bkz. sibling
    projedeki ayni gerekce).

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m ECC.run_ecc
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    from graf_ozellik_ortak import GaussianRBF, edge_geometri, scatter_mean_manual, NUM_ATOM_TYPES, N_RBF, EMB_DIM
    from egitim_ortak import EgitimAyarlari, model_factory_olustur, finetune_k_fold_egit, pretrain_encoder
    from paths import PRETRAINED_ENCODER_FILENAME
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m ECC.run_ecc")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "ECC"
HIDDEN = 64
N_LAYERS = 3  # dinamik agirlik matrisinin (HIDDEN x HIDDEN) maliyeti nedeniyle sinirli


class ECCEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import NNConv

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(N_LAYERS):
            edge_nn = nn.Sequential(
                nn.Linear(N_RBF, 32), nn.SiLU(), nn.Linear(32, HIDDEN * HIDDEN)
            )
            self.convs.append(NNConv(HIDDEN, HIDDEN, edge_nn, aggr="mean"))
            self.norms.append(nn.LayerNorm(HIDDEN))

        self.out_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        r, _ = edge_geometri(pos, edge_index)
        edge_attr = self.rbf(r)

        h = self.atom_emb(z)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.nn.functional.silu(norm(conv(h, edge_index, edge_attr)))

        graph_feat = scatter_mean_manual(h, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    pretrain_encoder(MODEL_ADI, lambda: ECCEncoder(), PRETRAIN_CKPT_DIR, batch_size=16)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, batch_size=16,  # NNConv agir - daha kucuk batch
                              ekstra_bilgi={"n_layers": N_LAYERS})
    model_factory = model_factory_olustur(lambda: ECCEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
