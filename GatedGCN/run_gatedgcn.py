"""
GatedGCN/run_gatedgcn.py
==========================
AMAC:
    Bresson & Laurent (2017) "Residual Gated Graph ConvNets" (GatedGCN) -
    MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr secicilik tahmini
    (Bilesen 5: on-egitim + ince-ayar).

    torch_geometric.nn.ResGatedGraphConv resmi katmani kullanilir: her
    kenar icin bir GECIS KAPISI (edge gate) eta_ij = sigmoid(W1*h_i +
    W2*h_j [+ W_e*e_ij]) ogrenilir ve komsu mesajlari bu kapiyla
    agirliklandirilarak toplanir.

MIMARI:
    Z-gommesi(64) -> 4 x ResGatedGraphConv(edge_dim=N_RBF) + LayerNorm +
    reziduel -> scatter-mean(atom -> MOF) -> Linear -> EMB_DIM=64 ->
    [gomme | aux(15)] -> egitim_ortak 'RegressionHead' basi ->
    [Xe, Kr, Xe/Kr, I2]

NOT (surum bagimliligi): ResGatedGraphConv'un edge_dim parametresi
torch_geometric >= 2.4 gerektirir.

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m GatedGCN.run_gatedgcn
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
    sys.exit("HATA: kok dizinden calistirin: python -m GatedGCN.run_gatedgcn")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "GatedGCN"
HIDDEN = 64
N_LAYERS = 4


class GatedGCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import ResGatedGraphConv

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)

        self.convs = nn.ModuleList([
            ResGatedGraphConv(HIDDEN, HIDDEN, edge_dim=N_RBF) for _ in range(N_LAYERS)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
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
    pretrain_encoder(MODEL_ADI, lambda: GatedGCNEncoder(), PRETRAIN_CKPT_DIR)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS})
    model_factory = model_factory_olustur(lambda: GatedGCNEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
