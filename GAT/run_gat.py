"""
GAT/run_gat.py
================
AMAC:
    Velickovic et al. (2018) "Graph Attention Networks", GATv2 (Brody et al.
    2022) duzeltmesiyle - MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr
    secicilik tahmini (Bilesen 5: on-egitim + ince-ayar).

    torch_geometric.nn.GATv2Conv resmi katmani kullanilir (edge_dim ile
    mesafe-RBF'i DOGRUDAN dikkat (attention) skoruna sokan kenar-farkinda
    varyant).

MIMARI:
    Z-gommesi(64) -> 4 x GATv2Conv(heads=4, concat=False, edge_dim=N_RBF) +
    LayerNorm + reziduel -> scatter-mean(atom -> MOF) -> Linear -> EMB_DIM=64
    -> [gomme | aux(15)] -> egitim_ortak 'RegressionHead' basi ->
    [Xe, Kr, Xe/Kr, I2]

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m GAT.run_gat
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
    sys.exit("HATA: kok dizinden calistirin: python -m GAT.run_gat")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "GAT"
HIDDEN = 64
N_LAYERS = 4
N_HEADS = 4


class GATEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)

        self.convs = nn.ModuleList([
            GATv2Conv(HIDDEN, HIDDEN // N_HEADS, heads=N_HEADS, concat=True,
                      edge_dim=N_RBF, dropout=0.1, add_self_loops=False)
            for _ in range(N_LAYERS)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        self.out_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        r, _ = edge_geometri(pos, edge_index)
        edge_attr = self.rbf(r)

        h = self.atom_emb(z)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.nn.functional.elu(norm(conv(h, edge_index, edge_attr)))

        graph_feat = scatter_mean_manual(h, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    pretrain_encoder(MODEL_ADI, lambda: GATEncoder(), PRETRAIN_CKPT_DIR)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS, "heads": N_HEADS})
    model_factory = model_factory_olustur(lambda: GATEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
