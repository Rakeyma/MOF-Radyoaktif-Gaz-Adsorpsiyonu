"""
DeeperGCN/run_deepergcn.py
=============================
AMAC:
    Li et al. (2020) "DeeperGCN: All You Need to Train Deeper GCNs" - MOF
    Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr secicilik tahmini (Bilesen 5:
    on-egitim + ince-ayar).

    torch_geometric.nn.GENConv (Generalized/differentiable-softmax
    agregasyonlu mesaj iletimi) + DeepGCNLayer (ResNet-tarzi 'res+' blok)
    resmi katmanlari kullanilir. 'res+' blogu, standart GCN'lerin 3-4
    katmandan sonra performans kaybettigi COK-KATMANLI (8 katman) rejimde
    egitilebilir kalmayi saglar - gozenekli MOF birim hucrelerinin genis
    (uzun linkerli) yapisinda UZUN-MENZILLI atom etkilesimlerini yakalamak
    icin faydalidir.

MIMARI:
    Z-gommesi(64) -> 8 x DeepGCNLayer(GENConv(edge_dim=N_RBF, aggr='softmax',
    learn_t=True), LayerNorm, ReLU, block='res+', dropout=0.1) -> scatter-mean
    (atom -> MOF) -> Linear -> EMB_DIM=64 -> [gomme | aux(15)] ->
    egitim_ortak 'RegressionHead' basi -> [Xe, Kr, Xe/Kr, I2]

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m DeeperGCN.run_deepergcn
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
    sys.exit("HATA: kok dizinden calistirin: python -m DeeperGCN.run_deepergcn")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "DeeperGCN"
HIDDEN = 64
N_LAYERS = 8  # DeeperGCN'in ana iddiasi: standart GCN'lerin basarisiz oldugu derinlikte calisabilmesi


class DeeperGCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GENConv, DeepGCNLayer

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)

        self.layers = nn.ModuleList()
        for i in range(N_LAYERS):
            conv = GENConv(HIDDEN, HIDDEN, aggr="softmax", t=1.0, learn_t=True,
                            num_layers=2, norm="layer", edge_dim=N_RBF)
            norm = nn.LayerNorm(HIDDEN, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            block = "plain" if i == 0 else "res+"
            self.layers.append(DeepGCNLayer(conv, norm, act, block=block, dropout=0.1))

        self.out_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        r, _ = edge_geometri(pos, edge_index)
        edge_attr = self.rbf(r)

        h = self.atom_emb(z)
        h = self.layers[0].conv(h, edge_index, edge_attr)
        for layer in self.layers[1:]:
            h = layer(h, edge_index, edge_attr)
        h = self.layers[0].act(self.layers[0].norm(h))

        graph_feat = scatter_mean_manual(h, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    pretrain_encoder(MODEL_ADI, lambda: DeeperGCNEncoder(), PRETRAIN_CKPT_DIR)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS})
    model_factory = model_factory_olustur(lambda: DeeperGCNEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
