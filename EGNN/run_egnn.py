"""
EGNN/run_egnn.py
===================
AMAC:
    Satorras, Hoogeboom & Welling (2021) "E(n) Equivariant Graph Neural
    Networks" (EGNN) - MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr
    secicilik tahmini (Bilesen 5: on-egitim + ince-ayar). SIFIRDAN, SAF
    PYTORCH (torch_geometric.nn KATMANI KULLANILMAZ - digerleri gibi, e3nn
    de GEREKMEZ).

    EGNN, TAM E(3) (rotasyon+oteleme+YANSIMA) ESDEGERLI (equivariant) EN
    BASIT mimarilerden biridir: vektör ozellikler (yon bilgisi) TUTULMAZ,
    mesajlar SADECE ATOMLAR ARASI MESAFENIN KARESINE (||x_i-x_j||^2 -
    rotasyon-degismez/invariant) bagli olarak hesaplanir.

    BU PROJEDE EGNN'IN ROLU: 11 model arasinda en hafif/hizli ileri-gecisli
    model oldugundan, TUM 4 XAI yontemi (GraphLIME/Edge_Attribution/
    SubgraphX/IntegratedGradients) icin XAI_HEDEF_MODEL olarak secilmistir
    (bkz. GraphLIME/run_graphlime.py'deki gerekce).

MESAJ ILETIMI (Satorras et al. 2021, Denklem 3-6; koordinat guncellemesi
KAPALI - bu bir OZELLIK tahmin gorevi, koordinat/kuvvet tahmini DEGIL):
    m_ij = phi_e(h_i, h_j, ||x_i-x_j||^2, RBF(r_ij))
    m_i  = (1/|N(i)|) * sum_{j in N(i)} m_ij         <- invariant toplama
    h_i' = h_i + phi_h(h_i, m_i)                      <- reziduel guncelleme

MIMARI:
    Z-gommesi(64) -> 4 x EGNNLayer -> scatter-mean(atom -> MOF) -> Linear
    -> EMB_DIM=64 -> [gomme | aux(15)] -> egitim_ortak 'RegressionHead'
    basi -> [Xe, Kr, Xe/Kr, I2]

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m EGNN.run_egnn
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    from graf_ozellik_ortak import (
        GaussianRBF, edge_geometri, scatter_mean_manual, NUM_ATOM_TYPES, N_RBF, EMB_DIM,
    )
    from egitim_ortak import EgitimAyarlari, model_factory_olustur, finetune_k_fold_egit, pretrain_encoder
    from paths import PRETRAINED_ENCODER_FILENAME
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m EGNN.run_egnn")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "EGNN"
HIDDEN = 64
N_LAYERS = 4


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1 + n_rbf, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, pos, edge_index, rbf):
        src, dst = edge_index
        d2 = (pos[src] - pos[dst]).pow(2).sum(-1, keepdim=True)  # ||x_i - x_j||^2 - rotasyon-degismez
        m_ij = self.edge_mlp(torch.cat([h[src], h[dst], d2, rbf], dim=-1))
        m_i = scatter_mean_manual(m_ij, dst, h.size(0))
        h_yeni = h + self.node_mlp(torch.cat([h, m_i], dim=-1))
        return self.norm(h_yeni)


class EGNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)
        self.layers = nn.ModuleList([EGNNLayer(HIDDEN, N_RBF) for _ in range(N_LAYERS)])
        self.out_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        r, _ = edge_geometri(pos, edge_index)
        rbf = self.rbf(r)

        h = self.atom_emb(z)
        for layer in self.layers:
            h = layer(h, pos, edge_index, rbf)

        graph_feat = scatter_mean_manual(h, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    pretrain_encoder(MODEL_ADI, lambda: EGNNEncoder(), PRETRAIN_CKPT_DIR)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS})
    model_factory = model_factory_olustur(lambda: EGNNEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
