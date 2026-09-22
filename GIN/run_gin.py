"""
GIN/run_gin.py
================
AMAC:
    Xu et al. (2019) "How Powerful are Graph Neural Networks?" - Graph
    Isomorphism Network (GIN), edge-feature-farkinda varyanti (GINE, Hu et
    al. 2020) ile - MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr secicilik
    tahmini (Bilesen 5: on-egitim + ince-ayar, bkz. egitim_ortak.py).

    torch_geometric.nn.GINEConv resmi katmani kullanilir (WL-testi kadar
    ayirt edici mesaj iletimi + toplam (sum) agregasyon, MLP guncelleme).

MIMARI:
    Z-gommesi(64) -> 4 x GINEConv(mlp=Lin-SiLU-Lin, edge_attr=RBF(mesafe)
    projeksiyonu, 64-boyutlu) + LayerNorm + reziduel -> scatter-mean(atom
    -> MOF) -> Linear -> EMB_DIM=64 -> [gomme | aux(15)] -> egitim_ortak
    'RegressionHead' basi (head) -> [Xe, Kr, Xe/Kr, I2]

    Egitim dongusu, K-fold, erken durdurma, checkpoint, transfer learning,
    permutation importance: TAMAMI egitim_ortak.py'den (paylasilan cerceve).

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m GIN.run_gin
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
    sys.exit(
        "HATA: 'graf_ozellik_ortak'/'egitim_ortak' bulunamadi. Kok dizininden calistirin:\n"
        "    cd \"MOF Radyoaktif Gaz Adsorpsiyonu\" && python -m GIN.run_gin"
    )

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "GIN"
HIDDEN = 64
N_LAYERS = 4


class GINEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)
        self.edge_proj = nn.Linear(N_RBF, HIDDEN)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(N_LAYERS):
            mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(HIDDEN))

        self.out_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        r, _ = edge_geometri(pos, edge_index)
        edge_attr = self.edge_proj(self.rbf(r))

        h = self.atom_emb(z)
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.nn.functional.silu(norm(conv(h, edge_index, edge_attr)))

        graph_feat = scatter_mean_manual(h, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    # BILESEN 5 / ASAMA A: devasa proxy veri seti (QMOF_PRETRAIN_CSV) uzerinde
    # SADECE encoder'i on-egitir (veri yoksa zarifce atlanir).
    pretrain_encoder(MODEL_ADI, lambda: GINEncoder(), PRETRAIN_CKPT_DIR)

    # BILESEN 5 / ASAMA B: gaz-adsorpsiyon hedefleriyle K-fold ince-ayar
    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS, "hidden": HIDDEN})
    model_factory = model_factory_olustur(lambda: GINEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
