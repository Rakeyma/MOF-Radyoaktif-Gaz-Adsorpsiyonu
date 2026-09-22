"""
GraphGPS/run_graphgps.py
===========================
AMAC:
    Rampasek et al. (2022) "Recipe for a General, Powerful, Scalable Graph
    Transformer" (GraphGPS) - MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr
    secicilik tahmini (Bilesen 5: on-egitim + ince-ayar).

    torch_geometric.nn.GPSConv resmi katmani kullanilir: HER katmanda,
    (1) YEREL bir mesaj-iletim aginin (burada GINEConv, kenar-farkinda) VE
    (2) KURESEL/TAM (global, tum atom ciftleri arasi) coklu-baslikli
    self-attention'in ciktilari TOPLANIR - GraphGPS'in temel iddiasi: yerel
    MPNN'lerin komsuluk-onyargisi (inductive bias) + Transformer'in uzun-
    menzilli etkilesim yakalama gucunu BIRLESTIRMEK. Bu, gozenekli MOF
    yapilarinda ozellikle FAYDALIDIR: bir gaz molekulunun bir gozenek
    penceresinden gecip gecemeyecegi, o pencereyi cevreleyen atomlarin
    LOKAL duzenlenimine (yerel MPNN) DEGIL, TÜM kavite/pencere sistemine
    (kuresel dikkat) bagli olabilir.

MIMARI:
    Z-gommesi(64) -> 4 x GPSConv(channels=64, conv=GINEConv, heads=4,
    attn_type='multihead', dropout=0.1) -> scatter-mean(atom -> MOF) ->
    Linear -> EMB_DIM=64 -> [gomme | aux(15)] -> egitim_ortak
    'RegressionHead' basi -> [Xe, Kr, Xe/Kr, I2]

    NOT: GPSConv'un kuresel self-attention terimi O(N^2) maliyetlidir; devasa
    gozenekli MOF birim hucrelerinde (N buyuk) bu maliyet artabilir - kesme-
    yaricapli (cutoff) lokal komsuluk sayesinde N tipik olarak birkac yuz
    atomu gecmez, pratik kalir.

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m GraphGPS.run_graphgps
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
    sys.exit("HATA: kok dizinden calistirin: python -m GraphGPS.run_graphgps")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "GraphGPS"
HIDDEN = 64
N_LAYERS = 4
N_HEADS = 4


class GraphGPSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GPSConv, GINEConv

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)
        self.edge_proj = nn.Linear(N_RBF, HIDDEN)

        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            local_mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
            local_conv = GINEConv(local_mlp)
            self.layers.append(GPSConv(HIDDEN, conv=local_conv, heads=N_HEADS,
                                        dropout=0.1, attn_type="multihead"))

        self.out_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        r, _ = edge_geometri(pos, edge_index)
        edge_attr = self.edge_proj(self.rbf(r))  # GINEConv icin dugum-boyutuyla eslesmis kenar ozelligi

        h = self.atom_emb(z)
        for gps in self.layers:
            h = gps(h, edge_index, batch, edge_attr=edge_attr)

        graph_feat = scatter_mean_manual(h, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    pretrain_encoder(MODEL_ADI, lambda: GraphGPSEncoder(), PRETRAIN_CKPT_DIR)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, batch_size=16,  # global attention agir - kucuk batch
                              ekstra_bilgi={"n_layers": N_LAYERS, "heads": N_HEADS})
    model_factory = model_factory_olustur(lambda: GraphGPSEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
