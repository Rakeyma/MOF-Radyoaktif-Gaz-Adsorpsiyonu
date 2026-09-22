"""
TFN/run_tfn.py
=================
AMAC:
    Thomas et al. (2018) "Tensor Field Networks: Rotation- and Translation-
    Equivariant Neural Networks for 3D Point Clouds" (TFN) - MOF Xe/Kr/I2
    adsorpsiyon kapasitesi + Xe/Kr secicilik tahmini (Bilesen 5: on-egitim
    + ince-ayar). SIFIRDAN, SAF PYTORCH (e3nn GEREKMEZ - l<=1 (SKALER +
    VEKTOR) dereceyle SINIRLI, HESAPCA OLCEKLENEBILIR bir TFN varyanti).

    EGNN'DEN FARKI: EGNN sadece SKALER (l=0) ozellik tasir ve mesajlari
    SADECE ||x_i-x_j||^2 (invariant mesafe) uzerinden hesaplar - yon bilgisi
    (l=1) YOKTUR. TFN ise l=0 (skaler s) VE l=1 (vektor v) ozellikleri
    BIRLIKTE tasir ve mesajlari GERCEK Clebsch-Gordan (CG) tensor-carpim
    kurallariyla birlestirir:
        l=0 ⊗ l=0 -> l=0   (skaler x skaler = skaler)
        l=0 ⊗ l=1 -> l=1   (skaler x vektor = vektor, filtre*yon)
        l=1 ⊗ l=1 -> l=0   (vektor . vektor = skaler, ic carpim)
        l=1 ⊗ l=1 -> l=1   (vektor x vektor = vektor, dis/cross carpim)
    Gozenekli MOF yapilarinda YON bilgisi (l=1) onemlidir: bir gozenek
    penceresinin ACIKLIK YONU, gaz molekulunun o pencereden GECEBILME
    olasiligini etkiler - EGNN'in skaler-sadece temsili bu yonelim
    bilgisini kaybeder, TFN korur.

MIMARI:
    Z-gommesi(64) -> 4 x TFNLayer(4 radyal filtreli CG mesaj + self-interaction
    guncelleme) -> scatter-mean(atom -> MOF, skaler+||vektor|| birlesimi)
    -> Linear -> EMB_DIM=64 -> [gomme | aux(15)] -> egitim_ortak
    'RegressionHead' basi -> [Xe, Kr, Xe/Kr, I2]

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m TFN.run_tfn
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
    sys.exit("HATA: kok dizinden calistirin: python -m TFN.run_tfn")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "TFN"
HIDDEN = 64
N_LAYERS = 4


def _radyal_filtre(n_rbf: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))


class TFNLayer(nn.Module):
    """Mesaj: tam l<=1 Clebsch-Gordan ciftlenim kumesi (0x0->0, 0x1->1,
    1x1->0, 1x1->1), her biri kendi ogrenilebilir radyal filtresiyle.
    Guncelleme: skaler kanal uzerinde nonlineerlik + vektor kanalini
    invariant bir kapiyla (gate) olcekleme (ekivaryantligi bozmaz)."""

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.W0 = _radyal_filtre(n_rbf, hidden)  # 0x0 -> 0
        self.W1 = _radyal_filtre(n_rbf, hidden)  # 0x1 -> 1
        self.W2 = _radyal_filtre(n_rbf, hidden)  # 1x1 -> 0
        self.W3 = _radyal_filtre(n_rbf, hidden)  # 1x1 -> 1
        self.update_s = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update_gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.norm_s = nn.LayerNorm(hidden)

    def forward(self, s, v, edge_index, rbf, r_hat):
        src, dst = edge_index
        n = s.size(0)

        s_j, v_j = s[src], v[src]                        # [E,F], [E,F,3]
        w0, w1, w2, w3 = self.W0(rbf), self.W1(rbf), self.W2(rbf), self.W3(rbf)

        m_s_00 = s_j * w0
        dot_vj_rhat = (v_j * r_hat.unsqueeze(1)).sum(-1)  # [E,F]
        m_s_11 = dot_vj_rhat * w2
        m_s = m_s_00 + m_s_11

        m_v_01 = (s_j * w1).unsqueeze(-1) * r_hat.unsqueeze(1)  # [E,F,3]
        r_hat_exp = r_hat.unsqueeze(1).expand_as(v_j)
        cross_vj_rhat = torch.cross(v_j, r_hat_exp, dim=-1)      # [E,F,3]
        m_v_11 = cross_vj_rhat * w3.unsqueeze(-1)
        m_v = m_v_01 + m_v_11

        ds = scatter_mean_manual(m_s, dst, n)
        dv = scatter_mean_manual(m_v, dst, n)

        s1 = s + ds
        v1 = v + dv

        v_norm = v1.norm(dim=-1)  # [N,F]
        s2 = self.norm_s(s1 + self.update_s(torch.cat([s1, v_norm], dim=-1)))
        gate = self.update_gate(torch.cat([s1, v_norm], dim=-1)).unsqueeze(-1)  # [N,F,1]
        v2 = v1 * gate

        return s2, v2


class TFNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)
        self.layers = nn.ModuleList([TFNLayer(HIDDEN, N_RBF) for _ in range(N_LAYERS)])
        self.out_proj = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        r, r_hat = edge_geometri(pos, edge_index)
        rbf = self.rbf(r)

        s = self.atom_emb(z)
        v = s.new_zeros(s.size(0), HIDDEN, 3)

        for layer in self.layers:
            s, v = layer(s, v, edge_index, rbf, r_hat)

        s_graph = scatter_mean_manual(s, batch, n_graphs)
        v_graph = scatter_mean_manual(v.norm(dim=-1), batch, n_graphs)
        return self.out_proj(torch.cat([s_graph, v_graph], dim=-1))


if __name__ == "__main__":
    pretrain_encoder(MODEL_ADI, lambda: TFNEncoder(), PRETRAIN_CKPT_DIR)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS, "l_max": 1})
    model_factory = model_factory_olustur(lambda: TFNEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
