"""
SE3_Transformer/run_se3_transformer.py
=========================================
AMAC:
    Fuchs et al. (2020) "SE(3)-Transformers: 3D Roto-Translation Equivariant
    Attention Networks" - MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr
    secicilik tahmini (Bilesen 5: on-egitim + ince-ayar). SIFIRDAN, SAF
    PYTORCH (e3nn GEREKMEZ - TFN/EGNN ile AYNI felsefe: l<=1 ile SINIRLI).

    TFN'DEN FARKI: TFN.py'deki mesaj TERIMLERI (0x0->0, 0x1->1, 1x1->0,
    1x1->1 Clebsch-Gordan ciftlenimleri) BURADA DA AYNEN kullanilir - ANCAK
    komsular uzerindeki toplama (aggregation) artik DUZ ORTALAMA (mean)
    DEGIL, COKLU-BASLIKLI (multi-head) bir DIKKAT (attention) agirligiyla
    YAPILIR:
        alpha_ij = softmax_j( (W_Q s_i) . (W_K [s_j, RBF(r_ij)]) / sqrt(d) )
    Bu, gozenekli MOF yapilarinda ozellikle anlamlidir: bir atomun (gozenek
    duvarindaki) HANGI komsularinin gaz-baglanma etkilesimi icin "daha
    onemli" oldugu (orn. acik metal bolgesine yakinlik) SABIT/duz ortalama
    yerine OGRENILIR (dinamik agirlik).

MIMARI:
    Z-gommesi(64) -> 4 x SE3TransformerLayer (4-baslikli ekivaryant dikkat
    + CG mesaj + self-interaction guncelleme) -> scatter-mean(atom -> MOF,
    skaler+||vektor|| birlesimi) -> Linear -> EMB_DIM=64 -> [gomme | aux(15)]
    -> egitim_ortak 'RegressionHead' basi -> [Xe, Kr, Xe/Kr, I2]

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m SE3_Transformer.run_se3_transformer
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    from graf_ozellik_ortak import (
        GaussianRBF, edge_geometri, scatter_mean_manual, segment_softmax,
        NUM_ATOM_TYPES, N_RBF, EMB_DIM,
    )
    from egitim_ortak import EgitimAyarlari, model_factory_olustur, finetune_k_fold_egit, pretrain_encoder
    from paths import PRETRAINED_ENCODER_FILENAME
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m SE3_Transformer.run_se3_transformer")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "SE3_Transformer"
HIDDEN = 64
N_LAYERS = 4
N_HEADS = 4


def _radyal_filtre(n_rbf: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))


class SE3TransformerLayer(nn.Module):
    def __init__(self, hidden: int, n_rbf: int, n_heads: int = N_HEADS):
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

        self.W0 = _radyal_filtre(n_rbf, hidden)  # 0x0 -> 0
        self.W1 = _radyal_filtre(n_rbf, hidden)  # 0x1 -> 1
        self.W2 = _radyal_filtre(n_rbf, hidden)  # 1x1 -> 0
        self.W3 = _radyal_filtre(n_rbf, hidden)  # 1x1 -> 1

        self.W_q = nn.Linear(hidden, hidden)
        self.W_k = nn.Sequential(nn.Linear(hidden + n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        self.update_s = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update_gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.norm_s = nn.LayerNorm(hidden)

    def forward(self, s, v, edge_index, rbf, r_hat):
        src, dst = edge_index
        n = s.size(0)
        s_j, v_j = s[src], v[src]
        w0, w1, w2, w3 = self.W0(rbf), self.W1(rbf), self.W2(rbf), self.W3(rbf)

        m_s_ij = s_j * w0 + (v_j * r_hat.unsqueeze(1)).sum(-1) * w2                     # [E,HIDDEN]
        r_hat_exp = r_hat.unsqueeze(1).expand_as(v_j)
        cross_vj_rhat = torch.cross(v_j, r_hat_exp, dim=-1)
        m_v_ij = (s_j * w1).unsqueeze(-1) * r_hat.unsqueeze(1) + cross_vj_rhat * w3.unsqueeze(-1)  # [E,HIDDEN,3]

        q = self.W_q(s).view(n, self.n_heads, self.head_dim)
        k = self.W_k(torch.cat([s_j, rbf], dim=-1)).view(-1, self.n_heads, self.head_dim)
        logits = (q[dst] * k).sum(-1) / (self.head_dim ** 0.5)   # [E, n_heads]
        alpha = segment_softmax(logits, dst, n)                   # [E, n_heads], sum_j alpha_ij = 1

        alpha_s = alpha.repeat_interleave(self.head_dim, dim=-1)  # [E, HIDDEN]
        m_s_weighted = m_s_ij * alpha_s
        m_v_weighted = m_v_ij * alpha_s.unsqueeze(-1)

        ds = s.new_zeros(n, self.hidden)
        ds.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden), m_s_weighted)
        dv = v.new_zeros(n, self.hidden, 3)
        idx3 = dst[:, None, None].expand(-1, self.hidden, 3)
        dv.scatter_add_(0, idx3, m_v_weighted)

        s1 = s + ds
        v1 = v + dv

        v_norm = v1.norm(dim=-1)
        s2 = self.norm_s(s1 + self.update_s(torch.cat([s1, v_norm], dim=-1)))
        gate = self.update_gate(torch.cat([s1, v_norm], dim=-1)).unsqueeze(-1)
        v2 = v1 * gate
        return s2, v2


class SE3TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)
        self.layers = nn.ModuleList([SE3TransformerLayer(HIDDEN, N_RBF, N_HEADS) for _ in range(N_LAYERS)])
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
    pretrain_encoder(MODEL_ADI, lambda: SE3TransformerEncoder(), PRETRAIN_CKPT_DIR, batch_size=16)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, batch_size=16,
                              ekstra_bilgi={"n_layers": N_LAYERS, "heads": N_HEADS, "l_max": 1})
    model_factory = model_factory_olustur(lambda: SE3TransformerEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
