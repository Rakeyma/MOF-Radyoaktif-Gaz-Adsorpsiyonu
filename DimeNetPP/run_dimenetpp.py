"""
DimeNetPP/run_dimenetpp.py
=============================
AMAC (OTONOM EKLENEN 11. MODEL - bkz. paths.MODEL_KLASORLERI dokstringi):
    Gasteiger, Giri, Margraf & Gunnemann (2020) "Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules" (DimeNet++) -
    MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr secicilik tahmini
    (Bilesen 5: on-egitim + ince-ayar).

    NEDEN OTONOM OLARAK EKLENDI (kullanicinin "ek SOTA mimari" gereksinimi):
    Xe/Kr/I2'nin MOF gozeneklerinde SIGABILMESI (boyut-elemesi/size-sieving)
    TEMEL OLARAK gozenek GEOMETRISINE - ozellikle Gozenek-Sinirlayici Cap
    (PLD) etrafindaki atomlarin ACISAL DUZENLENIMINE (bir "pencere"nin ne
    kadar dar/genis oldugu, komsu atomlarin BIRBIRINE GORE ACISI ile
    belirlenir) - baglidir. Bu projedeki DIGER 10 model (GraphGPS...SE3_
    Transformer) mesajlari YALNIZCA ikili (pairwise) mesafe/yon uzerinden
    kurar; DimeNet(++) ise UCLU (triplet, k-j-i) ETKILESIMLER uzerinden
    ACI (theta_kji) bilgisini DOGRUDAN mesaja katan, moleküler/gozenekli
    sistemlerde durum-of-the-art bir mimaridir - bu yuzden gozenek-boyutu-
    secici bir gorev icin FIZIKSEL OLARAK en dogrudan motive edilen 11.
    model olarak eklenmistir.

    SIFIRDAN, SAF PYTORCH (TFN/EGNN/SE3_Transformer ile AYNI felsefe):
    orijinal DimeNet(++) TAM kuresel Bessel/spherical-harmonic taban
    fonksiyonlari + torch_sparse gerektirir; burada, graf_ozellik_ortak.
    uc_yakin_komsu_acisi'nin urettigi (k,j,i) ucluleri ustunde, RADYAL
    taban icin AYNI GaussianRBF (digerleriyle tutarlilik icin), ACISAL
    taban icin ise basit bir FOURIER (cos(k*theta), k=0..7) genisletmesi
    kullanilir - tam kuresel harmoniklerin HESAPCA HAFIF bir yaklastirmasi
    (TFN/SE3'un e3nn'siz l<=1 yaklastirmasiyla AYNI ruhta).

MESAJ ILETIMI (basitlestirilmis DimeNet++ etkilesim blogu):
    m_ji^0   = MLP(h_j, h_i, RBF(r_ji))                        <- ilk kenar gommesi
    HER katmanda:
        t_kji = MLP(m_kj, RBF(r_kj), FourierACI(theta_kji))    <- ucgen (triplet) mesaji
        a_ji  = sum_{k in N(j)\{i}} t_kji                       <- j'ye giren tum ucgenlerin toplami
        m_ji  = m_ji + MLP(m_ji, a_ji)                          <- reziduel kenar guncellemesi
    h_i      = sum_j m_ji                                       <- kenar mesajlari -> dugum
    graph_feat = scatter_mean(h_i, batch) -> EMB_DIM

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m DimeNetPP.run_dimenetpp
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
    sys.exit("HATA: kok dizinden calistirin: python -m DimeNetPP.run_dimenetpp")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "DimeNetPP"
HIDDEN = 64
N_LAYERS = 4
N_FOURIER = 8  # acisal taban icin Fourier bileseni sayisi (basitlestirilmis kuresel harmonik)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros(dim_size, *src.shape[1:])
    if index.numel() == 0:
        return out
    out.scatter_add_(0, index.view(-1, *([1] * (src.dim() - 1))).expand_as(src), src)
    return out


class FourierAngularBasis(nn.Module):
    """theta [rad] -> [cos(0*theta), cos(1*theta), ..., cos((K-1)*theta)] -
    kuresel harmoniklerin (l=0..K-1, m=0 kesiti) HESAPCA HAFIF bir
    yaklastirmasi; DimeNet'in orijinal spherical-Bessel tabaninin ayni
    ROLUNU (aci-farkinda, SUREKLI/turevlenebilir bir genisletme) oynar."""

    def __init__(self, n_fourier: int = N_FOURIER):
        super().__init__()
        self.register_buffer("k", torch.arange(n_fourier, dtype=torch.float))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.cos(theta.unsqueeze(-1) * self.k)  # [T, n_fourier]


class DimeNetPPInteraction(nn.Module):
    def __init__(self, hidden: int, n_rbf: int, n_fourier: int):
        super().__init__()
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden + n_rbf + n_fourier, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, m: torch.Tensor, rbf_edge: torch.Tensor,
                idx_kj: torch.Tensor, idx_ji: torch.Tensor, abf: torch.Tensor, n_edges: int) -> torch.Tensor:
        if idx_kj.numel() == 0:
            # ucgen (triplet) YOK (orn. atom sayisi <=1 ya da izole atomlar) -
            # aci-farkinda terim katkisi olmadan, sadece kenar mesaji reziduel kalir.
            a_ji = m.new_zeros(n_edges, m.size(-1))
        else:
            t_kji = self.triplet_mlp(torch.cat([m[idx_kj], rbf_edge[idx_kj], abf], dim=-1))
            a_ji = _scatter_sum(t_kji, idx_ji, n_edges)
        m_yeni = m + self.edge_update_mlp(torch.cat([m, a_ji], dim=-1))
        return self.norm(m_yeni)


class DimeNetPPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)
        self.abf = FourierAngularBasis(N_FOURIER)
        self.edge_init = nn.Sequential(
            nn.Linear(2 * HIDDEN + N_RBF, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN),
        )
        self.layers = nn.ModuleList([
            DimeNetPPInteraction(HIDDEN, N_RBF, N_FOURIER) for _ in range(N_LAYERS)
        ])
        self.out_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, EMB_DIM))

    def forward(self, g: dict) -> torch.Tensor:
        z, pos, edge_index, batch, n_graphs = g["z"], g["pos"], g["edge_index"], g["batch"], g["n_graphs"]
        idx_kj, idx_ji, theta = g["idx_kj"], g["idx_ji"], g["theta"]

        src, dst = edge_index
        r, _ = edge_geometri(pos, edge_index)
        rbf_edge = self.rbf(r)
        abf = self.abf(theta) if theta.numel() > 0 else theta.new_zeros(0, N_FOURIER)

        h = self.atom_emb(z)
        m = self.edge_init(torch.cat([h[src], h[dst], rbf_edge], dim=-1))  # [E, HIDDEN] - ilk kenar (yonlu) mesaji

        n_edges = edge_index.size(1)
        for layer in self.layers:
            m = layer(m, rbf_edge, idx_kj, idx_ji, abf, n_edges)

        h_node = _scatter_sum(m, dst, h.size(0))  # kenar mesajlarini hedef dugume topla
        graph_feat = scatter_mean_manual(h_node, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    pretrain_encoder(MODEL_ADI, lambda: DimeNetPPEncoder(), PRETRAIN_CKPT_DIR)

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS, "n_fourier": N_FOURIER})
    model_factory = model_factory_olustur(lambda: DimeNetPPEncoder())
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME)
