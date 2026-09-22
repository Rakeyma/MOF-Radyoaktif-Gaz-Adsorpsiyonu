"""
PNA_GNN/run_pna_gnn.py
=========================
AMAC:
    Corso et al. (2020) "Principal Neighbourhood Aggregation for Graph
    Nets" (PNA) - MOF Xe/Kr/I2 adsorpsiyon kapasitesi + Xe/Kr secicilik
    tahmini (Bilesen 5: on-egitim + ince-ayar).

    torch_geometric.nn.PNAConv resmi katmani kullanilir: TEK bir agregatör
    yerine BIRDEN FAZLA agregatör (mean, min, max, std) ile BIRDEN FAZLA
    olcekleyici (identity, amplification, attenuation) CARPILARAK
    birlestirilir. PNAConv, VERI SETININ derece (degree) histogramini
    (deg) ONCEDEN bilmeyi GEREKTIRIR.

    BU PROJEDE IKI AYRI VERI SETI (on-egitim/QMOF proxy + ince-ayar/gaz)
    kullanildigindan, deg histogrami HER ASAMA icin O ASAMANIN KENDI graf
    onbellegi uzerinden AYRI AYRI hesaplanir (PNAConv'un dahili
    olcekleyici sabitleri (avg_log_degree) sadece Python float'lardir,
    ogrenilebilir agirliklarin sekli/anlami DEGISMEZ - bu yuzden on-egitim
    -> ince-ayar encoder agirlik transferi, deg histogrami farkli olsa
    bile GECERLIDIR).

MIMARI:
    Z-gommesi(64) -> 4 x PNAConv(aggregators=[mean,min,max,std],
    scalers=[identity,amplification,attenuation], deg=<asamanin kendi
    derece histogrami>, edge_dim=N_RBF) + LayerNorm + reziduel ->
    scatter-mean(atom -> MOF) -> Linear -> EMB_DIM=64 -> [gomme | aux(15)]
    -> egitim_ortak 'RegressionHead' basi -> [Xe, Kr, Xe/Kr, I2]

CALISTIRMA (kok dizinden, ON-EGITIM + INCE-AYARI TEK KOMUTTA yapar):
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python -m PNA_GNN.run_pna_gnn
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

try:
    from paths import GAS_FINETUNE_CSV, QMOF_PRETRAIN_CSV, PRETRAINED_ENCODER_FILENAME
    from graf_ozellik_ortak import GaussianRBF, edge_geometri, scatter_mean_manual, NUM_ATOM_TYPES, N_RBF, EMB_DIM
    from egitim_ortak import (
        EgitimAyarlari, model_factory_olustur, finetune_k_fold_egit, pretrain_encoder,
        grafik_onbellek_olustur, TARGET_COLUMNS, GROUP_COLUMN, PROXY_TARGET_COLUMN,
    )
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python -m PNA_GNN.run_pna_gnn")

PROJE_DIZINI = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJE_DIZINI / "checkpoints"
SONUC_DIR = PROJE_DIZINI / "sonuclar"
PRETRAIN_CKPT_DIR = PROJE_DIZINI / "pretrain_checkpoints"

MODEL_ADI = "PNA_GNN"
HIDDEN = 64
N_LAYERS = 4
AGGREGATORS = ["mean", "min", "max", "std"]
SCALERS = ["identity", "amplification", "attenuation"]


def derece_histogrami_hesapla(cache: dict) -> torch.Tensor:
    """PNAConv'un olcekleyicileri (scalers) icin gereken, veri setindeki
    dugum-basina-giren-kenar-sayisi (in-degree) histogramini hesaplar -
    PyG'nin resmi PNA ornegiyle (examples/pna.py) AYNI yontem."""
    max_degree = 0
    for g in cache.values():
        if g["edge_index"].numel() == 0:
            continue
        d = torch.bincount(g["edge_index"][1], minlength=g["n_atoms"])
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for g in cache.values():
        if g["edge_index"].numel() == 0:
            d = torch.zeros(g["n_atoms"], dtype=torch.long)
        else:
            d = torch.bincount(g["edge_index"][1], minlength=g["n_atoms"])
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


class PNAEncoder(nn.Module):
    def __init__(self, deg: torch.Tensor):
        super().__init__()
        from torch_geometric.nn import PNAConv

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES + 1, HIDDEN, padding_idx=0)
        self.rbf = GaussianRBF(N_RBF)

        self.convs = nn.ModuleList([
            PNAConv(HIDDEN, HIDDEN, aggregators=AGGREGATORS, scalers=SCALERS,
                     deg=deg, edge_dim=N_RBF, towers=4, pre_layers=1, post_layers=1)
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
            h = h + torch.nn.functional.relu(norm(conv(h, edge_index, edge_attr)))

        graph_feat = scatter_mean_manual(h, batch, n_graphs)
        return self.out_proj(graph_feat)


if __name__ == "__main__":
    # --- ON-EGITIM: QMOF proxy veri setinin KENDI derece histogramiyla ---
    if QMOF_PRETRAIN_CSV.exists():
        pre_df = pd.read_csv(QMOF_PRETRAIN_CSV, low_memory=False)
        pre_df = pre_df[pre_df["eslesme_durumu"] == "TAM"].dropna(subset=[PROXY_TARGET_COLUMN]).reset_index(drop=True)
        print(f"[{MODEL_ADI}] [ON-EGITIM] PNAConv icin derece histogrami hazirlaniyor...")
        pre_cache = grafik_onbellek_olustur(pre_df)
        pre_deg = derece_histogrami_hesapla(pre_cache)
        print(f"[{MODEL_ADI}] [ON-EGITIM] Derece histogrami hazir (maks derece={pre_deg.numel() - 1}).")
        pretrain_encoder(MODEL_ADI, lambda: PNAEncoder(pre_deg), PRETRAIN_CKPT_DIR, df=pre_df, cache=pre_cache)
    else:
        print(f"[{MODEL_ADI}] [ON-EGITIM ATLANDI] '{QMOF_PRETRAIN_CSV}' bulunamadi.")

    # --- INCE-AYAR: gaz-adsorpsiyon veri setinin KENDI derece histogramiyla ---
    if not GAS_FINETUNE_CSV.exists():
        sys.exit(f"HATA: '{GAS_FINETUNE_CSV}' bulunamadi. Once boru hatti (1-5) calistirilmali.")
    df = pd.read_csv(GAS_FINETUNE_CSV, low_memory=False)
    df = df[df["eslesme_durumu"] == "TAM"].dropna(subset=[GROUP_COLUMN]).reset_index(drop=True)
    df = df[df[TARGET_COLUMNS].notna().any(axis=1)].reset_index(drop=True)

    print(f"[{MODEL_ADI}] [INCE-AYAR] PNAConv icin derece histogrami hazirlaniyor (graf onbellegi kuruluyor)...")
    cache = grafik_onbellek_olustur(df)
    deg = derece_histogrami_hesapla(cache)
    print(f"[{MODEL_ADI}] [INCE-AYAR] Derece histogrami hazir (maks derece={deg.numel() - 1}).")

    ayarlar = EgitimAyarlari(model_adi=MODEL_ADI, ekstra_bilgi={"n_layers": N_LAYERS, "max_degree": deg.numel() - 1})
    model_factory = model_factory_olustur(lambda: PNAEncoder(deg))
    finetune_k_fold_egit(ayarlar, model_factory, CHECKPOINT_DIR, SONUC_DIR,
                          pretrained_encoder_path=PRETRAIN_CKPT_DIR / PRETRAINED_ENCODER_FILENAME,
                          df=df, cache=cache)
