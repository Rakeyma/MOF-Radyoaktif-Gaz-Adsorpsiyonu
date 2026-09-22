"""
egitim_ortak.py
=================
AMAC:
    "Üç Boyutlu Kristal Malzemeler/egitim_ortak.py" icindeki egitim/
    degerlendirme/K-fold/permutation-importance mantigini TEK bir PAYLASILAN
    modulde toplamak, boylece 11 model klasorunun HER BIRI (GraphGPS,
    PNA_GNN, GIN, GAT, GatedGCN, DeeperGCN, ECC, TFN, EGNN, SE3_Transformer,
    DimeNetPP) SADECE KENDI ENCODER MIMARISINI (algoritmayi) tanimlar;
    egitim dongusu, K-fold bolme, erken durdurma, checkpoint kaydi, metrik
    hesabi, TRANSFER LEARNING ve permutation importance MANTIGI HERKES ICIN
    AYNIDIR ve BURADA TEK SEFER dogru yazilir.

BU PROJENIN "Üç Boyutlu Kristal Malzemeler"DEN IKI TEMEL FARKI:

    (1) COK-HEDEFLI (multi-target) REGRESYON: hedef artik TEK bir skaler
        (formation energy) DEGIL, DORT radyoaktif-gaz buyuklugudur
        (TARGET_COLUMNS, asagida). Literatur-madencilikli (NLP) etiketler
        SEYREK olabilir (bir MOF icin sadece Xe kapasitesi bilinip Kr/I2
        bilinmeyebilir) - bu yuzden kayip fonksiyonu MASKELI (NaN hedefler
        o ornek/hedef icin kayba KATKIDA BULUNMAZ, bkz. _maskeli_mse).

    (2) TRANSFER LEARNING (Bilesen 5): iki asamali egitim saglanir:
        (A) pretrain_encoder(): encoder'i DEVASA (~100k) proxy-hedefli
            (formation energy benzeri) QMOF_PRETRAIN_CSV uzerinde tek-
            hedefli regresyonla ON-EGITIR, SADECE encoder agirliklarini
            (PRETRAINED_ENCODER_FILENAME) diske kaydeder (regresyon basi
            atilir - hedef gorev farkli).
        (B) finetune_k_fold_egit(): asil K-fold dongusu; her fold'un
            basinda YENI bir encoder ornegi olusturulup (varsa) on-egitimli
            agirliklarla baslatilir, ilk FREEZE_ENCODER_EPOCHS epoch
            boyunca encoder DONDURULUR (sadece regresyon basi/head
            egitilir - klasik 'linear probing then fine-tuning' semasi),
            sonra TUM ag birlikte ince-ayar (fine-tune) edilir.

ORTAK ARAYUZ (TUM 11 model bu sozlesmeye uymalidir - "Üç Boyutlu Kristal
Malzemeler"DEKI POZISYONEL arayuzden FARKLI, SOZLUK-tabanli, cunku
DimeNetPP ekstra (idx_kj/idx_ji/theta) alanlara ihtiyac duyar):
    encoder.forward(g: dict[str, Tensor]) -> Tensor[n_graphs, EMB_DIM]
        g icerir: z, pos, edge_index, batch, n_graphs (+ DimeNetPP icin
        idx_kj, idx_ji, theta - digerleri bu alanlari YOK SAYAR).
    RegressionHead(encoder, n_outputs) bu gommeyi AUX_DIM boyutundaki
    gozeneklilik/kompozisyon ozellik vektoruyle birlestirip 3 katmanli bir
    MLP-bas (head) ile n_outputs hedefi (pretrain: 1, finetune: 4) tahmin
    eder.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

try:
    from paths import GAS_FINETUNE_CSV, QMOF_PRETRAIN_CSV, PRETRAINED_ENCODER_FILENAME
    from ortak_ozellikler import (
        AUX_FEATURE_COLUMNS, AUX_DIM, AUX_GRUPLARI, AUX_GRUP_IDX,
        aux_ham_matris, AuxOlcekleyici,
    )
    from graf_ozellik_ortak import cif_den_3b_graf, batch_graphs, CUTOFF, EMB_DIM
except ImportError:
    sys.exit(
        "HATA: 'paths'/'ortak_ozellikler'/'graf_ozellik_ortak' bulunamadi. "
        "Kok dizininden calistirin:\n"
        "    cd \"MOF Radyoaktif Gaz Adsorpsiyonu\" && python -m GIN.run_gin"
    )

# --------------------------------------------------------------------------
# HEDEF DEGISKENLER (Bilesen 5 - INCE-AYAR/finetune asamasi)
#   xe_uptake_mmol_g    : Ksenon adsorpsiyon kapasitesi (mmol/g, 1 bar/298K referans)
#   kr_uptake_mmol_g    : Kripton adsorpsiyon kapasitesi (mmol/g, ayni kosullar)
#   xe_kr_selectivity   : IAST/Henry-orani turevi Xe/Kr secicilik (boyutsuz)
#   i2_uptake_mmol_g    : Iyot (I2) adsorpsiyon kapasitesi (mmol/g) - nukleer
#                         atik gazi arıtma icin (kemisorpsiyon agirlikli)
# --------------------------------------------------------------------------
TARGET_COLUMNS = ["xe_uptake_mmol_g", "kr_uptake_mmol_g", "xe_kr_selectivity", "i2_uptake_mmol_g"]
TARGET_UNITS = {
    "xe_uptake_mmol_g": "mmol/g", "kr_uptake_mmol_g": "mmol/g",
    "xe_kr_selectivity": "(-)", "i2_uptake_mmol_g": "mmol/g",
}
N_TARGETS = len(TARGET_COLUMNS)

# ON-EGITIM (pretrain) proxy hedefi - QMOF-benzeri devasa veri setinden,
# formation energy (DFT olusum enerjisi, eV/atom) - GNN'in "hangi 3B atomik
# duzenlerin enerjisel olarak nasil davrandigini" ogrenmesini saglayan,
# ETIKET AZLIGI OLMAYAN (label-rich) bir vekil (proxy) gorev.
PROXY_TARGET_COLUMN = "formation_energy_eV_atom_proxy"

STRUCT_PATH_COLUMN = "graf_cif_path"
GROUP_COLUMN = "base_mof_id"   # ayni temel MOF'tan turetilen (augment edilmis) tum ornekler AYNI fold'da kalmalidir
ID_COLUMN = "sample_id"
NAME_COLUMN = "mof_name"

# --------------------------------------------------------------------------
# ORTAK VARSAYILAN HIPERPARAMETRELER (11 model arasi adil karsilastirma icin
# egitim protokolu SABIT tutulur - SADECE encoder mimarisi degisir)
# --------------------------------------------------------------------------
SEED = 42
K_FOLDS = 5
BATCH_SIZE = 32
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 12
LR = 1e-3
WEIGHT_DECAY = 1e-5
LR_ETA_MIN = 1e-6
HIDDEN_DIM = 128
DROPOUT = 0.1

# Transfer learning (Bilesen 5) varsayilanlari
FREEZE_ENCODER_EPOCHS = 5     # finetune'un ILK N epoch'unda encoder DONDURULUR (sadece head egitilir)
PRETRAIN_MAX_EPOCHS = 40
PRETRAIN_PATIENCE = 8
PRETRAIN_VAL_FRAC = 0.1

# Ortam degiskeni override'lari - kucuk/hizli bir dogrulama (smoke-test)
# kosusu icin PRODUCTION varsayilanlarini DEGISTIRMEDEN gecici olarak
# kucultmeye yarar. Hicbiri tanimli degilse davranis TAMAMEN ayni kalir.
K_FOLDS = int(os.environ.get("KFOLD_OVERRIDE", K_FOLDS))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS_OVERRIDE", MAX_EPOCHS))
EARLY_STOP_PATIENCE = int(os.environ.get("PATIENCE_OVERRIDE", EARLY_STOP_PATIENCE))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE_OVERRIDE", BATCH_SIZE))
FREEZE_ENCODER_EPOCHS = int(os.environ.get("FREEZE_ENCODER_EPOCHS_OVERRIDE", FREEZE_ENCODER_EPOCHS))
PRETRAIN_MAX_EPOCHS = int(os.environ.get("PRETRAIN_MAX_EPOCHS_OVERRIDE", PRETRAIN_MAX_EPOCHS))
PRETRAIN_PATIENCE = int(os.environ.get("PRETRAIN_PATIENCE_OVERRIDE", PRETRAIN_PATIENCE))
USE_PRETRAINED = os.environ.get("USE_PRETRAINED_OVERRIDE", "1").strip() != "0"


@dataclass
class EgitimAyarlari:
    model_adi: str
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    k_folds: int = K_FOLDS
    seed: int = SEED
    cutoff: float = CUTOFF
    hidden_dim: int = HIDDEN_DIM
    dropout: float = DROPOUT
    freeze_encoder_epochs: int = FREEZE_ENCODER_EPOCHS
    use_pretrained: bool = USE_PRETRAINED
    ekstra_bilgi: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1) REGRESYON BASI (head) - TUM modeller icin ORTAK, sadece encoder degisir.
#    n_outputs=1 -> ON-EGITIM (pretrain, proxy hedef)
#    n_outputs=N_TARGETS -> INCE-AYAR (finetune, gaz adsorpsiyon hedefleri)
# ---------------------------------------------------------------------------
class RegressionHead(nn.Module):
    def __init__(self, encoder: nn.Module, n_outputs: int,
                 hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = encoder
        self.n_outputs = n_outputs
        giris = EMB_DIM + AUX_DIM
        self.head = nn.Sequential(
            nn.Linear(giris, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_outputs),
        )

    def _kristal_gommesi(self, g_batch: dict, device: str) -> torch.Tensor:
        g_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g_batch.items()}
        return self.encoder(g_dev)

    def forward(self, g_batch: dict, aux: torch.Tensor, device: str) -> torch.Tensor:
        fea = self._kristal_gommesi(g_batch, device)
        x = torch.cat([fea, aux.to(device)], dim=-1)
        return self.head(x)  # [B, n_outputs] (n_outputs=1 icin de son eksen KORUNUR, squeeze cagiran taraftadir)

    def set_encoder_egitilebilir(self, egitilebilir: bool) -> None:
        """Transfer learning 'freeze/unfreeze' - Bilesen 5, bkz. fold_calistir."""
        for p in self.encoder.parameters():
            p.requires_grad_(egitilebilir)


EncoderFactory = Callable[[], nn.Module]


def model_factory_olustur(encoder_factory: EncoderFactory, n_outputs: int = N_TARGETS,
                           hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT) -> Callable[[], RegressionHead]:
    """encoder_factory() -> nn.Module dondurur; bunu RegressionHead icine
    saran bir model_factory() dondurur (her fold icin SIFIRDAN, egitilmemis
    bir model ornegi uretmek icin)."""
    def _factory():
        return RegressionHead(encoder_factory(), n_outputs=n_outputs, hidden_dim=hidden_dim, dropout=dropout)
    return _factory


# ---------------------------------------------------------------------------
# 2) VERI SETI + COLLATE (COK-HEDEFLI, NaN-toleransli)
# ---------------------------------------------------------------------------
class MOFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache: dict, aux_matris: np.ndarray,
                 hedef_ort: np.ndarray, hedef_std: np.ndarray,
                 hedef_kolonlari: list[str], yol_kolon: str = STRUCT_PATH_COLUMN):
        self.df = df.reset_index(drop=True)
        self.cache = cache
        self.aux_matris = aux_matris
        self.hedef_ort = hedef_ort
        self.hedef_std = hedef_std
        self.hedef_kolonlari = hedef_kolonlari
        self.yol_kolon = yol_kolon
        self.hedef_ham = df[hedef_kolonlari].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        hedef_std = (self.hedef_ham[i] - self.hedef_ort) / self.hedef_std  # NaN kalir (maskeli kayipta elenir)
        return self.cache[row[self.yol_kolon]], self.aux_matris[i], hedef_std


def collate_fn(batch):
    g_list, auxlar, hedefler = zip(*batch)
    return (
        batch_graphs(list(g_list)),
        torch.tensor(np.stack(auxlar), dtype=torch.float),
        torch.tensor(np.stack(hedefler), dtype=torch.float),  # [B, n_hedef], NaN icerebilir
    )


def grafik_onbellek_olustur(df: pd.DataFrame, cutoff: float = CUTOFF,
                             yol_kolon: str = STRUCT_PATH_COLUMN) -> dict:
    yollar = sorted(set(df[yol_kolon]))
    cache = {}
    for i, yol in enumerate(yollar, 1):
        cache[yol] = cif_den_3b_graf(yol, cutoff=cutoff)
        if i % 2000 == 0 or i == len(yollar):
            print(f"    graf onbellegi: {i}/{len(yollar)}")
    return cache


# ---------------------------------------------------------------------------
# 3) MASKELI (NaN-toleransli) COK-HEDEFLI KAYIP
# ---------------------------------------------------------------------------
def maskeli_mse(tahmin: torch.Tensor, hedef: torch.Tensor) -> torch.Tensor:
    """hedef [B,K] NaN icerebilir (seyrek etiketler - orn. bir MOF icin I2
    kapasitesi literaturde yok). HER hedef sutunu AYRI AYRI (sadece
    gecerli/non-NaN ornekler uzerinden) ortalanir, sonra K hedefin
    ortalamasi alinir - boylece bir hedefin ornek SAYISI (etiket yogunlugu)
    digerinin kayip olcegini BASKILAMAZ (adil coklu-gorev dengesi)."""
    mask = ~torch.isnan(hedef)
    hedef_dolu = torch.nan_to_num(hedef, nan=0.0)
    se = (tahmin - hedef_dolu).pow(2) * mask
    per_target_n = mask.sum(dim=0).clamp(min=1)
    per_target_loss = se.sum(dim=0) / per_target_n
    gecerli_hedef = mask.any(dim=0)
    if gecerli_hedef.sum() == 0:
        return tahmin.sum() * 0.0
    return per_target_loss[gecerli_hedef].mean()


# ---------------------------------------------------------------------------
# 4) EGITIM / DEGERLENDIRME
# ---------------------------------------------------------------------------
def epoch_calistir(model, loader, optimizer, device, egitim: bool):
    model.train() if egitim else model.eval()
    toplam_kayip, n = 0.0, 0
    tum_g, tum_t = [], []

    with torch.set_grad_enabled(egitim):
        for g_b, aux, hedef in loader:
            tahmin = model(g_b, aux, device)
            hedef = hedef.to(device)
            kayip = maskeli_mse(tahmin, hedef)
            if egitim:
                optimizer.zero_grad()
                kayip.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            bn = hedef.size(0)
            toplam_kayip += kayip.item() * bn
            n += bn
            tum_g.append(hedef.detach().cpu().numpy())
            tum_t.append(tahmin.detach().cpu().numpy())

    return toplam_kayip / max(n, 1), np.concatenate(tum_g), np.concatenate(tum_t)


def tahminleri_topla(model, loader, device, hedef_ort, hedef_std):
    model.eval()
    gercekler, tahminler = [], []
    with torch.no_grad():
        for g_b, aux, hedef in loader:
            t = model(g_b, aux, device).cpu().numpy()
            gercekler.append(hedef.numpy() * hedef_std + hedef_ort)
            tahminler.append(t * hedef_std + hedef_ort)
    return np.concatenate(gercekler), np.concatenate(tahminler)


def metrik_hesapla_tekli(gercek: np.ndarray, tahmin: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    gecerli = ~np.isnan(gercek)
    if gecerli.sum() < 2:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "n_ornek": int(gecerli.sum())}
    g, t = gercek[gecerli], tahmin[gecerli]
    return {
        "MAE": float(mean_absolute_error(g, t)),
        "RMSE": float(np.sqrt(mean_squared_error(g, t))),
        "R2": float(r2_score(g, t)) if gecerli.sum() >= 2 and np.std(g) > 1e-10 else float("nan"),
        "n_ornek": int(gecerli.sum()),
    }


def metrik_hesapla_coklu(gercek: np.ndarray, tahmin: np.ndarray, hedef_kolonlari: list[str]) -> dict:
    """[N,K] gercek/tahmin matrislerinden HER hedef icin ayri metrik +
    tum-hedeflerin ortalamasi ('overall') dondurur."""
    sonuc = {}
    for k, ad in enumerate(hedef_kolonlari):
        sonuc[ad] = metrik_hesapla_tekli(gercek[:, k], tahmin[:, k])
    gecerli_r2 = [v["R2"] for v in sonuc.values() if np.isfinite(v["R2"])]
    gecerli_mae = [v["MAE"] for v in sonuc.values() if np.isfinite(v["MAE"])]
    sonuc["overall"] = {
        "MAE": float(np.mean(gecerli_mae)) if gecerli_mae else float("nan"),
        "R2": float(np.mean(gecerli_r2)) if gecerli_r2 else float("nan"),
        "n_ornek": int(gercek.shape[0]),
    }
    return sonuc


def _hedef_istatistik(df_train: pd.DataFrame, hedef_kolonlari: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ham = df_train[hedef_kolonlari].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    with np.errstate(invalid="ignore"):
        ort = np.nanmean(ham, axis=0)
        std = np.nanstd(ham, axis=0)
    ort[~np.isfinite(ort)] = 0.0
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return ort, std


# ---------------------------------------------------------------------------
# 5) TEK FOLD (Bilesen 5: on-egitimli encoder yukleme + dondur/coz semasi)
# ---------------------------------------------------------------------------
def fold_calistir(fold_no: int, ayarlar: EgitimAyarlari, model_factory,
                   df: pd.DataFrame, train_grp: set, test_grp: set,
                   cache: dict, device: str, checkpoint_dir: Path, sonuc_dir: Path,
                   pretrained_encoder_path: Path | None = None) -> dict:
    from sklearn.model_selection import GroupShuffleSplit

    print(f"\n{'=' * 70}\n{ayarlar.model_adi} — FOLD {fold_no}/{ayarlar.k_folds}\n{'=' * 70}")
    df_test = df[df[GROUP_COLUMN].isin(test_grp)].reset_index(drop=True)
    df_tv = df[df[GROUP_COLUMN].isin(train_grp)].reset_index(drop=True)
    gss = GroupShuffleSplit(1, test_size=0.15, random_state=ayarlar.seed + fold_no)
    ti, vi = next(gss.split(df_tv, groups=df_tv[GROUP_COLUMN]))
    df_train = df_tv.iloc[ti].reset_index(drop=True)
    df_val = df_tv.iloc[vi].reset_index(drop=True)

    print(f"  Train: {len(df_train)} satir, {df_train[GROUP_COLUMN].nunique()} temel MOF")
    print(f"  Val  : {len(df_val)} satir, {df_val[GROUP_COLUMN].nunique()} temel MOF")
    print(f"  Test : {len(df_test)} satir, {df_test[GROUP_COLUMN].nunique()} temel MOF")

    hedef_ort, hedef_std = _hedef_istatistik(df_train, TARGET_COLUMNS)

    aux_olcekleyici = AuxOlcekleyici().fit(aux_ham_matris(df_train))
    aux_train = aux_olcekleyici.transform(aux_ham_matris(df_train))
    aux_val = aux_olcekleyici.transform(aux_ham_matris(df_val))
    aux_test = aux_olcekleyici.transform(aux_ham_matris(df_test))

    def mk(df_, aux_, shuffle):
        ds = MOFDataset(df_, cache, aux_, hedef_ort, hedef_std, TARGET_COLUMNS)
        return DataLoader(ds, ayarlar.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    train_loader = mk(df_train, aux_train, True)
    val_loader = mk(df_val, aux_val, False)
    test_loader = mk(df_test, aux_test, False)

    model = model_factory().to(device)

    # --- BILESEN 5 / TRANSFER LEARNING: on-egitimli encoder agirliklarini yukle ---
    on_egitimli_yuklendi = False
    if ayarlar.use_pretrained and pretrained_encoder_path is not None and pretrained_encoder_path.exists():
        enc_state = torch.load(pretrained_encoder_path, map_location=device, weights_only=False)
        try:
            model.encoder.load_state_dict(enc_state)
            on_egitimli_yuklendi = True
            print(f"  [Transfer Learning] On-egitimli encoder yuklendi: {pretrained_encoder_path.name}")
        except Exception as e:
            print(f"  [Transfer Learning] UYARI: on-egitimli agirliklar yuklenemedi ({e}); rastgele baslatiliyor.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=ayarlar.lr, weight_decay=ayarlar.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, ayarlar.max_epochs, LR_ETA_MIN)

    if fold_no == 1:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  Toplam parametre: {n_p:,}")

    ckpt = checkpoint_dir / f"fold{fold_no}_best_model.pt"
    en_iyi_val_mae = float("inf")
    sabir = 0
    gecmis = []

    for epoch in range(1, ayarlar.max_epochs + 1):
        # --- BILESEN 5 / dondur-sonra-coz (freeze-then-unfreeze) semasi ---
        if on_egitimli_yuklendi and ayarlar.freeze_encoder_epochs > 0:
            model.set_encoder_egitilebilir(epoch > ayarlar.freeze_encoder_epochs)

        t0 = time.time()
        tl, tg, tt = epoch_calistir(model, train_loader, optimizer, device, True)
        vl, vg, vt = epoch_calistir(model, val_loader, optimizer, device, False)
        scheduler.step()
        sure = time.time() - t0

        tm = metrik_hesapla_coklu(tg * hedef_std + hedef_ort, tt * hedef_std + hedef_ort, TARGET_COLUMNS)
        vm = metrik_hesapla_coklu(vg * hedef_std + hedef_ort, vt * hedef_std + hedef_ort, TARGET_COLUMNS)
        gecmis.append({
            "epoch": epoch, "train_loss": tl, "val_loss": vl,
            "train_mae": tm["overall"]["MAE"], "val_mae": vm["overall"]["MAE"],
            "train_r2": tm["overall"]["R2"], "val_r2": vm["overall"]["R2"],
            "lr": optimizer.param_groups[0]["lr"], "sure_sn": sure,
            "encoder_donuk_mu": on_egitimli_yuklendi and epoch <= ayarlar.freeze_encoder_epochs,
        })
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}/{ayarlar.max_epochs} | train_loss={tl:.4f} val_loss={vl:.4f} | "
                f"train_MAE={tm['overall']['MAE']:.4f} val_MAE={vm['overall']['MAE']:.4f} | "
                f"train_R2={tm['overall']['R2']:.3f} val_R2={vm['overall']['R2']:.3f} | "
                f"LR={optimizer.param_groups[0]['lr']:.2e} | {sure:.1f}s"
            )

        val_mae_karsilastirma = vm["overall"]["MAE"] if np.isfinite(vm["overall"]["MAE"]) else float("inf")
        if val_mae_karsilastirma < en_iyi_val_mae:
            en_iyi_val_mae = val_mae_karsilastirma
            sabir = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_mae": val_mae_karsilastirma,
                "hedef_ort": hedef_ort, "hedef_std": hedef_std,
                "aux_ortalama": aux_olcekleyici.ortalama, "aux_std": aux_olcekleyici.std,
            }, ckpt)
        else:
            sabir += 1
            if sabir >= ayarlar.early_stop_patience:
                print(f"  Erken durdurma (epoch {epoch}, patience={ayarlar.early_stop_patience}).")
                break

    pd.DataFrame(gecmis).to_csv(sonuc_dir / f"egitim_gecmisi_fold{fold_no}.csv", index=False)

    best = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    print(f"  En iyi epoch: {best['epoch']} (val_MAE={en_iyi_val_mae:.4f})")

    fold_m, test_sonuc = {}, None
    for ad, ld, sub_df in [("train", train_loader, df_train),
                            ("val", val_loader, df_val),
                            ("test", test_loader, df_test)]:
        g, t = tahminleri_topla(model, ld, device, hedef_ort, hedef_std)
        m = metrik_hesapla_coklu(g, t, TARGET_COLUMNS)
        fold_m[ad] = m
        print(f"  {ad:5s} -> MAE={m['overall']['MAE']:.4f}  R2={m['overall']['R2']:.3f}  (n={m['overall']['n_ornek']})")
        if ad == "test":
            test_sonuc = pd.DataFrame({
                "fold": fold_no, ID_COLUMN: sub_df[ID_COLUMN].values,
                GROUP_COLUMN: sub_df[GROUP_COLUMN].values,
                NAME_COLUMN: sub_df[NAME_COLUMN].values if NAME_COLUMN in sub_df.columns else sub_df[GROUP_COLUMN].values,
            })
            for k, kol in enumerate(TARGET_COLUMNS):
                test_sonuc[f"gercek_{kol}"] = g[:, k]
                test_sonuc[f"tahmin_{kol}"] = t[:, k]
            for c in AUX_FEATURE_COLUMNS:
                if c in sub_df.columns and c not in test_sonuc.columns:
                    test_sonuc[c] = sub_df[c].values

    return {
        "fold_no": fold_no, "en_iyi_epoch": best["epoch"],
        "n_grup_train": int(df_train[GROUP_COLUMN].nunique()),
        "n_grup_val": int(df_val[GROUP_COLUMN].nunique()),
        "n_grup_test": int(df_test[GROUP_COLUMN].nunique()),
        "on_egitimli_kullanildi": on_egitimli_yuklendi,
        "metrikler": fold_m, "test_tahminleri": test_sonuc,
    }


# ---------------------------------------------------------------------------
# 6) TAM K-FOLD DONGUSU (INCE-AYAR / finetune - her modelin __main__
#    blogundan cagirdigi ANA fonksiyon)
# ---------------------------------------------------------------------------
def finetune_k_fold_egit(ayarlar: EgitimAyarlari, model_factory, checkpoint_dir: Path, sonuc_dir: Path,
                          pretrained_encoder_path: Path | None = None,
                          df: pd.DataFrame | None = None, cache: dict | None = None) -> pd.DataFrame:
    from sklearn.model_selection import KFold

    torch.manual_seed(ayarlar.seed)
    np.random.seed(ayarlar.seed)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sonuc_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        if not GAS_FINETUNE_CSV.exists():
            sys.exit(f"HATA: '{GAS_FINETUNE_CSV}' bulunamadi. Once boru hatti (1-5) calistirilmali.")
        df = pd.read_csv(GAS_FINETUNE_CSV, low_memory=False)
        df = df[df["eslesme_durumu"] == "TAM"].reset_index(drop=True)

    onceki_n = len(df)
    df = df.dropna(subset=[GROUP_COLUMN]).reset_index(drop=True)
    df = df[df[TARGET_COLUMNS].notna().any(axis=1)].reset_index(drop=True)  # en az 1 hedefi olan satirlar
    if len(df) < onceki_n:
        print(f"[{ayarlar.model_adi}] UYARI: {onceki_n - len(df)} satir grup/hedef eksikligi nedeniyle elendi.")
    print(f"[{ayarlar.model_adi}] Veri: {len(df)} satir, {df[GROUP_COLUMN].nunique()} temel MOF")
    for kol in TARGET_COLUMNS:
        print(f"    {kol}: {df[kol].notna().sum()} etiketli ornek")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{ayarlar.model_adi}] Cihaz: {device}")

    if cache is None:
        print(f"[{ayarlar.model_adi}] CIF -> 3B graf onbellegi hazirlaniyor (cutoff={ayarlar.cutoff}A)...")
        cache = grafik_onbellek_olustur(df, ayarlar.cutoff)
        atom_sayilari = [v["n_atoms"] for v in cache.values()]
        print(f"  {len(cache)} yapi (ort={np.mean(atom_sayilari):.1f}, maks={max(atom_sayilari)} atom)")

    uniq_grp = np.array(sorted(df[GROUP_COLUMN].unique()))
    print(f"[{ayarlar.model_adi}] Benzersiz temel-MOF sayisi: {len(uniq_grp)}")
    k_eff = min(ayarlar.k_folds, len(uniq_grp)) if len(uniq_grp) >= 2 else 1
    kf = KFold(k_eff, shuffle=True, random_state=ayarlar.seed)

    fold_sonuclari = []
    for fold_no, (tr_pos, te_pos) in enumerate(kf.split(uniq_grp), 1):
        s = fold_calistir(fold_no, ayarlar, model_factory, df,
                           set(uniq_grp[tr_pos]), set(uniq_grp[te_pos]),
                           cache, device, checkpoint_dir, sonuc_dir,
                           pretrained_encoder_path=pretrained_encoder_path)
        fold_sonuclari.append(s)

    rows = []
    for s in fold_sonuclari:
        row = {"fold": s["fold_no"], "en_iyi_epoch": s["en_iyi_epoch"],
               "n_grup_train": s["n_grup_train"], "n_grup_val": s["n_grup_val"], "n_grup_test": s["n_grup_test"],
               "on_egitimli_kullanildi": s["on_egitimli_kullanildi"]}
        for ad in ("train", "val", "test"):
            row[f"{ad}_overall_MAE"] = s["metrikler"][ad]["overall"]["MAE"]
            row[f"{ad}_overall_R2"] = s["metrikler"][ad]["overall"]["R2"]
            for kol in TARGET_COLUMNS:
                row[f"{ad}_{kol}_MAE"] = s["metrikler"][ad][kol]["MAE"]
                row[f"{ad}_{kol}_R2"] = s["metrikler"][ad][kol]["R2"]
        rows.append(row)
    kdf = pd.DataFrame(rows)
    kdf.to_csv(sonuc_dir / "kfold_metrikleri.csv", index=False)

    oof_df = pd.concat([s["test_tahminleri"] for s in fold_sonuclari], ignore_index=True)
    oof_df.to_csv(sonuc_dir / "test_tahminleri_oof.csv", index=False)
    oof_g = oof_df[[f"gercek_{k}" for k in TARGET_COLUMNS]].values
    oof_t = oof_df[[f"tahmin_{k}" for k in TARGET_COLUMNS]].values
    oof_m = metrik_hesapla_coklu(oof_g, oof_t, TARGET_COLUMNS)

    print("\n" + "=" * 70)
    print(f"{ayarlar.model_adi} — K-FOLD ({k_eff}) OZET")
    print("=" * 70)
    print(kdf.to_string(index=False))
    print(f"\nPooled OOF ({len(oof_df)} satir), hedef bazinda:")
    for kol in TARGET_COLUMNS:
        m = oof_m[kol]
        print(f"  {kol:20s} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.3f} (n={m['n_ornek']})")
    print(f"  {'OVERALL':20s} MAE={oof_m['overall']['MAE']:.4f} R2={oof_m['overall']['R2']:.3f}")

    meta = {
        "model_adi": ayarlar.model_adi, "asama": "finetune",
        "girdi": str(GAS_FINETUNE_CSV),
        "k_folds": k_eff, "n_grup_toplam": int(len(uniq_grp)),
        "n_ornek_toplam": int(df[ID_COLUMN].nunique()),
        "on_egitimli_kullanildi": bool(any(s["on_egitimli_kullanildi"] for s in fold_sonuclari)),
        "pooled_out_of_fold_metrikleri": oof_m,
        "hiperparametreler": {
            "cutoff": ayarlar.cutoff, "batch_size": ayarlar.batch_size,
            "max_epochs": ayarlar.max_epochs, "early_stop_patience": ayarlar.early_stop_patience,
            "lr": ayarlar.lr, "weight_decay": ayarlar.weight_decay, "aux_dim": AUX_DIM,
            "hidden_dim": ayarlar.hidden_dim, "dropout": ayarlar.dropout, "seed": ayarlar.seed,
            "emb_dim": EMB_DIM, "freeze_encoder_epochs": ayarlar.freeze_encoder_epochs,
            "target_columns": TARGET_COLUMNS, **ayarlar.ekstra_bilgi,
        },
    }
    (sonuc_dir / "metrikler.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOOF CSV   -> {(sonuc_dir / 'test_tahminleri_oof.csv').resolve()}")
    print(f"Metrikler -> {(sonuc_dir / 'metrikler.json').resolve()}")
    print(f"\n[{ayarlar.model_adi}] Ince-ayar (finetune) tamamlandi.")
    return kdf


# alias - eski adla (k_fold_egit) cagiran/beklenen kodlarla uyum icin
k_fold_egit = finetune_k_fold_egit


# ---------------------------------------------------------------------------
# 7) BILESEN 5 / ASAMA A: ON-EGITIM (pretraining) - devasa proxy veri seti
#    uzerinde SADECE encoder agirliklarini ogrenip diske kaydeder.
# ---------------------------------------------------------------------------
def pretrain_encoder(model_adi: str, encoder_factory: EncoderFactory, pretrain_ckpt_dir: Path,
                      df: pd.DataFrame | None = None, cache: dict | None = None,
                      max_epochs: int = PRETRAIN_MAX_EPOCHS, patience: int = PRETRAIN_PATIENCE,
                      batch_size: int = BATCH_SIZE, lr: float = LR, seed: int = SEED) -> Path:
    """QMOF_PRETRAIN_CSV (~100k, PROXY_TARGET_COLUMN = formation energy
    benzeri DFT hedefi) uzerinde encoder'i tek-hedefli regresyonla ON-EGITIR.
    K-fold YAPILMAZ (bu bir genellenebilir TEMSIL ogrenme asamasidir, nihai
    degerlendirme DEGIL) - basit tek train/val bolmesi + erken durdurma
    yeterlidir. SADECE encoder.state_dict() kaydedilir (regresyon basi
    (head) ATILIR - finetune asamasinda YENIDEN, gaz-hedeflerine ozel
    olarak, sifirdan olusturulur)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pretrain_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = pretrain_ckpt_dir / PRETRAINED_ENCODER_FILENAME

    if df is None:
        if not QMOF_PRETRAIN_CSV.exists():
            print(f"[{model_adi}] [ON-EGITIM ATLANDI] '{QMOF_PRETRAIN_CSV}' bulunamadi.")
            return out_path
        df = pd.read_csv(QMOF_PRETRAIN_CSV, low_memory=False)
        df = df[df["eslesme_durumu"] == "TAM"].reset_index(drop=True)

    df = df.dropna(subset=[PROXY_TARGET_COLUMN, STRUCT_PATH_COLUMN]).reset_index(drop=True)
    print(f"[{model_adi}] [ON-EGITIM] Proxy veri: {len(df)} satir (hedef: {PROXY_TARGET_COLUMN})")
    if len(df) < 50:
        print(f"[{model_adi}] [ON-EGITIM ATLANDI] Yeterli proxy veri yok (n={len(df)} < 50).")
        return out_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cache is None:
        print(f"[{model_adi}] [ON-EGITIM] CIF -> 3B graf onbellegi hazirlaniyor...")
        cache = grafik_onbellek_olustur(df, CUTOFF, STRUCT_PATH_COLUMN)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * PRETRAIN_VAL_FRAC))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    hedef_ort, hedef_std = _hedef_istatistik(df_train, [PROXY_TARGET_COLUMN])
    aux_olcekleyici = AuxOlcekleyici().fit(aux_ham_matris(df_train))
    aux_train = aux_olcekleyici.transform(aux_ham_matris(df_train))
    aux_val = aux_olcekleyici.transform(aux_ham_matris(df_val))

    ds_train = MOFDataset(df_train, cache, aux_train, hedef_ort, hedef_std, [PROXY_TARGET_COLUMN])
    ds_val = MOFDataset(df_val, cache, aux_val, hedef_ort, hedef_std, [PROXY_TARGET_COLUMN])
    train_loader = DataLoader(ds_train, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size, shuffle=False, collate_fn=collate_fn)

    model = RegressionHead(encoder_factory(), n_outputs=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, LR_ETA_MIN)

    en_iyi_val = float("inf")
    sabir = 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tl, _, _ = epoch_calistir(model, train_loader, optimizer, device, True)
        vl, vg, vt = epoch_calistir(model, val_loader, optimizer, device, False)
        scheduler.step()
        vm = metrik_hesapla_tekli((vg * hedef_std + hedef_ort)[:, 0], (vt * hedef_std + hedef_ort)[:, 0])
        if epoch == 1 or epoch % 5 == 0:
            print(f"  [ON-EGITIM] Epoch {epoch:3d}/{max_epochs} | train_loss={tl:.4f} val_loss={vl:.4f} "
                  f"val_MAE={vm['MAE']:.4f} val_R2={vm['R2']:.3f} | {time.time() - t0:.1f}s")
        if vl < en_iyi_val:
            en_iyi_val = vl
            sabir = 0
            torch.save(model.encoder.state_dict(), out_path)
        else:
            sabir += 1
            if sabir >= patience:
                print(f"  [ON-EGITIM] Erken durdurma (epoch {epoch}).")
                break

    print(f"[{model_adi}] [ON-EGITIM] Encoder agirliklari kaydedildi -> {out_path.resolve()}")
    return out_path


# ---------------------------------------------------------------------------
# 8) PERMUTATION IMPORTANCE (grafik.py dosyalarinin cagirdigi ORTAK fonksiyon)
# ---------------------------------------------------------------------------
def permutation_importance_hesapla(ayarlar: EgitimAyarlari, model_factory,
                                    checkpoint_dir: Path, sonuc_dir: Path,
                                    device: str | None = None, perm_batch_size: int = 32,
                                    perm_seed: int = 42) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df_oof = pd.read_csv(sonuc_dir / "test_tahminleri_oof.csv")
    df_full = pd.read_csv(GAS_FINETUNE_CSV)
    df_full = df_full[df_full["eslesme_durumu"] == "TAM"].reset_index(drop=True)

    rng = np.random.default_rng(perm_seed)
    delta_yapi_list = []
    delta_grup_list = {ad: [] for ad in AUX_GRUPLARI}

    def coklu_mae(hedef_mat, tahmin_mat):
        vals = []
        for k in range(hedef_mat.shape[1]):
            m = metrik_hesapla_tekli(hedef_mat[:, k], tahmin_mat[:, k])
            if np.isfinite(m["MAE"]):
                vals.append(m["MAE"])
        return float(np.mean(vals)) if vals else float("nan")

    for fold_no in sorted(df_oof["fold"].unique()):
        ckpt_path = checkpoint_dir / f"fold{fold_no}_best_model.pt"
        if not ckpt_path.exists():
            print(f"  [ATLANDI] fold {fold_no}: checkpoint bulunamadı.")
            continue

        ids = set(df_oof.loc[df_oof["fold"] == fold_no, ID_COLUMN].unique())
        df_test = df_full[df_full[ID_COLUMN].isin(ids)].reset_index(drop=True)
        if len(df_test) == 0:
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = model_factory().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        cache = {y: cif_den_3b_graf(y, ayarlar.cutoff) for y in sorted(set(df_test[STRUCT_PATH_COLUMN]))}

        olcekleyici = AuxOlcekleyici()
        olcekleyici.ortalama = ckpt["aux_ortalama"]
        olcekleyici.std = ckpt["aux_std"]
        aux_matris = olcekleyici.transform(aux_ham_matris(df_test))

        ds = MOFDataset(df_test, cache, aux_matris, ckpt["hedef_ort"], ckpt["hedef_std"], TARGET_COLUMNS)
        loader = DataLoader(ds, perm_batch_size, shuffle=False, collate_fn=collate_fn)

        tum_fea, tum_aux, tum_hedef = [], [], []
        with torch.no_grad():
            for g_b, aux, hedef in loader:
                fea = model._kristal_gommesi(g_b, device)
                tum_fea.append(fea)
                tum_aux.append(aux.to(device))
                tum_hedef.append(hedef)

        fea = torch.cat(tum_fea)
        aux = torch.cat(tum_aux)
        hedef = torch.cat(tum_hedef).numpy() * ckpt["hedef_std"] + ckpt["hedef_ort"]

        def head_ile_tahmin_et(fea_t, aux_t):
            with torch.no_grad():
                x = torch.cat([fea_t, aux_t], dim=-1)
                tahmin_std = model.head(x).cpu().numpy()
            return tahmin_std * ckpt["hedef_std"] + ckpt["hedef_ort"]

        taban_tahmin = head_ile_tahmin_et(fea, aux)
        taban_mae = coklu_mae(hedef, taban_tahmin)

        perm_idx = torch.from_numpy(rng.permutation(len(fea)))
        mae_yapi = coklu_mae(hedef, head_ile_tahmin_et(fea[perm_idx], aux))
        delta_yapi_list.append(mae_yapi - taban_mae)

        for grup_adi, idxler in AUX_GRUP_IDX.items():
            aux_karisik = aux.clone()
            perm_idx2 = torch.from_numpy(rng.permutation(len(aux)))
            for j in idxler:
                aux_karisik[:, j] = aux[perm_idx2, j]
            mae_grup = coklu_mae(hedef, head_ile_tahmin_et(fea, aux_karisik))
            delta_grup_list[grup_adi].append(mae_grup - taban_mae)

        print(f"  Fold {fold_no}: taban_MAE(overall)={taban_mae:.4f}")

    sonuclar = {"Crystal Structure": float(np.mean(delta_yapi_list)) if delta_yapi_list else 0.0}
    for grup_adi, degerler in delta_grup_list.items():
        sonuclar[grup_adi] = float(np.mean(degerler)) if degerler else 0.0
    return dict(sorted(sonuclar.items(), key=lambda item: item[1]))


# ---------------------------------------------------------------------------
# 9) EGITIM GECMISI CSVLERINI OKUMA (grafik.py'nin egitim-kaybi grafigi icin)
# ---------------------------------------------------------------------------
def egitim_gecmisi_oku(sonuc_dir: Path) -> dict[int, pd.DataFrame]:
    gecmisler = {}
    for f in sorted(sonuc_dir.glob("egitim_gecmisi_fold*.csv")):
        fold_no = int(f.stem.replace("egitim_gecmisi_fold", ""))
        gecmisler[fold_no] = pd.read_csv(f)
    return gecmisler


# ---------------------------------------------------------------------------
# 10) TEK-COAGRI GRAFIK URETIMI (her modelin grafik.py'si bunu cagirir)
#     4 hedefin HER BIRI icin (gercek_vs_tahmin, residual, confusion_matrix,
#     pore-secicilik-tutarliligi) + 1 ortak (egitim_kaybi) + feature_importance
# ---------------------------------------------------------------------------
def tum_grafikleri_uret(ayarlar: EgitimAyarlari, model_factory, checkpoint_dir: Path,
                         sonuc_dir: Path, grafik_dir: Path, panel: str = "") -> None:
    import grafik_ortak as go

    if panel:
        go.set_panel(panel)
    grafik_dir.mkdir(parents=True, exist_ok=True)

    oof_csv = sonuc_dir / "test_tahminleri_oof.csv"
    if not oof_csv.exists():
        sys.exit(f"HATA: '{oof_csv}' bulunamadi. Once run_<model>.py calistirin.")

    df = pd.read_csv(oof_csv)
    print(f"[{ayarlar.model_adi}] OOF veri yuklendi: {df.shape[0]} satir.")

    for kol in TARGET_COLUMNS:
        G, T = f"gercek_{kol}", f"tahmin_{kol}"
        etiket = f"{ayarlar.model_adi}_{kol}"
        go.ciz_fold_sacilim(df, G, T, kol, "", grafik_dir / f"gercek_vs_tahmin_{etiket}.tif")
        go.residual_dagilim_grafigi(df, G, T, kol, "", grafik_dir / f"residual_dagilim_{etiket}.tif")
        go.confusion_matrix_grafigi(df, G, T, kol, "", grafik_dir / f"confusion_matrix_{etiket}.tif")

    basarili = go.gozeneklilik_secicilik_tutarlilik_grafigi(
        df, ayarlar.model_adi, "", grafik_dir / f"pore_secicilik_tutarlilik_{ayarlar.model_adi}.tif")
    if not basarili:
        print("  [ATLANDI] Gozeneklilik-secicilik tutarlılığı için gerekli sütun/değerler yok.")

    gecmis = egitim_gecmisi_oku(sonuc_dir)
    if gecmis:
        go.egitim_kaybi_grafigi(gecmis, ayarlar.model_adi, "", grafik_dir / f"egitim_kaybi_{ayarlar.model_adi}.tif")
    else:
        print("  [ATLANDI] Eğitim geçmişi CSV'leri bulunamadı.")

    print(f"[{ayarlar.model_adi}] Hedef-basina grafik turleri -> {grafik_dir.resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{ayarlar.model_adi}] Permutation importance hesaplanıyor (cihaz: {device})...")
    skorlar = permutation_importance_hesapla(ayarlar, model_factory, checkpoint_dir, sonuc_dir, device)
    go.permutation_importance_grafigi(skorlar, ayarlar.model_adi, "", grafik_dir / "feature_importance.tif")
    print(f"[{ayarlar.model_adi}] Önem skorları: {skorlar}")
    print(f"[{ayarlar.model_adi}] Tüm grafikler -> {grafik_dir.resolve()}")
