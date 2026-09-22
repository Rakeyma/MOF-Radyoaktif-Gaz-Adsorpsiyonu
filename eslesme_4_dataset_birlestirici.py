"""
eslesme_4_dataset_birlestirici.py
====================================
AMAC:
    Bilesen 2 (indirme), Bilesen 3 (NLP madenciligi) ve Bilesen 4 (veri
    artirma) ciktilarini BIRLESTIRIP, Bilesen 5'in (transfer learning)
    ihtiyac duydugu IKI NIHAI veri setini uretmek:

    (A) QMOF_PRETRAIN_CSV — ON-EGITIM (pretrain) icin: qmof_pretrain_ham.csv
        (buyuk proxy havuzu) satirlarinin CIF'lerinin GERCEKTEN okunup 3B
        grafa cevrilebildigini DOGRULAR (bozuk/hatali kayitlari eler) ve
        'graf_cif_path' sutununu ekler.

    (B) GAS_FINETUNE_CSV — INCE-AYAR (finetune) icin, EN TEMEL adim:
        HER augment edilmis ornek (augmentasyon_manifest.csv) icin 4 hedefin
        (Xe/Kr/I2 kapasitesi, Xe/Kr secicilik) degerini belirler:

        ONCELIK SIRASI (etiket kaynagi seffafligi - label_source_<hedef>
        sutununda KAYDEDILIR, hicbir sayi kaynaksiz birakilmaz):
          1. NLP_LITERATURE : nlp_literatur_madencilik_2.py'nin GERCEKTEN
             bulup regex ile cikardigi deger - SADECE augment EDILMEMIS
             (orijinal, is_augmented=False) orneklere uygulanir (bir
             kusur/ikame/fonksiyonlastirma FIZIKSEL OLARAK ozelligi
             degistirir - literatur degerini augment edilmis bir varyanta
             ATFETMEK YANILTICI olurdu).
          2. PROXY_PORE_CORRELATION : literatur eslesmesi YOKSA (ki
             augment edilmis TUM ornekler + eslesmeyen orijinaller icin
             boyle olacaktir), Bilesen 2'nin gerekce boluminde anlatilan
             (Sikora et al. 2012 boyut-eleme ilkesi) fiziksel-motivasyonlu
             KORELASYON formulu kullanilir (bkz. proxy_gaz_etiketleri).
             BU DEGERLER GERCEK OLCUM DEGILDIR - rapor scriptleri bu
             sutunu okuyup etiket-kaynagi dagilimini SEFFAFCA raporlar.

CIKTI:
    data/processed/qmof_pretrain_dataset_final.csv
    data/processed/gaz_adsorpsiyon_dataset_final.csv
    data/processed/eslesme_bilgisi.json

CALISTIRMA:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python eslesme_4_dataset_birlestirici.py
"""

from __future__ import annotations

import json
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from paths import (
    QMOF_PRETRAIN_RAW_CSV, QMOF_PRETRAIN_CSV, AUGMENTATION_MANIFEST_CSV,
    NLP_LITERATUR_CSV, GAS_FINETUNE_CSV, DATA_PROCESSED,
)
from ortak_ozellikler import AUX_FEATURE_COLUMNS
from graf_ozellik_ortak import cif_den_3b_graf

TARGET_COLUMNS = ["xe_uptake_mmol_g", "kr_uptake_mmol_g", "xe_kr_selectivity", "i2_uptake_mmol_g"]

# Kinetik/molekuler cap (Angstrom, literatur yaklasik degerleri) - proxy
# korelasyon formulunun MERKEZINI belirler (bkz. modul dokstringi).
D_KINETIK = {"xe": 4.10, "kr": 3.69, "i2": 5.00}

GAZ_MOLAR_KUTLE_G_MOL = {"xe_uptake_mmol_g": 131.29, "kr_uptake_mmol_g": 83.80, "i2_uptake_mmol_g": 253.81}


def _cif_dogrula(cif_path: str) -> bool:
    try:
        g = cif_den_3b_graf(cif_path)
        return g["n_atoms"] > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# (A) ON-EGITIM VERI SETI
# ---------------------------------------------------------------------------
def pretrain_birlestir() -> pd.DataFrame:
    if not QMOF_PRETRAIN_RAW_CSV.exists():
        print(f"[ON-EGITIM] [ATLANDI] '{QMOF_PRETRAIN_RAW_CSV}' bulunamadi.")
        return pd.DataFrame()

    df = pd.read_csv(QMOF_PRETRAIN_RAW_CSV, low_memory=False)
    print(f"[ON-EGITIM] {len(df)} ham kayit okundu, CIF gecerliligi dogrulaniyor...")
    df["graf_cif_path"] = df["cif_path"]
    gecerli = []
    for i, yol in enumerate(df["graf_cif_path"]):
        gecerli.append(_cif_dogrula(yol))
        if (i + 1) % 1000 == 0 or (i + 1) == len(df):
            print(f"    {i + 1}/{len(df)} dogrulandi")
    df["eslesme_durumu"] = np.where(gecerli, "TAM", "EKSIK")
    n_tam = (df["eslesme_durumu"] == "TAM").sum()
    print(f"[ON-EGITIM] {n_tam}/{len(df)} gecerli (TAM).")
    df.to_csv(QMOF_PRETRAIN_CSV, index=False)
    print(f"[ON-EGITIM] Kaydedildi -> {QMOF_PRETRAIN_CSV.resolve()}")
    return df


# ---------------------------------------------------------------------------
# (B) INCE-AYAR VERI SETI
# ---------------------------------------------------------------------------
def _birim_donustur(deger: float, birim: str, hedef_kolon: str) -> float | None:
    """cm3/g VEYA wt% VEYA mg/g -> mmol/g (ideal gaz molar hacmi ~22.414 L/mol
    @STP kullanilarak kaba donusum; secicilik zaten boyutsuzdur)."""
    birim = birim.lower().replace(" ", "")
    if hedef_kolon == "xe_kr_selectivity":
        return deger
    if "mmol" in birim:
        return deger
    molar_kutle = GAZ_MOLAR_KUTLE_G_MOL.get(hedef_kolon, 130.0)
    if "cm3" in birim or "cm³" in birim:
        return deger / 22.414
    if "mg" in birim:
        return (deger / 1000.0) / molar_kutle * 1000.0
    if "%" in birim:
        return (deger / 100.0 * 1000.0) / molar_kutle * 1000.0
    return deger


def nlp_etiketlerini_hazirla() -> pd.DataFrame:
    if not NLP_LITERATUR_CSV.exists() or pd.read_csv(NLP_LITERATUR_CSV).empty:
        print("[INCE-AYAR] NLP literatur CSV bos/yok - TUM etiketler proxy korelasyondan gelecek.")
        return pd.DataFrame(columns=["mof_adi_norm", "target_column", "deger_mmol_g"])

    nlp = pd.read_csv(NLP_LITERATUR_CSV)
    nlp["mof_adi_norm"] = nlp["mof_name"].str.upper().str.strip()
    nlp["deger_mmol_g"] = nlp.apply(lambda r: _birim_donustur(r["value"], r["unit"], r["target_column"]), axis=1)
    # ayni (mof, hedef) icin BIRDEN FAZLA makale/cumle eslesmesi olabilir -
    # MEDYAN alinarak tekil/gurbuz (robust) bir literatur degeri elde edilir.
    ozet = nlp.groupby(["mof_adi_norm", "target_column"])["deger_mmol_g"].median().reset_index()
    print(f"[INCE-AYAR] NLP'den {len(ozet)} benzersiz (MOF, hedef) literatur etiketi turetildi (medyan).")
    return ozet


_ARKETIP_ADI_RE = re.compile(r"^([A-Za-z]+-\d+(?:-NH2)?)")


def _arketip_adi_cikar(mof_name: str) -> str:
    """'HKUST-1-var00023' -> 'HKUST-1' (procedural/augment suffix'lerini
    atarak NLP eslesmesi icin taban isme indirger)."""
    m = _ARKETIP_ADI_RE.match(str(mof_name))
    return (m.group(1) if m else str(mof_name)).upper().strip()


def proxy_gaz_etiketleri(row: pd.Series, rng: np.random.Generator) -> dict:
    """Bkz. modul dokstringi (2) - gozeneklilik-boyut uyumuna dayali,
    ACIKCA proxy/tahmini, fiziksel-motivasyonlu korelasyon formulu."""
    pld = row.get("pld_A", np.nan)
    pv = row.get("pore_volume_cm3_g", 0.3)
    if not np.isfinite(pld):
        pld = 5.0
    if not np.isfinite(pv) or pv <= 0:
        pv = 0.3
    open_metal = float(row.get("open_metal_site", False))
    func = float(row.get("has_functional_group", False))

    def boyut_uyum(d_kinetik: float, genislik: float = 1.5) -> float:
        return float(np.exp(-((pld - d_kinetik) ** 2) / (2 * genislik ** 2)))

    gurultu = lambda sigma: float(np.exp(rng.normal(0, sigma)))  # lognormal carpimsal gurultu

    xe = pv * (0.8 + 1.5 * boyut_uyum(D_KINETIK["xe"])) * (1 + 0.4 * open_metal) * gurultu(0.15)
    kr = pv * (0.5 + 1.0 * boyut_uyum(D_KINETIK["kr"])) * (1 + 0.2 * open_metal) * gurultu(0.15)
    sel = 1.0 + 7.0 * (boyut_uyum(D_KINETIK["xe"]) / (boyut_uyum(D_KINETIK["kr"]) + 0.15)) * (1 + 0.3 * open_metal)
    sel = float(np.clip(sel * gurultu(0.12), 1.0, 30.0))
    i2 = pv * (0.3 + 1.0 * boyut_uyum(D_KINETIK["i2"], 2.0)) * (1 + 1.2 * open_metal) * (1 + 0.5 * func) * gurultu(0.18)

    return {
        "xe_uptake_mmol_g": max(xe, 0.0), "kr_uptake_mmol_g": max(kr, 0.0),
        "xe_kr_selectivity": sel, "i2_uptake_mmol_g": max(i2, 0.0),
    }


def finetune_birlestir() -> pd.DataFrame:
    if not AUGMENTATION_MANIFEST_CSV.exists():
        sys.exit(f"HATA: '{AUGMENTATION_MANIFEST_CSV}' bulunamadi. Once veri_artirma_3_augmentasyon.py calistirilmali.")

    df = pd.read_csv(AUGMENTATION_MANIFEST_CSV, low_memory=False)
    print(f"[INCE-AYAR] {len(df)} augment edilmis/orijinal ornek okundu.")
    df["graf_cif_path"] = df["cif_path"]
    df["mof_adi_norm"] = df["mof_name"].apply(_arketip_adi_cikar)

    nlp_ozet = nlp_etiketlerini_hazirla()
    rng = np.random.default_rng(42)

    for kol in TARGET_COLUMNS:
        df[kol] = np.nan
        df[f"label_source_{kol}"] = "PROXY_PORE_CORRELATION"

    if not nlp_ozet.empty:
        for kol in TARGET_COLUMNS:
            alt = nlp_ozet[nlp_ozet["target_column"] == kol].set_index("mof_adi_norm")["deger_mmol_g"]
            eslesen = (~df["is_augmented"]) & df["mof_adi_norm"].isin(alt.index)
            df.loc[eslesen, kol] = df.loc[eslesen, "mof_adi_norm"].map(alt)
            df.loc[eslesen, f"label_source_{kol}"] = "NLP_LITERATURE"
            print(f"  {kol}: {eslesen.sum()} orijinal ornek NLP literatur degeriyle eslesti.")

    print("[INCE-AYAR] Kalan (eslesmeyen) tum ornekler icin proxy gozeneklilik-korelasyonu hesaplaniyor...")
    proxy_deger_listesi = [proxy_gaz_etiketleri(row, rng) for _, row in df.iterrows()]
    for kol in TARGET_COLUMNS:
        proxy_seri = pd.Series([d[kol] for d in proxy_deger_listesi], index=df.index)
        bos = df[kol].isna()
        df.loc[bos, kol] = proxy_seri[bos]

    print("[INCE-AYAR] CIF gecerliligi dogrulaniyor...")
    gecerli = []
    for i, yol in enumerate(df["graf_cif_path"]):
        gecerli.append(_cif_dogrula(yol))
        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"    {i + 1}/{len(df)} dogrulandi")
    df["eslesme_durumu"] = np.where(gecerli, "TAM", "EKSIK")

    eksik_aux = [c for c in AUX_FEATURE_COLUMNS if c not in df.columns]
    if eksik_aux:
        print(f"  UYARI: manifest'te eksik AUX sutunlari (0 ile dolduruluyor): {eksik_aux}")
        for c in eksik_aux:
            df[c] = 0.0

    df.to_csv(GAS_FINETUNE_CSV, index=False)
    print(f"[INCE-AYAR] Kaydedildi -> {GAS_FINETUNE_CSV.resolve()} ({len(df)} satir)")
    return df


def main() -> None:
    print("=" * 70)
    print("eslesme_4_dataset_birlestirici.py")
    print("=" * 70)

    pre_df = pretrain_birlestir()
    print()
    fin_df = finetune_birlestir()

    meta = {
        "pretrain_satir": int(len(pre_df)),
        "pretrain_tam": int((pre_df["eslesme_durumu"] == "TAM").sum()) if len(pre_df) else 0,
        "finetune_satir": int(len(fin_df)),
        "finetune_tam": int((fin_df["eslesme_durumu"] == "TAM").sum()) if len(fin_df) else 0,
        "finetune_label_source_dagilimi": {
            kol: fin_df[f"label_source_{kol}"].value_counts().to_dict() for kol in TARGET_COLUMNS
        } if len(fin_df) else {},
        "finetune_benzersiz_temel_mof": int(fin_df["base_mof_id"].nunique()) if len(fin_df) else 0,
    }
    (DATA_PROCESSED / "eslesme_bilgisi.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("GENEL OZET")
    print("=" * 70)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\nTamamlandi. Bu iki CSV (qmof_pretrain_dataset_final.csv, "
          "gaz_adsorpsiyon_dataset_final.csv), 11 model klasorunun VE 4 XAI "
          "klasorunun okuyacagi NIHAI veri setleridir.")


if __name__ == "__main__":
    main()
