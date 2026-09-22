"""
veri_indirici_1_jarvis_core_mof.py
=====================================
BILESEN 2 — Veritabani/API Kazima (Database API Scraping)
=============================================================
AMAC:
    Metal-Organik Cerceve (MOF) kristal yapilarini (CIF + tam 3B kartezyen
    koordinat), Hugging Face `datasets` kutuphanesi (JARVIS-QMOF benzeri
    kaynak) VE/VEYA CoRE-MOF'un genel-erisimli veritabanindan DEVASA
    olcekte, TAMAMEN OTOMATIK olarak indirip PyTorch Geometric'e hazir
    (paths.CIF_DIR altina yazilan CIF + paths.RAW_MOF_CSV/QMOF_PRETRAIN_RAW_CSV
    tablosu) hale getirmek. Kullanicidan HICBIR manuel indirme BEKLENMEZ.

UC KATMANLI KAYNAK STRATEJISI ("Üç Boyutlu Kristal Malzemeler/
mp_veri_indirici_1.py"deki 'IKI kaynak, HICBIR ZAMAN durma' ilkesinin
UCE genisletilmis hali - agsiz/kapali bir ortamda BILE calisir):
    (A) HUGGING FACE `datasets` (BIRINCIL): HF_DATASET_ID ortam degiskeni
        (veya HF_DATASET_ADAYLARI listesindeki adaylar sirayla) ile
        `datasets.load_dataset(...)` denenir. Basarili olursa, kayitlardaki
        CIF/yapi metni + (varsa) DFT formation energy / band gap / pore
        alanlari doğrudan kullanilir.
    (B) CoRE-MOF ACIK CSV (IKINCIL): HF basarisiz/erisilemezse, CoRE-MOF
        projesinin GitHub'da barindirilan genel-erisimli (anahtarsiz)
        ozet CSV'sine (hesaplanmis gozeneklilik: LCD/PLD/yuzey alani/
        gozeneklilik orani) `requests` ile erisim denenir.
    (C) PROSEDUREL YEDEK (UCUNCUL, HER ZAMAN calisir - agsiz ortamda bile):
        (A) ve (B) basarisiz olursa (ag yok/zaman asimi/paket eksik),
        script HICBIR ZAMAN durmaz: 12 IYI-BILINEN, gercek MOF ailesinin
        (HKUST-1, MOF-5/IRMOF-1, ZIF-8, UiO-66, Mg-MOF-74, NU-1000, vb.)
        GERCEK birim hucre simetri sinifi/kabaca kafes parametreleri
        temel alinarak PROSEDUREL (basitlestirilmis dugum-baglayici
        iskelet) varyantlar uretilir - bu yapilar TAM crystallografik
        hassasiyette DEGILDIR (bu bir DFT-dogrulugu veri kaynagi degil,
        GNN boru hattinin UCTAN UCA calisabilmesini saglayan, ACIKCA
        etiketlenmis - source='PROCEDURAL_FALLBACK' - bir yedektir).
    HER SATIRDA 'source' sutunu HANGI KATMANDAN geldigini SEFFAFCA belirtir
    (rapor scriptleri bu sutunu okuyup veri kaynagi dagilimini raporlar -
    hicbir sayi/yapi kaynak-belirsiz birakilmaz).

GOZENEKLILIK (proxy) OZELLIKLERI:
    Gercek Zeo++ (Willems et al. 2012) hesaplamasi bu ortamda mevcut
    DEGILDIR (harici/derlenmis bir binary gerektirir) - bunun yerine
    `estimate_pore_proxies()` HAFIF bir GEOMETRIK YAKLASTIRMA kullanir
    (van der Waals hacim orani -> gozeneklilik orani -> yuzey alani/LCD/PLD
    kabaca turetilir). Kullanicinin Bilesen-2 gereksinimi ("Xe/Kr
    adsorpsiyon verisi eksikse pore volume/void fraction/surface area gibi
    proxy ozellikleri esle") tam olarak BUDUR - dogrudan gaz-adsorpsiyon
    olcumu YERINE, adsorpsiyonla GUCLU KORELE oldugu bilinen gozeneklilik
    buyuklukleri kullanilir.

CIKTI:
    data/raw/mof_ham_veri.csv          (ince-ayar/finetune icin temel MOF havuzu)
    data/raw/qmof_pretrain_ham.csv     (on-egitim/pretrain icin BUYUK proxy havuz)
    data/raw/kaynak_bilgisi.json
    data/processed/cif/<base_mof_id>.cif

CALISTIRMA:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python veri_indirici_1_jarvis_core_mof.py

AYARLAR (ortam degiskenleri):
    MAX_MATERIALS           Finetune-havuzu icin kac TEMEL MOF (varsayilan 600)
    MAX_MATERIALS_PRETRAIN  Pretrain-havuzu icin kac yapi (varsayilan 6000;
                             GERCEK QMOF ~20-100k icerir, kucuk/hizli dogrulama
                             kosumlari icin varsayilan kucultulmustur - buyutmek
                             icin bu degiskeni artirin, kod DEGISIKLIGI gerekmez)
    HF_DATASET_ID            Hugging Face dataset kimligi (bos ise adaylar denenir)
    DATA_SOURCE              "auto" | "huggingface" | "core_mof" | "procedural"
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from paths import DATA_RAW, RAW_MOF_CSV, RAW_MOF_META, CIF_DIR, QMOF_PRETRAIN_RAW_CSV, HF_TOKEN

# --------------------------------------------------------------------------
# AYARLAR
# --------------------------------------------------------------------------
def _int_oku(ad: str, varsayilan: int) -> int:
    ham = os.environ.get(ad, str(varsayilan)).strip().lower()
    if ham in ("all", "none", "sinirsiz", "-1"):
        return varsayilan  # 'sinirsiz' bu projede procedural yedegin makul kalmasi icin varsayilana sabitlenir
    try:
        return int(ham)
    except ValueError:
        return varsayilan


MAX_MATERIALS = _int_oku("MAX_MATERIALS", 600)
MAX_MATERIALS_PRETRAIN = _int_oku("MAX_MATERIALS_PRETRAIN", 6000)
DATA_SOURCE = os.environ.get("DATA_SOURCE", "auto").strip().lower()
HF_DATASET_ID = os.environ.get("HF_DATASET_ID", "").strip() or None
# "dataset_id" veya "dataset_id::config_adi" formatinda - ilki GERCEK, dogrulanmis
# bir CoRE-MOF turevi (GCMC-simule edilmis CIF + CO2/CH4 adsorpsiyon verisi
# iceren jablonkagroup/core_mof_no_topo, 'raw_data' konfigurasyonu); digerleri
# JARVIS-QMOF benzeri adaylar icin YEDEK denemeler (Hub'da bulunmayabilir -
# bkz. modul dokstringi 'HICBIR ZAMAN durmaz' ilkesi).
HF_DATASET_ADAYLARI = [
    "jablonkagroup/core_mof_no_topo::raw_data",
    "n0w0f/MOFdiff", "jarvis-tools/qmof", "hyunp2/MOF-CIF", "chenxwh/qmof",
]
CORE_MOF_CSV_URL = os.environ.get(
    "CORE_MOF_CSV_URL",
    "https://raw.githubusercontent.com/gregchung/gregchung.github.io/master/CoRE-MOF/2019-ASR-public.csv",
)

RNG_SEED = 42

# --------------------------------------------------------------------------
# 12 IYI-BILINEN MOF AILESI (procedural yedek icin GERCEK yayimlanmis
# kabaca kafes/simetri/kimya bilgisi - Chem. Rev./yayimlanmis CSD
# girdilerinden BILINEN yaklasik degerler; TAM crystallografik hassasiyette
# DEGILDIR, bkz. modul dokstringi).
# --------------------------------------------------------------------------
ARKETIPLER = {
    "HKUST-1":    dict(a=26.343, metal="Cu", metal_z=29, linker_elems=["C", "O"], n_nodes=8, open_metal_site=True,  linker_len=6.9),
    "MOF-5":      dict(a=25.669, metal="Zn", metal_z=30, linker_elems=["C", "O"], n_nodes=8, open_metal_site=False, linker_len=9.0),
    "IRMOF-3":    dict(a=25.832, metal="Zn", metal_z=30, linker_elems=["C", "N", "O"], n_nodes=8, open_metal_site=False, linker_len=9.0, functional=True),
    "ZIF-8":      dict(a=16.991, metal="Zn", metal_z=30, linker_elems=["C", "N"], n_nodes=8, open_metal_site=False, linker_len=6.0),
    "UiO-66":     dict(a=20.751, metal="Zr", metal_z=40, linker_elems=["C", "O"], n_nodes=8, open_metal_site=False, linker_len=6.6),
    "UiO-66-NH2": dict(a=20.803, metal="Zr", metal_z=40, linker_elems=["C", "N", "O"], n_nodes=8, open_metal_site=False, linker_len=6.6, functional=True),
    "Mg-MOF-74":  dict(a=25.850, metal="Mg", metal_z=12, linker_elems=["C", "O"], n_nodes=6, open_metal_site=True,  linker_len=5.6, hex=True, c=6.86),
    "Ni-MOF-74":  dict(a=25.960, metal="Ni", metal_z=28, linker_elems=["C", "O"], n_nodes=6, open_metal_site=True,  linker_len=5.6, hex=True, c=6.87),
    "NU-1000":    dict(a=40.312, metal="Zr", metal_z=40, linker_elems=["C", "O"], n_nodes=8, open_metal_site=False, linker_len=16.0, hex=True, c=16.7),
    "MIL-53":     dict(a=6.608,  metal="Al", metal_z=13, linker_elems=["C", "O"], n_nodes=4, open_metal_site=True,  linker_len=6.6, ortho=True, b=16.7, c=12.7),
    "MIL-101":    dict(a=88.869, metal="Cr", metal_z=24, linker_elems=["C", "O"], n_nodes=8, open_metal_site=True,  linker_len=6.6),
    "ZIF-67":     dict(a=16.960, metal="Co", metal_z=27, linker_elems=["C", "N"], n_nodes=8, open_metal_site=False, linker_len=6.0),
}

_LINKER_Z = {"C": 6, "N": 7, "O": 8, "H": 1}


def _procedural_yapi(arketip_adi: str, param: dict, jitter_rng: np.random.Generator):
    """Bir MOF arketipinin GERCEK simetri ailesine (kubik/hegzagonal/
    ortorombik) ve kabaca kafes olcegine sadik, basitlestirilmis (dugum +
    dogrusal baglayici iskelet) periyodik bir pymatgen Structure uretir.
    Metal dugumler kufesin YUKSEK-SIMETRI (kose+yuz-merkezli benzeri)
    konumlarina, baglayicilar (linker) EN YAKIN iki dugum arasindaki
    dogru uzerine yerlestirilir - TAM crystallografik cozumleme DEGIL,
    ama gercekci HACIM/GOZENEK OLCEGINE sahip, gecerli bir 3B periyodik
    yapi (bkz. modul dokstringi 'PROSEDUREL YEDEK' basligi)."""
    from pymatgen.core import Structure, Lattice

    a = param["a"] * (1.0 + jitter_rng.uniform(-0.015, 0.015))
    if param.get("hex"):
        c = param["c"] * (1.0 + jitter_rng.uniform(-0.02, 0.02))
        lattice = Lattice.hexagonal(a, c)
    elif param.get("ortho"):
        b = param["b"] * (1.0 + jitter_rng.uniform(-0.02, 0.02))
        c = param["c"] * (1.0 + jitter_rng.uniform(-0.02, 0.02))
        lattice = Lattice.orthorhombic(a, b, c)
    else:
        lattice = Lattice.cubic(a)

    # dugum konumlari: FCC-benzeri kose+yuz-merkezi frac koordinat kumesi
    # (n_nodes'e gore kesilir) - gercek MOF topolojilerinin cogu FCC/HCP
    # turevi dugum diziliminde olma egilimindedir.
    aday_frac = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ])
    n_nodes = min(param["n_nodes"], len(aday_frac))
    dugum_frac = aday_frac[:n_nodes] + jitter_rng.normal(0, 0.01, size=(n_nodes, 3))

    species, coords_cart = [], []
    for f in dugum_frac:
        species.append(param["metal_z"])
        coords_cart.append(lattice.get_cartesian_coords(f % 1.0))

    # baglayicilar: HER dugum ciftinden (i<j) EN YAKIN olanlari, dogru
    # uzerine linker_len'e gore esit-araliklandirilmis atomlarla doldur.
    dugum_cart = np.array(coords_cart)
    mesafeler = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.linalg.norm(dugum_cart[i] - dugum_cart[j])
            mesafeler.append((d, i, j))
    mesafeler.sort(key=lambda t: t[0])
    hedef_len = param["linker_len"]
    baglanti_sayisi = 0
    for d, i, j in mesafeler:
        if abs(d - hedef_len) / hedef_len > 0.6:
            continue  # sadece hedef baglayici uzunluguna makul yakin ciftler baglanir
        n_ara = max(2, int(round(d / 1.4)))
        for k in range(1, n_ara):
            t = k / n_ara
            elem = param["linker_elems"][k % len(param["linker_elems"])]
            coords_cart.append(dugum_cart[i] * (1 - t) + dugum_cart[j] * t)
            species.append(_LINKER_Z[elem])
        baglanti_sayisi += 1
        if baglanti_sayisi >= n_nodes * 2:  # asiri-baglanmayi onle (makul koordinasyon sayisi)
            break

    coords_frac = [lattice.get_fractional_coords(np.array(c)) % 1.0 for c in coords_cart]
    struct = Structure(lattice, species, coords_frac, coords_are_cartesian=False)
    return struct


def estimate_pore_proxies(struct) -> dict:
    """Zeo++ OLMADAN (bu ortamda mevcut degil), HAFIF bir GEOMETRIK
    yaklastirmayla gozeneklilik proxy ozelliklerini turetir:
        void_fraction ~ 1 - (atom vdW hacimleri toplami / hucre hacmi)
        pore_volume_cm3_g = void_fraction * hucre_hacmi_cm3 / kutle_g
        surface_area ~ atom-basina ortalama vdW yuzeyi * (atom sayisi/hacim)
        LCD/PLD ~ hucrenin en kisa kafes vektorunden atom yaricapi cikarilarak
    Bu bir YAKLASTIRMADIR (rapor scriptleri bunu 'PROXY_GEOMETRIC' olarak
    etiketler, gercek Zeo++/deneysel deger OLARAK SUNULMAZ)."""
    from pymatgen.core.periodic_table import Element

    vol_A3 = struct.volume
    vdw_toplam = 0.0
    for site in struct:
        try:
            r = Element(site.specie.symbol).van_der_waals_radius or 1.5
        except Exception:
            r = 1.5
        vdw_toplam += (4.0 / 3.0) * np.pi * (r ** 3)

    void_fraction = float(np.clip(1.0 - vdw_toplam / max(vol_A3, 1e-6), 0.05, 0.95))
    kutle_g = struct.composition.weight * 1.66053907e-24  # amu -> gram
    hucre_cm3 = vol_A3 * 1e-24
    yogunluk = kutle_g / max(hucre_cm3, 1e-30)
    pore_volume_cm3_g = void_fraction * hucre_cm3 / max(kutle_g, 1e-30)

    n_atom = max(len(struct), 1)
    ort_r = 1.7
    gravimetric_sa = void_fraction * (4 * np.pi * ort_r ** 2) * n_atom / max(kutle_g, 1e-30) * 1e-20 * 6.02214076e23 / n_atom
    volumetric_sa = gravimetric_sa * yogunluk

    # sqrt-olcekleme: dogrudan carpimdan (kisa_kafes * void_fraction) daha AZ
    # agresif kuculme uygular - kucuk (asimetrik birim hucreli, orn. HF'ten
    # gelen GERCEK CoRE-MOF girdileri gibi ~6-9A) hucrelerde bile anlamli
    # (sifira/tabana COKMEYEN) bir LCD/PLD varyasyonu korunur.
    kisa_kafes = min(struct.lattice.a, struct.lattice.b, struct.lattice.c)
    lcd = float(np.clip(kisa_kafes * np.sqrt(max(void_fraction, 0.01)), 1.5, 40.0))
    pld = float(np.clip(lcd * 0.55, 1.2, lcd))

    return {
        "pore_volume_cm3_g": float(pore_volume_cm3_g),
        "void_fraction": void_fraction,
        "gravimetric_surface_area_m2_g": float(max(gravimetric_sa, 0.0)),
        "volumetric_surface_area_m2_cm3": float(max(volumetric_sa, 0.0)),
        "lcd_A": lcd, "pld_A": pld,
        "density_g_cm3": float(yogunluk),
    }


def kompozisyon_ozellikleri(struct) -> dict:
    from pymatgen.core.periodic_table import Element
    zler = [site.specie.Z for site in struct]
    en = []
    for z in set(zler):
        try:
            en.append(Element.from_Z(z).X)
        except Exception:
            pass
    metal_z_kumesi = {v["metal_z"] for v in ARKETIPLER.values()}
    metal_sayisi = sum(1 for z in zler if z in metal_z_kumesi or Element.from_Z(z).is_metal)
    return {
        "nsites": len(struct), "nelements": len(set(zler)),
        "mean_atomic_number": float(np.mean(zler)),
        "mean_atomic_radius_A": float(np.mean([Element.from_Z(z).atomic_radius or 1.5 for z in zler])),
        "mean_electronegativity_diff": float(np.std(en)) if len(en) > 1 else 0.0,
        "metal_fraction": float(metal_sayisi / max(len(zler), 1)),
    }


# ---------------------------------------------------------------------------
# (A) HUGGING FACE `datasets` KAYNAGI
# ---------------------------------------------------------------------------
def huggingface_dene(max_n: int) -> list[dict] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [HuggingFace] 'datasets' paketi kurulu degil, atlaniyor.")
        return None

    adaylar = [HF_DATASET_ID] if HF_DATASET_ID else HF_DATASET_ADAYLARI
    for aday in adaylar:
        if not aday:
            continue
        dataset_id, _, config_adi = aday.partition("::")
        try:
            print(f"  [HuggingFace] deneniyor: {dataset_id}" + (f" (config={config_adi})" if config_adi else "") + " ...")
            args = [dataset_id] + ([config_adi] if config_adi else [])
            ds = load_dataset(*args, split="train", streaming=True, token=HF_TOKEN)
            kayitlar = []
            for i, rec in enumerate(ds):
                if i >= max_n:
                    break
                kayitlar.append(rec)
            if kayitlar:
                print(f"  [HuggingFace] BASARILI: '{dataset_id}' -> {len(kayitlar)} kayit.")
                return kayitlar
        except Exception as e:
            print(f"  [HuggingFace] '{dataset_id}' basarisiz ({type(e).__name__}: {e}).")
            continue
    print("  [HuggingFace] Hicbir aday veri seti erisilebilir degil.")
    return None


# ---------------------------------------------------------------------------
# (B) CoRE-MOF ACIK CSV KAYNAGI
# ---------------------------------------------------------------------------
def core_mof_dene(max_n: int) -> pd.DataFrame | None:
    try:
        import requests
    except ImportError:
        print("  [CoRE-MOF] 'requests' paketi kurulu degil, atlaniyor.")
        return None
    try:
        print(f"  [CoRE-MOF] deneniyor: {CORE_MOF_CSV_URL} ...")
        r = requests.get(CORE_MOF_CSV_URL, timeout=15)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        print(f"  [CoRE-MOF] BASARILI: {len(df)} satir indirildi.")
        return df.head(max_n)
    except Exception as e:
        print(f"  [CoRE-MOF] basarisiz ({type(e).__name__}: {e}).")
        return None


# ---------------------------------------------------------------------------
# ANA TOPLAMA MANTIGI
# ---------------------------------------------------------------------------
def mof_havuzu_olustur(max_n: int, havuz_adi: str, rng_seed: int) -> pd.DataFrame:
    print(f"\n[{havuz_adi}] Hedef: {max_n} yapi")
    rng = np.random.default_rng(rng_seed)
    kayitlar = []

    hf_basarili = False
    if DATA_SOURCE in ("auto", "huggingface"):
        hf_kayitlar = huggingface_dene(max_n)
        if hf_kayitlar:
            hf_basarili = True
            # NOT: HF kayit semasi veri setine gore DEGISIR - burada 'cif'/'structure'
            # metin alanini pymatgen ile ayristirmaya calisiyoruz; basarisiz olan
            # tekil kayitlar (semaya uymayan) sessizce atlanip procedural yedekle
            # tamamlanir (Bilesen 2'nin "asla durma" ilkesi TEKIL KAYIT duzeyinde de gecerlidir).
            from pymatgen.core import Structure
            KJ_MOL_TO_EV = 1.0 / 96.485
            for i, rec in enumerate(hf_kayitlar):
                try:
                    cif_metni = rec.get("cif") or rec.get("structure") or rec.get("cif_str")
                    if not cif_metni:
                        continue
                    # jablonkagroup/core_mof_no_topo (VE benzer HF veri setleri) CIF metnini
                    # bir '[CIF]' etiketiyle ONEKLENDIRIR - pymatgen'in CIF ayristiricisi bunu
                    # taniyamaz, bu yuzden GERCEK CIF govdesi baslamadan ONCE silinir.
                    cif_metni = re.sub(r"^\s*\[CIF\]\s*\n?", "", cif_metni)
                    cif_metni = re.sub(r"\s*\[/CIF\]\s*$", "", cif_metni)
                    struct = Structure.from_str(cif_metni, fmt="cif")
                    formula_adi = struct.composition.reduced_formula

                    # GERCEK DFT olusum enerjisi bu veri setinde YOKTUR - onun yerine,
                    # GERCEK GCMC-simule edilmis CO2 adsorpsiyon isisi (Widom-eklenmis
                    # molekul yontemi, kJ/mol -> eV) enerjisel bir VEKIL (proxy) olarak
                    # kullanilir (kullanicinin "formation energy VEYA elektronik/enerjisel
                    # ozellik" alternatifine uyar) - formation_energy_kaynak sutunuyla
                    # bu ACIKCA belirtilir (bkz. main() icindeki eslesme mantigi).
                    proxy_ham = (rec.get("formation_energy_per_atom") or rec.get("formation_energy")
                                 or rec.get("outputs.pure_CO2_widomHOA"))
                    proxy_ev = float(proxy_ham) * KJ_MOL_TO_EV if proxy_ham not in (None, "None") else None

                    kayitlar.append({"struct": struct, "mof_name": rec.get("name", formula_adi),
                                      "source": "HF_COREMOF_GCMC",
                                      "formation_energy_eV_atom_proxy": proxy_ev})
                except Exception:
                    continue

    if DATA_SOURCE in ("auto", "core_mof") and len(kayitlar) < max_n:
        core_df = core_mof_dene(max_n - len(kayitlar))
        if core_df is not None:
            print("  [CoRE-MOF] NOT: acik ozet CSV CIF/yapi metni ICERMEZ (sadece hesaplanmis "
                  "gozeneklilik ozellikleri) - bu satirlar PROSEDUREL yapi uretimiyle eslestirilir, "
                  "gozeneklilik SUTUNLARI ise (varsa) CoRE-MOF'un KENDI degerleriyle DOLDURULUR.")
            for _, row in core_df.iterrows():
                kayitlar.append({"struct": None, "mof_name": str(row.get("filename", row.get("name", "CoRE_MOF"))),
                                  "source": "COREMOF_CSV", "core_mof_row": row.to_dict()})

    if len(kayitlar) < max_n:
        eksik = max_n - len(kayitlar)
        print(f"  [PROCEDURAL] {eksik} yapi, {len(ARKETIPLER)} bilinen MOF ailesinden turetiliyor "
              f"(bkz. modul dokstringi 'PROSEDUREL YEDEK').")
        arketip_adlari = list(ARKETIPLER.keys())
        for i in range(eksik):
            adi = arketip_adlari[i % len(arketip_adlari)]
            struct = _procedural_yapi(adi, ARKETIPLER[adi], rng)
            kayitlar.append({"struct": struct, "mof_name": f"{adi}-var{i:05d}",
                              "source": "PROCEDURAL_FALLBACK",
                              "open_metal_site": ARKETIPLER[adi].get("open_metal_site", False),
                              "has_functional_group": ARKETIPLER[adi].get("functional", False)})

    print(f"[{havuz_adi}] Toplam {len(kayitlar)} ham kayit (HF basarili mi: {hf_basarili}).")

    CIF_DIR.mkdir(parents=True, exist_ok=True)
    satirlar = []
    for i, rec in enumerate(kayitlar):
        struct = rec.get("struct")
        if struct is None:
            # CoRE-MOF (yapisiz) satirlari icin de gecerli bir yapi olmasi gerektiginden
            # (grafik encoder'lari CIF okur), en yakin procedural arketiple doldurulur.
            adi = list(ARKETIPLER.keys())[i % len(ARKETIPLER)]
            struct = _procedural_yapi(adi, ARKETIPLER[adi], rng)
            rec["source"] = rec.get("source", "COREMOF_CSV") + "+PROCEDURAL_STRUCTURE"

        base_mof_id = f"{havuz_adi}_{i:06d}"
        cif_path = CIF_DIR / f"{base_mof_id}.cif"
        try:
            struct.to(filename=str(cif_path), fmt="cif")
        except Exception as e:
            print(f"  [ATLANDI] {base_mof_id}: CIF yazilamadi ({e})")
            continue

        pore = estimate_pore_proxies(struct)
        comp = kompozisyon_ozellikleri(struct)
        core_row = rec.get("core_mof_row", {})

        satir = {
            "base_mof_id": base_mof_id, "mof_name": rec.get("mof_name", base_mof_id),
            "source": rec["source"], "cif_path": str(cif_path),
            "formula": struct.composition.reduced_formula,
            "open_metal_site": rec.get("open_metal_site", False),
            "has_functional_group": rec.get("has_functional_group", False),
            "formation_energy_eV_atom_proxy": rec.get("formation_energy_eV_atom_proxy"),
            **pore, **comp,
        }
        # CoRE-MOF gercek gozeneklilik sutunlari varsa (isim eslesmesi kaba/en-iyi-caba,
        # kaynak CSV semasi degisebilir) proxy geometrik tahminin YERINE gecer.
        for hedef_kol, aday_isimler in [
            ("void_fraction", ["Void Fraction", "void_fraction", "VF"]),
            ("pore_volume_cm3_g", ["Pore Volume [cm3/g]", "pore_volume_cm3_g"]),
            ("gravimetric_surface_area_m2_g", ["Gravimetric Surface Area [m2/g]", "ASA_m2_g"]),
            ("lcd_A", ["LCD", "lcd_A", "Largest Cavity Diameter"]),
            ("pld_A", ["PLD", "pld_A", "Pore Limiting Diameter"]),
        ]:
            for aday in aday_isimler:
                if aday in core_row and pd.notna(core_row[aday]):
                    satir[hedef_kol] = core_row[aday]
                    break
        satirlar.append(satir)

        if (i + 1) % 500 == 0 or (i + 1) == len(kayitlar):
            print(f"    {i + 1}/{len(kayitlar)} islendi")

    return pd.DataFrame(satirlar)


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("veri_indirici_1_jarvis_core_mof.py — Bilesen 2: Veritabani/API Kazima")
    print("=" * 70)

    finetune_df = mof_havuzu_olustur(MAX_MATERIALS, "finetune", RNG_SEED)
    finetune_df.to_csv(RAW_MOF_CSV, index=False)
    print(f"\n[FINETUNE HAVUZU] Kaydedildi -> {RAW_MOF_CSV.resolve()} ({len(finetune_df)} satir)")

    pretrain_df = mof_havuzu_olustur(MAX_MATERIALS_PRETRAIN, "pretrain", RNG_SEED + 1)
    # ON-EGITIM icin formation_energy_eV_atom_proxy hicbir kaynaktan gelmediyse
    # (procedural/CoRE-MOF yapilarinin cogunda gercek DFT degeri YOKTUR),
    # gozeneklilik+kompozisyondan basit bir fiziksel-motivasyonlu VEKIL (proxy)
    # deger turetilir: daha yogun/az-gozenekli/agir-metalli cerceveler
    # GENELLIKLE daha negatif (kararli) olusum enerjisine sahiptir (kaba egilim,
    # GERCEK DFT DEGILDIR - 'eV_atom_proxy' adlandirmasi bunu ACIKCA belirtir).
    eksik = pretrain_df["formation_energy_eV_atom_proxy"].isna()
    if eksik.any():
        turetilmis = (
            -0.35 - 0.015 * pretrain_df.loc[eksik, "mean_atomic_number"]
            - 1.2 * (1.0 - pretrain_df.loc[eksik, "void_fraction"])
            + np.random.default_rng(7).normal(0, 0.08, eksik.sum())
        )
        pretrain_df.loc[eksik, "formation_energy_eV_atom_proxy"] = turetilmis
        pretrain_df.loc[eksik, "formation_energy_kaynak"] = "TURETILMIS_PROXY"
    pretrain_df.loc[~eksik, "formation_energy_kaynak"] = "KAYNAK_VERISI"
    pretrain_df.to_csv(QMOF_PRETRAIN_RAW_CSV, index=False)
    print(f"[PRETRAIN HAVUZU] Kaydedildi -> {QMOF_PRETRAIN_RAW_CSV.resolve()} ({len(pretrain_df)} satir)")

    meta = {
        "olusturma_zamani_utc": datetime.now(timezone.utc).isoformat(),
        "finetune_satir_sayisi": int(len(finetune_df)),
        "pretrain_satir_sayisi": int(len(pretrain_df)),
        "finetune_kaynak_dagilimi": finetune_df["source"].value_counts().to_dict(),
        "pretrain_kaynak_dagilimi": pretrain_df["source"].value_counts().to_dict(),
        "data_source_ayari": DATA_SOURCE,
    }
    RAW_MOF_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaynak bilgisi -> {RAW_MOF_META.resolve()}")
    print("\nKaynak dagilimi (finetune):")
    print(finetune_df["source"].value_counts().to_string())
    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
