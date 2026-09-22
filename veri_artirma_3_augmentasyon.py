"""
veri_artirma_3_augmentasyon.py
=================================
BILESEN 4 — Veri Artirma (Structural Data Augmentation)
=============================================================
AMAC:
    veri_indirici_1_jarvis_core_mof.py'nin ürettiği TEMEL MOF havuzunu
    (data/raw/mof_ham_veri.csv, tipik olarak birkaç yuz yapi), pymatgen
    kullanarak FIZIKSEL OLARAK ANLAMLI 4 tur yapisal degisiklik uygulayip
    BIN(LER)CE ornege genisletmek:

    (a) EKSIK-BAGLAYICI KUSURU (missing-linker defect): linker (metal
        OLMAYAN) atomlarin rastgele bir kesrini (defect_fraction) siler -
        gercek MOF sentezinde YAYGIN gozlenen bir kusur turudur (Bennett &
        Cheetham 2014, "Defective MOFs: Chemistry & Applications") ve
        GOZENEK HACMINI/erisebilirligini DOGRUDAN etkileyerek gaz
        adsorpsiyonunu degistirir.
    (b) METAL DUGUM IKAMESI (metal node substitution): merkezi metal
        atomunu kimyasal olarak BENZER bir metalle degistirir (orn.
        Cu<->Ni<->Co, Zn<->Mg<->Cd, Zr<->Hf) - izoretiküler MOF
        kimyasinda bilinen bir strateji, farkli acik-metal-bolgesi
        elektronik/boyut ozellikleri uretir.
    (c) FONKSIYONEL GRUP EKLENMESI (-CH3 / -NH2): rastgele secilen linker
        karbon atomlarina bir sarkan (pendant) substituent atomu (basit
        C-proxy veya N-proxy, bkz. asagida) eklenir - IRMOF-3/UiO-66-NH2
        gibi GERCEK fonksiyonlastirilmis MOF ailelerinin (Component 2'deki
        ARKETIPLER'de zaten temsil edilen) yapisal mantigini TÜM veri
        setine genelleyecek sekilde uygular; fonksiyonel gruplar gozenek
        hacmini DARALTIR ve kimyasal secicilikte rol oynar.
    (d) 3B UZAMSAL GAUSSIAN GURULTU: tum atom kartezyen konumlarina kucuk
        (sigma=0.03-0.15 A, kovalent bag uzunlugundan KUCUK) rastgele
        yer degistirme eklenir - sentez/isil bozukluk (thermal disorder)
        etkisinin BASIT bir simulasyonu, modelin KESIN/idealize
        koordinatlara asiri-uyumunu (overfitting) azaltir.

    NOT (basitlestirme seffafligi): -CH3/-NH2 eklemesi TAM bir molekuler-
    mekanik optimizasyonla (H atomlari dahil tam geometri) DEGIL, TEK bir
    "substituent-proxy" agir atomuyla (C icin metil, N icin amino)
    temsil edilir - Component 2'deki procedural-yapi basitlestirme
    seviyesiyle TUTARLI bir tasarim tercihi (amac DFT-hassasiyetinde
    kimya degil, GNN'in ogrenebilecegi GERCEKCI-OLCEKTE yapisal
    varyasyon uretmektir).

    HER augment edilmis ornek KENDI base_mof_id'sini KORUR (egitim_ortak.
    GROUP_COLUMN) - boylece K-fold bolme, AYNI temel MOF'tan turetilen
    TUM varyantlari AYNI fold'da tutar (sizinti/leakage onlenir).

CIKTI:
    data/processed/augmented_cif/<sample_id>.cif
    data/processed/augmentasyon_manifest.csv

CALISTIRMA:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python veri_artirma_3_augmentasyon.py

AYARLAR (ortam degiskenleri):
    N_AUGMENT_PER_BASE   Temel MOF basina kac EK (augmented) varyant (varsayilan 6
                          -> ~600 temel MOF x (1 orijinal + 6 varyant) = ~4200 ornek)
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from paths import RAW_MOF_CSV, AUGMENTED_CIF_DIR, AUGMENTATION_MANIFEST_CSV, DATA_PROCESSED

try:
    from veri_indirici_1_jarvis_core_mof import estimate_pore_proxies, kompozisyon_ozellikleri
except ImportError:
    sys.exit("HATA: kok dizinden calistirin: python veri_artirma_3_augmentasyon.py")

N_AUGMENT_PER_BASE = int(os.environ.get("N_AUGMENT_PER_BASE", "6"))
SEED = 42

METAL_IKAME_GRUPLARI = {
    "Cu": ["Ni", "Co", "Zn"], "Zn": ["Mg", "Co", "Cd"], "Zr": ["Hf", "Ce"],
    "Mg": ["Ni", "Co", "Zn"], "Ni": ["Co", "Cu", "Zn"], "Cr": ["Fe", "Al"],
    "Al": ["Cr", "Fe", "Ga"], "Co": ["Ni", "Zn", "Mn"],
}
FONKSIYONEL_GRUP_PROXY = {"-CH3": "C", "-NH2": "N"}
DEFECT_FRAC_ARALIGI = (0.03, 0.18)
NOISE_SIGMA_ARALIGI_A = (0.03, 0.15)


def _metal_z_bul(struct) -> int | None:
    """Yapidaki EN AGIR (Z en buyuk) atomu 'metal dugum' kabul eder - bu
    projedeki TUM arketip/procedural yapilarda dogru varsayimdir (linkerler
    C/N/O/H, metaller HER ZAMAN daha agirdir)."""
    zler = [site.specie.Z for site in struct]
    return max(zler) if zler else None


def eksik_baglayici_kusuru(struct, rng: np.random.Generator, metal_z: int) -> tuple:
    from pymatgen.core import Structure
    frac = rng.uniform(*DEFECT_FRAC_ARALIGI)
    linker_idx = [i for i, site in enumerate(struct) if site.specie.Z != metal_z]
    n_sil = int(len(linker_idx) * frac)
    n_sil = min(n_sil, max(len(linker_idx) - 3, 0))  # en az birkac linker atomu kalsin
    silinecek = set(rng.choice(linker_idx, size=n_sil, replace=False)) if n_sil > 0 else set()

    species, coords = [], []
    for i, site in enumerate(struct):
        if i in silinecek:
            continue
        species.append(site.specie.Z)
        coords.append(site.frac_coords)
    yeni = Structure(struct.lattice, species, coords, coords_are_cartesian=False)
    return yeni, frac, n_sil


def metal_dugum_ikamesi(struct, rng: np.random.Generator, metal_z: int) -> tuple:
    from pymatgen.core import Structure, Element
    metal_sembol = Element.from_Z(metal_z).symbol
    adaylar = METAL_IKAME_GRUPLARI.get(metal_sembol)
    if not adaylar:
        return struct, None
    yeni_sembol = rng.choice(adaylar)
    yeni_z = Element(yeni_sembol).Z

    species, coords = [], []
    for site in struct:
        z = yeni_z if site.specie.Z == metal_z else site.specie.Z
        species.append(z)
        coords.append(site.frac_coords)
    yeni = Structure(struct.lattice, species, coords, coords_are_cartesian=False)
    return yeni, yeni_sembol


def fonksiyonel_grup_ekle(struct, rng: np.random.Generator, metal_z: int) -> tuple:
    from pymatgen.core import Structure, Element
    grup = rng.choice(list(FONKSIYONEL_GRUP_PROXY.keys()))
    proxy_sembol = FONKSIYONEL_GRUP_PROXY[grup]
    proxy_z = Element(proxy_sembol).Z

    karbon_idx = [i for i, site in enumerate(struct) if site.specie.Z == 6]
    if not karbon_idx:
        return struct, None
    n_ekle = max(1, len(karbon_idx) // 6)
    hedefler = rng.choice(karbon_idx, size=min(n_ekle, len(karbon_idx)), replace=False)

    species = [site.specie.Z for site in struct]
    coords_cart = [site.coords for site in struct]
    for h in hedefler:
        yon = rng.normal(0, 1, 3)
        yon = yon / (np.linalg.norm(yon) + 1e-9)
        yeni_konum = struct[h].coords + yon * 1.5  # ~1.5A sarkan substituent bagi
        species.append(proxy_z)
        coords_cart.append(yeni_konum)

    frac = [struct.lattice.get_fractional_coords(c) % 1.0 for c in coords_cart]
    yeni = Structure(struct.lattice, species, frac, coords_are_cartesian=False)
    return yeni, grup


def gaussian_gurultu_ekle(struct, rng: np.random.Generator) -> tuple:
    from pymatgen.core import Structure
    sigma = rng.uniform(*NOISE_SIGMA_ARALIGI_A)
    species = [site.specie.Z for site in struct]
    coords_cart = [site.coords + rng.normal(0, sigma, 3) for site in struct]
    frac = [struct.lattice.get_fractional_coords(c) % 1.0 for c in coords_cart]
    yeni = Structure(struct.lattice, species, frac, coords_are_cartesian=False)
    return yeni, sigma


def bir_varyant_uret(struct, rng: np.random.Generator) -> dict:
    """Rastgele 1-3 augmentasyon TURUNU (defect/substitution/functional/noise)
    BIRLESTIRIR (gercek sentezde birden fazla kusur/degisim BIR ARADA
    bulunabilir) ve uygulanan islemleri metadata olarak dondurur."""
    from pymatgen.core import Structure

    metal_z = _metal_z_bul(struct)
    islemler = list(rng.choice(
        ["defect", "substitution", "functional", "noise"],
        size=rng.integers(1, 4), replace=False,
    ))
    if "noise" not in islemler:  # gurultu HER varyantta uygulanir (gercekci minimum varyasyon)
        islemler.append("noise")

    meta = {"defect_fraction": 0.0, "n_removed": 0, "metal_substituted_to": None,
            "functional_group": None, "noise_sigma_A": 0.0, "augmentation_ops": ",".join(islemler)}

    for op in islemler:
        if op == "defect" and metal_z is not None:
            struct, frac, n_sil = eksik_baglayici_kusuru(struct, rng, metal_z)
            meta["defect_fraction"], meta["n_removed"] = frac, n_sil
        elif op == "substitution" and metal_z is not None:
            struct, yeni_sembol = metal_dugum_ikamesi(struct, rng, metal_z)
            meta["metal_substituted_to"] = yeni_sembol
            metal_z = None if yeni_sembol is None else metal_z  # ikame basarisizsa eski z ile devam
        elif op == "functional":
            struct, grup = fonksiyonel_grup_ekle(struct, rng, metal_z)
            meta["functional_group"] = grup
        elif op == "noise":
            struct, sigma = gaussian_gurultu_ekle(struct, rng)
            meta["noise_sigma_A"] = sigma

    return {"struct": struct, **meta}


def main() -> None:
    if not RAW_MOF_CSV.exists():
        sys.exit(f"HATA: '{RAW_MOF_CSV}' bulunamadi. Once veri_indirici_1_jarvis_core_mof.py calistirilmali.")

    from pymatgen.core import Structure

    print("=" * 70)
    print("veri_artirma_3_augmentasyon.py — Bilesen 4: Veri Artirma")
    print("=" * 70)

    base_df = pd.read_csv(RAW_MOF_CSV)
    print(f"Temel MOF havuzu: {len(base_df)} yapi, augment/temel = {N_AUGMENT_PER_BASE} "
          f"-> hedef toplam ~{len(base_df) * (1 + N_AUGMENT_PER_BASE)} ornek")

    AUGMENTED_CIF_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    kayitlar = []

    for bi, (_, row) in enumerate(base_df.iterrows()):
        try:
            struct0 = Structure.from_file(row["cif_path"])
        except Exception as e:
            print(f"  [ATLANDI] {row['base_mof_id']}: CIF okunamadi ({e})")
            continue

        # --- varyant 0: ORIJINAL (augment edilmemis) yapi da veri setine dahildir ---
        varyantlar = [{"struct": struct0, "defect_fraction": 0.0, "n_removed": 0,
                       "metal_substituted_to": None, "functional_group": None,
                       "noise_sigma_A": 0.0, "augmentation_ops": "none"}]
        for _ in range(N_AUGMENT_PER_BASE):
            varyantlar.append(bir_varyant_uret(struct0, rng))

        for vi, v in enumerate(varyantlar):
            sample_id = f"{row['base_mof_id']}_v{vi:02d}"
            cif_path = AUGMENTED_CIF_DIR / f"{sample_id}.cif"
            try:
                v["struct"].to(filename=str(cif_path), fmt="cif")
            except Exception as e:
                print(f"  [ATLANDI] {sample_id}: CIF yazilamadi ({e})")
                continue

            pore = estimate_pore_proxies(v["struct"])
            comp = kompozisyon_ozellikleri(v["struct"])
            kayitlar.append({
                "sample_id": sample_id, "base_mof_id": row["base_mof_id"],
                "mof_name": row["mof_name"], "source": row["source"],
                "cif_path": str(cif_path), "formula": v["struct"].composition.reduced_formula,
                "is_augmented": vi > 0,
                "defect_fraction": v["defect_fraction"], "n_removed": v["n_removed"],
                "metal_substituted_to": v["metal_substituted_to"],
                "functional_group": v["functional_group"], "noise_sigma_A": v["noise_sigma_A"],
                "augmentation_ops": v["augmentation_ops"],
                "open_metal_site": row.get("open_metal_site", False) and v["metal_substituted_to"] is None,
                "has_functional_group": bool(row.get("has_functional_group", False) or v["functional_group"]),
                **pore, **comp,
            })

        if (bi + 1) % 100 == 0 or (bi + 1) == len(base_df):
            print(f"  {bi + 1}/{len(base_df)} temel MOF islendi ({len(kayitlar)} toplam ornek uretildi)")

    manifest = pd.DataFrame(kayitlar)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(AUGMENTATION_MANIFEST_CSV, index=False)

    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)
    print(f"Toplam ornek       : {len(manifest)}")
    print(f"  - Orijinal       : {(~manifest['is_augmented']).sum()}")
    print(f"  - Augment edilmis: {manifest['is_augmented'].sum()}")
    print(f"Benzersiz temel MOF: {manifest['base_mof_id'].nunique()}")
    print("\nUygulanan islem dagilimi:")
    print(manifest["augmentation_ops"].value_counts().to_string())
    print(f"\nKaydedildi -> {AUGMENTATION_MANIFEST_CSV.resolve()}")

    meta = {
        "n_augment_per_base": N_AUGMENT_PER_BASE, "toplam_ornek": int(len(manifest)),
        "benzersiz_temel_mof": int(manifest["base_mof_id"].nunique()),
        "augmentation_ops_dagilimi": manifest["augmentation_ops"].value_counts().to_dict(),
    }
    (DATA_PROCESSED / "augmentasyon_bilgisi.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
