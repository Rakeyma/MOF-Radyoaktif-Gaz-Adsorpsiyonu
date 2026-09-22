"""
ortak_ozellikler.py
=====================
AMAC:
    Butun model klasorlerinin (GraphGPS/, GIN/, GAT/, ..., DimeNetPP/) ayni
    sekilde kullandigi, MOF gozeneklilik/kompozisyon YARDIMCI (auxiliary)
    sayisal ozellik vektorunu TEK bir yerden hazirlamak - "Üç Boyutlu
    Kristal Malzemeler/ortak_ozellikler.py" ile BIREBIR AYNI mimari ilke,
    sadece ozellik listesi MOF/gozenekli-malzeme alanlarina uyarlanmistir.

NEDEN BU OZELLIKLER (fiziksel gerekce):
    Xe/Kr/I2 adsorpsiyon kapasitesi VE secicilik (selectivity), literatürde
    (Sikora et al. 2012; Simon et al. 2015 "Materials Genome in Action:
    High-throughput Screening of MOFs for Xe/Kr separation") BASLICA
    GOZENEK GEOMETRISI (gozeneklilik/void fraction, LCD/PLD, yuzey alani)
    VE metal-linker kimyasi (acik metal bolgeleri, fonksiyonel gruplar) ile
    ACIKLANIR - bu yuzden AUX vektoru bu buyuklukleri (grafik encoder'in
    ATOMISTIK/3B geometriden ogrendigi temsili TAMAMLAYAN, hazir-hesaplanmis
    makroskopik gozeneklilik bilgisi olarak) icerir.

NEDEN MERKEZI BIR MODUL:
    Eksik deger (NaN) ISTATISTIKLERI (ortalama/std) SADECE o fold'un TRAIN
    kismindan hesaplanmalidir (sizinti/leakage olmasin diye) - 11 model +
    4 XAI dosyasinda bu mantigi ayri ayri yazmak yerine TEK bir fonksiyonda
    standartlastirilmistir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MOF gozeneklilik (Zeo++/pymatgen tarzi) + kompozisyon ozellikleri - HEDEF
# (Xe/Kr/I2 kapasitesi, Xe/Kr secicilik) DISINDA, model girdisi olarak
# kullanilacak sayisal/mantiksal ozellikler.
AUX_FEATURE_COLUMNS = [
    "pore_volume_cm3_g",              # cm^3/g - Zeo++ tarzi gravimetrik gozenek hacmi
    "void_fraction",                  # 0-1 - hesaplanan (helyum-prob) gozeneklilik orani
    "gravimetric_surface_area_m2_g",  # m^2/g - BET-benzeri gravimetrik yuzey alani
    "volumetric_surface_area_m2_cm3", # m^2/cm^3 - hacimsel yuzey alani
    "lcd_A",                          # Angstrom - En Buyuk Kavite Capi (Largest Cavity Diameter)
    "pld_A",                          # Angstrom - Gozenek-Sinirlayici Cap (Pore Limiting Diameter)
    "density_g_cm3",                  # g/cm^3 - cerceve yogunlugu
    "nsites",                         # birim hucredeki atom sayisi
    "nelements",                      # benzersiz element sayisi
    "open_metal_site",                # bool -> 0/1 - acik metal bolgesi var mi (guclu Xe/I2 baglanma)
    "has_functional_group",           # bool -> 0/1 - linker fonksiyonlastirilmis mi (-CH3/-NH2, veri artirmadan)
    "mean_electronegativity_diff",    # kompozisyondan (Pauling)
    "mean_atomic_radius_A",           # kompozisyondan (pymatgen)
    "mean_atomic_number",             # kompozisyondan (pymatgen)
    "metal_fraction",                 # metal atomlarinin toplam atoma orani (0-1)
]

AUX_DIM = len(AUX_FEATURE_COLUMNS)

_IKILI_ESLEME = {
    "Yes": 1.0, "No": 0.0, "True": 1.0, "False": 0.0,
    True: 1.0, False: 0.0, "yes": 1.0, "no": 0.0,
    "true": 1.0, "false": 0.0, 1: 1.0, 0: 0.0,
}

BOOL_KOLONLAR = ("open_metal_site", "has_functional_group")


def _ikili_sutunu_sayisallastir(seri: pd.Series) -> pd.Series:
    """'open_metal_site'/'has_functional_group' gibi True/False (ya da
    Yes/No) sutunlari 0.0/1.0'a cevirir; taninmayan/bos degerler NaN kalir
    (sonradan AuxOlcekleyici tarafindan train-ortalamasiyla doldurulur)."""
    if seri.dtype == bool:
        return seri.astype(float)
    return seri.map(_IKILI_ESLEME).astype(float)


def aux_ham_matris(df: pd.DataFrame) -> np.ndarray:
    """AUX_FEATURE_COLUMNS'u df'den ham (standardize edilmemis, NaN
    icerebilen) numpy dizisine cevirir. Satir sirasi df ile birebir aynidir.
    Sekil: (n_satir, AUX_DIM)."""
    parcalar = []
    for c in AUX_FEATURE_COLUMNS:
        seri = df[c]
        if c in BOOL_KOLONLAR:
            seri = _ikili_sutunu_sayisallastir(seri)
        else:
            seri = pd.to_numeric(seri, errors="coerce")
        parcalar.append(seri.values.astype(np.float64))
    return np.stack(parcalar, axis=1)


class AuxOlcekleyici:
    """TRAIN foldundan ogrenilen ortalama/std ile standardize eder ve eksik
    degerleri TRAIN ortalamasiyla doldurur (standardize sonrasi 0.0 olur -
    yani 'bilgi yok' -> notr girdi). fit() SADECE train verisiyle
    cagrilmalidir (fold'lar arasi/test'e sizinti olmamasi icin)."""

    def __init__(self) -> None:
        self.ortalama: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, train_ham: np.ndarray) -> "AuxOlcekleyici":
        with np.errstate(invalid="ignore"):
            ortalama = np.nanmean(train_ham, axis=0)
            std = np.nanstd(train_ham, axis=0)
        # TUMU NaN olan bir sutun icin (orn. bir kaynak MOF veritabaninda
        # LCD/PLD hic gelmeyebilir) nanmean/nanstd DE NaN doner - bu,
        # transform()'da NaN'i "TRAIN ortalamasiyla doldur" adimini bozup
        # NaN'in tum modele SIZMASINA yol acardi. Boyle sutunlar icin
        # ortalama=0.0/std=1.0 (= "hicbir bilgi yok, notr girdi").
        ortalama[~np.isfinite(ortalama)] = 0.0
        std[~np.isfinite(std) | (std < 1e-8)] = 1.0
        self.ortalama = ortalama
        self.std = std
        return self

    def transform(self, ham: np.ndarray) -> np.ndarray:
        doldurulmus = np.where(np.isnan(ham), self.ortalama, ham)
        return (doldurulmus - self.ortalama) / self.std

    def fit_transform(self, train_ham: np.ndarray) -> np.ndarray:
        return self.fit(train_ham).transform(train_ham)


# permutation-importance / GraphLIME gruplandirmasi icin ortak kategori
# haritasi (11 modelin grafik.py'si VE XAI scriptleri bu sozlugu import eder).
AUX_GRUPLARI = {
    "Pore Geometry": ["pore_volume_cm3_g", "void_fraction", "lcd_A", "pld_A"],
    "Surface Area": ["gravimetric_surface_area_m2_g", "volumetric_surface_area_m2_cm3"],
    "Structural / Size": ["density_g_cm3", "nsites", "nelements", "metal_fraction"],
    "Chemical Modification": ["open_metal_site", "has_functional_group"],
    "Composition-Derived": ["mean_electronegativity_diff", "mean_atomic_radius_A", "mean_atomic_number"],
}
AUX_GRUP_IDX = {ad: [AUX_FEATURE_COLUMNS.index(c) for c in kolonlar] for ad, kolonlar in AUX_GRUPLARI.items()}
