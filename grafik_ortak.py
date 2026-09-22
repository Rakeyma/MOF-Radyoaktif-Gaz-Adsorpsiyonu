"""
grafik_ortak.py
======================
AMAC:
    Her model klasorundeki `grafik.py` tarafindan kullanilan, modele-OZGU
    OLMAYAN grafik fonksiyonlarini (predicted vs true, residual distribution,
    training-loss curves, confusion matrix, pore-geometry/selectivity
    consistency) TEK bir yerden saglamak - 11 model + 4 XAI klasorunde ayni
    kodu tekrar etmemek icin.

    STIL: "Üç Boyutlu Kristal Malzemeler/grafik_ortak.py" ile BIREBIR AYNI
    (rcParams, dpi=600, panel etiketleme, .tif+.png ikili kayit, TÜM METIN
    INGILIZCE VE BOLD) - KULLANICI GEREKSINIMI OLARAK SERTCE (hardcoded)
    UYGULANIR, hicbir grafik fonksiyonu bu ayarlari degistiremez:
        - savefig.dpi = 600 (TÜM .tif VE .png kayıtları)
        - format='tiff' (.tif, ana/asil format)
        - font.weight='bold', axes.labelweight='bold', axes.titleweight='bold'
        - TÜM eksen/başlık/lejant metinleri İNGİLİZCE

    Feature importance (permutation importance) grafigi de burada - 11
    model AYNI arayuze (encoder + aux vektoru) sahip oldugundan (egitim_ortak.py
    sayesinde), permutation importance hesabi TEK bir yerde (egitim_ortak.py
    icinde) yapilir; bu dosya sadece SONUCU cizer.

DEGISKENLIK: HEDEF-BASINA (per-target) GRAFIKLER
    "Üç Boyutlu Kristal Malzemeler" TEK bir skaler hedef (formation energy)
    icin sabit bir eksen etiketi kullanirdi. Bu projede 4 hedef (Xe/Kr/I2
    kapasitesi + Xe/Kr secicilik) oldugundan, gercek_vs_tahmin/residual/
    confusion_matrix fonksiyonlari bir 'kol' (hedef sutun adi) parametresi
    alir ve DISPLAY_LABELS/TARGET_UNITS sozluklerinden dogru eksen etiketini
    ve birimi otomatik secer (egitim_ortak.tum_grafikleri_uret 4 hedefin
    HER BIRI icin bu fonksiyonlari ayri ayri cagirir).

FIZIKSEL TUTARLILIK TESTI - "Üç Boyutlu Kristal Malzemeler"deki termodinamik
tutarlilik grafiginin bu projedeki analogu:
    O projede Δχ (elektronegatiflik farki) ile formation energy arasindaki
    BEKLENEN ISARETLI korelasyon test ediliyordu. Burada, gozeneklilik
    literaturunun (Sikora et al. 2012) TEMEL BULGUSU test edilir: Gozenek-
    Sinirlayici Cap (PLD), hedef gazin kinetik capina (Xe: 4.0-4.4 A, Kr:
    3.6-3.8 A) NE KADAR YAKINSA, boyut-elemesi (size-sieving) o kadar
    GUCLUDUR ve Xe/Kr secicilik o kadar YUKSEK olma egilimindedir - yani
    |PLD - d_kinetik(Xe)| ile xe_kr_selectivity arasinda BEKLENEN egilim
    NEGATIFTIR (kucuk fark = yuksek secicilik). Bu fonksiyon, gercek VE
    tahmin edilen secicilik degerlerinin bu egilimi AYNI YONDE yakalayip
    yakalamadigini kontrol eder.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Once seaborn temasi (izgara/palet/arka plan) uygulanir, SONRA asagidaki
# rcParams.update() ile eksen-yazisi/DPI ayarlari ONUN UZERINE yazilir.
sns.set_theme(style="whitegrid", palette="deep")

# Eksen yazılarının büyük ve BOLD olması için global ayarlar (600 dpi .tif +
# .png ikili kayıt, İngilizce eksen/metin - tüm figürler bu ayarları miras
# alır; KULLANICI GEREKSİNİMİ - bu bloğa dokunulmamalıdır).
plt.rcParams.update({
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'font.weight': 'bold',
    'font.family': 'DejaVu Sans',
    'savefig.dpi': 600,
    'figure.dpi': 150,
})

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Hedef sutun adi -> insan-okunur (Ingilizce, bold-uyumlu LaTeX alt-simgeli) etiket
DISPLAY_LABELS = {
    "xe_uptake_mmol_g": "Xe Uptake (mmol/g)",
    "kr_uptake_mmol_g": "Kr Uptake (mmol/g)",
    "xe_kr_selectivity": "Xe/Kr Selectivity (-)",
    "i2_uptake_mmol_g": r"I$_2$ Uptake (mmol/g)",
}

# KISA versiyon - uzun eksen basliklari (orn. "Mean |Integrated Gradients|
# (I$_2$ Uptake (mmol/g))") ic ice/okunmaz hale geldiginde (kullanici geri
# bildirimi) kullanilir - XAI grafiklerinin y/x eksen basliklarinda tercih
# edilir.
KISA_ETIKETLER = {
    "xe_uptake_mmol_g": "Xe (mmol/g)",
    "kr_uptake_mmol_g": "Kr (mmol/g)",
    "xe_kr_selectivity": "Xe/Kr Sel.",
    "i2_uptake_mmol_g": r"I$_2$ (mmol/g)",
}


def _etiket(kol: str) -> str:
    return DISPLAY_LABELS.get(kol, kol)


def olcekle_ve_etiketle(degerler: np.ndarray, taban_etiket: str) -> tuple[np.ndarray, str]:
    """PAYLASILAN duzeltme (kullanici geri bildirimi): matplotlib'in
    `ticklabel_format(style="sci")` ile urettigi OTOMATIK 'offset metni'
    (orn. "x10^-2"), AYRI/KAYAN bir metin kutusu olarak eksenin kosesine
    yerlestirilir - UZUN eksen basliklariyla (orn. "Residuals (Predicted -
    True), I2 Uptake (mmol/g)") CAKISIP BOZUK/OKUNAMAZ gorunebiliyordu
    (bkz. kullanici geri bildirimi - "mmol uzeri bir sey... bozuk yazilmis").

    BU FONKSIYON O AYRI/KAYAN METIN KUTUSUNU TAMAMEN ORTADAN KALDIRIR:
    degerleri UYGUN bir 10^k ile YENIDEN OLCEKLER ve olcek carpanini
    DOGRUDAN eksen basliginin ICINE (TEK bir metin parcasi olarak) gomer -
    boylece cakisabilecek IKINCI bir metin ogesi hic OLUSTURULMAZ.
    Dondurur: (yeniden_olceklenmis_degerler, yeni_baslik)."""
    degerler = np.asarray(degerler, dtype=float)
    maxabs = float(np.max(np.abs(degerler))) if degerler.size else 0.0
    if maxabs == 0.0 or 1e-2 <= maxabs < 1e2:
        return degerler, taban_etiket  # zaten okunakli olcekte, dokunma
    k = int(np.floor(np.log10(maxabs)))
    olcek = 10.0 ** k
    return degerler / olcek, f"{taban_etiket}  (×$\\mathdefault{{10^{{{k}}}}}$)"


def _sikitir_xlim(ax, degerler, eksen: str = "x", pay_orani: float = 0.15) -> None:
    """PAYLASILAN duzeltme (kullanici geri bildirimi - GraphLIME/Edge_
    Attribution/IntegratedGradients element-onem bar grafiklerinin HEPSINDE
    tekrarlanan sorun): eksen, GERCEK deger araliginin COK OTESINE (bos/
    anlamsiz araliklara) genisliyordu -> simdi SADECE gercek min/max (0
    dahil) etrafinda kucuk bir pay ile SIKI bir limit ayarlanir. NOT: olcek/
    bilimsel-gosterim ARTIK BURADA YAPILMAZ - cagiran taraf, cizimden ONCE
    `olcekle_ve_etiketle()` ile degerleri kendisi yeniden olceklemelidir
    (bkz. o fonksiyonun dokstringi - cakisan 'offset metni' sorununu onler)."""
    degerler = np.asarray(degerler, dtype=float)
    if degerler.size == 0:
        return
    v_min, v_max = float(min(0.0, degerler.min())), float(max(0.0, degerler.max()))
    yayilim = v_max - v_min
    if yayilim < 1e-12:
        return  # tum degerler pratikte sifir - varsayilan gorunume dokunma
    pay = yayilim * pay_orani
    if eksen == "x":
        ax.set_xlim(v_min - pay, v_max + pay)
    else:
        ax.set_ylim(v_min - pay, v_max + pay)


# ---------------------------------------------------------------------------
# PANEL ETİKETİ — (a), (b), ... modele özel, axes dışı sağ üst köşe
# ---------------------------------------------------------------------------
_current_panel: str = ""


def set_panel(etiket: str) -> None:
    global _current_panel
    _current_panel = etiket


def panel_ekle(fig: plt.Figure) -> None:
    if not _current_panel:
        return
    axs = fig.get_axes()
    if axs:
        axs[0].annotate(
            _current_panel,
            xy=(1, 1), xycoords="axes fraction",
            xytext=(0, 5), textcoords="offset points",
            ha="right", va="bottom",
            fontsize=14, fontweight="bold",
        )
    else:
        fig.text(0.98, 0.99, _current_panel,
                  ha="right", va="top", fontsize=14, fontweight="bold")


def _dinamik_esikler(degerler: np.ndarray) -> list[float]:
    """Q1/medyan/Q3 - veri-guduml sinif esikleri."""
    degerler = degerler[~np.isnan(degerler)]
    q1, q2, q3 = np.percentile(degerler, [25, 50, 75])
    return [float(q1), float(q2), float(q3)]


def _sinif_etiketleri(esikler: list[float]) -> list[str]:
    e1, e2, e3 = esikler
    return [f"< {e1:.2f}", f"{e1:.2f} to {e2:.2f}", f"{e2:.2f} to {e3:.2f}", f"> {e3:.2f}"]


def kaydet(fig: plt.Figure, cikti_yolu) -> None:
    """Her sekli HEM .tif (asil, yuksek kalite, 600 dpi) HEM .png (goruntuleme
    ve Word raporuna gomme icin daha kolay, ayni 600 dpi) olarak kaydeder -
    KULLANICI GEREKSINIMI: dpi=600 + format='tiff' HER ZAMAN sabit, hicbir
    cagrida degistirilemez."""
    from pathlib import Path as _Path

    cikti_yolu = _Path(cikti_yolu)
    fig.savefig(cikti_yolu, dpi=600, bbox_inches="tight", format="tiff")
    fig.savefig(cikti_yolu.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def tahmin_dogru_grafigi(df: pd.DataFrame, gercek_kolon: str, tahmin_kolon: str,
                          kol: str, baslik: str, cikti_yolu) -> None:
    alt = df.dropna(subset=[gercek_kolon, tahmin_kolon])
    gercek, tahmin = alt[gercek_kolon].values, alt[tahmin_kolon].values
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gercek, tahmin, s=10, alpha=0.35, edgecolors="none", color="steelblue")
    lim_min, lim_max = min(gercek.min(), tahmin.min()), max(gercek.max(), tahmin.max())
    pay = 0.05 * (lim_max - lim_min + 1e-9)
    ax.plot([lim_min - pay, lim_max + pay], [lim_min - pay, lim_max + pay],
            "r--", lw=1.5, label="y = x (Perfect Prediction)")
    ax.set_xlabel(f"True {_etiket(kol)}")
    ax.set_ylabel(f"Predicted {_etiket(kol)}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    panel_ekle(fig)
    kaydet(fig, cikti_yolu)


def ciz_fold_sacilim(df: pd.DataFrame, gercek_kolon: str, tahmin_kolon: str,
                      kol: str, baslik: str, cikti_yolu,
                      fold_kolon: str = "fold") -> None:
    """fold renkli scatter - tahmin_dogru_grafigi ile aynı figsize=(6,6) ölçeği.
    NaN (etiketsiz) satirlar bu hedef icin OTOMATIK atlanir (seyrek coklu-
    hedef etiketleme - bkz. egitim_ortak.maskeli_mse)."""
    from sklearn.metrics import r2_score, mean_absolute_error
    alt_all = df.dropna(subset=[gercek_kolon, tahmin_kolon])
    if len(alt_all) < 2:
        print(f"  [ATLANDI] '{kol}' icin yeterli etiketli ornek yok (n={len(alt_all)}).")
        return
    foldlar = sorted(alt_all[fold_kolon].unique()) if fold_kolon in alt_all.columns else [1]
    renkler = plt.cm.tab10(np.linspace(0, 0.9, len(foldlar)))
    gercek = alt_all[gercek_kolon].values
    tahmin = alt_all[tahmin_kolon].values

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, renk in zip(foldlar, renkler):
        alt = alt_all[alt_all[fold_kolon] == fold] if fold_kolon in alt_all.columns else alt_all
        ax.scatter(alt[gercek_kolon], alt[tahmin_kolon],
                   s=10, alpha=0.35, color=renk, label=f"Fold {fold}")
    vmin = min(gercek.min(), tahmin.min())
    vmax = max(gercek.max(), tahmin.max())
    margin = (vmax - vmin) * 0.07 + 1e-9
    lim = [vmin - margin, vmax + margin]
    ax.plot(lim, lim, "k--", lw=1.2, label="Perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    r2 = r2_score(gercek, tahmin) if np.std(gercek) > 1e-10 else float("nan")
    mae = mean_absolute_error(gercek, tahmin)
    ax.text(0.05, 0.93, f"R²={r2:.3f}, MAE={mae:.3f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xlabel(f"True {_etiket(kol)}")
    ax.set_ylabel(f"Predicted {_etiket(kol)}")
    ax.legend(markerscale=2, fontsize=9, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    panel_ekle(fig)
    kaydet(fig, cikti_yolu)


def residual_dagilim_grafigi(df: pd.DataFrame, gercek_kolon: str, tahmin_kolon: str,
                              kol: str, baslik: str, cikti_yolu) -> None:
    alt_df = df.dropna(subset=[gercek_kolon, tahmin_kolon])
    if len(alt_df) < 2:
        print(f"  [ATLANDI] '{kol}' residual dagilimi icin yeterli etiketli ornek yok.")
        return
    residual_ham = alt_df[tahmin_kolon].values - alt_df[gercek_kolon].values
    # Kullanici geri bildirimi ("mmol uzeri bir sey... bozuk yazilmis"):
    # matplotlib'in otomatik bilimsel-gosterim 'offset metni' UZUN eksen
    # basligiyla CAKISIYORDU - degerler ONCEDEN yeniden olceklenir ve olcek
    # DOGRUDAN basligin icine gomulur (bkz. olcekle_ve_etiketle dokstringi),
    # AYRI/cakisabilecek bir metin ogesi hic olusturulmaz.
    # Kullanici geri bildirimi: eksen basliklarindaki YONTEMSEL/aciklayici
    # ekler (orn. "(Predicted - True)" isaret-kurali notu) KALDIRILIR -
    # makale yazilirken bu aciklamalar metinde/dipnotta verilecek, grafik
    # SADECE buyukluk+birimi tasir.
    residual, x_etiket = olcekle_ve_etiketle(residual_ham, f"Residual, {_etiket(kol)}")
    p1, p99 = np.percentile(residual, 1), np.percentile(residual, 99)
    data = residual[(residual >= p1) & (residual <= p99)]
    if len(data) < 2:
        data = residual
    counts, bin_edges = np.histogram(data, bins=min(40, max(5, len(data))))
    rng = bin_edges[-1] - bin_edges[0] + 1e-9
    xlim_l = bin_edges[0] - rng * 0.18
    xlim_r = bin_edges[-1] + rng * 0.18
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(data, bins=bin_edges, color="steelblue", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlim(xlim_l, xlim_r)
    ax.set_xlabel(x_etiket)
    ax.set_ylabel("Frequency")
    # Kullanici istegi: en yuksek (ortadaki/tepe) sutunun degeri, sutunun
    # HEMEN USTUNDE yazili olsun - TUM sutunlari etiketlemek (40 sutuna
    # kadar) okunmaz/cakisik olurdu, bu yuzden sadece tepe (mod) sutunu
    # etiketlenir.
    tepe_idx = int(np.argmax(counts))
    if counts[tepe_idx] > 0:
        tepe_x = (bin_edges[tepe_idx] + bin_edges[tepe_idx + 1]) / 2
        ax.text(tepe_x, counts[tepe_idx], f"{int(counts[tepe_idx])}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(top=counts.max() * 1.12)
    ax.text(0.02, 0.95, f"mean={residual_ham.mean():.3g}\nstd={residual_ham.std():.3g}",
            transform=ax.transAxes, va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.tight_layout()
    panel_ekle(fig)
    kaydet(fig, cikti_yolu)


def confusion_matrix_grafigi(df: pd.DataFrame, gercek_kolon: str, tahmin_kolon: str,
                              kol: str, baslik: str, cikti_yolu) -> None:
    alt = df.dropna(subset=[gercek_kolon, tahmin_kolon])
    if len(alt) < 4:
        print(f"  [ATLANDI] '{kol}' confusion matrix icin yeterli etiketli ornek yok.")
        return
    esikler = _dinamik_esikler(alt[gercek_kolon].values)
    sinif_etiketleri = _sinif_etiketleri(esikler)
    y_true = np.digitize(alt[gercek_kolon].values, esikler)
    y_pred = np.digitize(alt[tahmin_kolon].values, esikler)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(sinif_etiketleri))))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=sinif_etiketleri, yticklabels=sinif_etiketleri,
                annot_kws={"fontsize": 11, "fontweight": "bold"}, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    ax.set_xlabel(f"Predicted Class ({_etiket(kol)})")
    ax.set_ylabel(f"True Class ({_etiket(kol)})")
    fig.tight_layout()
    panel_ekle(fig)
    kaydet(fig, cikti_yolu)


def gozeneklilik_secicilik_tutarlilik_grafigi(df: pd.DataFrame, etiket: str, baslik: str,
                                               cikti_yolu, pld_kolon: str = "pld_A",
                                               d_kinetik_xe: float = 4.1) -> bool:
    """Bkz. modul dokstringi - |PLD - d_kinetik(Xe)| (boyut-uyum farki) ile
    Xe/Kr secicilik arasindaki BEKLENEN NEGATIF egilimin gercek VE tahmin
    edilen degerlerde AYNI ISARETTE olup olmadigini kontrol eder ("Üç
    Boyutlu Kristal Malzemeler"deki termodinamik-tutarlilik testinin
    doğrudan analogu)."""
    from scipy.stats import pearsonr

    G, T = "gercek_xe_kr_selectivity", "tahmin_xe_kr_selectivity"
    if pld_kolon not in df.columns or G not in df.columns or T not in df.columns:
        return False

    alt = df.dropna(subset=[G, T, pld_kolon]).copy()
    if len(alt) < 5:
        return False

    alt["boyut_uyum_farki"] = (alt[pld_kolon] - d_kinetik_xe).abs()
    dx = alt["boyut_uyum_farki"].values
    gercek = alt[G].values
    tahmin = alt[T].values

    if np.std(dx) < 1e-10 or np.std(gercek) < 1e-10 or np.std(tahmin) < 1e-10:
        return False

    r_gercek, _ = pearsonr(dx, gercek)
    r_tahmin, _ = pearsonr(dx, tahmin)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dx, gercek, s=18, alpha=0.4, color="steelblue", label="True Selectivity")
    ax.scatter(dx, tahmin, s=18, alpha=0.4, color="darkorange", label="Predicted Selectivity")

    for vals, renk in [(gercek, "steelblue"), (tahmin, "darkorange")]:
        katsayi = np.polyfit(dx, vals, 1)
        x_cizgi = np.linspace(dx.min(), dx.max(), 50)
        ax.plot(x_cizgi, np.polyval(katsayi, x_cizgi), "--", color=renk, lw=1.5, alpha=0.85)

    ax.set_xlabel(r"|PLD $-$ Xe Kinetic Diameter| (Å)")
    ax.set_ylabel("Xe/Kr Selectivity (-)")
    ax.text(0.95, 0.95,
            f"r(True)={r_gercek:.3f}\nr(Pred)={r_tahmin:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.legend(loc="lower left", fontsize=10)  # kullanici geri bildirimi: sol-ust'teki r(True)/r(Pred) kutusuyla cakismasin diye sol-alta alindi

    fig.tight_layout()
    panel_ekle(fig)
    kaydet(fig, cikti_yolu)
    return True


def egitim_kaybi_grafigi(gecmis_dfs: dict[int, pd.DataFrame], etiket: str, baslik: str,
                          cikti_yolu, kayip_kolon_train: str = "train_loss",
                          kayip_kolon_val: str = "val_loss") -> None:
    """Eğitim/validasyon kayıp eğrileri - her fold için ayrı çizgi (train:
    düz, val: kesik)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    renkler = plt.cm.tab10(np.linspace(0, 0.9, max(len(gecmis_dfs), 1)))
    for (fold_no, gdf), renk in zip(sorted(gecmis_dfs.items()), renkler):
        ax.plot(gdf["epoch"], gdf[kayip_kolon_train], color=renk, lw=1.6,
                label=f"Fold {fold_no} (Train)")
        ax.plot(gdf["epoch"], gdf[kayip_kolon_val], color=renk, lw=1.6, ls="--",
                label=f"Fold {fold_no} (Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    panel_ekle(fig)
    kaydet(fig, cikti_yolu)


def permutation_importance_grafigi(skorlar: dict, etiket: str, baslik: str, cikti_yolu,
                                    bilimsel_notasyon: bool = False, sifir_esik: float = 1e-9) -> None:
    import json as _json
    from pathlib import Path as _Path

    _ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
              "XI", "XII", "XIII", "XIV", "XV"]

    _Path(cikti_yolu).with_name("perm_importance.json").write_text(
        _json.dumps(skorlar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    anlamli = {k: v for k, v in skorlar.items() if abs(v) > sifir_esik}
    if not anlamli:
        anlamli = dict(skorlar)

    isimler = list(anlamli.keys())
    degerler = [anlamli[k] for k in isimler]

    sirali = sorted(zip(degerler, isimler), reverse=True)
    deg_s = [d for d, _ in sirali]
    ad_s = [a for _, a in sirali]
    roman = _ROMAN[:len(ad_s)]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.barh(roman, deg_s, color="darkorange")
    ax.set_xlabel("Importance (ΔMAE)")
    for i, v in enumerate(deg_s):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
    pos_vals = [v for v in deg_s if v > 0]
    if pos_vals:
        ax.set_xlim(right=max(pos_vals) * 1.40)
    if bilimsel_notasyon:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2), useMathText=True)

    _Path(cikti_yolu).with_name("Feature_Importance.txt").write_text(
        "\n".join(f"{r} = {a}" for r, a in zip(roman, ad_s)), encoding="utf-8"
    )

    fig.tight_layout()
    panel_ekle(fig)
    kaydet(fig, cikti_yolu)
