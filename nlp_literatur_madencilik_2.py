"""
nlp_literatur_madencilik_2.py
================================
BILESEN 3 — NLP Tabanli Veri Madenciligi (Literature Mining)
==================================================================
AMAC:
    CrossRef VE arXiv API'lerini (ikisi de ANAHTARSIZ/genel-erisimli)
    kullanarak malzeme bilimi/nukleer muhendislik literaturunde "Xenon
    adsorption capacity MOF", "Krypton selectivity", "iodine capture
    metal-organic framework" gibi anahtar kelimelerle arama yapmak,
    basliklardan/ozetlerden (abstract) MOF isimlerini (orn. HKUST-1) VE
    sayisal kapasite degerlerini (mmol/g, cm3/g, wt%) DUZENLI IFADE
    (regex) tabanli bir NLP boru hattiyla cikarip bir pandas DataFrame'e
    kaydetmek.

DURUSTLUK ILKESI (ONEMLI - "Üç Boyutlu Kristal Malzemeler" projesindeki
"asla uydurma, HER SAYI gercek bir kaynaktan gelir" ilkesiyle AYNI):
    Bu script YALNIZCA GERCEKTEN API'lerden DONEN ozet/baslik metninden
    regex ile cikardigi degerleri kaydeder. Ag erisimi YOKSA (bu sandbox
    ortaminda oldugu gibi olabilir) VEYA hicbir API sonuc dondurmezse,
    script SENTETIK/UYDURULMUS literatur degeri YAZMAZ - bunun yerine
    BOS (sadece baslik satirli) bir CSV birakir ve durumu ACIKCA
    raporlar. Asagi akis (eslesme_4_dataset_birlestirici.py), NLP'den
    gelen GERCEK-eslesmis etiketleri varsa ONCELIKLENDIRIR, yoksa
    TAMAMEN Bilesen-2'nin gozeneklilik-proxy korelasyonuna (Bilesen 5
    icin dogal bir "seyrek etiket" senaryosu, egitim_ortak.maskeli_mse
    tarafindan zaten desteklenir) DUSER - pipeline HICBIR ZAMAN
    literatur verisine SERT-BAGIMLI degildir.

CIKARIM YONTEMI:
    1) MOF ADI: regex ile "Buyuk-harf + tire + sayi" kalibi (orn. HKUST-1,
       ZIF-8, UiO-66, MIL-101, NU-1000) + bilinen-onek beyaz listesi
       (MOF, ZIF, UiO, HKUST, MIL, PCN, NU, IRMOF, SIFSIX, CPO, JUC, PCP)
       ile YUKSEK-KESINLIK filtrelemesi.
    2) GAZ TURU: cumle icinde "xenon"/"Xe"/"krypton"/"Kr"/"iodine"/"I2"
       gecen anahtar kelime penceresi.
    3) SAYISAL DEGER: gaz-anahtar-kelimesine YAKIN (ayni cumle) bir
       "sayi + birim (mmol/g, cm3/g, wt%, mg/g)" veya "selectivity ~ sayi"
       kalibi.
    Her satir HANGI makaleden (DOI/arXiv ID) VE hangi cumleden geldigini
    (context_sentence) tasir - izlenebilirlik icin.

CIKTI:
    data/raw/nlp_literatur_madencilik_sonuclari.csv

CALISTIRMA:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python nlp_literatur_madencilik_2.py

AYARLAR (ortam degiskenleri):
    CROSSREF_MAILTO   CrossRef 'kibar havuz' (polite pool) icin e-posta (opsiyonel, hizli/guvenilir yanit icin onerilir)
    NLP_MAX_PER_QUERY Sorgu basina kac kayit cekilsin (varsayilan 40)
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from paths import DATA_RAW, NLP_LITERATUR_CSV, CROSSREF_MAILTO

NLP_MAX_PER_QUERY = int(os.environ.get("NLP_MAX_PER_QUERY", "40"))
ISTEK_TIMEOUT_SN = 20

SORGULAR = [
    "Xenon adsorption capacity metal-organic framework",
    "Krypton selectivity metal-organic framework",
    "Xe Kr separation MOF",
    "iodine capture metal-organic framework",
    "radioactive noble gas capture MOF nuclear",
    "Xe/Kr adsorption porous material off-gas",
]

# --------------------------------------------------------------------------
# REGEX KALIPLARI
# --------------------------------------------------------------------------
_MOF_ONEK_BEYAZ_LISTESI = (
    "MOF", "ZIF", "UiO", "HKUST", "MIL", "PCN", "NU", "IRMOF",
    "SIFSIX", "CPO", "JUC", "PCP", "ZJU", "NOTT", "SNU", "FIR",
)
_MOF_ADI_RE = re.compile(r"\b([A-Za-z]{2,8}-\d{1,4}[a-zA-Z0-9\-]*)\b")

_GAZ_ANAHTAR = {
    "xe_uptake_mmol_g": re.compile(r"\b(xenon|Xe)\b", re.IGNORECASE),
    "kr_uptake_mmol_g": re.compile(r"\b(krypton|Kr)\b", re.IGNORECASE),
    "i2_uptake_mmol_g": re.compile(r"\b(iodine|I2|I₂)\b", re.IGNORECASE),
}
_SECICILIK_RE = re.compile(r"\b(Xe/Kr|Kr/Xe)\b.{0,40}?selectivity[^.]{0,40}?(\d+\.?\d*)", re.IGNORECASE)
_DEGER_BIRIM_RE = re.compile(
    r"(\d+\.?\d*)\s*(mmol\s*/?\s*g|mmol\s*g[\-⁻]?1|cm3\s*/?\s*g|cm³\s*/?\s*g|mg\s*/?\s*g|wt\s*\.?\s*%)",
    re.IGNORECASE,
)


def _cumlelere_ayir(metin: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", metin or "")


def mof_isimlerini_cikar(cumle: str) -> list[str]:
    adaylar = _MOF_ADI_RE.findall(cumle)
    return [a for a in adaylar if a.split("-")[0].upper() in _MOF_ONEK_BEYAZ_LISTESI
            or any(a.upper().startswith(o) for o in _MOF_ONEK_BEYAZ_LISTESI)]


def kayitlari_cikar(baslik: str, ozet: str, kaynak_api: str, kaynak_id: str, sorgu: str) -> list[dict]:
    kayitlar = []
    tam_metin = f"{baslik or ''}. {ozet or ''}"
    for cumle in _cumlelere_ayir(tam_metin):
        mof_adlari = mof_isimlerini_cikar(cumle)
        if not mof_adlari:
            continue
        for hedef_kolon, gaz_re in _GAZ_ANAHTAR.items():
            if not gaz_re.search(cumle):
                continue
            deger_eslesme = _DEGER_BIRIM_RE.search(cumle)
            if not deger_eslesme:
                continue
            for mof_adi in mof_adlari:
                kayitlar.append({
                    "mof_name": mof_adi, "target_column": hedef_kolon,
                    "value": float(deger_eslesme.group(1)), "unit": deger_eslesme.group(2).strip(),
                    "context_sentence": cumle.strip()[:300],
                    "source_api": kaynak_api, "source_id": kaynak_id, "query": sorgu,
                })
        sel_eslesme = _SECICILIK_RE.search(cumle)
        if sel_eslesme:
            for mof_adi in mof_adlari:
                kayitlar.append({
                    "mof_name": mof_adi, "target_column": "xe_kr_selectivity",
                    "value": float(sel_eslesme.group(2)), "unit": "(-)",
                    "context_sentence": cumle.strip()[:300],
                    "source_api": kaynak_api, "source_id": kaynak_id, "query": sorgu,
                })
    return kayitlar


# ---------------------------------------------------------------------------
# CrossRef (anahtarsiz, 'kibar havuz' icin mailto onerilir)
# ---------------------------------------------------------------------------
def crossref_ara(sorgu: str, max_n: int) -> list[dict]:
    import requests

    params = {"query": sorgu, "rows": max_n, "select": "DOI,title,abstract"}
    headers = {"User-Agent": f"MOF-XAI-Pipeline/1.0 (mailto:{CROSSREF_MAILTO or 'anonymous@example.com'})"}
    try:
        r = requests.get("https://api.crossref.org/works", params=params, headers=headers, timeout=ISTEK_TIMEOUT_SN)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"    [CrossRef] '{sorgu}' basarisiz ({type(e).__name__}: {e})")
        return []

    kayitlar = []
    for it in items:
        baslik = " ".join(it.get("title", []) or [])
        ozet = re.sub(r"<[^>]+>", " ", it.get("abstract", "") or "")  # JATS XML etiketlerini temizle
        kayitlar.extend(kayitlari_cikar(baslik, ozet, "CrossRef", it.get("DOI", ""), sorgu))
    print(f"    [CrossRef] '{sorgu}': {len(items)} makale tarandi, {len(kayitlar)} aday kayit bulundu.")
    return kayitlar


# ---------------------------------------------------------------------------
# arXiv (anahtarsiz, Atom XML)
# ---------------------------------------------------------------------------
def arxiv_ara(sorgu: str, max_n: int) -> list[dict]:
    import requests
    import xml.etree.ElementTree as ET

    params = {"search_query": f"all:{sorgu}", "start": 0, "max_results": max_n}
    try:
        r = requests.get("http://export.arxiv.org/api/query", params=params, timeout=ISTEK_TIMEOUT_SN)
        r.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        entries = root.findall("atom:entry", ns)
    except Exception as e:
        print(f"    [arXiv] '{sorgu}' basarisiz ({type(e).__name__}: {e})")
        return []

    kayitlar = []
    for entry in entries:
        baslik = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        ozet = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        arxiv_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        kayitlar.extend(kayitlari_cikar(baslik, ozet, "arXiv", arxiv_id, sorgu))
    print(f"    [arXiv] '{sorgu}': {len(entries)} makale tarandi, {len(kayitlar)} aday kayit bulundu.")
    return kayitlar


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("nlp_literatur_madencilik_2.py — Bilesen 3: NLP Literatur Madenciligi")
    print("=" * 70)

    tum_kayitlar: list[dict] = []
    for sorgu in SORGULAR:
        print(f"\nSorgu: '{sorgu}'")
        tum_kayitlar.extend(crossref_ara(sorgu, NLP_MAX_PER_QUERY))
        tum_kayitlar.extend(arxiv_ara(sorgu, NLP_MAX_PER_QUERY))

    if not tum_kayitlar:
        print("\n[UYARI] Hicbir API'den kullanilabilir sonuc alinamadi (ag erisimi olmayabilir). "
              "SENTETIK/uydurulmus deger YAZILMAYACAK - bos (sadece baslik satirli) bir CSV birakiliyor. "
              "eslesme_4_dataset_birlestirici.py bu durumda TAMAMEN Bilesen-2 gozeneklilik-proxy "
              "korelasyonuna dusecek (bkz. bu dosyanin dokstringindeki 'DURUSTLUK ILKESI').")
        df = pd.DataFrame(columns=["mof_name", "target_column", "value", "unit",
                                    "context_sentence", "source_api", "source_id", "query"])
    else:
        df = pd.DataFrame(tum_kayitlar).drop_duplicates(
            subset=["mof_name", "target_column", "source_id", "context_sentence"]
        ).reset_index(drop=True)
        print(f"\nToplam {len(df)} benzersiz (mof_adi, hedef, kaynak) uclusu cikarildi.")
        print("\nHedef-basina kayit sayisi:")
        print(df["target_column"].value_counts().to_string())
        print("\nEn sik gecen MOF adlari (ilk 15):")
        print(df["mof_name"].value_counts().head(15).to_string())

    df.to_csv(NLP_LITERATUR_CSV, index=False)
    print(f"\nKaydedildi -> {NLP_LITERATUR_CSV.resolve()} ({len(df)} satir)")
    print("\nTamamlandi.")


if __name__ == "__main__":
    main()
