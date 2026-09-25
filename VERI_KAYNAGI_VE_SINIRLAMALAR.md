# Veri Kaynağı, Sentetiklik Durumu ve Sınırlamalar

Bu belge, projedeki **her bir veri parçasının nereden geldiğini** ve
**ne kadarının gerçek, ne kadarının sentetik/vekil (proxy) olduğunu**
açıkça ortaya koyar. Aşağıdaki tüm sayılar, depodaki gerçek veri
dosyalarından (`data/raw/kaynak_bilgisi.json`,
`data/processed/eslesme_bilgisi.json`,
`data/processed/gaz_adsorpsiyon_dataset_final.csv`) **okunarak
doğrulanmıştır** — kod yorumlarından kopyalanmamıştır.

---

## ⚠️ En Önemli Uyarı (önce bunu okuyun)

**Modellerin tahmin etmeyi öğrendiği 4 hedefin (Xe/Kr/I₂ kapasitesi ve
Xe/Kr seçicilik) etiketlerinin %100'ü SENTETİKTİR.** Bunlar deneysel
ölçüm değildir, GCMC simülasyonu değildir, literatürden alınmış değildir:
gözeneklilik tanımlayıcılarından **kapalı-form bir formülle üretilmiştir**
(bkz. §3).

Dolayısıyla raporlardaki **R² ≈ 0.93 gibi yüksek skorlar, "bu model gerçek
Xe/Kr adsorpsiyonunu tahmin edebiliyor" ANLAMINA GELMEZ.** Bu skorlar,
makine öğrenmesi boru hattının uçtan uca doğru çalıştığını gösterir —
malzeme keşfi sonucu olarak sunulamaz. Ayrıntı için bkz. §5
"Döngüsellik Problemi".

---

## 1. Kristal Yapılar (model GİRDİSİ, X)

| Veri seti | Örnek | Kaynak | Gerçek mi? |
|---|---|---|---|
| İnce-ayar, temel MOF'lar | 300 | Hugging Face `jablonkagroup/core_mof_no_topo` (CoRE-MOF türevi, CC BY 4.0) | ✅ **GERÇEK** deneysel kristal yapılar (CIF + 3B koordinat) |
| İnce-ayar, toplam örnek | 2100 | 300 gerçek temel + 1800 veri-artırma varyantı | ⚠️ %14 gerçek, **%86 sentetik türev** |
| Ön-eğitim | 1500 | 1164 HF CoRE-MOF + 336 `PROCEDURAL_FALLBACK` | ⚠️ %78 gerçek, **%22 tamamen sentetik** |

**Veri artırma (`veri_artirma_3_augmentasyon.py`):** her temel MOF'tan 6
varyant üretilir — eksik-bağlayıcı kusuru (defect), metal-düğüm
değiştirme (substitution), fonksiyonel grup ekleme (-CH₃/-NH₂) ve 3B
Gauss koordinat gürültüsü. Bunlar **gerçek kristal yapılar değildir**,
gerçek yapılardan türetilmiş bozulmuş kopyalardır. Sızıntıyı önlemek için
aynı temel MOF'tan türeyen tüm varyantlar aynı K-Fold parçasında tutulur
(`base_mof_id` ile gruplama).

**`PROCEDURAL_FALLBACK`:** API erişimi yetersiz kaldığında, 12 bilinen
MOF ailesinden tohumlanarak **prosedürel olarak üretilmiş** basitleştirilmiş
düğüm+bağlayıcı iskeletleridir — kristalografik olarak rafine edilmiş
yapılar değildir.

---

## 2. Gözeneklilik Tanımlayıcıları (PLD, LCD, gözenek hacmi, yüzey alanı)

**Bunlar da ölçüm/Zeo++ değeri DEĞİLDİR.** Bu ortamda Zeo++ mevcut
olmadığından, `veri_indirici_1_jarvis_core_mof.py` içindeki
`estimate_pore_proxies()` fonksiyonu bunları yapıdan **kaba geometrik
yaklaşımla** hesaplar:

- `void_fraction ≈ 1 − (atomların vdW hacimleri toplamı / hücre hacmi)`
- `pore_volume = void_fraction × hücre hacmi / kütle`
- `LCD ≈ en kısa kafes vektörü × √(void_fraction)`, `PLD ≈ LCD × 0.55`

Gerçek bir gözenek-geometrisi hesabı (Zeo++ küresel-prob taraması) bundan
**niteliksel olarak farklı** sonuçlar verir. Bu değerler deneysel BET
yüzey alanı veya ölçülmüş PLD olarak sunulamaz.

---

## 3. Hedef Etiketler (model ÇIKTISI, y) — %100 SENTETİK

`data/processed/eslesme_bilgisi.json` ve nihai veri setinden doğrulanan
gerçek dağılım:

| Hedef | `PROXY_PORE_CORRELATION` | `NLP_LITERATURE` |
|---|---|---|
| `xe_uptake_mmol_g` | **2100 / 2100 (%100)** | 0 |
| `kr_uptake_mmol_g` | **2100 / 2100 (%100)** | 0 |
| `xe_kr_selectivity` | **2100 / 2100 (%100)** | 0 |
| `i2_uptake_mmol_g` | **2100 / 2100 (%100)** | 0 |

### Etiketleri üreten formül (`eslesme_4_dataset_birlestirici.py`)

```
boyut_uyum(d) = exp(−(PLD − d)² / (2 × 1.5²))        # d: kinetik çap
                                                      # Xe 4.10 Å, Kr 3.69 Å, I₂ 5.00 Å
xe  = pore_volume × (0.8 + 1.5 × boyut_uyum(Xe)) × (1 + 0.4 × open_metal) × lognormal(0.15)
kr  = pore_volume × (0.5 + 1.0 × boyut_uyum(Kr)) × (1 + 0.2 × open_metal) × lognormal(0.15)
sel = clip(1 + 7 × boyut_uyum(Xe)/(boyut_uyum(Kr)+0.15) × (1 + 0.3 × open_metal), 1, 30)
i2  = pore_volume × (0.3 + 1.0 × boyut_uyum(I₂)) × (1 + 1.2 × open_metal) × (1 + 0.5 × func)
```

Formülün **fiziksel motivasyonu gerçektir** (Sikora et al. 2012'nin
boyut-eleme ilkesi: PLD hedef gazın kinetik çapına yaklaştıkça seçicilik
artar), ancak **üretilen sayılar gerçek değildir** — bir eğilimin
kapalı-form taklididir, üzerine lognormal çarpımsal gürültü eklenmiştir.

### NLP literatür madenciliği neden 0 etiket sağladı?

`nlp_literatur_madencilik_2.py` CrossRef/arXiv'den **sadece 2 satır**
çıkarabildi (her ikisi de "MOF-11" için) ve bunlar veri setindeki hiçbir
yapıyla eşleşmediği için **hiçbiri kullanılmadı**.

Ayrıca bu 2 satırın kendisi de **hatalıdır**: kaynak cümlede
`3.46 mmol/g` değeri Xe kapasitesine, Kr ise `350 cm³/g`'a aittir; regex
aynı `3.46` değerini hem `xe_uptake` hem `kr_uptake` olarak atamıştır.
Yani NLP bileşeni şu an güvenilir etiket üretmemektedir.

---

## 4. Ön-Eğitim Hedefi (`formation_energy_eV_atom_proxy`)

| Kaynak | Satır | Ne? |
|---|---|---|
| `KAYNAK_VERISI` | 1164 (%78) | HF veri setindeki **gerçek GCMC-simüle** CO₂ Widom adsorpsiyon ısısından türetilmiş |
| `TURETILMIS_PROXY` | 336 (%22) | Gözeneklilik+kompozisyondan sentetik formül |

Adındaki `_proxy` eki kasıtlıdır: bu **gerçek DFT oluşum enerjisi
değildir**. Ön-eğitim aşamasının amacı zaten nihai değerlendirme değil,
genellenebilir bir yapı temsili (embedding) öğrenmektir.

---

## 5. Döngüsellik Problemi (sonuçları yorumlarken kritik)

Etiket formülünün girdileri şunlardır: **`pld_A`, `pore_volume_cm3_g`,
`open_metal_site`, `has_functional_group`**.

Bu dört değişkenin **dördü de** modele yardımcı (aux) girdi özelliği
olarak **doğrudan verilmektedir** (`ortak_ozellikler.AUX_FEATURE_COLUMNS`).

Yani model, kendi girdilerinden hesaplanan kapalı-form bir formülü geri
çözmeyi öğrenmektedir. Bunun sonuçlarda bıraktığı izler net olarak
görülmektedir:

- **R² tavanı fizikle değil, enjekte edilen gürültüyle belirlenir**
  (σ = 0.12–0.18 lognormal). Uptake hedeflerinde R² ≈ 0.90–0.93 çıkması
  bunun doğrudan sonucudur.
- **Permutation importance**: `Pore Geometry` ΔMAE ≈ 0.146'ya karşılık
  `Crystal Structure` ΔMAE ≈ 0.002 — yaklaşık **73 kat** fark. Model 3B
  atomistik geometriye neredeyse hiç dayanmıyor, formülün girdilerine
  dayanıyor.
- **Fiziksel tutarlılık grafiği**: r(Tahmin) ≈ −0.99 iken r(Gerçek) ≈
  −0.82. Model, gürültülü "gerçek" etiketlerden **daha temiz** bir
  korelasyon üretiyor — çünkü altta yatan gürültüsüz formülü öğrenmiş
  durumda.

**Sonuç:** Bu koşum, 10 GNN mimarisinin + 4 XAI yönteminin + transfer
öğrenmenin uçtan uca doğru çalıştığını gösteren geçerli bir **altyapı
doğrulamasıdır**. Gerçek Xe/Kr/I₂ adsorpsiyon tahmin başarısı
**gösterilmemiştir** ve bu sayılar bir yayında malzeme-keşfi sonucu
olarak sunulamaz.

---

## 6. Bunu gerçek bir çalışmaya dönüştürmek için gerekenler

1. **Gerçek etiket kaynağı bağlanmalı** — örn. CoRE-MOF/CSD için yayınlanmış
   GCMC Xe/Kr izoterm veri setleri, ya da bu proje kapsamında RASPA ile
   yapılacak GCMC simülasyonları. Proxy formülü tamamen devre dışı bırakılmalı.
2. **Gözeneklilik tanımlayıcıları Zeo++ ile hesaplanmalı** (geometrik
   yaklaşım yerine).
3. **Döngüsellik kırılmalı:** etiket üretiminde kullanılan hiçbir değişken
   modele girdi olarak verilmemeli (gerçek etiketlere geçilince bu sorun
   kendiliğinden ortadan kalkar).
4. **NLP çıkarımı düzeltilmeli** — mevcut regex, aynı sayıyı farklı
   hedeflere atayabiliyor; birim dönüşümü ve hedef-eşleştirme
   doğrulanmalı.
5. **DimeNet++ eğitilmeli** — implemente edilmiş ancak bu koşumda
   eğitilmemiştir (11 mimariden 10'u eğitilmiştir).
