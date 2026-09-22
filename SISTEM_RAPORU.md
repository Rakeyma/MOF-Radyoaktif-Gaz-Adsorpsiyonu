# SISTEM_RAPORU.md — MOF Radyoaktif Gaz Adsorpsiyonu

Bu belge, `İki Boyutlu Malzemeler` / `Üç Boyutlu Kristal Malzemeler`
projeleriyle **aynı mimari mantık ve raporlama stilini** izleyen bu yeni
kardeş projenin (`MOF Radyoaktif Gaz Adsorpsiyonu`) tam teknik dökümüdür.

## 0. Hedef

Metal-Organik Çerçevelerde (MOF) **Xe/Kr adsorpsiyon kapasitesi**, **Xe/Kr
seçicilik** ve **I₂ adsorpsiyon kapasitesi** tahmini — nükleer atık gaz
yönetimi (off-gas treatment) uygulaması için. 4 hedef, TEK BİR ortak
mimarideki (`egitim_ortak.py`) çok-görevli (multi-task) regresyon başıyla
BİRLİKTE tahmin edilir.

## 1. Beş Bileşenli Veri Boru Hattı

| # | Script | Bileşen | Ne yapar |
|---|--------|---------|----------|
| 2 | `veri_indirici_1_jarvis_core_mof.py` | DB/API kazıma | Hugging Face `datasets` (**gerçek** `jablonkagroup/core_mof_no_topo` — CoRE-MOF türevi CIF + GCMC CO₂/CH₄ adsorpsiyon verisi) → CoRE-MOF açık CSV → 12 bilinen MOF ailesinden prosedürel yedek (ASLA durmaz). Pore volume/void fraction/surface area/LCD/PLD proxy'leri geometrik olarak tahmin edilir. |
| 3 | `nlp_literatur_madencilik_2.py` | NLP madenciliği | CrossRef + arXiv (anahtarsız) API'lerinden Xe/Kr/I₂ anahtar kelimeleriyle arama, regex ile MOF adı + sayısal değer çıkarımı. Sonuç yoksa **UYDURMAZ**, boş CSV bırakır. |
| 4 | `veri_artirma_3_augmentasyon.py` | Veri artırma | pymatgen: eksik-bağlayıcı kusuru, metal-düğüm ikamesi, -CH₃/-NH₂ fonksiyonel grup, 3B Gaussian gürültü. Yüzlerce temel MOF'u binlerce örneğe genişletir. |
| 5 | `eslesme_4_dataset_birlestirici.py` | Birleştirme | NLP-eşleşen (orijinal) etiketleri ÖNCELİKLENDİRİR, kalanı gözeneklilik-proxy korelasyonuyla doldurur (`label_source_<hedef>` ile şeffaf). İki nihai CSV üretir: `qmof_pretrain_dataset_final.csv` (ön-eğitim) + `gaz_adsorpsiyon_dataset_final.csv` (ince-ayar). |
| 1 | `grafik_ortak.py` | Raporlama standardı | TÜM grafikler 600 DPI `.tif`, bold + İngilizce font/eksen/başlık/lejant (kullanıcı gereksinimi, `plt.rcParams` içinde sabit). |

## 2. Transfer Learning (Bileşen 5, `egitim_ortak.py`)

- **Aşama A — Ön-eğitim** (`pretrain_encoder`): her model klasörünün
  `__main__` bloğu önce `QMOF_PRETRAIN_CSV` üzerinde (proxy hedef:
  `formation_energy_eV_atom_proxy` — gerçek DFT yoksa GCMC CO₂ ısı-of-
  adsorpsiyon değerinden türetilir) encoder'ı tek-hedefli regresyonla
  eğitir, SADECE `encoder.state_dict()`'i `pretrain_checkpoints/encoder_pretrained.pt`
  olarak kaydeder (n<50 ise nazikçe atlanır).
- **Aşama B — İnce-ayar** (`finetune_k_fold_egit`): her fold'da yeni bir
  `RegressionHead(encoder, n_outputs=4)` kurulur, varsa ön-eğitimli
  ağırlıklar `encoder`'a yüklenir, ilk `FREEZE_ENCODER_EPOCHS` (varsayılan
  5) epoch boyunca encoder DONDURULUR (linear probing), sonra tüm ağ
  birlikte ince-ayar edilir. K-fold gruplama `base_mof_id` üzerinden
  yapılır (aynı temel MOF'tan türetilen TÜM augment varyantları AYNI
  fold'da kalır — sızıntı önlenir).
- Kayıp fonksiyonu **maskeli** çok-hedefli MSE'dir (`maskeli_mse`) — bir
  hedefin (örn. I₂) etiketi olmayan örnekler o hedef için kayba katkı
  YAPMAZ, diğer 3 hedef normal şekilde öğrenilir.

## 3. Ortak Grafik Temsili (`graf_ozellik_ortak.py`)

CIF → periyodik brute-force radius graph (cutoff=**8.0 Å**, dense
kristallerden geniş — gözenekli MOF'lar için), `z`/`pos`/`edge_index` +
DimeNet++ için (k,j,i) üçlü açı listesi (`idx_kj`/`idx_ji`/`theta`,
graf-önbelleğe alınırken BİR KEZ hesaplanır, forward-pass başına DEĞİL).
Ortak arayüz: `encoder.forward(g: dict) -> Tensor[n_graphs, 64]`.

## 4. 11 GNN Mimarisi

| Model | Katman | Not |
|---|---|---|
| GraphGPS | `GPSConv` (yerel GINEConv + küresel attention) | |
| PNA-GNN | `PNAConv` (4 agregatör × 3 ölçekleyici) | derece histogramı her aşama için ayrı hesaplanır |
| GIN | `GINEConv` | |
| GAT | `GATv2Conv` | kenar-farkında |
| GatedGCN | `ResGatedGraphConv` | |
| DeeperGCN | `GENConv` + `DeepGCNLayer` (8 katman, res+) | |
| ECC | `NNConv` (dinamik ağırlık üretimi) | 3 katman, küçük batch |
| TFN | sıfırdan, l≤1 Clebsch-Gordan | e3nn gerekmez |
| EGNN | sıfırdan, invariant-mesafe mesajı | **en hafif — 4 XAI yönteminin hedefi** |
| SE(3)-Transformer | sıfırdan, l≤1 CG + çok-başlıklı ekivaryant dikkat | |
| **DimeNet++** (otonom eklendi) | sıfırdan, yönlü/açısal mesaj iletimi (Fourier açısal taban) | gözenek-penceresi geometrisi boyut-seçici adsorpsiyonun BİRİNCİL belirleyicisi olduğundan seçildi |

## 5. 4 XAI Yöntemi (EGNN üzerine)

| Yöntem | Mekanizma |
|---|---|
| GraphLIME | Bernoulli atom maskesi + ağırlıklı Lasso (hedef-başına) |
| Edge Attribution | Vanilla gradient + Integrated Gradients, kenar (RBF) maskesi üzerinde |
| SubgraphX | MCTS (UCT seçim, atom-çıkarma genişleme, çok-hedefli standardize ödül) |
| **Integrated Gradients** (otonom eklendi) | GERÇEK sürekli girdiler (atom 3B konumu + gözeneklilik özellik vektörü) üzerinde, maskesiz, tamlık aksiyomuna uygun IG — diğer 3 yöntemin ayrık/maske-tabanlı yaklaşımını TAMAMLAR |

## 6. Dürüstlük / Sınırlamalar

- Gaz-adsorpsiyon etiketleri KARMA kaynaklıdır: gerçek NLP-madencilik
  (nadir, sadece augment edilmemiş orijinallere uygulanır) + gözeneklilik-
  proxy korelasyon (çoğunluk). `label_source_<hedef>` sütunlarında şeffaf.
- Prosedürel yedek yapılar (canlı API erişimi yoksa) basitleştirilmiş
  düğüm+bağlayıcı iskeletleridir, TAM kristalografik çözünürlükte DEĞİLDİR.
- Gözeneklilik proxy özellikleri (pore volume/void fraction/LCD/PLD) Zeo++
  YERİNE hafif bir geometrik yaklaştırma (`estimate_pore_proxies`) kullanır.
- Varsayılan `MAX_MATERIALS`/`MAX_MATERIALS_PRETRAIN`/`N_AUGMENT_PER_BASE`
  hızlı doğrulama için KÜÇÜLTÜLMÜŞTÜR; üretim ölçeğine çıkmak sadece ortam
  değişkeni büyütmesi gerektirir, kod değişikliği GEREKMEZ.
- Bu ortamda pipeline GERÇEKTEN hem küçük hem de orta ölçekte uçtan uca
  çalıştırılıp doğrulanmıştır (bkz. "Doğrulama Kaydı" ve "Orta-Ölçek Koşum
  Sonuçları" altında).
- **BİLİNEN PERFORMANS SINIRLAMASI (DimeNetPP):** `graf_ozellik_ortak.
  uc_yakin_komsu_acisi` (üçlü/triplet açı listesi) saf Python döngüsü
  kullanır (bkz. dosyanın kendi dokstringi) - küçük birim hücreli GERÇEK
  CoRE-MOF yapılarında (8.0 Å kesme yarıçapı görece büyük periyodik
  genişletme gerektirdiğinden) bu döngü BEKLENENDEN ÇOK YAVAŞ kalabilir.
  Orta-ölçek koşumda diğer 10 model 7-36 dakikada tamamlanırken DimeNetPP
  ~105 dakika sonra (donmadan, gerçekten CPU harcayarak, `fold1_best_model.pt`
  üretecek kadar ilerlemişken) atlanmıştır - bu bir KOD HATASI değil, ÜRETIM
  ÖLÇEĞİNDE (10⁴-10⁵ yapı) vektörleştirilmiş bir triplet-inşa fonksiyonuyla
  optimize edilmesi gereken bilinen bir performans sınırlamasıdır.

  **Güncelleme — `uc_yakin_komsu_acisi` VEKTÖRLEŞTİRİLDİ:** yukarıdaki
  fonksiyon, Python-seviyeli kenar döngüsü OLMADAN, tamamen tensor
  işlemleriyle (grup-genişletme/`repeat_interleave` tekniği) yeniden
  yazıldı; 20 rastgele + gerçek-veri testinde ESKİ (yavaş) sürümle BİREBİR
  AYNI (kj,ji) çiftlerini ürettiği doğrulandı. Sonuç: CIF→3B-graf+üçlü-açı
  önbelleği aşaması 30 gerçek yapıda ortalama **31.5 ms/yapı**'ya indi (en
  büyük yapı: 161 atom, 164.864 üçlü, yine de <100ms) — eskiden TEK bir
  yapı dakikalarca sürebiliyordu. Buna rağmen DimeNetPP'nin KENDİ eğitim
  döngüsü (4 katmanlı, `HIDDEN=64` boyutunda, binlerce üçlü/yapı üzerinde
  `triplet_mlp` + `_scatter_sum` hesaplayan mimari) bu veri ölçeğinde
  (K=3 fold × 15 epoch, ~3600 yapı) yine de diğer 10 modelden (7-36 dk)
  belirgin şekilde daha maliyetli kalmıştır (~100 dk'da hâlâ fold 1
  tamamlanmamıştı, ikinci kez atlandı) - bu artık bir GRAF-İNŞA darboğazı
  DEĞİL, mimarinin kendi ileri/geri-yayılım maliyetidir (üretim ölçeğinde
  `N_LAYERS`/`HIDDEN` küçültme veya `batch_size` ayarıyla iyileştirilebilir).
  Diğer 10 model bu değişiklikten ETKİLENMEMİŞTİR (aynı gerçek sonuçlar).

## 7. Doğrulama Kaydı (bu ortamda gerçekten çalıştırıldı)

`MAX_MATERIALS=15`, `MAX_MATERIALS_PRETRAIN=25-60`, `N_AUGMENT_PER_BASE=4`,
`KFOLD_OVERRIDE=2`, `MAX_EPOCHS_OVERRIDE=2` ile TÜM 5 veri-boru-hattı
scripti + TÜM 11 model (`run_*.py` + `grafik.py`) + TÜM 4 XAI yöntemi
(`run_*.py` + `grafik.py`) + `model_karsilastirma.py`/`_grafik.py` +
`rapor_olustur.py`/`bilgi_raporu_olustur.py` GERÇEKTEN çalıştırıldı, GPU
(CUDA) tespit edildi, gerçek Hugging Face verisi (`jablonkagroup/
core_mof_no_topo`) indirildi, transfer learning (`[Transfer Learning]
On-egitimli encoder yuklendi`) doğrulandı, 600 DPI `.tif` grafikler ve
gömülü-görsel `.docx` raporlar üretildi. 2 gerçek bug bulunup düzeltildi:
(1) HF veri setinin CIF metnindeki `[CIF]`/`[/CIF]` etiketleri pymatgen'in
CIF ayrıştırıcısını bozuyordu → regex ile temizlendi; (2) LCD/PLD proxy
formülü küçük birim hücrelerde (gerçek CoRE-MOF verisi) sabit alt-sınıra
çöküyordu → `sqrt`-ölçekli daha az agresif formüle güncellendi.

## 8. Orta-Ölçek Koşum Sonuçları (2026-08-12/13, gerçek veri)

`MAX_MATERIALS=300` (gerçek HF `jablonkagroup/core_mof_no_topo` CoRE-MOF
yapıları), `MAX_MATERIALS_PRETRAIN=1500`, `N_AUGMENT_PER_BASE=6` (→ 2100
ince-ayar örneği), `KFOLD_OVERRIDE=3`, `MAX_EPOCHS_OVERRIDE=15` ile TAM bir
koşum yapıldı (toplam ~4.5 saat, GPU/CUDA üzerinde). 10/11 model (DimeNetPP
performans nedeniyle atlandı, bkz. §6) + 4/4 XAI yöntemi + 2/2 rapor
BAŞARIYLA tamamlandı, hiçbir çalışma-zamanı hatası oluşmadı.

**Genel model sıralaması (4 hedefin ortalaması, pooled out-of-fold, n=8400):**

| Sıra | Model | R² | MAE |
|---|---|---|---|
| 1 | PNA_GNN | 0.854 | 0.1226 |
| 2 | EGNN | 0.854 | 0.1225 |
| 3 | SE3_Transformer | 0.854 | 0.1225 |
| 4 | DeeperGCN | 0.854 | 0.1232 |
| 5 | GatedGCN | 0.854 | 0.1228 |
| 6 | TFN | 0.853 | 0.1231 |
| 7 | GraphGPS | 0.852 | 0.1240 |
| 8 | GIN | 0.852 | 0.1234 |
| 9 | ECC | 0.851 | 0.1239 |
| 10 | GAT | 0.851 | 0.1238 |

Hedef bazında: Xe R²≈0.93, Kr R²≈0.93, I₂ R²≈0.90, Xe/Kr seçicilik R²≈0.65
(en zor hedef — beklenen, çünkü seçicilik iki ayrı kapasitenin oranından
türetilir ve gürültüyü katlar). 10 model arasındaki fark ÇOK KÜÇÜKTÜR
(R²=0.851-0.854) — bu, bu ölçekte (2100 örnek, çoğunluğu proxy-korelasyon
etiketli) mimari seçiminden çok VERİ/ETİKET kalitesinin performansı
belirlediğini gösterir (dürüstlük notu: etiketlerin tamamı bu koşumda
`PROXY_PORE_CORRELATION` kaynaklıydı — gerçek NLP-literatür eşleşmesi bu
belirli 300 rastgele CoRE-MOF örnekleminde bulunamadı, bu BEKLENEN bir
durumdur çünkü CoRE-MOF girdileri kimyasal-formül isimli olup literatürdeki
kısaltılmış MOF adlarıyla (HKUST-1 vb.) örtüşmez).

Tam sonuçlar: `model_karsilastirma_sonuclari.txt/.csv`,
`model_karsilastirma_grafikler/*.tif`, `MOF_Radyoaktif_Gaz_Adsorpsiyonu_
Raporu.docx` (72 MB, gerçek gömülü 600 DPI görsellerle).

## 9. Çalıştırma

Bkz. [`README.md`](README.md) — hızlı başlangıç komutları.
