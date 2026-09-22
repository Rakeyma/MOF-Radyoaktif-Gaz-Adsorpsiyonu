# MOF Radyoaktif Gaz Adsorpsiyonu

Metal-Organik Çerçevelerde (MOF) **Ksenon (Xe) / Kripton (Kr) / İyot (I₂)**
radyoaktif gaz adsorpsiyon kapasitesi ve **Xe/Kr seçiciliği** tahmini —
nükleer atık gaz yönetimi için uçtan uca bir boru hattı.

11 GNN mimarisi (GraphGPS, PNA-GNN, GIN, GAT, GatedGCN, DeeperGCN, ECC, TFN,
EGNN, SE(3)-Transformer + otonom eklenen **DimeNet++**) + 4 XAI yöntemi
(GraphLIME, Edge Attribution, SubgraphX + otonom eklenen **Integrated
Gradients**), **Transfer Learning** (QMOF-benzeri proxy veri setinde
ön-eğitim → artırılmış gaz-adsorpsiyon veri setinde ince-ayar) ile.

**Tam belge için bkz. [`SISTEM_RAPORU.md`](SISTEM_RAPORU.md)** — mimari, 5
bileşenli veri boru hattı, her modelin nasıl çalıştığı, XAI yöntemleri,
çalıştırma komutları ve dürüst sınırlamalar orada anlatılmaktadır.

Hızlı başlangıç:

```bash
pip install -r requirements.txt
cp .env.example .env   # (opsiyonel) CROSSREF_MAILTO / HF_TOKEN

# 1) Veri boru hattı (Bileşen 2-5)
python veri_indirici_1_jarvis_core_mof.py       # DB/API kazıma (HF datasets + CoRE-MOF + prosedürel yedek)
python nlp_literatur_madencilik_2.py            # NLP literatür madenciliği (CrossRef + arXiv)
python veri_artirma_3_augmentasyon.py           # pymatgen tabanlı veri artırma (binlerce örnek)
python eslesme_4_dataset_birlestirici.py        # nihai pretrain + finetune veri setleri

# 2) Her model (ön-eğitim + ince-ayar TEK KOMUTTA) + grafikleri
python -m GIN.run_gin && python -m GIN.grafik    # ... diğer 10 model için aynı desen
python -m DimeNetPP.run_dimenetpp && python -m DimeNetPP.grafik

# 3) Karşılaştırma + XAI (EGNN eğitimi bittikten sonra)
python model_karsilastirma.py && python model_karsilastirma_grafik.py
python -m GraphLIME.run_graphlime && python -m GraphLIME.grafik
python -m Edge_Attribution.run_edge_attribution && python -m Edge_Attribution.grafik
python -m SubgraphX.run_subgraphx && python -m SubgraphX.grafik
python -m IntegratedGradients.run_integrated_gradients && python -m IntegratedGradients.grafik

# 4) Raporlar
python rapor_olustur.py            # dinamik sonuç raporu (.docx)
python bilgi_raporu_olustur.py     # statik metodoloji raporu (.docx)
```

Üretim/büyük ölçeğe çıkmak için sadece ortam değişkenlerini büyütün (kod
değişikliği gerekmez): `MAX_MATERIALS`, `MAX_MATERIALS_PRETRAIN`,
`N_AUGMENT_PER_BASE`, `KFOLD_OVERRIDE`, `MAX_EPOCHS_OVERRIDE`, vb. — bkz.
her scriptin başındaki "AYARLAR" bölümü.
