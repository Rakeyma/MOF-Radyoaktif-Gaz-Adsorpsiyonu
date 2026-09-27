"""
kaynakca.py
=============
Her iki rapor scriptinin (rapor_olustur.py ve bilgi_raporu_olustur.py) PAYLASTIGI
TEK kaynakca. Ayni kunyeyi iki dosyada ayri ayri tutmak, birinde guncelleme
yapilip digerinde yapilmamasi riskini dogururdu; atif NUMARALARI da bu tek
listedeki siradan turetildigi icin iki rapor ayni numarayi ayni esere verir.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# KAYNAKÇA (akademik biçim: metin-içi [N] atıfları + sonda "Kaynaklar" bölümü)
# ---------------------------------------------------------------------------
# NOT: künyeler literatürden derlenmiştir; YAYINA GÖNDERMEDEN ÖNCE cilt/sayfa
# bilgileri orijinal kaynaklardan DOĞRULANMALIDIR. Sıra = metinde ilk geçiş
# sırası değil, konusal gruplama (veri seti -> alan -> mimariler -> XAI ->
# araçlar); atıf numaraları bu listedeki sıradan OTOMATİK türetilir.
KAYNAKLAR: list[tuple[str, str]] = [
    ("jablonka2023",
     "Jablonka, K. M.; Rosen, A. S.; Krishnapriyan, A. S.; Smit, B. An Ecosystem for "
     "Digital Reticular Chemistry. ACS Cent. Sci. 2023, 9 (4), 563-581. "
     "https://doi.org/10.1021/acscentsci.2c01177."),
    ("chung2014",
     "Chung, Y. G.; Camp, J.; Haranczyk, M.; Sikora, B. J.; Bury, W.; Krungleviciute, V.; "
     "Yildirim, T.; Farha, O. K.; Sholl, D. S.; Snurr, R. Q. Computation-Ready, "
     "Experimental Metal-Organic Frameworks: A Tool To Enable High-Throughput Screening of "
     "Nanoporous Crystals. Chem. Mater. 2014, 26 (21), 6185-6192. "
     "https://doi.org/10.1021/cm502594j."),
    ("chung2019",
     "Chung, Y. G.; Haldoupis, E.; Bucior, B. J.; Haranczyk, M.; Lee, S.; Zhang, H.; "
     "Vogiatzis, K. D.; Milisavljevic, M.; Ling, S.; Camp, J. S.; et al. Advances, Updates, "
     "and Analytics for the Computation-Ready, Experimental Metal-Organic Framework "
     "Database: CoRE MOF 2019. J. Chem. Eng. Data 2019, 64 (12), 5985-5998. "
     "https://doi.org/10.1021/acs.jced.9b00835."),
    ("sikora2012",
     "Sikora, B. J.; Wilmer, C. E.; Greenfield, M. L.; Snurr, R. Q. Thermodynamic Analysis "
     "of Xe/Kr Selectivity in over 137,000 Hypothetical Metal-Organic Frameworks. "
     "Chem. Sci. 2012, 3, 2217-2223. https://doi.org/10.1039/C2SC01097F."),
    ("simon2015",
     "Simon, C. M.; Mercado, R.; Schnell, S. K.; Smit, B.; Haranczyk, M. What Are the Best "
     "Materials To Separate a Xenon/Krypton Mixture? Chem. Mater. 2015, 27 (12), 4459-4475. "
     "https://doi.org/10.1021/acs.chemmater.5b01475."),
    ("willems2012",
     "Willems, T. F.; Rycroft, C. H.; Kazi, M.; Meza, J. C.; Haranczyk, M. Algorithms and "
     "Tools for High-Throughput Geometry-Based Analysis of Crystalline Porous Materials. "
     "Microporous Mesoporous Mater. 2012, 149 (1), 134-141."),
    ("rampasek2022",
     "Rampasek, L.; Galkin, M.; Dwivedi, V. P.; Luu, A. T.; Wolf, G.; Beaini, D. Recipe for "
     "a General, Powerful, Scalable Graph Transformer. Advances in Neural Information "
     "Processing Systems (NeurIPS) 2022."),
    ("corso2020",
     "Corso, G.; Cavalleri, L.; Beaini, D.; Lio, P.; Velickovic, P. Principal Neighbourhood "
     "Aggregation for Graph Nets. Advances in Neural Information Processing Systems "
     "(NeurIPS) 2020."),
    ("xu2019",
     "Xu, K.; Hu, W.; Leskovec, J.; Jegelka, S. How Powerful Are Graph Neural Networks? "
     "International Conference on Learning Representations (ICLR) 2019."),
    ("brody2022",
     "Brody, S.; Alon, U.; Yahav, E. How Attentive Are Graph Attention Networks? "
     "International Conference on Learning Representations (ICLR) 2022."),
    ("bresson2017",
     "Bresson, X.; Laurent, T. Residual Gated Graph ConvNets. arXiv:1711.07553, 2017."),
    ("li2020",
     "Li, G.; Xiong, C.; Thabet, A.; Ghanem, B. DeeperGCN: All You Need to Train Deeper "
     "GCNs. arXiv:2006.07739, 2020."),
    ("simonovsky2017",
     "Simonovsky, M.; Komodakis, N. Dynamic Edge-Conditioned Filters in Convolutional "
     "Neural Networks on Graphs. IEEE Conference on Computer Vision and Pattern Recognition "
     "(CVPR) 2017."),
    ("thomas2018",
     "Thomas, N.; Smidt, T.; Kearnes, S.; Yang, L.; Li, L.; Kohlhoff, K.; Riley, P. Tensor "
     "Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point "
     "Clouds. arXiv:1802.08219, 2018."),
    ("satorras2021",
     "Satorras, V. G.; Hoogeboom, E.; Welling, M. E(n) Equivariant Graph Neural Networks. "
     "International Conference on Machine Learning (ICML) 2021."),
    ("fuchs2020",
     "Fuchs, F. B.; Worrall, D. E.; Fischer, V.; Welling, M. SE(3)-Transformers: 3D "
     "Roto-Translation Equivariant Attention Networks. Advances in Neural Information "
     "Processing Systems (NeurIPS) 2020."),
    ("gasteiger2020",
     "Gasteiger, J.; Giri, S.; Margraf, J. T.; Gunnemann, S. Fast and Uncertainty-Aware "
     "Directional Message Passing for Non-Equilibrium Molecules (DimeNet++). "
     "arXiv:2011.14115, 2020."),
    ("ribeiro2016",
     "Ribeiro, M. T.; Singh, S.; Guestrin, C. \"Why Should I Trust You?\": Explaining the "
     "Predictions of Any Classifier. ACM SIGKDD International Conference on Knowledge "
     "Discovery and Data Mining (KDD) 2016."),
    ("huang2020",
     "Huang, Q.; Yamada, M.; Tian, Y.; Singh, D.; Chang, Y. GraphLIME: Local Interpretable "
     "Model Explanations for Graph Neural Networks. arXiv:2001.06216, 2020."),
    ("yuan2021",
     "Yuan, H.; Yu, H.; Wang, J.; Li, K.; Ji, S. On Explainability of Graph Neural Networks "
     "via Subgraph Explorations. International Conference on Machine Learning (ICML) 2021."),
    ("sundararajan2017",
     "Sundararajan, M.; Taly, A.; Yan, Q. Axiomatic Attribution for Deep Networks. "
     "International Conference on Machine Learning (ICML) 2017."),
    ("tibshirani1996",
     "Tibshirani, R. Regression Shrinkage and Selection via the Lasso. J. R. Stat. Soc. "
     "Series B 1996, 58 (1), 267-288."),
    ("ong2013",
     "Ong, S. P.; Richards, W. D.; Jain, A.; Hautier, G.; Kocher, M.; Cholia, S.; Gunter, "
     "D.; Chevrier, V. L.; Persson, K. A.; Ceder, G. Python Materials Genomics (pymatgen): "
     "A Robust, Open-Source Python Library for Materials Analysis. Comput. Mater. Sci. "
     "2013, 68, 314-319."),
    ("pedregosa2011",
     "Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; "
     "et al. Scikit-learn: Machine Learning in Python. J. Mach. Learn. Res. 2011, 12, "
     "2825-2830."),
    ("paszke2019",
     "Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; et al. "
     "PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in "
     "Neural Information Processing Systems (NeurIPS) 2019."),
]

_KAYNAK_NO = {anahtar: i for i, (anahtar, _) in enumerate(KAYNAKLAR, 1)}


def atif(*anahtarlar: str) -> str:
    """Metin-içi atıf dizesi üretir: atif('sikora2012') -> '[4]',
    atif('xu2019', 'brody2022') -> '[9, 10]'. Bilinmeyen anahtar için hata
    verir (sessizce yanlış numara basmaktansa erken patlaması istenir).

    Numaralar ARTAN sırada yazılır - akademik biçim gereği; cagiran taraf
    anahtarlari hangi sirada verirse versin cikti '[18, 19]' olur, '[19, 18]'
    olmaz."""
    return "[" + ", ".join(str(n) for n in sorted(_KAYNAK_NO[a] for a in anahtarlar)) + "]"
