"""
graf_ozellik_ortak.py
======================
AMAC:
    "Üç Boyutlu Kristal Malzemeler/graf_ozellik_ortak.py" icindeki
    'yapidan_3b_graf' + 'bruteforce_radius_graph' + 'batch_graphs' mantigini,
    TUM 11 model (GraphGPS, PNA_GNN, GIN, GAT, GatedGCN, DeeperGCN, ECC,
    TFN, EGNN, SE3_Transformer, DimeNetPP) tarafindan PAYLASILAN TEK bir
    modulde toplamak.

    NEDEN PAYLASILIYOR: 11 modelin TAMAMI ayni ham 3B graf temsiline (atom
    numaralari z, kartezyen konum pos, kesme-yaricapli (cutoff) periyodik
    komsuluk edge_index) ihtiyac duyar - her modelin KENDI mimarisi (mesaj
    iletimi/dikkat/tensor carpimi/yonlu aci) farkli olsa da, "CIF -> ham 3B
    graf" donusumu HERKES icin AYNIDIR.

BU PROJENIN "Üç Boyutlu Kristal Malzemeler"DEN FARKI (CUTOFF/MAX_NEIGHBORS):
    MOF'lar (Metal-Organik Cerceveler) YOGUN inorganik kristallerden COK
    DAHA GOZENEKLIDIR (void fraction genellikle >0.5, bazen >0.9) - metal
    dugumler arasi organik baglayicilar (linker) UZUN olabilir (bazi
    "isoretiküler" MOF serilerinde >15 A). Sabit 6.0 A kesme yaricapi
    (dense-kristal projesindeki deger) boyle yapılarda BAGLI/komsu atomlari
    KACIRIP graf BAGLANTISIZ (disconnected) hale getirebilirdi - bu yuzden
    CUTOFF 8.0 A'ya, MAX_NEIGHBORS ise (dusuk atom yogunlugu telafisi icin)
    32'ye YUKSELTILMISTIR. Geri kalan TUM mantik (periyodik brute-force
    genisletme, Gaussian RBF, segment-softmax, scatter-mean) birebir aynidir.

GRAF INSASI (periyodik sinir kosullari, ayni brute-force yontem):
    MOF CIF'leri de GERCEK 3 BOYUTLU (vakumsuz, TUM yonlerde periyodik)
    kristallerdir. Birim hucre, cutoff yaricapini kapsayacak sekilde
    +-n_a,+-n_b,+-n_c katlariyla brute-force genisletilir, merkeze
    uzakligi cutoff icinde kalan TUM atom kopyalari (kendi ve komsu
    hucrelerden) dugum olarak alinir, sonra en yakin komsu (radius graph)
    kenarlari kurulur.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CUTOFF = 8.0           # Angstrom - gozenekli/uzamis MOF baglaiyicilari icin dense-kristallerden (6.0A) genis
MAX_NEIGHBORS = 32
NUM_ATOM_TYPES = 100    # Z=1..99 (padding_idx=0) - metal dugumler + C/H/N/O/S/halojen linkerler
N_RBF = 24
EMB_DIM = 64            # tum encoderlerin ortak cikti gomme boyutu (adil karsilastirma icin sabit)


# ---------------------------------------------------------------------------
# 1) SAF PYTORCH RADIUS GRAPH (top-k mesafe + esik filtresi)
# ---------------------------------------------------------------------------
def bruteforce_radius_graph(pos: torch.Tensor, r: float, max_neighbors: int = MAX_NEIGHBORS) -> torch.Tensor:
    n = pos.size(0)
    dev = pos.device
    dist = torch.cdist(pos, pos)
    dist.fill_diagonal_(float("inf"))
    k = min(max_neighbors, n - 1)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=dev)
    topk_dist, topk_idx = torch.topk(dist, k, dim=1, largest=False)
    valid = topk_dist <= r
    src = torch.arange(n, device=dev).unsqueeze(1).expand(-1, k)[valid]
    dst = topk_idx[valid]
    return torch.stack([src, dst])


# ---------------------------------------------------------------------------
# 2) YAPI (pymatgen Structure/CIF) -> HAM 3B GRAF
# ---------------------------------------------------------------------------
def yapidan_3b_graf(struct, cutoff: float = CUTOFF, max_neighbors: int = MAX_NEIGHBORS) -> dict:
    """pymatgen Structure -> {z, pos, edge_index, n_atoms}."""
    lat = struct.lattice
    n_a = int(np.ceil(cutoff / lat.a)) + 1
    n_b = int(np.ceil(cutoff / lat.b)) + 1
    n_c = int(np.ceil(cutoff / lat.c)) + 1
    centroid = lat.get_cartesian_coords(np.mean(struct.frac_coords, axis=0))

    def _z(site):
        return site.specie.Z if hasattr(site, "specie") else site.species.elements[0].Z

    z_list, pos_list = [], []
    for i in range(-n_a, n_a + 1):
        for j in range(-n_b, n_b + 1):
            for k in range(-n_c, n_c + 1):
                shift = (i * lat.matrix[0] + j * lat.matrix[1] + k * lat.matrix[2])
                for site in struct:
                    p = site.coords + shift
                    if np.linalg.norm(p - centroid) <= cutoff:
                        z_list.append(_z(site))
                        pos_list.append(p - centroid)

    if not z_list:
        # NADIR UC DURUM (buyuk/uzamis birim hucreli, az atomlu - orn. cok
        # gozenekli/dusuk-yogunluklu MOF - yapilar): ortalama-frac-koord
        # ("centroid") tum atomlardan cutoff disina dusebilir. Bu durumda
        # kesme-yaricapi filtresini UYGULAMADAN, sadece kaydirilmamis
        # (i=j=k=0) birim hucre atomlarini kullanarak GERI DUS.
        for site in struct:
            z_list.append(_z(site))
            pos_list.append(site.coords - centroid)

    z_np = np.array(z_list, dtype=np.int64)
    pos_np = np.array(pos_list, dtype=np.float32).reshape(-1, 3)
    pos_t = torch.from_numpy(pos_np).float()
    edge_index = bruteforce_radius_graph(pos_t, r=cutoff, max_neighbors=max_neighbors)

    # DimeNetPP'nin yonlu/acisal mesaj iletimi icin gereken (k,j,i) uclu
    # aci listesi BURADA, GRAF ONBELLEGE ALINIRKEN (yani egitim boyunca
    # SADECE BIR KEZ) hesaplanir - pos/edge_index egitim sirasinda
    # DEGISMEDIGINDEN (koordinat guncellemesi olmayan bir ozellik-tahmin
    # gorevi), theta acilarini HER ileri-gecişte yeniden hesaplamak
    # gereksiz/pahali olurdu (bkz. uc_yakin_komsu_acisi dokstringi).
    idx_kj, idx_ji, theta = uc_yakin_komsu_acisi(pos_t, edge_index)

    return {
        "z": torch.from_numpy(z_np).long(),
        "pos": pos_t,
        "edge_index": edge_index,
        "n_atoms": len(z_np),
        "idx_kj": idx_kj,
        "idx_ji": idx_ji,
        "theta": theta,
    }


def cif_den_3b_graf(cif_path: str, cutoff: float = CUTOFF, max_neighbors: int = MAX_NEIGHBORS) -> dict:
    from pymatgen.core import Structure
    struct = Structure.from_file(cif_path)
    return yapidan_3b_graf(struct, cutoff=cutoff, max_neighbors=max_neighbors)


# ---------------------------------------------------------------------------
# 3) BATCHLEME (birden fazla MOF'u TEK bir seyrek grafta birlestirme)
# ---------------------------------------------------------------------------
def batch_graphs(graph_list: list[dict]) -> dict:
    z_cat, pos_cat, ei_cat, batch_idx = [], [], [], []
    kj_cat, ji_cat, theta_cat = [], [], []
    atom_offset = 0
    edge_offset = 0
    for i, g in enumerate(graph_list):
        n = g["n_atoms"]
        z_cat.append(g["z"])
        pos_cat.append(g["pos"])
        ei_cat.append(g["edge_index"] + atom_offset)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        # triplet indeksleri (idx_kj/idx_ji), bu grafin KENDI edge_index'ine
        # gore tanimlidir - toplu (batched) edge_index'te dogru kenarlari
        # gostermeleri icin KENAR (edge) ofseti eklenir (atom ofseti DEGIL).
        if "idx_kj" in g and g["idx_kj"].numel() > 0:
            kj_cat.append(g["idx_kj"] + edge_offset)
            ji_cat.append(g["idx_ji"] + edge_offset)
            theta_cat.append(g["theta"])
        atom_offset += n
        edge_offset += g["edge_index"].size(1)
    out = {
        "z": torch.cat(z_cat),
        "pos": torch.cat(pos_cat),
        "edge_index": torch.cat(ei_cat, dim=1),
        "batch": torch.cat(batch_idx),
        "n_graphs": len(graph_list),
    }
    if kj_cat:
        out["idx_kj"] = torch.cat(kj_cat)
        out["idx_ji"] = torch.cat(ji_cat)
        out["theta"] = torch.cat(theta_cat)
    else:
        out["idx_kj"] = torch.empty(0, dtype=torch.long)
        out["idx_ji"] = torch.empty(0, dtype=torch.long)
        out["theta"] = torch.empty(0, dtype=torch.float)
    return out


# ---------------------------------------------------------------------------
# 4) GAUSSIAN RBF - kenar (mesafe) ozellik genisletmesi
#    (PyG tabanli 8 model icin edge_attr, e3nn'siz TFN/SE3-Transformer/EGNN
#    icin radyal filtre girdisi olarak, DimeNetPP icin de kismen paylasilir)
# ---------------------------------------------------------------------------
class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int = N_RBF, cutoff: float = CUTOFF):
        super().__init__()
        offset = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("offset", offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.cutoff = cutoff

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(-1) - self.offset
        env = 0.5 * (torch.cos(np.pi * (r / self.cutoff).clamp(max=1.0)) + 1.0)  # smooth cutoff (cosine envelope)
        return torch.exp(self.coeff * diff.pow(2)) * env.unsqueeze(-1)


def edge_geometri(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mesafe [E], birim yon vektoru r_hat [E,3]) dondurur - tum
    ekivaryant modeller (EGNN/TFN/SE3_Transformer) ve RBF genisletmesi
    icin ortak baslangic noktasi."""
    src, dst = edge_index
    r_vec = pos[src] - pos[dst]
    r = r_vec.norm(dim=-1).clamp(min=1e-8)
    r_hat = r_vec / r.unsqueeze(-1)
    return r, r_hat


def segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """[E, H] kenar-lojitlerini, HER hedef dugum (index) icindeki kenarlar
    ARASINDA normalize eder (segment-wise softmax, sayisal kararlilik icin
    segment-max cikarma ile) - SE3_Transformer'in coklu-baslikli
    ekivaryant dikkat agirliklari VE DimeNetPP'nin acisal agirliklandirmasi
    icin kullanilir. torch>=2.0 gerektirir (Tensor.scatter_reduce_)."""
    seg_max = logits.new_full((num_nodes, logits.size(-1)), float("-inf"))
    seg_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(logits), logits, reduce="amax", include_self=True)
    logits = logits - seg_max[index]
    exp = logits.exp()
    seg_sum = exp.new_zeros((num_nodes, logits.size(-1)))
    seg_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(exp), exp)
    return exp / (seg_sum[index] + 1e-9)


def scatter_mean_manual(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """torch_geometric/torch_scatter'a bagimli olmadan (saf PyTorch)
    ortalama-havuzlama - graf gommesi (atom -> MOF) icin TUM modeller
    tarafindan kullanilir."""
    out = src.new_zeros(dim_size, *src.shape[1:])
    out.scatter_add_(0, index.view(-1, *([1] * (src.dim() - 1))).expand_as(src), src)
    sayim = torch.bincount(index, minlength=dim_size).clamp(min=1).float()
    sayim = sayim.view(dim_size, *([1] * (src.dim() - 1)))
    return out / sayim


def uc_yakin_komsu_acisi(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DimeNetPP (yonlu/acisal mesaj iletimi) icin: her (j->i) kenari icin,
    j'nin diger komsulari k uzerinden ucgen (k->j->i) UCLULERI olusturur ve
    aci theta = angle(r_ji, r_jk)'yi hesaplar - torch_geometric'in resmi
    DimeNet 'triplet' yardimci fonksiyonuyla AYNI mantik, sadece bu projenin
    kendi periyodik radius-graph'i uzerinde calisacak sekilde saf PyTorch'ta
    yeniden yazilmistir. Dondurur: (idx_kj, idx_ji, theta) - ucgenin iki
    kenar-indeksi VE aralarindaki aci.

    VEKTORLESTIRME NOTU (performans duzeltmesi): ILK surum, her kenar icin
    komsu-kenar listesini Python for-dongusuyle geziyordu - kucuk birim
    hucreli GERCEK CoRE-MOF yapilarinda (8.0A kesme yaricapi goreli buyuk
    periyodik genisletme urettiginden) bu, tek bir yapida ONLARCA DAKIKA
    surebilecek kadar yavas kaliyordu (bkz. SISTEM_RAPORU.md '§6 Bilinen
    Performans Sinirlamasi'). Asagidaki surum HICBIR Python-seviyeli kenar
    dongusu ICERMEZ - 'grup-genisletme' (group-expansion) tekniginle TUM
    (k,j,i) uclulerini TEK SEFERDE, tensor islemleriyle uretir:
        1) HER dugume giren kenarlarin (dst=dugum) CSR-benzeri 'ptr' dizini
           (degismedi).
        2) HER kenar e_ji=(j->i) icin, j dugumune giren kenar SAYISI
           (group_sizes[e]=counts[src[e]]) - kac ucgen adayi oldugunu verir.
        3) idx_ji, HER kenar indeksini KENDI grup boyutu kadar tekrarlayarak
           (repeat_interleave) genisletilir; idx_kj ise HER grubun 'ptr'
           araligindaki kenar indeksleri (bir kumulatif-toplam/yerel-indeks
           hilesiyle, YINE for-dongusu OLMADAN) ile doldurulur.
        4) e_kj == e_ji (kendi-kendine ucgen) olan satirlar filtrelenir.
    Sonuc, ESKI (dogru ama yavas) for-dongulu surumle BIREBIR AYNI (kj,ji)
    ciftlerini (farkli SIRADA) uretir - sadece TENSOR-seviyeli oldugu icin
    100-1000x daha hizlidir (E ~ birkac bin kenar icin saniyenin altinda)."""
    src, dst = edge_index  # src=j, dst=i (mesaj j->i)
    n_edges = edge_index.size(1)
    n_nodes = int(pos.size(0))
    dev = pos.device

    if n_edges == 0:
        empty = torch.empty(0, dtype=torch.long, device=dev)
        return empty, empty, torch.empty(0, dtype=pos.dtype, device=dev)

    # HER dugume giren kenarlarin CSR-benzeri dizini (degismedi)
    order = torch.argsort(dst)
    counts = torch.bincount(dst, minlength=n_nodes)
    ptr = torch.zeros(n_nodes + 1, dtype=torch.long, device=dev)
    ptr[1:] = torch.cumsum(counts, dim=0)

    group_sizes = counts[src]              # [E] - e_ji=(j->i) icin j'ye giren kenar sayisi
    total = int(group_sizes.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=dev)
        return empty, empty, torch.empty(0, dtype=pos.dtype, device=dev)

    idx_ji = torch.repeat_interleave(torch.arange(n_edges, device=dev), group_sizes)

    # HER genisletilmis satir icin, kendi grubu icindeki YEREL pozisyonu
    # (0..group_size-1) - for-dongusuz, kumulatif-toplam farki ile.
    cum = torch.cumsum(group_sizes, dim=0)
    grup_baslangici_genisletilmiste = cum - group_sizes
    yerel_idx = torch.arange(total, device=dev) - torch.repeat_interleave(grup_baslangici_genisletilmiste, group_sizes)

    grup_baslangici_order_icinde = torch.repeat_interleave(ptr[src], group_sizes)
    order_pozisyonlari = grup_baslangici_order_icinde + yerel_idx
    idx_kj = order[order_pozisyonlari]

    gecerli = idx_kj != idx_ji  # kendi-kendine ucgen (e_kj == e_ji) elenir
    idx_kj = idx_kj[gecerli]
    idx_ji = idx_ji[gecerli]

    if idx_kj.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=dev)
        return empty, empty, torch.empty(0, dtype=pos.dtype, device=dev)

    r_ji = pos[dst[idx_ji]] - pos[src[idx_ji]]   # i - j
    r_kj = pos[dst[idx_kj]] - pos[src[idx_kj]]   # j - k
    cos_theta = (r_ji * r_kj).sum(-1) / (r_ji.norm(dim=-1) * r_kj.norm(dim=-1) + 1e-8)
    theta = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    return idx_kj, idx_ji, theta
