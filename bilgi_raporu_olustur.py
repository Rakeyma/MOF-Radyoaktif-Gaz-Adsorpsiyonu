"""
bilgi_raporu_olustur.py
=========================
STATİK bir metodoloji raporu üretir — rapor_olustur.py'nin AKSİNE, hiçbir
CSV/JSON çıktısını OKUMAZ; sadece bu projenin mimarisini/tasarım
kararlarını (5 bileşen, 11 model, 4 XAI yöntemi, transfer learning şeması)
sabit metin olarak açıklar. "Üç Boyutlu Kristal Malzemeler/
bilgi_raporu_olustur.py" ile AYNI ilke (dinamik DEĞİL, methodology-only).

STİL: rapor_olustur.py ile TUTARLI - başlıklar (title + tüm add_heading
seviyeleri) DAHİL rapor TAMAMEN siyah-beyazdır. §2'de veri kaynağının
(Hugging Face jablonkagroup/core_mof_no_topo, CoRE-MOF türevi) tam atfı
verilir.

Çalıştırma:
    cd "MOF Radyoaktif Gaz Adsorpsiyonu" && python bilgi_raporu_olustur.py
Çıktı: MOF_Radyoaktif_Gaz_Adsorpsiyonu_Bilgi_RAPORU.docx
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

from egitim_ortak import (
    K_FOLDS, MAX_EPOCHS, EARLY_STOP_PATIENCE, LR, WEIGHT_DECAY, BATCH_SIZE,
    HIDDEN_DIM, FREEZE_ENCODER_EPOCHS, PRETRAIN_MAX_EPOCHS, PRETRAIN_PATIENCE, SEED,
)
from graf_ozellik_ortak import CUTOFF, EMB_DIM

PROJECT_ROOT = Path(__file__).resolve().parent


def h(doc, text, level=1):
    """TÜM başlıklar SİYAH kalın metin - rapor_olustur.py ile TUTARLI
    (rapor tamamen siyah-beyaz, grafikler hariç)."""
    sizes = {0: 18, 1: 16, 2: 14, 3: 12}
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = True; r.font.size = Pt(sizes.get(level, 12))
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def para(doc, text, size=10, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text); r.font.size = Pt(size); r.bold = bold
    return p


def bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text); r.font.size = Pt(size)
    return p


def citation_box(doc, text, size=9.5):
    tablo = doc.add_table(rows=1, cols=1)
    tablo.style = "Table Grid"
    cell = tablo.rows[0].cells[0]
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(text); r.font.size = Pt(size); r.italic = True
    return tablo


def references_list(doc, refs, size=8.5):
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        r = p.add_run(f"[{i}] {ref}")
        r.font.size = Pt(size)


def main() -> None:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10)

    h(doc, "MOF Radioactive Gas Adsorption — Methodology Report", level=0)
    para(doc, "Static architecture/methodology document (does not read live results — see "
              "'MOF_Radyoaktif_Gaz_Adsorpsiyonu_Raporu.docx' for actual run outputs).", bold=True)

    h(doc, "1. Objective")
    para(doc, "Predict Xenon (Xe) and Krypton (Kr) adsorption capacity, Xe/Kr selectivity, and Iodine "
              "(I2) adsorption capacity in Metal-Organic Frameworks (MOFs) for nuclear waste off-gas "
              "management, using an end-to-end pipeline covering data acquisition, literature mining, "
              "structural data augmentation, 11 Graph Neural Network architectures with transfer "
              "learning, and 4 Explainable AI methods.")

    h(doc, "2. Data Source & Citation")
    para(doc, "Base MOF structures (CIF + full 3D coordinates) and GCMC-simulated CO2/CH4 adsorption "
              "proxy properties are fetched from the Hugging Face dataset 'jablonkagroup/"
              "core_mof_no_topo' (CC BY 4.0), a CoRE-MOF-derived corpus. Any use of this pipeline's "
              "output data must cite the sources below.")
    doc.add_paragraph()
    citation_box(doc, "Dataset: jablonkagroup/core_mof_no_topo. Hugging Face Datasets. "
                       "License: CC BY 4.0. https://huggingface.co/datasets/jablonkagroup/core_mof_no_topo")
    doc.add_paragraph()
    references_list(doc, [
        "Jablonka, K. M.; Rosen, A. S.; Krishnapriyan, A. S.; Smit, B. An Ecosystem for Digital "
        "Reticular Chemistry. ACS Cent. Sci. 2023, 9 (4), 563-581. https://doi.org/10.1021/acscentsci.2c01177.",
        "Chung, Y. G. et al. Computation-Ready, Experimental Metal-Organic Frameworks: A Tool To "
        "Enable High-Throughput Screening of Nanoporous Crystals. Chem. Mater. 2014, 26 (21), "
        "6185-6192. https://doi.org/10.1021/cm502594j.",
        "Chung, Y. G. et al. Advances, Updates, and Analytics for the Computation-Ready, Experimental "
        "Metal-Organic Framework Database: CoRE MOF 2019. J. Chem. Eng. Data 2019, 64 (12), "
        "5985-5998. https://doi.org/10.1021/acs.jced.9b00835.",
        "Sikora, B. J.; Wilmer, C. E.; Greenfield, M. L.; Snurr, R. Q. Thermodynamic Analysis of "
        "Xe/Kr Selectivity in over 137,000 Hypothetical Metal-Organic Frameworks. Chem. Sci. 2012, "
        "3, 2217-2223. https://doi.org/10.1039/C2SC01097F (physical basis of the pore-size/"
        "selectivity proxy correlation, see eslesme_4_dataset_birlestirici.py).",
    ])

    h(doc, "3. Five-Component Pipeline")
    bullet(doc, "Component 1 — Reporting standard: all plots hardcoded to 600 DPI .tif, bold English "
                "fonts/labels/legends/titles (grafik_ortak.py rcParams). In .docx reports, only "
                "headings carry color; all body text/tables are black-and-white — only the embedded "
                "plots remain in color.")
    bullet(doc, "Component 2 — Database/API scraping (veri_indirici_1_jarvis_core_mof.py): Hugging Face "
                "`datasets` (jablonkagroup/core_mof_no_topo, see §2) as primary source, CoRE-MOF open "
                "CSV as secondary, and a procedural generator seeded from 12 well-known real MOF "
                "families as a tertiary fallback that NEVER blocks the pipeline. Pore volume / void "
                "fraction / surface area / LCD / PLD are estimated via a lightweight geometric proxy "
                "when not directly supplied.")
    bullet(doc, "Component 3 — NLP literature mining (nlp_literatur_madencilik_2.py): CrossRef + arXiv "
                "APIs (keyless), regex-based extraction of MOF names + numeric capacity values "
                "(mmol/g, cm3/g, wt%) from real abstracts/titles near Xe/Kr/I2 keywords. Writes an empty "
                "CSV (never fabricated values) when no extractable text is returned.")
    bullet(doc, "Component 4 — Structural data augmentation (veri_artirma_3_augmentasyon.py, pymatgen): "
                "missing-linker defects, metal-node substitution, -CH3/-NH2 functional group addition, "
                "and 3D Gaussian coordinate noise — expanding hundreds of base MOFs into thousands of "
                "augmented samples while preserving base_mof_id for leakage-safe K-fold grouping.")
    bullet(doc, "Component 5 — Transfer learning (egitim_ortak.py): Stage A pretrains each encoder on a "
                "large proxy dataset (formation-energy/heat-of-adsorption proxy target) and saves ONLY "
                "the encoder weights. Stage B loads those weights, freezes the encoder for the first N "
                "epochs (linear probing), then unfreezes for full fine-tuning on the augmented gas-"
                "adsorption dataset (4 targets, masked multi-task loss for sparse labels).")

    h(doc, "4. Hyperparameters (Code Defaults)")
    para(doc, "This is a STATIC methodology document, so the values below are the CODE "
              "DEFAULTS (egitim_ortak.py) shared by all 11 models, imported programmatically "
              "(never hand-typed) so this text cannot drift from the source code. Every "
              "default can be overridden per-run via an environment variable without "
              "touching the code (e.g. for a fast smoke-test). The hyperparameters ACTUALLY "
              "used in a given run (which may differ from these defaults if an override was "
              "set) are auto-documented — read live from that run's "
              "<Model>/sonuclar/metrikler.json — in §1 of the companion dynamic report "
              "'MOF_Radyoaktif_Gaz_Adsorpsiyonu_Raporu.docx'.", size=9)
    doc.add_paragraph()
    for label, val, env_var in [
        ("K (K-Fold count)", K_FOLDS, "KFOLD_OVERRIDE"),
        ("Fine-tune max epochs", MAX_EPOCHS, "MAX_EPOCHS_OVERRIDE"),
        ("Early-stopping patience", EARLY_STOP_PATIENCE, "PATIENCE_OVERRIDE"),
        ("Encoder freeze duration (epochs)", FREEZE_ENCODER_EPOCHS, "FREEZE_ENCODER_EPOCHS_OVERRIDE"),
        ("Learning rate (AdamW)", LR, "—"),
        ("Weight decay", WEIGHT_DECAY, "—"),
        ("Batch size", BATCH_SIZE, "BATCH_SIZE_OVERRIDE"),
        ("Hidden dim (encoder width)", HIDDEN_DIM, "—"),
        ("Output embedding dim", EMB_DIM, "—"),
        ("Radius-graph cutoff", f"{CUTOFF} Å", "—"),
        ("Random seed", SEED, "—"),
        ("Pretrain max epochs", PRETRAIN_MAX_EPOCHS, "PRETRAIN_MAX_EPOCHS_OVERRIDE"),
        ("Pretrain patience", PRETRAIN_PATIENCE, "PRETRAIN_PATIENCE_OVERRIDE"),
    ]:
        bullet(doc, f"{label}: {val}  (env override: {env_var})")

    h(doc, "5. Terminology Glossary")
    para(doc, "Plain-language definitions of abbreviations/terms used throughout both "
              "reports:", size=9)
    doc.add_paragraph()
    for terim, acilim, aciklama in [
        ("PLD", "Pore Limiting Diameter", "Diameter of the largest sphere that can pass "
         "all the way through a MOF's pore network — the diffusion 'bottleneck'. If a "
         "gas molecule's kinetic diameter exceeds the PLD, it cannot pass through that pore."),
        ("LCD", "Largest Cavity Diameter", "Diameter of the largest sphere that fits inside "
         "the MOF's largest internal cavity. Different from PLD: LCD measures internal pore "
         "SIZE, PLD measures the inter-pore passage bottleneck (LCD ≥ PLD always)."),
        ("Kinetic Diameter", "—", "Effective size of a gas molecule governing its diffusion/"
         "sieving behavior (Xe ≈ 4.0-4.4 Å, Kr ≈ 3.6-3.8 Å); compared against PLD to predict "
         "size-sieving strength."),
        ("Void Fraction", "—", "Fraction (0-1) of a MOF unit cell's volume that is empty/"
         "accessible pore space."),
        ("Open Metal Site", "—", "A metal center with unsaturated coordination that can bind "
         "a gas molecule directly once solvent is removed — strengthens Xe/I2 adsorption."),
        ("Xe/Kr Selectivity", "—", "Ratio describing how preferentially a MOF adsorbs Xe over "
         "Kr — the key metric for separating radioactive off-gas isotopes."),
        ("R² / MAE / RMSE / MedianAE / MaxError / PearsonR", "regression metrics",
         "R²=variance explained (1.0=perfect); MAE=mean absolute error (same units as "
         "target); RMSE=root-mean-squared error (penalizes large errors more than MAE); "
         "MedianAE=typical error, outlier-robust; MaxError=single worst-case error; "
         "PearsonR=linear correlation strength/direction (-1 to +1)."),
        ("K-Fold / OOF (Out-of-Fold)", "—", "Data split into K parts; each fold is held out "
         "as test while the rest train. An 'OOF prediction' is only ever produced while a "
         "sample was in the TEST fold — no sample is ever evaluated with its own training "
         "data, which is why pooled OOF metrics are a fair overfitting check."),
        ("Permutation Importance (ΔMAE)", "—", "A feature group's values are randomly "
         "shuffled across samples and the resulting MAE degradation is measured; larger "
         "positive ΔMAE = the model depends on that feature group more."),
        ("Feature-importance chart Roman numerals (I, II, III, ...)", "—", "Y-axis labels "
         "are assigned by IMPORTANCE RANK (most to least important), so which numeral maps "
         "to which feature group differs per model/run. The exact per-model mapping is "
         "written to <Model>/sonuclar/grafikler/Feature_Importance.txt and reproduced as a "
         "table under each model's chart in §6.7 of the dynamic report."),
        ("Aux (auxiliary) feature groups", "Pore Geometry / Surface Area / Structural-Size / "
         "Chemical Modification / Composition-Derived / Crystal Structure",
         "The 6 categories used in permutation importance and GraphLIME: 'Crystal Structure' "
         "is the 3D atomistic arrangement learned by the graph encoder; the other 5 are "
         "precomputed porosity/composition numeric features (Pore Geometry = pore_volume/"
         "void_fraction/LCD/PLD; Surface Area = gravimetric+volumetric surface area; "
         "Structural/Size = density/atom+element count/metal fraction; Chemical "
         "Modification = open metal site+functional group; Composition-Derived = mean "
         "electronegativity difference/atomic radius/atomic number)."),
    ]:
        para(doc, f"{terim}" + (f" ({acilim})" if acilim != "—" else ""), bold=True, size=9)
        para(doc, aciklama, size=8.5)
        doc.add_paragraph()

    h(doc, "6. Eleven GNN Architectures (Shared Interface)")
    para(doc, "All encoders implement encoder.forward(g: dict) -> Tensor[n_graphs, 64] and share "
              "graf_ozellik_ortak.py (periodic 3D radius graph, cutoff=8.0 Å) + egitim_ortak.py "
              "(K-fold, early stopping, checkpointing, transfer learning, permutation importance).")
    for line in [
        "GraphGPS — GPSConv (local GINEConv + global multi-head attention).",
        "PNA-GNN — PNAConv (mean/min/max/std aggregators x identity/amplification/attenuation scalers).",
        "GIN — GINEConv (edge-feature-aware Graph Isomorphism Network).",
        "GAT — GATv2Conv (edge-distance-aware attention).",
        "GatedGCN — ResGatedGraphConv (learned edge gating).",
        "DeeperGCN — GENConv + DeepGCNLayer 'res+' blocks (8 layers).",
        "ECC — NNConv (dynamic edge-conditioned filter/weight generation).",
        "TFN — Tensor Field Network, from scratch, l≤1 Clebsch-Gordan coupling (no e3nn).",
        "EGNN — E(n)-Equivariant GNN, from scratch, invariant-distance message passing (lightest — chosen for XAI).",
        "SE(3)-Transformer — from scratch, l≤1 CG coupling + multi-head equivariant attention.",
        "DimeNet++ (autonomously added) — from scratch directional/angular message passing over (k,j,i) "
        "triplets (Fourier angular basis, fully vectorized triplet construction), chosen because "
        "pore-window angular geometry directly governs size-selective Xe/Kr/I2 sieving.",
    ]:
        bullet(doc, line)

    h(doc, "7. Four XAI Methods (applied to EGNN — lightest model)")
    for line in [
        "GraphLIME — local linear (Lasso) surrogate over Bernoulli atom masks, per target.",
        "Edge Attribution — vanilla gradient saliency + Integrated Gradients on an edge (RBF-output) mask.",
        "SubgraphX — Monte Carlo Tree Search over connected atom subsets, UCT selection, multi-target "
        "standardized-space reward.",
        "Integrated Gradients (autonomously added) — true continuous-input IG directly on atomic 3D "
        "coordinates and porosity/composition features (not a mask), satisfying the completeness axiom.",
    ]:
        bullet(doc, line)

    h(doc, "8. Honesty / Limitations")
    bullet(doc, "Gas-adsorption labels are a MIX of real NLP-mined literature values (rare, only applied "
                "to non-augmented originals) and physically-motivated proxy correlations — never "
                "presented as ground-truth experimental data (see 'label_source_<target>' columns).")
    bullet(doc, "Procedural fallback structures (when live API access is unavailable) are simplified "
                "node+linker skeletons, not crystallographically refined structures.")
    bullet(doc, "Default MAX_MATERIALS/MAX_MATERIALS_PRETRAIN/N_AUGMENT_PER_BASE are kept small for fast "
                "validation; scaling to production size requires only environment-variable changes.")

    out_path = PROJECT_ROOT / "MOF_Radyoaktif_Gaz_Adsorpsiyonu_Bilgi_RAPORU.docx"
    doc.save(str(out_path))
    print(f"Bilgi raporu kaydedildi -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
