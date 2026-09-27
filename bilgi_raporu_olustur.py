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
from paths import GITHUB_REPO_URL
from kaynakca import KAYNAKLAR, atif
from veri_indirici_1_jarvis_core_mof import MAX_MATERIALS, MAX_MATERIALS_PRETRAIN, ARKETIPLER
from veri_artirma_3_augmentasyon import N_AUGMENT_PER_BASE

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
    doc.add_paragraph()

    h(doc, "Abstract")
    para(doc, "This document specifies the methodology of an end-to-end graph neural network "
              "(GNN) pipeline for predicting xenon and krypton adsorption capacity, Xe/Kr "
              "selectivity and iodine (I2) adsorption capacity in metal-organic frameworks "
              "(MOFs), motivated by radioactive off-gas separation in nuclear waste "
              "management. The pipeline comprises five components: a fixed publication-grade "
              "reporting standard, database/API acquisition of MOF structures, NLP-based "
              "literature mining, pymatgen-based structural augmentation, and a two-stage "
              "transfer-learning protocol (proxy-target pretraining followed by multi-task "
              "fine-tuning under a leakage-safe, structure-grouped K-fold split). Eleven GNN "
              "architectures spanning message-passing, attention-based, edge-conditioned and "
              "E(3)-equivariant families are implemented behind a single shared encoder "
              "interface, and four complementary explainability methods are applied to the "
              "lightest of them. This document describes what the pipeline DOES; the "
              "companion dynamic report states what a given run actually PRODUCED, including "
              "which architectures were trained and how the target labels were generated. "
              "Readers should consult §9 before citing any metric: in the published run all "
              "target labels are synthetic.", size=9)
    doc.add_paragraph()

    h(doc, "1. Objective")
    para(doc, "Predict Xenon (Xe) and Krypton (Kr) adsorption capacity, Xe/Kr selectivity, and Iodine "
              "(I2) adsorption capacity in Metal-Organic Frameworks (MOFs) for nuclear waste off-gas "
              "management, using an end-to-end pipeline covering data acquisition, literature mining, "
              "structural data augmentation, 11 Graph Neural Network architectures with transfer "
              "learning, and 4 Explainable AI methods.")

    h(doc, "2. Data Source & Citation")
    para(doc, f"Base MOF structures (CIF + full 3D coordinates) are fetched from the Hugging Face "
              f"dataset 'jablonkagroup/core_mof_no_topo' (CC BY 4.0) {atif('jablonka2023')}, a "
              f"CoRE-MOF-derived corpus {atif('chung2014', 'chung2019')}. "
              "That dataset carries no DFT formation energy, so the pretraining target is taken "
              "from its GCMC-simulated CO2 heat of adsorption (Widom insertion, kJ/mol converted "
              "to eV) — used purely as an ENERGETIC PROXY and named accordingly "
              "('formation_energy_eV_atom_proxy'). No Xe, Kr or I2 adsorption values exist in this "
              f"source; see §9 for how the gas-adsorption labels are actually produced. The "
              f"dataset itself must be cited with the record below in any publication that uses "
              f"this pipeline's output; the full reference list is given at the end of this "
              f"document.")
    doc.add_paragraph()
    citation_box(doc, "Dataset: jablonkagroup/core_mof_no_topo. Hugging Face Datasets. "
                       "License: CC BY 4.0. https://huggingface.co/datasets/jablonkagroup/core_mof_no_topo")

    h(doc, "3. Five-Component Pipeline")
    bullet(doc, "Component 1 — Reporting standard: all plots hardcoded to 600 DPI .tif (each also "
                "saved as .png), bold English fonts/labels/legends/titles (grafik_ortak.py rcParams). "
                "Both .docx reports are entirely BLACK-AND-WHITE — headings included (every heading "
                "run is explicitly set to RGB 0,0,0) and tables use plain 'Table Grid' borders with "
                "no fills or zebra striping; the only colour in either document comes from the "
                "embedded matplotlib figures.")
    bullet(doc, f"Component 2 — Database/API scraping (veri_indirici_1_jarvis_core_mof.py): Hugging "
                f"Face `datasets` (jablonkagroup/core_mof_no_topo, see §2) as primary source, CoRE-MOF "
                f"open CSV as secondary, and a procedural generator seeded from {len(ARKETIPLER)} "
                f"well-known real MOF families as a tertiary fallback that NEVER blocks the pipeline. "
                f"Pore volume / void fraction / surface area / LCD / PLD are NOT taken from Zeo++ or "
                f"experiment (Zeo++ is not available in this environment) — they are estimated from "
                f"the structure by a lightweight geometric approximation "
                f"(estimate_pore_proxies(): van-der-Waals volume packing for void fraction, shortest "
                f"lattice vector scaled by sqrt(void fraction) for LCD, and PLD = 0.55 x LCD). "
                f"They must not be cited as measured BET areas or pore diameters.")
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
              "DEFAULTS (egitim_ortak.py), imported programmatically (never hand-typed) so "
              "this text cannot drift from the source code. Every default can be overridden "
              "per-run via an environment variable without touching the code (e.g. for a fast "
              "smoke-test), and an override may be applied to one model but not another — so "
              "these defaults must NOT be read as 'what every model actually used'. The "
              "hyperparameters ACTUALLY used in a given run are auto-documented — read live "
              "from each model's <Model>/sonuclar/metrikler.json, with any value that differs "
              "between models broken out per-model — in §1 of the companion dynamic report "
              "'MOF_Radyoaktif_Gaz_Adsorpsiyonu_Raporu.docx'. That report is also the only "
              "place that states which architectures were actually TRAINED in a given run; "
              "this document describes every architecture IMPLEMENTED in the repository.", size=9)
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
        ("Confusion-matrix classes", "—", "These are regression targets, so the confusion "
         "matrices do NOT use fixed physical thresholds. Classes are derived from each "
         "target's own QUARTILE distribution of true values (Q1/median/Q3), giving 4 "
         "equal-sized relative bins: <Q1 / Q1-median / median-Q3 / >Q3 ('low / mid-low / "
         "mid-high / high'). They are recomputed per target and per model, so the class "
         "boundaries differ between figures — each figure's actual numeric boundaries are "
         "printed in §6 of the dynamic report (see grafik_ortak._dinamik_esikler)."),
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
        f"GraphGPS {atif('rampasek2022')} — GPSConv (local GINEConv + global multi-head attention).",
        f"PNA-GNN {atif('corso2020')} — PNAConv (mean/min/max/std aggregators x identity/amplification/attenuation scalers).",
        f"GIN {atif('xu2019')} — GINEConv (edge-feature-aware Graph Isomorphism Network).",
        f"GAT {atif('brody2022')} — GATv2Conv (edge-distance-aware attention).",
        f"GatedGCN {atif('bresson2017')} — ResGatedGraphConv (learned edge gating).",
        f"DeeperGCN {atif('li2020')} — GENConv + DeepGCNLayer 'res+' blocks (8 layers).",
        f"ECC {atif('simonovsky2017')} — NNConv (dynamic edge-conditioned filter/weight generation).",
        f"TFN {atif('thomas2018')} — Tensor Field Network, from scratch, l≤1 Clebsch-Gordan coupling (no e3nn).",
        f"EGNN {atif('satorras2021')} — E(n)-Equivariant GNN, from scratch, invariant-distance message passing (lightest — chosen for XAI).",
        f"SE(3)-Transformer {atif('fuchs2020')} — from scratch, l≤1 CG coupling + multi-head equivariant attention.",
        f"DimeNet++ {atif('gasteiger2020')} (autonomously added) — from scratch directional/angular message passing over (k,j,i) "
        "triplets (Fourier angular basis, fully vectorized triplet construction), chosen because "
        "pore-window angular geometry directly governs size-selective Xe/Kr/I2 sieving.",
    ]:
        bullet(doc, line)

    h(doc, "7. Four XAI Methods (applied to EGNN — lightest model)")
    for line in [
        f"GraphLIME {atif('huang2020', 'ribeiro2016')} — local linear surrogate over Bernoulli atom masks, fitted per target with a "
        "CROSS-VALIDATED Lasso (LassoCV). The regularization strength is selected per sample from "
        "the data rather than fixed: with a hardcoded alpha, the masking-induced prediction deltas "
        "of these large MOFs (72-172 atoms, whose per-atom effect is heavily diluted by 4 message-"
        "passing layers + LayerNorm) fell below the penalty threshold and every coefficient "
        "collapsed to exactly zero, producing empty explanations.",
        "Edge Attribution — vanilla gradient saliency + Integrated Gradients on an edge (RBF-output) mask.",
        f"SubgraphX {atif('yuan2021')} — Monte Carlo Tree Search over connected atom subsets, UCT selection, multi-target "
        "standardized-space reward.",
        f"Integrated Gradients {atif('sundararajan2017')} (autonomously added) — true continuous-input IG directly on atomic 3D "
        "coordinates and porosity/composition features (not a mask), satisfying the completeness axiom.",
    ]:
        bullet(doc, line)

    h(doc, "8. Code & Data Availability")
    para(doc, "The complete pipeline — all source code, the generated dataset, model "
              "checkpoints, per-model result CSVs and figures — is publicly available at:")
    doc.add_paragraph()
    citation_box(doc, GITHUB_REPO_URL)
    doc.add_paragraph()
    para(doc, "Repository contents and deliberate exclusions:", bold=True, size=9)
    bullet(doc, "Included: every .py script, the final datasets under data/ (raw + processed, "
                "including the CIF structures), per-model checkpoints (.pt), result CSVs/JSONs "
                "(metrikler.json, test_tahminleri_oof.csv, kfold_metrikleri.csv), XAI outputs, "
                "and all figures in .png form.", size=9)
    bullet(doc, "Excluded via .gitignore (regenerable from the code, and too large for git): the "
                "600 DPI .tif figures (40-90 MB each, whose .png twins ARE included), the "
                "~98 MB results .docx that embeds them, and one raw per-edge XAI dump "
                "(edge_attribution_kenarlar.csv, 79 MB). Re-running the corresponding "
                "grafik.py / rapor_olustur.py reproduces all of them.", size=9)
    bullet(doc, "Data provenance and the synthetic-vs-real status of every input and label is "
                "documented in VERI_KAYNAGI_VE_SINIRLAMALAR.md in the repository root — read "
                "it before citing any metric from these reports (see also §9).", size=9)

    h(doc, "9. Honesty / Limitations")
    bullet(doc, "CRITICAL — the target labels in the published run are 100% SYNTHETIC: all "
                "2100 rows of all four targets are 'PROXY_PORE_CORRELATION', generated by a "
                "closed-form formula over porosity descriptors plus lognormal noise. They are "
                "not experimental, not GCMC-simulated and not literature-derived. Moreover the "
                "formula's inputs (pld_A, pore_volume_cm3_g, open_metal_site, "
                "has_functional_group) are themselves fed to the model as aux input features, "
                "so the model is inverting a formula from its own inputs. The reported R2 "
                "therefore validates that the ML pipeline runs correctly end-to-end — it does "
                "NOT demonstrate real Xe/Kr/I2 adsorption prediction and must not be presented "
                "as a materials-discovery result.")
    bullet(doc, "The label pipeline is DESIGNED to mix NLP-mined literature values (applied only to "
                "non-augmented originals) with physically-motivated proxy correlations, and records "
                "the origin of every row in a 'label_source_<target>' column. In practice the NLP "
                "stage has so far contributed zero usable labels — it extracted 2 candidate values, "
                "neither of which matched a structure in the dataset, and both were mis-parsed (the "
                "same 3.46 mmol/g was assigned to Xe and Kr, while the source sentence reports Kr as "
                "350 cm3/g). The regex extraction therefore needs fixing before this component can "
                "contribute real labels; until then every label is proxy-generated, as stated above.")
    bullet(doc, "Procedural fallback structures (when live API access is unavailable) are simplified "
                "node+linker skeletons, not crystallographically refined structures.")
    bullet(doc, f"Dataset size is capped by environment variables whose code defaults are "
                f"MAX_MATERIALS={MAX_MATERIALS} (fine-tune base MOFs), "
                f"MAX_MATERIALS_PRETRAIN={MAX_MATERIALS_PRETRAIN} and "
                f"N_AUGMENT_PER_BASE={N_AUGMENT_PER_BASE}. These are modest by design so the whole "
                f"pipeline can be validated quickly; scaling up requires only environment-variable "
                f"changes, no code edits. Note the caps are upper bounds — a run yields fewer rows "
                f"if the upstream source returns fewer usable structures (the published run "
                f"obtained 300 fine-tune base MOFs and 1500 pretrain structures).")

    h(doc, "References")
    para(doc, "Bibliographic records were compiled from the literature; volume/page details "
              "should be verified against the originals before submission. This list is shared "
              "with the companion dynamic report (kaynakca.py), so a given number refers to the "
              "same work in both documents.", size=8.5)
    doc.add_paragraph()
    references_list(doc, [kunye for _, kunye in KAYNAKLAR], size=9)

    out_path = PROJECT_ROOT / "MOF_Radyoaktif_Gaz_Adsorpsiyonu_Bilgi_RAPORU.docx"
    doc.save(str(out_path))
    print(f"Bilgi raporu kaydedildi -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
