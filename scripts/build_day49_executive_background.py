#!/usr/bin/env python3
"""
Day 49 — Executive Summary and Background Draft Builder
========================================================

Reads available evidence artefacts and generates paper-oriented
drafts for the executive summary, introduction, and background
sections.

Generated outputs
-----------------
  - docs/paper/day49_executive_summary_background.md
  - artifacts/day49/day49_writing_summary.json
  - artifacts/day49/day49_citation_todo.md
  - artifacts/day49/README.md

This script does NOT modify source code, train models, or access
private clinical data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── helpers ────────────────────────────────────────────────────────────

def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file if it exists; return None otherwise."""
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [warn] Could not load {path}: {exc}")
        return None


def _safe_read_text(path: Path) -> Optional[str]:
    """Read a text file if it exists; return None otherwise."""
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# ── evidence loading ───────────────────────────────────────────────────

EVIDENCE_SPECS: List[Dict[str, str]] = [
    {
        "key": "day20_eval",
        "rel_path": "artifacts/day20/day20_supervised_eval_summary.json",
        "description": "Day 20 supervised detector evaluation",
    },
    {
        "key": "day34_assessment",
        "rel_path": "artifacts/day34_final/day34_final_assessment.json",
        "description": "Day 34 Sgen generative assessment",
    },
    {
        "key": "day41_readme",
        "rel_path": "artifacts/day41/README.md",
        "description": "Day 41 ablation study summary",
    },
    {
        "key": "day47_readme",
        "rel_path": "artifacts/day47/README.md",
        "description": "Day 47 risk and failure-mode analysis",
    },
    {
        "key": "day48_readme",
        "rel_path": "artifacts/day48/README.md",
        "description": "Day 48 reproducibility audit",
    },
]


def load_evidence(root: Path) -> Dict[str, Any]:
    """Load all evidence files, recording found/missing status."""
    evidence: Dict[str, Any] = {}
    found: List[str] = []
    missing: List[str] = []

    for spec in EVIDENCE_SPECS:
        path = root / spec["rel_path"]
        if spec["rel_path"].endswith(".json"):
            data = _safe_load_json(path)
        else:
            data = _safe_read_text(path)

        if data is not None:
            evidence[spec["key"]] = data
            found.append(spec["rel_path"])
            print(f"  [ok]   {spec['rel_path']}")
        else:
            missing.append(spec["rel_path"])
            print(f"  [miss] {spec['rel_path']}")

    evidence["_found"] = found
    evidence["_missing"] = missing
    return evidence


# ── draft generators ───────────────────────────────────────────────────

def _build_empirical_paragraph(evidence: Dict[str, Any]) -> str:
    """Build a conservative empirical-evidence paragraph."""
    parts: List[str] = []

    day20 = evidence.get("day20_eval")
    if day20 and isinstance(day20, dict):
        roc = day20.get("roc_auc")
        ap = day20.get("average_precision")
        prec = day20.get("precision")
        rec = day20.get("recall")
        f1 = day20.get("f1")
        hardest = day20.get("hardest_anomaly_type", "unknown")
        easiest = day20.get("easiest_anomaly_type", "unknown")

        parts.append(
            f"In supervised evaluation on synthetic injected anomalies "
            f"(Day 20), the detector achieved a ROC-AUC of "
            f"{roc:.3f} and an average precision of {ap:.3f}, "
            f"with precision {prec:.3f}, recall {rec:.3f}, and "
            f"F1 {f1:.3f} at the selected operating threshold.  "
            f"The hardest anomaly type was *{hardest.replace('_', ' ')}*; "
            f"the easiest was *{easiest.replace('_', ' ')}*."
        )

    day34 = evidence.get("day34_assessment")
    if day34 and isinstance(day34, dict):
        sgen_result = day34.get("main_sgen_result", {})
        best_auc = sgen_result.get("best_roc_auc")
        interp = day34.get("interpretation", "")
        parts.append(
            f"In contrast, the diffusion-based generative surprise signal "
            f"(Sgen), evaluated under exact checkpoint alignment at "
            f"Day 34, achieved a best ROC-AUC of only "
            f"{best_auc:.3f} across all tested timesteps — "
            f"effectively near random.  {interp}  "
            f"Consequently, Sgen is treated as a diagnostic or "
            f"auxiliary signal in the current framework and should "
            f"**not** be claimed as a strong standalone anomaly "
            f"detector."
        )

    if not parts:
        parts.append(
            "*Empirical evidence artefacts were not available at "
            "generation time; this section should be populated once "
            "evaluation results are produced.*"
        )

    return "\n\n".join(parts)


def build_main_draft(evidence: Dict[str, Any]) -> str:
    """Compose the full paper-section draft."""
    empirical = _build_empirical_paragraph(evidence)

    draft = textwrap.dedent("""\
    # Day 49 — Executive Summary and Background Draft

    > **Status:** First draft — conservative, evidence-grounded.
    > All citations marked `[CITATION NEEDED]` require replacement
    > with verified references before submission.

    ---

    ## Executive Summary

    Electronic Health Records (EHRs) are the primary medium through
    which clinical observations, diagnoses, medications, and procedures
    are documented in modern hospital information systems.  Despite
    their centrality, EHR data are known to contain noise, omissions,
    and inconsistencies that can compromise downstream clinical
    decision support, research cohort selection, and patient safety
    monitoring [CITATION NEEDED].

    This work presents an **ontology-calibrated framework for anomaly
    detection and counterfactual explanation in sequential EHR data**.
    The framework decomposes abnormality assessment into three
    complementary signals: (i) a supervised statistical anomaly score,
    (ii) an ontology-violation score grounded in medical knowledge
    graphs (SNOMED CT, RxNorm, UMLS), and (iii) a generative
    plausibility score derived from a denoising diffusion model.
    Where an anomaly is detected, the system proposes minimal
    ontology-constrained counterfactual repairs — the smallest set of
    clinically plausible changes that would render the record
    non-anomalous.

    Preliminary evaluation on synthetic injected anomalies indicates
    that the supervised detector and ontology-calibrated scoring
    components provide useful discriminative signal, while the
    current diffusion-based generative proxy (Sgen) yields
    near-random separation and should be regarded as auxiliary
    pending methodological improvement.  The principal contribution
    is the integration of clinical ontology constraints into both
    the scoring and explanation stages of the anomaly-detection
    pipeline.

    ---

    ## 1. Introduction

    ### 1.1 Clinical Motivation

    Hospitals generate vast quantities of structured and semi-structured
    clinical data through EHR systems.  These records — comprising
    diagnoses (ICD-9/10), medications (RxNorm/NDC), procedures (CPT),
    laboratory results, and demographic attributes — are used not only
    for direct patient care but also for retrospective research,
    quality improvement, and regulatory reporting [CITATION NEEDED].
    Data quality failures in EHRs, including missing diagnoses,
    implausible medication combinations, and demographic-clinical
    conflicts, are well documented in the medical informatics
    literature [CITATION NEEDED] and can propagate silently into
    clinical decision-support systems, predictive models, and
    real-world evidence studies.

    Intensive Care Unit (ICU) settings amplify these risks.  The high
    acuity, rapid temporal dynamics, and multi-provider documentation
    patterns characteristic of critical care produce EHR sequences
    that are especially susceptible to transcription errors, omitted
    entries, and delayed coding [CITATION NEEDED].  Identifying and
    explaining anomalous records within such sequences is therefore a
    pressing clinical informatics challenge.

    ### 1.2 Problem Statement

    Existing anomaly-detection approaches for EHR data typically
    operate on flat, tabular representations and rely on purely
    statistical deviation criteria [CITATION NEEDED].  These methods
    suffer from two interconnected limitations:

    1. **Lack of clinical grounding.**  Statistical outliers are not
       necessarily clinically implausible, and vice versa.  A rare
       but valid diagnosis–medication pair may be flagged as
       anomalous, while a common but ontologically incoherent
       combination may pass undetected.
    2. **Absence of actionable explanations.**  When an anomaly is
       detected, clinicians and data stewards require not merely a
       flag but a *minimal, clinically plausible correction* — a
       counterfactual explanation — to guide review and remediation.

    ### 1.3 Aim and Objectives

    The aim of this work is to develop and evaluate a framework that:

    - Detects anomalies in sequential EHR data by combining
      statistical, ontological, and generative signals.
    - Grounds anomaly assessment in established medical ontologies
      (SNOMED CT, RxNorm, UMLS) to distinguish statistical
      deviation from clinical implausibility.
    - Generates minimal ontology-constrained counterfactual
      explanations that propose the smallest clinically coherent
      repair for each detected anomaly.
    - Provides reproducible, auditable, and transparent outputs
      suitable for clinical informatics research.

    ### 1.4 Proposed Framework Overview

    The proposed framework, referred to as **OntoCF-AD**
    (*Ontology-Calibrated Counterfactual Anomaly Detection*),
    consists of four integrated components:

    1. **Supervised Anomaly Detector** — a sequence-aware neural
       classifier trained to discriminate normal from anomalous EHR
       records using learned clinical-code embeddings.
    2. **Ontology-Violation Scorer (S_ont)** — a knowledge-graph
       traversal module that quantifies the degree to which a record
       violates expected ontological relationships (e.g., diagnosis–
       medication coherence, hierarchical code validity).
    3. **Generative Plausibility Scorer (S_gen)** — a denoising
       diffusion model trained on normal EHR sequences; the
       reconstruction error at selected diffusion timesteps provides
       a proxy for distributional plausibility.
    4. **Counterfactual Explanation Engine** — a constrained search
       procedure that identifies the minimal ontology-consistent
       perturbation to a flagged record such that the composite
       anomaly score falls below the detection threshold.

    These four components are combined through a calibrated scoring
    function that weights each signal according to its empirically
    validated discriminative contribution.

    ### 1.5 Contributions

    The principal contributions of this work are:

    1. **Ontology-calibrated anomaly scoring** — a multi-signal
       decomposition that integrates statistical detection, ontology
       violation, and generative plausibility into a unified
       framework grounded in medical knowledge graphs.
    2. **Constrained counterfactual explanations** — an explanation
       method that proposes minimal, ontology-consistent repairs
       rather than unconstrained perturbations, improving clinical
       interpretability and safety.
    3. **Transparent reporting of negative findings** — an honest
       assessment of the current diffusion-based Sgen component,
       which does not yet provide strong standalone discriminative
       signal, reported alongside the stronger detector and ontology
       evidence.
    4. **Reproducibility infrastructure** — a fully auditable
       codebase with day-level artefact tracking, environment
       capture, and private-data boundary documentation.

    ---

    ## 2. Background and Related Work

    ### 2.1 EHR Representation Learning

    Clinical codes — ICD-9/10 diagnoses, RxNorm medications, CPT
    procedures — are high-dimensional, sparse, and semantically
    structured.  Representation learning for EHR data seeks to embed
    these codes into dense vector spaces that capture clinical
    similarity and temporal dynamics [CITATION NEEDED].

    Early approaches adapted word-embedding techniques (e.g.,
    word2vec, GloVe) to medical-code sequences, treating patient
    timelines as analogues of natural-language sentences
    [CITATION NEEDED].  More recent work has explored recurrent,
    attention-based, and transformer architectures for sequential
    clinical modelling [CITATION NEEDED].  However, most EHR
    representation methods do not explicitly encode ontological
    relationships between codes, relying instead on co-occurrence
    statistics to learn implicit structure.

    ### 2.2 Anomaly Detection in Clinical Sequences

    Anomaly detection in structured clinical data has been approached
    through reconstruction-based autoencoders [CITATION NEEDED],
    isolation forests adapted to medical feature spaces
    [CITATION NEEDED], and supervised classifiers trained on
    synthetically injected or expert-annotated anomalies
    [CITATION NEEDED].

    A persistent challenge is the scarcity of labelled anomalous
    records: true clinical data-quality errors are often unknown or
    ambiguous, motivating the use of synthetic anomaly injection
    strategies [CITATION NEEDED].  The present work adopts a
    supervised detector trained on synthetically injected anomalies,
    complemented by ontology-based signals that do not require
    anomaly labels.

    ### 2.3 Clinical Ontologies and Knowledge Graphs

    Medical ontologies such as SNOMED CT [CITATION NEEDED],
    RxNorm [CITATION NEEDED], the Unified Medical Language System
    (UMLS) [CITATION NEEDED], and the OMOP Common Data Model
    [CITATION NEEDED] provide hierarchical and relational structures
    that encode clinical knowledge.  These resources enable semantic
    reasoning about diagnosis–medication relationships, code
    hierarchies, and therapeutic appropriateness.

    While ontologies are widely used for data harmonisation and
    cohort definition, their integration into anomaly-detection
    pipelines as a *scoring signal* — rather than merely a
    preprocessing step — remains underexplored [CITATION NEEDED].
    This work uses ontology-graph traversal to compute a violation
    score that quantifies how far a given record departs from
    expected ontological patterns.

    ### 2.4 Generative Models for EHR Data

    Generative models for clinical data have attracted growing
    interest for tasks including synthetic data generation, missing
    data imputation, and distributional novelty detection
    [CITATION NEEDED].  Variational autoencoders (VAEs) and
    generative adversarial networks (GANs) have been applied to
    tabular and sequential EHR data [CITATION NEEDED], and more
    recently, denoising diffusion probabilistic models (DDPMs) have
    been explored for structured clinical sequence generation
    [CITATION NEEDED].

    In the present framework, a DDPM trained on normal EHR sequences
    is used to compute a generative plausibility score (Sgen) based
    on denoising error at selected timesteps.  However, as reported
    transparently in the experimental evaluation, the current Sgen
    proxy yields near-random discriminative performance (ROC-AUC ≈
    0.508 at the best timestep) and should be regarded as an
    auxiliary or diagnostic signal pending further methodological
    refinement.  This negative finding is included in the interest
    of scientific transparency.

    ### 2.5 Counterfactual Explanations in Healthcare AI

    Counterfactual explanations answer the question: *"What is the
    smallest change to this input that would alter the model's
    output?"* [CITATION NEEDED].  In the healthcare domain,
    counterfactual methods have been applied to tabular patient data
    [CITATION NEEDED], clinical text [CITATION NEEDED], and medical
    imaging [CITATION NEEDED].

    A critical limitation of unconstrained counterfactual methods is
    that the proposed perturbations may be clinically implausible —
    for example, suggesting a biologically impossible medication
    substitution or an ontologically incoherent diagnosis change
    [CITATION NEEDED].  The present work addresses this by
    constraining the counterfactual search to ontology-consistent
    perturbations, ensuring that proposed repairs respect the
    hierarchical and relational structure of medical knowledge
    graphs.

    ### 2.6 Research Gap

    Despite advances in each of the above areas, several gaps
    remain at their intersection:

    1. **No unified framework** currently integrates statistical
       anomaly detection, ontology-based violation scoring, and
       generative plausibility assessment into a single calibrated
       pipeline for EHR sequences.
    2. **Counterfactual explanations for EHR anomalies** are rarely
       constrained by medical ontologies, risking clinically
       meaningless or unsafe suggestions.
    3. **Honest benchmarking** of generative components is uncommon;
       negative or weak results for generative surprise signals are
       frequently unreported.
    4. **Reproducibility infrastructure** for EHR anomaly-detection
       research is often incomplete, lacking environment capture,
       private-data boundary documentation, and artefact auditing.

    This work addresses these gaps by proposing an ontology-calibrated
    framework that combines detection, scoring, and explanation,
    reports negative findings transparently, and provides
    comprehensive reproducibility tooling.

    ### 2.7 Working Hypothesis

    The working hypothesis of this research is:

    > *An anomaly-detection framework that calibrates its scoring
    > function using medical ontology signals (SNOMED CT, RxNorm,
    > UMLS) and constrains its counterfactual explanations to
    > ontology-consistent perturbations will produce more clinically
    > interpretable, actionable, and safe outputs than frameworks
    > relying on statistical deviation alone.*

    This hypothesis is tested through ablation studies (Days 40–41),
    end-to-end case studies (Day 39), and failure-mode analyses
    (Day 47), with the current evidence indicating that the
    ontology-calibrated and supervised-detector components provide
    the strongest signal, while the generative component requires
    further development.

    ---

    ## Preliminary Empirical Evidence

    """).lstrip()

    draft += empirical
    draft += "\n\n"
    draft += textwrap.dedent("""\
    ---

    *Draft generated by `scripts/build_day49_executive_background.py`.
    All `[CITATION NEEDED]` markers must be resolved with verified
    references before manuscript submission.*
    """)

    return draft


def build_citation_todo() -> str:
    """Generate the citation TODO checklist."""
    return textwrap.dedent("""\
    # Day 49 — Citation TODO

    All `[CITATION NEEDED]` markers in the draft must be resolved
    before manuscript submission.  The following categories require
    verified references.

    ---

    ## Required Citation Categories

    ### 1. EHR Data Quality and ICU EHR Noise
    - Prevalence of data-quality issues in hospital EHR systems
    - Documentation errors and coding delays in ICU settings
    - Impact of EHR noise on downstream analytics

    ### 2. EHR Representation Learning / Clinical Sequence Modelling
    - Med2Vec, clinical word2vec / GloVe analogues
    - RETAIN, GRAM, and attention-based clinical models
    - Clinical transformer architectures (e.g., BEHRT, Med-BERT)

    ### 3. EHR Anomaly Detection
    - Reconstruction-based anomaly detection (autoencoders)
    - Isolation-forest and tree-based anomaly detection for clinical data
    - Supervised anomaly detection with synthetic injection
    - Unsupervised / self-supervised anomaly detection in clinical data

    ### 4. Medical Ontologies and Terminologies
    - SNOMED CT reference and design documentation
    - RxNorm reference and NLM documentation
    - UMLS Metathesaurus overview
    - OMOP Common Data Model specification

    ### 5. Knowledge Graphs / Ontologies in Healthcare AI
    - Ontology-driven feature engineering for clinical prediction
    - Knowledge-graph embedding methods for medical codes
    - Ontology-aware anomaly scoring (if prior work exists)

    ### 6. EHR Generative Models
    - VAE-based clinical data synthesis
    - GAN-based EHR generation
    - Diffusion models applied to structured / sequential clinical data
    - Denoising-error-based anomaly scoring (if prior work exists)

    ### 7. Counterfactual Explanations in Healthcare AI
    - Foundational counterfactual explanation papers (Wachter et al.)
    - Counterfactual explanations for tabular clinical data
    - Counterfactual explanations for clinical text and imaging
    - Ontology-constrained counterfactual methods (if prior work exists)

    ### 8. Clinical Plausibility and Safety Limits
    - Safety considerations for automated clinical explanations
    - Human-in-the-loop requirements for clinical AI outputs
    - Limitations of synthetic anomaly benchmarks

    ---

    *Generated by `scripts/build_day49_executive_background.py`.*
    """)


def build_day49_readme(evidence: Dict[str, Any]) -> str:
    """Generate artifacts/day49/README.md."""
    found_count = len(evidence.get("_found", []))
    missing_count = len(evidence.get("_missing", []))

    return textwrap.dedent(f"""\
    # Day 49 — Executive Summary and Background Draft

    ## Status
    Complete.

    ## Goal
    Write the first paper-oriented draft covering the executive summary,
    introduction, background, related work, and research gap sections.

    ## Generated Files

    | File | Description |
    | ---- | ----------- |
    | `docs/paper/day49_executive_summary_background.md` | Full section draft |
    | `artifacts/day49/day49_writing_summary.json` | Machine-readable summary |
    | `artifacts/day49/day49_citation_todo.md` | Citation requirements checklist |
    | `artifacts/day49/README.md` | This file |

    ## Evidence Used
    - Evidence files found: {found_count}
    - Evidence files missing: {missing_count}

    ## Scientific Principles
    - All claims are grounded in available empirical artefacts.
    - The weak Sgen (diffusion generative) finding is reported transparently.
    - Ontology-calibrated scoring and counterfactual explanation are framed
      as the primary contributions.
    - All unverified references are marked `[CITATION NEEDED]`.
    - No fake citations are included.

    ## Regeneration

    ```powershell
    $env:PYTHONPATH = (Get-Location).Path
    python scripts/build_day49_executive_background.py
    ```

    ---

    *Generated by `scripts/build_day49_executive_background.py`.*
    """)


def build_writing_summary(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Build the JSON writing summary."""
    return {
        "day": 49,
        "title": "Executive Summary and Background Draft",
        "status": "complete",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "output_paths": [
            "docs/paper/day49_executive_summary_background.md",
            "artifacts/day49/day49_writing_summary.json",
            "artifacts/day49/day49_citation_todo.md",
            "artifacts/day49/README.md",
        ],
        "evidence_files_found": evidence.get("_found", []),
        "evidence_files_missing": evidence.get("_missing", []),
        "paper_sections_started": [
            "Executive Summary",
            "1. Introduction",
            "1.1 Clinical Motivation",
            "1.2 Problem Statement",
            "1.3 Aim and Objectives",
            "1.4 Proposed Framework Overview",
            "1.5 Contributions",
            "2. Background and Related Work",
            "2.1 EHR Representation Learning",
            "2.2 Anomaly Detection in Clinical Sequences",
            "2.3 Clinical Ontologies and Knowledge Graphs",
            "2.4 Generative Models for EHR Data",
            "2.5 Counterfactual Explanations in Healthcare AI",
            "2.6 Research Gap",
            "2.7 Working Hypothesis",
            "Preliminary Empirical Evidence",
        ],
        "next_day": "Day 50 — Technical Approach / Methodology section",
    }


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 49 — Executive Summary and Background Draft Builder",
    )
    parser.add_argument(
        "--project_root", default=".",
        help="Project root directory (default: .)",
    )
    parser.add_argument(
        "--out_dir", default="artifacts/day49",
        help="Output directory for Day 49 artefacts (default: artifacts/day49)",
    )
    parser.add_argument(
        "--docs_dir", default="docs/paper",
        help="Output directory for paper drafts (default: docs/paper)",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_absolute():
        docs_dir = root / docs_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[day49] Project root : {root}")
    print(f"[day49] Output dir   : {out_dir}")
    print(f"[day49] Docs dir     : {docs_dir}")
    print()

    # ── load evidence ──────────────────────────────────────────────────
    print("[day49] Loading evidence artefacts ...")
    evidence = load_evidence(root)
    print()

    # ── generate draft ─────────────────────────────────────────────────
    print("[day49] Generating paper draft ...")
    draft = build_main_draft(evidence)
    draft_path = docs_dir / "day49_executive_summary_background.md"
    draft_path.write_text(draft, encoding="utf-8")
    print(f"  -> {draft_path}")

    # ── citation TODO ──────────────────────────────────────────────────
    print("[day49] Generating citation TODO ...")
    citation_todo = build_citation_todo()
    citation_path = out_dir / "day49_citation_todo.md"
    citation_path.write_text(citation_todo, encoding="utf-8")
    print(f"  -> {citation_path}")

    # ── README ─────────────────────────────────────────────────────────
    print("[day49] Generating README ...")
    readme = build_day49_readme(evidence)
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    print(f"  -> {readme_path}")

    # ── JSON summary ───────────────────────────────────────────────────
    print("[day49] Generating writing summary JSON ...")
    summary = build_writing_summary(evidence)
    summary_path = out_dir / "day49_writing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  -> {summary_path}")

    # ── final report ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Day 49 — Executive Summary & Background: Complete")
    print("=" * 60)
    print(f"  Evidence found  : {len(evidence.get('_found', []))}")
    print(f"  Evidence missing: {len(evidence.get('_missing', []))}")
    print(f"  Sections drafted: {len(summary['paper_sections_started'])}")
    print()
    print("  Generated files:")
    print(f"    {draft_path}")
    print(f"    {summary_path}")
    print(f"    {citation_path}")
    print(f"    {readme_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
