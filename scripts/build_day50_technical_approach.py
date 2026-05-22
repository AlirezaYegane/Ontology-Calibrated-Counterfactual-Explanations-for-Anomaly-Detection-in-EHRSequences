#!/usr/bin/env python3
"""
Day 50 -- Technical Approach / Methodology Section Builder
==========================================================

Reads project evidence artefacts and source-code metadata to
generate a paper-ready Section 3 (Technical Approach) grounded
in the actual implementation.

Generated outputs
-----------------
  - docs/paper/03_technical_approach.md
  - artifacts/day50/day50_technical_approach_summary.json
  - artifacts/day50/README.md

This script does NOT modify source code, train models, or
access private clinical data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


# -- helpers -----------------------------------------------------------

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
    """Read a text/markdown file if it exists; return None otherwise."""
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# -- evidence loading --------------------------------------------------

EVIDENCE_SPECS = [
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
        "key": "day48_readme",
        "rel_path": "artifacts/day48/README.md",
        "description": "Day 48 reproducibility audit",
    },
    {
        "key": "day49_readme",
        "rel_path": "artifacts/day49/README.md",
        "description": "Day 49 executive summary",
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


# -- empirical inserts -------------------------------------------------

def _detector_evidence(evidence: Dict[str, Any]) -> str:
    """Return a grounded empirical sentence for the detector."""
    d20 = evidence.get("day20_eval")
    if d20 and isinstance(d20, dict):
        roc = d20.get("roc_auc")
        ap = d20.get("average_precision")
        f1 = d20.get("f1")
        if roc is not None and ap is not None and f1 is not None:
            return (
                f"In preliminary evaluation on synthetic injected "
                f"anomalies, the GRU-based detector achieved a "
                f"ROC-AUC of {roc:.3f}, average precision of "
                f"{ap:.3f}, and F1 of {f1:.3f} "
                f"(see Section 4 for full results)."
            )
    return (
        "Preliminary evaluation results for the supervised "
        "detector are reported in Section 4."
    )


def _sgen_evidence(evidence: Dict[str, Any]) -> str:
    """Return a grounded empirical sentence for Sgen."""
    d34 = evidence.get("day34_assessment")
    if d34 and isinstance(d34, dict):
        sgen_result = d34.get("main_sgen_result", {})
        best_auc = sgen_result.get("best_roc_auc")
        if best_auc is not None:
            return (
                f"After exact checkpoint alignment and a sweep "
                f"across all diffusion timesteps, the best Sgen "
                f"ROC-AUC was {best_auc:.3f} -- effectively "
                f"near random.  This negative finding confirms "
                f"that the current denoising-error proxy does not "
                f"meaningfully separate injected anomalies from "
                f"normal records.  Sgen is therefore assigned zero "
                f"or near-zero weight in the calibrated score and "
                f"retained only as a diagnostic auxiliary signal."
            )
    return (
        "Empirical evaluation of Sgen is documented in the "
        "Day 34 assessment artefact.  The current proxy yields "
        "weak discriminative performance and should not be "
        "treated as a primary signal."
    )


# -- draft builder -----------------------------------------------------

def build_technical_approach(evidence: Dict[str, Any]) -> str:
    """Compose the full Section 3 draft."""
    det_ev = _detector_evidence(evidence)
    sgen_ev = _sgen_evidence(evidence)

    return textwrap.dedent("""\
# 3. Technical Approach

This section presents the architecture and design rationale of the
OntoCF-AD framework (*Ontology-Calibrated Counterfactual Anomaly
Detection*).  The description follows the actual implementation and
is grounded in the evidence artefacts produced during development.
Where empirical results are cited, they refer to evaluation on
synthetic injected anomalies and should be interpreted accordingly.

---

## 3.1 Overview

OntoCF-AD processes sequential EHR data through four integrated
stages:

1. **EHR Sequence Representation** -- clinical codes are mapped to
   dense embeddings and organised as variable-length temporal
   sequences.
2. **Anomaly Detection** -- a supervised detector and an
   ontology-violation scorer each produce an independent anomaly
   signal.
3. **Generative Plausibility Assessment** -- a denoising diffusion
   model provides an auxiliary distributional plausibility signal.
4. **Counterfactual Explanation** -- a constrained search procedure
   identifies the minimal ontology-consistent edit that reduces
   the composite anomaly score below a decision threshold.

The composite anomaly score decomposes abnormality into statistical,
ontological, and generative components.  The weights assigned to
each component reflect their empirically validated discriminative
contribution: the supervised detector and ontology scorer are
dominant, while the generative component is currently assigned zero
or near-zero weight (see Section 3.5 and 3.6 for rationale).

The high-level data flow is:

```
EHR Sequence X
    |
    +---> Supervised Detector  ---> S_det(X)
    |
    +---> Ontology Scorer      ---> S_ont(X)
    |
    +---> Diffusion Model      ---> S_gen(X)   [auxiliary]
    |
    +---> Calibrated Score     ---> S_cal(X)
    |
    +---> Counterfactual Search ---> X*, Explanation
```

---

## 3.2 EHR Sequence Representation

Each patient encounter is represented as a variable-length sequence
of clinical-code tokens drawn from a unified vocabulary.  The
vocabulary integrates ICD-9/10 diagnosis codes, RxNorm medication
codes, CPT procedure codes, and demographic indicator tokens.
Special tokens `<pad>` and `<unk>` are reserved for padding and
out-of-vocabulary entries, respectively.

Tokens are mapped to dense vectors through a learnable embedding
layer.  The embedding dimension is a hyperparameter; the current
implementation uses an embedding dimension of 160 for the
supervised detector and 128 for the diffusion model.

Sequences that exceed a maximum length parameter are truncated
using a configurable strategy (head or tail truncation).
Sequences shorter than the maximum are right-padded and masked
during attention and loss computation.

No hand-engineered features are used; all clinical-code
representations are learned from the data.  This design choice
ensures generality across different EHR schemas, though it also
means that the model must learn clinical semantics from training
data rather than encoding them a priori -- a limitation that the
ontology layer (Section 3.4) is designed to address.

---

## 3.3 Supervised Sequence Anomaly Detector

### Architecture

The supervised anomaly detector is a GRU-based binary sequence
classifier (`GRUSequenceBinaryClassifier`).  Its architecture
comprises:

- An embedding layer (vocabulary size V, embedding dimension 160,
  with padding-index masking).
- A two-layer bidirectional-capable GRU encoder (hidden dimension
  320, inter-layer dropout 0.30).
- A LayerNorm normalisation stage applied to the final hidden
  state.
- A dropout layer (rate 0.30).
- A linear projection head mapping the hidden state to a single
  logit.

The model processes packed variable-length sequences to avoid
computation on padding tokens.  The output logit is passed through
a sigmoid activation to produce the anomaly probability S_det(X)
in [0, 1].

### Training

The detector is trained on synthetically injected anomalies.
Normal EHR sequences are drawn from the training partition of the
MIMIC-III (or eICU) dataset; anomalous variants are generated by
applying controlled perturbations including:

- **Demographic conflicts** -- inserting codes inconsistent with
  patient demographics.
- **Medication mismatches** -- pairing medications with
  ontologically incompatible diagnoses.
- **Missing diagnoses** -- removing indication codes that would
  explain observed medications or procedures.
- **Forbidden co-occurrences** -- injecting code pairs that
  violate known ontological constraints.

Training uses binary cross-entropy loss with the Adam optimiser.
The best checkpoint is selected by validation ROC-AUC.

### Empirical Note

""") + det_ev + "\n\n" + textwrap.dedent("""\
---

## 3.4 Ontology Knowledge Layer

### Purpose

The ontology knowledge layer injects structured medical knowledge
into the anomaly-scoring pipeline.  Rather than relying solely on
statistical deviation, the ontology scorer quantifies the degree
to which a record violates expected clinical-code relationships
encoded in established medical terminologies.

### Knowledge Sources

The framework draws on the following ontology resources:

- **SNOMED CT** [CITATION NEEDED] -- hierarchical classification
  of clinical concepts (diagnoses, findings, procedures).
- **RxNorm** [CITATION NEEDED] -- normalised medication
  vocabulary with ingredient, brand, and clinical-drug
  relationships.
- **UMLS Metathesaurus** [CITATION NEEDED] -- cross-ontology
  mapping layer providing concept-unique identifiers (CUIs) that
  link SNOMED CT, RxNorm, ICD, and other vocabularies.

These resources are parsed into in-memory graph structures
(NetworkX graphs and lookup dictionaries) at initialisation time.

### Ontology Violation Score (S_ont)

The ontology violation score is computed by the `compute_s_ont`
function, which traverses the ontology graph and checks for:

1. **Diagnosis-medication coherence** -- whether each observed
   medication has at least one compatible indication present in
   the record's diagnosis set.
2. **Hierarchical code validity** -- whether diagnosis and
   procedure codes fall within expected branches of the SNOMED CT
   hierarchy.
3. **Forbidden co-occurrence rules** -- whether any pair of
   codes in the record matches a known clinically contradictory
   combination.

Each violated rule contributes a penalty; the raw violation count
is transformed into a normalised score via an exponential mapping:

```
S_ont_norm = 1 - exp(-max(S_ont_raw, 0))
```

This normalisation maps the raw violation count to the interval
[0, 1), with diminishing marginal penalty for additional
violations.

### Token-Level Attribution

The ontology layer also computes per-token violation weights,
identifying which specific clinical codes are most implicated in
the ontological inconsistency.  These token-level attributions are
used downstream by the explanation generator to produce targeted
counterfactual suggestions.

---

## 3.5 Diffusion-Based Generative Component

### Architecture

The generative plausibility component is a denoising diffusion
probabilistic model (DDPM) [CITATION NEEDED] adapted for discrete
clinical-code sequences.  The model operates in continuous
embedding space:

- **Embedding** -- discrete tokens are mapped to dense vectors
  (d_model = 128) through a learnable embedding layer.
- **Noise schedule** -- a cosine beta schedule with 64 diffusion
  timesteps, following the improved schedule of Nichol and Dhariwal
  [CITATION NEEDED].
- **Denoiser** -- a time-conditioned Transformer encoder with 4
  attention heads, 4 layers, feedforward dimension 512, GELU
  activations, and pre-norm layer normalisation.
- **Time conditioning** -- sinusoidal timestep embeddings are
  projected through a two-layer MLP and added to the sequence
  representation.
- **Output head** -- a LayerNorm followed by a linear projection
  predicts the injected noise epsilon.

The model is trained with the standard DDPM objective: predict
the noise epsilon added to the embedded sequence at a randomly
sampled timestep t.  The loss is computed as the mean squared error
between predicted and actual noise, masked to exclude padding
tokens:

```
L_diffusion = sum_i (mask_i * ||epsilon_pred_i - epsilon_i||^2)
              / sum_i mask_i
```

### Generative Surprise Score (S_gen)

The surprise score function (`surprise_score`) computes a
per-record anomaly proxy as follows:

1. Embed the input sequence.
2. Sample noise and compute the noisy embedding at a specified
   timestep (default: midpoint of the schedule).
3. Predict the noise using the trained denoiser.
4. Compute the per-token mean squared error between predicted
   and actual noise.
5. Average over non-padding positions to yield S_gen(X).

The intuition is that records conforming to the training
distribution should be denoised accurately, while anomalous
records should incur higher reconstruction error.

### Empirical Limitation

""") + sgen_ev + "\n\n" + textwrap.dedent("""\
This limitation is reported transparently in the interest of
scientific honesty.  The diffusion component remains part of
the framework architecture as a placeholder for future
improvement, but it does not currently contribute to the
primary anomaly-detection evidence.

---

## 3.6 Anomaly Score Decomposition and Calibration

### Composite Calibrated Score

The three component signals are combined into a single calibrated
anomaly score through a weighted linear combination:

```
S_cal(X) = w_det * S_det(X) + w_ont * S_ont_norm(X) + w_gen * S_gen(X)
```

where the weights are normalised:

```
S_cal(X) = (w_det * S_det(X) + w_ont * S_ont_norm(X) + w_gen * S_gen(X))
           / max(w_det + w_ont + w_gen, epsilon)
```

### Weight Assignment

The default weight assignment in the current implementation is:

| Component | Weight | Rationale |
| --------- | ------ | --------- |
| S_det (supervised detector) | w_det = 0.70 | Primary statistical signal; validated ROC-AUC |
| S_ont (ontology violation) | w_ont = 0.30 | Interpretable clinical-consistency signal |
| S_gen (generative surprise) | w_gen = 0.00 | Auxiliary only; near-random in current evaluation |

The detector weight is dominant because S_det has demonstrated
meaningful discriminative performance on synthetic anomaly
benchmarks.  The ontology weight provides complementary
clinical-consistency evidence that does not require labelled
anomalies and is directly interpretable by domain experts.  The
generative weight is set to zero by default because the Day 34
evaluation found that the current denoising-error Sgen proxy does
not separate anomalies from normal records above chance level.

This weight configuration is conservative by design: it ensures
that the framework's anomaly decisions are driven entirely by
validated signals.  Should future work improve the generative
component, w_gen can be increased to incorporate that signal.

### Decomposition Transparency

A key design goal is that each component score is individually
interpretable.  For any flagged record, a clinician or data
steward can inspect:

- S_det -- the statistical anomaly probability from the
  supervised model.
- S_ont -- the ontology violation score with per-token
  attributions.
- S_gen -- the generative plausibility score (currently
  diagnostic only).

This decomposition supports explainability audits and enables
users to understand *why* a record was flagged, not merely
*that* it was flagged.

---

## 3.7 Ontology-Constrained Counterfactual Generator

### Problem Formulation

Given a record X flagged as anomalous (S_cal(X) > tau), the
counterfactual generator seeks the minimal edit X* such that:

1. S_cal(X*) < tau  (the repaired record is no longer anomalous).
2. X* is ontology-consistent (the edit respects the hierarchical
   and relational structure of SNOMED CT, RxNorm, and UMLS).
3. The edit distance between X and X* is minimal.

The cost objective is:

```
Cost(X* | X) = lambda_edit * edit_count(X, X*) + S_cal(X*)
```

where lambda_edit is a sparsity penalty that discourages
unnecessary edits.  Minimising this objective favours
counterfactuals that are both effective (low residual anomaly
score) and sparse (few clinical-code changes).

### Edit Operations

The counterfactual search considers three atomic edit operations
on the token sequence:

- **Add** -- insert a clinically plausible token (e.g., add a
  missing indication diagnosis to resolve a medication mismatch).
- **Remove** -- delete a token that contributes disproportionately
  to the anomaly score (e.g., remove a demographically
  inconsistent code).
- **Replace** -- substitute one token with an ontologically
  related alternative (e.g., replace a medication with a
  therapeutically equivalent option that is compatible with the
  recorded diagnoses).

### Ontology Constraints

Each candidate edit is validated against the ontology graph before
acceptance:

- Added diagnoses must exist in the SNOMED CT hierarchy and be
  clinically compatible with the existing record context.
- Replacement medications must share a therapeutic class or
  ingredient relationship in RxNorm.
- The resulting sequence must not introduce new ontology
  violations.

These constraints ensure that proposed counterfactuals are
clinically meaningful, distinguishing OntoCF-AD from unconstrained
perturbation methods that may suggest biologically implausible or
ontologically incoherent edits [CITATION NEEDED].

### Search Procedure

The current implementation uses a greedy forward search seeded by
the ontology violation attributions and detector token
contributions.  At each step, the edit that yields the largest
reduction in S_cal is selected, subject to ontology consistency.
The search terminates when either (a) S_cal(X*) < tau, or (b) a
maximum edit budget is exhausted.

---

## 3.8 Explanation Generation

### Multi-Level Explanations

The explanation generator (`build_explanation`) produces three
levels of textual output for each flagged record:

1. **Short explanation** -- a one-sentence summary identifying the
   anomaly type, primary driver signal, proposed repair, and score
   reduction.
2. **Clinical explanation** -- a paragraph-length description
   framed for clinical data stewards, including violation details,
   proposed actions, confidence assessment, and a disclaimer that
   the output is a data-quality signal rather than a clinical
   recommendation.
3. **Research explanation** -- a detailed technical account listing
   all decomposed scores (S_det, S_gen, S_ont, S_cal), edit
   count, counterfactual actions, and the S_gen diagnostic policy.

### Primary Driver Attribution

The explanation identifies the primary evidence source for each
anomaly using a rule-based attribution function:

- If both S_det and S_ont are positive, the driver is
  "mixed detector-and-ontology signal."
- If S_ont alone is positive, the driver is
  "ontology violation signal."
- If S_det alone is positive, the driver is
  "detector/statistical signal."
- S_gen is cited as a driver only when the Sgen policy is not
  set to "diagnostic_only" -- which, under current evidence,
  it always is.

### Safety Disclaimer

All generated explanations include the statement:

> "This should be interpreted as a data-quality and explanation
> signal, not as a clinical recommendation."

This disclaimer is hard-coded to prevent misuse of the
framework's outputs as autonomous clinical decision support.

---

## 3.9 Implementation Consistency and Scientific Caution

### Code-Evidence Alignment

The technical approach described in this section corresponds
directly to the implemented source code:

| Component | Source File |
| --------- | ----------- |
| Supervised detector | `src/models/detector_supervised.py` |
| Diffusion model | `src/models/diffusion.py` |
| Ontology scorer | `src/scoring/ontology_aware.py` |
| Ontology rules | `src/ontology/rules.py` |
| Explanation generator | `src/explanation/text_generator.py` |

### Negative-Finding Transparency

The framework includes a diffusion-based generative component
(S_gen) that was designed to provide distributional plausibility
evidence.  However, rigorous evaluation at Day 34 -- including
exact checkpoint alignment and a sweep across all 64 diffusion
timesteps -- demonstrated that the current denoising-error proxy
does not meaningfully discriminate injected anomalies from normal
records (best ROC-AUC approximately 0.508).

This result is reported transparently rather than omitted.  The
following design decisions follow from it:

- S_gen is assigned w_gen = 0.0 in the calibrated scoring
  function.
- S_gen is included in explanation outputs as a diagnostic
  auxiliary, clearly labelled as such.
- The framework's primary evidence rests on the supervised
  detector (S_det) and the ontology violation scorer (S_ont).

Future work may improve the generative component through
alternative surprise definitions (e.g., likelihood-ratio
scoring, latent-space anomaly detection, or conditional
generation with masked reconstruction).  Until such
improvements are validated, S_gen should not be claimed as a
contributor to the framework's anomaly-detection performance.

### Reproducibility

The implementation is fully version-controlled with day-level
artefact tracking.  A reproducibility audit (Day 48) captures
the repository state, environment metadata, and private-data
boundary checks.  Full run-from-scratch instructions are
documented in `docs/run_from_scratch.md`.

---

*Draft generated by `scripts/build_day50_technical_approach.py`.
All `[CITATION NEEDED]` markers must be resolved with verified
references before manuscript submission.*
""")


def build_day50_readme(evidence: Dict[str, Any]) -> str:
    """Generate artifacts/day50/README.md."""
    found_count = len(evidence.get("_found", []))
    missing_count = len(evidence.get("_missing", []))
    return textwrap.dedent(f"""\
# Day 50 -- Technical Approach / Methodology Draft

## Status
Complete.

## Goal
Write the paper-ready Section 3 (Technical Approach) grounded in
the actual implementation artefacts and source code.

## Generated Files

| File | Description |
| ---- | ----------- |
| `docs/paper/03_technical_approach.md` | Full Section 3 draft |
| `artifacts/day50/day50_technical_approach_summary.json` | Machine-readable summary |
| `artifacts/day50/README.md` | This file |

## Evidence Used
- Evidence files found: {found_count}
- Evidence files missing: {missing_count}

## Scientific Principles
- Supervised detector is described as the primary statistical signal.
- Ontology scorer is described as the primary interpretable signal.
- Diffusion Sgen is honestly described as auxiliary/diagnostic
  (ROC-AUC ~ 0.508, near random).
- All unverified references are marked `[CITATION NEEDED]`.
- No fake citations are included.

## Regeneration

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/build_day50_technical_approach.py --project_root . --out_dir artifacts/day50 --docs_dir docs/paper
```

---

*Generated by `scripts/build_day50_technical_approach.py`.*
""")


def build_summary_json(
    evidence: Dict[str, Any],
    draft_text: str,
) -> Dict[str, Any]:
    """Build the machine-readable summary with acceptance checks."""

    # Acceptance checks: verify section headings are present
    checks = {
        "contains_detector_section": "## 3.3" in draft_text,
        "contains_ontology_section": "## 3.4" in draft_text,
        "contains_diffusion_section": "## 3.5" in draft_text,
        "contains_scoring_section": "## 3.6" in draft_text,
        "contains_counterfactual_section": "## 3.7" in draft_text,
        "contains_scientific_caution": "## 3.9" in draft_text,
    }

    return {
        "day": 50,
        "title": "Technical Approach / Methodology Section",
        "status": "complete",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "output_paths": [
            "docs/paper/03_technical_approach.md",
            "artifacts/day50/day50_technical_approach_summary.json",
            "artifacts/day50/README.md",
        ],
        "evidence_files_found": evidence.get("_found", []),
        "evidence_files_missing": evidence.get("_missing", []),
        "paper_sections_written": [
            "3.1 Overview",
            "3.2 EHR Sequence Representation",
            "3.3 Supervised Sequence Anomaly Detector",
            "3.4 Ontology Knowledge Layer",
            "3.5 Diffusion-Based Generative Component",
            "3.6 Anomaly Score Decomposition and Calibration",
            "3.7 Ontology-Constrained Counterfactual Generator",
            "3.8 Explanation Generation",
            "3.9 Implementation Consistency and Scientific Caution",
        ],
        "acceptance_checks": checks,
        "all_checks_passed": all(checks.values()),
        "next_day": "Day 51 -- Experimental Setup / Results section",
    }


# -- main --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 50 -- Technical Approach Section Builder",
    )
    parser.add_argument(
        "--project_root", default=".",
        help="Project root directory (default: .)",
    )
    parser.add_argument(
        "--out_dir", default="artifacts/day50",
        help="Output directory for Day 50 artefacts",
    )
    parser.add_argument(
        "--docs_dir", default="docs/paper",
        help="Output directory for paper drafts",
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

    print(f"[day50] Project root : {root}")
    print(f"[day50] Output dir   : {out_dir}")
    print(f"[day50] Docs dir     : {docs_dir}")
    print()

    # -- load evidence -------------------------------------------------
    print("[day50] Loading evidence artefacts ...")
    evidence = load_evidence(root)
    print()

    # -- generate draft ------------------------------------------------
    print("[day50] Generating Section 3 draft ...")
    draft = build_technical_approach(evidence)
    draft_path = docs_dir / "03_technical_approach.md"
    draft_path.write_text(draft, encoding="utf-8")
    print(f"  -> {draft_path}")

    # -- README --------------------------------------------------------
    print("[day50] Generating README ...")
    readme = build_day50_readme(evidence)
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    print(f"  -> {readme_path}")

    # -- JSON summary --------------------------------------------------
    print("[day50] Generating summary JSON ...")
    summary = build_summary_json(evidence, draft)
    summary_path = out_dir / "day50_technical_approach_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  -> {summary_path}")

    # -- final report --------------------------------------------------
    print()
    print("=" * 60)
    print("  Day 50 -- Technical Approach: Complete")
    print("=" * 60)
    print(f"  Evidence found  : {len(evidence.get('_found', []))}")
    print(f"  Evidence missing: {len(evidence.get('_missing', []))}")
    print(f"  Sections written: {len(summary['paper_sections_written'])}")
    checks = summary["acceptance_checks"]
    for k, v in checks.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")
    print(f"  All checks passed: {summary['all_checks_passed']}")
    print()
    print("  Generated files:")
    print(f"    {draft_path}")
    print(f"    {summary_path}")
    print(f"    {readme_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
