# Ontology-Calibrated Counterfactual Explanations for Anomaly Detection in EHR Sequences

## 1. Overview

This repository studies a deceptively simple question: when a patient's coded
hospital record contains a combination of codes that *should not go together*, can we
(a) flag it and (b) explain the flag with a minimal, medically coherent edit that would
make the record consistent again?

The honest short answer this project arrived at is: **a real medical-ontology rule
engine does most of the useful work.** A curated set of ontology constraints (built on
real SNOMED CT / RxNorm / UMLS data) is what actually separates coherent records from
incoherent ones on our benchmark, and it is also what lets us propose a valid repair.
The sequence "anomaly detector" and the diffusion-based generative surprise term that
the project originally leaned on turned out **not** to help — and we report that plainly
rather than hiding it.

This README is written to be read start-to-finish by a supervisor, a reviewer, a future
maintainer, or anyone trying to reproduce the aggregate numbers. It tells you what works,
what does not, and where the boundary of the evidence is.

## 2. What this project does

Given electronic health record (EHR) data from an ICU admission, the pipeline:

1. **Builds code sequences.** Each admission becomes an ordered sequence of clinical
   tokens (diagnoses, procedures, medications) plus a small set of model-visible
   demographic fields (`gender`, `age_group`).
2. **Maps tokens to a real ontology.** ICD-9/10 diagnosis codes are crosswalked to
   SNOMED CT, and drug names to RxNorm ingredients, using authoritative UMLS
   Metathesaurus mappings.
3. **Scores each record for relational incoherence** with an ontology rule engine
   (`S_ont`). The rules encode three families of clinical constraint: demographic
   incompatibility, medication-indication context, and forbidden co-occurrence.
4. **Optionally combines** the ontology score with an unsupervised sequence detector
   score (`S_det`) into a transparent calibrated score (`S_cal`). In the final method
   this combination is **not** used, because the detector does not add value (see below).
5. **Explains a flagged record** by searching for the smallest set of ontology-valid
   edits (remove / add-context / generalize) that drives the ontology score back down —
   a *counterfactual repair*. The search never reads how the anomaly was made.

## 3. What changed during the research

This project did not end where it started, and that is the point of the write-up.

| Originally expected | What the evidence forced |
|---|---|
| A deep sequence detector would carry detection | The unsupervised detector scored **below chance** on the leakage-controlled benchmark; the anomalies are *relational*, not next-token-surprising |
| Diffusion "generative surprise" (`Sgen`) would be a pillar | `Sgen` scored below chance, was mode-collapsed, and *harmed* the combined score — **removed from the core** |
| The headline would be a calibrated `S_det + S_ont` combination | The combination is significantly *worse* than ontology-only; the recommended method is **ontology-only** |
| "Ontology-calibrated" was assumed | A **real** ontology had to be parsed, mapped, and shown to beat the legacy ICD-prefix rules before the word was used |
| The first benchmark measured anomaly reasoning | It was **circular** — a single token-presence feature recovered the label (~0.94). It was rebuilt as leakage-controlled **benchmark-v2** |

The result is a smaller but genuinely defensible contribution: a leakage-controlled
benchmark, a real ontology scorer that beats the legacy baseline, and a leakage-free
counterfactual repair method — accompanied by two transparent negative results.

## 4. Final scientific position

The final method is **ontology-centered**:

```
S_main = S_ont
```

A calibrated-score infrastructure exists and is fully implemented:

```
S_cal = (w_det · S_det + w_ont · S_ont) / (w_det + w_ont),  with  w_gen = 0
```

but the **recommended paper method is ontology-only**, because adding the detector
significantly lowers ROC-AUC. The generative term is excluded from the core (`w_gen = 0`)
and appears only as a negative diagnostic.

All final evidence is on **MIMIC-IV benchmark-v2 only**. External validation on eICU is
**blocked by a schema mismatch** (eICU uses APACHE/body-system tokens that do not map to
the ICD/SNOMED/RxNorm ontology) and is documented as future work, not a current claim.

## 5. Main results

Leakage-controlled **benchmark-v2** (MIMIC-IV), test split (n = 6,307; anomaly rate
21.8%). Threshold selected on the validation split and applied unchanged to test.
95% confidence intervals are bootstrap; the source is
[`artifacts/phase7/final_evaluation.json`](artifacts/phase7/final_evaluation.json).

| Variant | ROC-AUC | 95% CI | AP | F1 |
|---|---:|---|---:|---:|
| **ontology_only_real** (main method) | **0.7881** | [0.774, 0.802] | 0.542 | 0.635 |
| legacy_baseline (ICD-prefix rules) | 0.7358 | [0.720, 0.751] | 0.543 | 0.593 |
| detector_only_full | 0.4525 | [0.436, 0.470] | 0.190 | 0.361 |
| combined_real_without_sgen | 0.7036 | [0.687, 0.720] | 0.404 | 0.460 |
| Sgen (diagnostic only, excluded) | 0.4868 | — | — | — |

Paired-bootstrap significance
([`final_stat_tests.json`](artifacts/phase7/final_stat_tests.json)):

- **Real ontology − legacy = +0.052 ROC-AUC**, CI [0.033, 0.072], p ≈ 0 → the real
  ontology **significantly beats** the legacy ICD-prefix baseline.
- **Combined − ontology-only = −0.085 ROC-AUC**, CI [−0.096, −0.073], p ≈ 0 → adding the
  detector **significantly hurts**. The detector is a non-additive, below-chance signal.

Ablation ([`ablation_results.json`](artifacts/phase7/ablation_results.json)): no single
rule family reaches the full 0.788 — demographic-only 0.599, medication-only 0.571,
forbidden-only 0.625 — so the three families **synergize**.

## 6. Counterfactual explanation results

Leakage-free ontology-guided repair on the benchmark-v2 test anomalies. The generator
reads only model-visible sequence + demographics + the real ontology + scorer feedback;
it never sees the corruption answer key. Source:
[`counterfactual_final.json`](artifacts/phase7/counterfactual_final.json).

| Metric | Value |
|---|---:|
| Test anomalies attempted | 1,376 |
| Ontology-flagged anomalies | 939 |
| **Repair success among ontology-flagged** | **89.99%** |
| Repair success overall | 61.4% |
| Mean ΔS_ont | 0.644 |
| Median edits | 1 |

By rule type, repair is strongest where the ontology has a clean handle:
missing-required-context 100%, mutual-exclusion 100%, demographic mismatch 65.7%
(edit-budget-limited on records with many obstetric codes). The ~33% of anomalies the
ontology does not flag are **detection** gaps, not repair failures — the repair method
can only fix what the scorer can see.

## 7. What is *not* claimed

To keep the boundary of the evidence explicit, this project does **not** claim:

- that the sequence detector improves anomaly detection (it is below chance here);
- that the combined detector + ontology score beats ontology-only (it is worse);
- that diffusion / generative surprise (`Sgen`) helps (removed from the core);
- external generalization — eICU external validation is **blocked** by schema mismatch;
- clinical deployment readiness;
- clinician-validated repairs (validity is ontology-violation resolution, not a clinical
  action study);
- state-of-the-art deep anomaly detection.

A note on **code capability vs. paper-supported claims**: the repository *implements* the
full calibrated score, the detector, and a diffusion diagnostic. Those exist so the
negative results are reproducible. The *paper-supported* method is the narrower,
ontology-centered one above.

## 8. Data and licensing

This project uses restricted, credentialed clinical data and licensed medical ontologies.
**No raw or processed patient data, no ontology dumps, no vocabularies, and no
per-record score files are committed to git.** Only aggregate artifacts are tracked.

| Resource | Access | Used for |
|---|---|---|
| MIMIC-IV | PhysioNet credentialed access (CITI training + DUA) | EHR sequences, benchmark-v2 |
| eICU (GOSSIS) | PhysioNet credentialed access | External-validation attempt (blocked) |
| UMLS Metathesaurus | UTS license (free account) | ICD→SNOMED / drug→RxNorm crosswalks |
| SNOMED CT | Affiliate license (via UMLS/UTS) | Diagnosis concept hierarchy |
| RxNorm | Public (NLM), obtained via UMLS release | Medication ingredient mapping |

See [`docs/reproducibility/data_access.md`](docs/reproducibility/data_access.md) for the
exact tables, files, and account steps.

## 9. Repository structure

```
src/
  preprocessing/       Sequence extraction (MIMIC-III/IV, eICU) + ontology mapping
  ontology/            Real ontology engine: index, rule packs, loader, scorer wiring
  scoring/             Calibrated score (S_ont, S_det, S_cal; w_gen = 0)
  explanations/        Leakage-free counterfactual repair generator
  training/            Unsupervised detector training (full-scale)
  experiments/         Config system, experiment tracking, shared eval helpers
  evaluation/          Generative decision gate (diagnostic), evaluators
scripts/               CLI entry points for every phase (build, train, evaluate)
configs/               Phase 6 detector configs (smoke / full / h200)
tests/                 Pytest suite (phase1b … phase8)
docs/
  paper/               Manuscript sections + combined final_manuscript.md
  reproducibility/     Runbook, environment, data access, artifact manifest
artifacts/
  phase0 … phase7/     Aggregate-only evidence (JSON / MD / CSV / figure data)
  phase8/              Final manifest, paper-asset index, phase8 summary
REPRODUCIBILITY.md     Top-level reproducibility entry point
```

Licensed/derived material lives under `data/processed/`, `ontologies/raw/`,
`ontologies/processed/`, and per-run checkpoint/vocab folders — **all git-ignored**.

## 10. Setup

Requires Python 3.10+. A CUDA-capable GPU is only needed to *retrain* the detector; the
aggregate results and tests run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
pip install pytest
# GPU-specific torch build (only needed to retrain the detector):
# pip install -r docs/setup/requirements-torch-cu128.txt
```

Point `dataset_roots.yaml` at your local copies of MIMIC-IV / eICU if you intend to
rebuild data. You do not need the data to run the test suite.

## 11. Data preparation

Only needed if you hold PhysioNet credentials and want to rebuild from scratch.

```bash
# Extract MIMIC-IV admissions into code sequences
python -m src.preprocessing.extract_mimiciv \
    --input-dir <mimic4_root> \
    --output-path data/processed/mimiciv_sequences.parquet \
    --stats-path data/processed/mimiciv_stats.json

# Map tokens into ontology space (ICD→SNOMED, drug→RxNorm)
python -m src.preprocessing.map_sequences_to_ont \
    --sequences-dir data/processed \
    --maps-dir ontologies/umls_maps \
    --output-dir data/processed
```

## 12. Ontology preparation

```bash
# Parse licensed SNOMED CT RF2 and RxNorm RRF into local processed files
python scripts/parse_snomed.py --snomed-dir <snomed_rf2_dir> --output-dir ontologies/processed
python scripts/parse_rxnorm.py --rxnorm-dir <rxnorm_rrf_dir>  --output-dir ontologies/processed

# Build the ICD→SNOMED and drug→RxCUI mapping dictionaries from UMLS MRCONSO
python -m src.preprocessing.build_umls_maps   --mrconso <MRCONSO.RRF> --output-dir ontologies/umls_maps
python -m src.preprocessing.build_rxnorm_maps --rxnconso <RXNCONSO.RRF> --output-dir ontologies/umls_maps
```

Coverage achieved on MIMIC-IV: diagnosis 0.80, medication 0.78
([`artifacts/phase2b_fix/coverage_report.json`](artifacts/phase2b_fix/coverage_report.json)).
All parsed ontology outputs are git-ignored.

## 13. Benchmark-v2

Benchmark-v2 is the **final, leakage-controlled** benchmark. Each anomaly is a violation
of a *relationship* between fields, while every individual model-visible token stays
common — this removes the circularity of the first benchmark (where one token-presence
feature recovered the label at ~0.94).

- **Demographic incompatibility** — flip `gender` on a record that already holds
  sex-specific codes. *Zero tokens injected*; the conflict lives in the (code, gender)
  relationship.
- **Medication-indication mismatch** — remove the indication diagnosis a present drug
  requires (e.g., insulin without diabetes). The drug token stays common.
- **Forbidden co-occurrence** — add a curated mutually-exclusive partner (type-2 vs
  type-1 diabetes). The added token is itself common.

Splits are subject-level with **zero overlap**; the train split is clean-normal-only.
The strongest label-free trivial signal is **0.6127 < 0.80** (the non-circularity gate).

```bash
python scripts/build_benchmark_v2.py          # rebuild (needs processed MIMIC-IV)
python scripts/diagnose_anomaly_triviality.py  # re-check the non-circularity gate
```

## 14. Running final evaluation

Reproduces the Section 5 table, the statistical tests, ablations, and paper tables.
These read local benchmark-v2 splits (git-ignored) and write aggregate artifacts.

```bash
python scripts/run_phase7_final_evaluation.py       # main results + CIs
python scripts/run_phase7_ablations.py              # rule-family + score-component ablations
python scripts/run_phase7_tables.py                 # table1..table5 CSVs
python scripts/run_phase7_external_validation_check.py   # eICU schema check (reports blocked)
```

## 15. Running counterfactual evaluation

```bash
python scripts/run_phase7_counterfactual_final.py   # leakage-free repair on test anomalies
```

Writes [`artifacts/phase7/counterfactual_final.json`](artifacts/phase7/counterfactual_final.json).
Leakage protections are exercised by `tests/test_phase4_counterfactual_leakage.py`.

## 16. Running tests

```bash
python -m pytest            # full suite
python -m pytest tests/test_phase8_readme.py -v   # Phase 8 finalization checks only
```

The suite runs on CPU and does not require the restricted data (data-dependent tests skip
cleanly when local splits are absent).

## 17. Reproducibility package

The end-to-end reproducibility story lives in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and
[`docs/reproducibility/`](docs/reproducibility/):

- [`runbook.md`](docs/reproducibility/runbook.md) — step-by-step rerun order.
- [`environment.md`](docs/reproducibility/environment.md) — Python / package / seed setup.
- [`data_access.md`](docs/reproducibility/data_access.md) — obtaining MIMIC-IV, eICU, UMLS.
- [`artifact_manifest.md`](docs/reproducibility/artifact_manifest.md) — what each committed
  artifact contains.
- [`phase8_reproducibility_guide.md`](docs/reproducibility/phase8_reproducibility_guide.md)
  — the finalization guide.

A machine-readable manifest of safe aggregate artifacts is in
[`artifacts/phase8/artifact_manifest.json`](artifacts/phase8/artifact_manifest.json).

**For reviewers.** All committed artifacts are aggregate-level only. Raw clinical data,
processed patient-level records, ontology dumps, model checkpoints, vocabularies, and
per-record score files are intentionally excluded from git. Every number in this README is
backed by a committed aggregate JSON/CSV under `artifacts/phase7/`, and the claim ledger
is [`docs/paper/final_claims_matrix.md`](docs/paper/final_claims_matrix.md).

## 18. Known limitations

- **MIMIC-IV only.** All final evidence is on one source database.
- **Synthetic, constructed anomalies.** Benchmark-v2 injects relational anomalies into
  real records; it is not a corpus of clinician-confirmed real-world errors.
- **No clinician validation.** Counterfactual "validity" means the ontology violation is
  resolved with a minimal, coherent edit — not that a clinician endorsed the action.
- **Curated rule packs.** The three rule families are hand-authored and auditable, not
  learned; their coverage is deliberately scoped.
- **External validation blocked.** eICU's APACHE/body-system tokens do not map to the
  ICD/SNOMED/RxNorm ontology (0/500 sampled records fire any rule).
- **Detector is a negative result.** It is reported for transparency, not as a component.

## 19. Phase roadmap

| Phase | What it delivered |
|---|---|
| 0 | Scientific claim reset; honest story + claims/contribution matrices |
| 1 / 1b | Triviality audit of the old benchmark; leakage-controlled **benchmark-v2** |
| 2 / 2b | Real UMLS / SNOMED CT / RxNorm parsing, mapping, coverage repair |
| 3 / 3b | Detector + calibrated scoring rebuild; auditable ontology rule packs |
| 4 | Leakage-free counterfactual repair |
| 5 | Generative / `Sgen` decision gate → **removed from core** |
| 6 | Full-scale training + experiment infrastructure; one full GPU detector run |
| 7 | Final evaluation, ablations, statistics, tables, figures, claim decisions |
| **8** | **This phase: final paper, humanized README, reproducibility package** |

Future work (not a current claim): an APACHE→ICD/SNOMED crosswalk (or another
ontology-compatible external dataset) to unblock external validation, and a clinician
study of repair usefulness.

## 20. Citation, acknowledgements, and licensing

This is a Master's research project by **Alireza Yegane**, supervised by **Professor Xuyun
Zhang**, Macquarie University, Faculty of Science and Engineering.

If you refer to this work, please cite the manuscript in
[`docs/paper/final_manuscript.md`](docs/paper/final_manuscript.md). Bibliographic
references for prior work and data/ontology resources are collected in
[`docs/paper/references.md`](docs/paper/references.md); entries still pending full
verification are listed in
[`docs/paper/references_todo.md`](docs/paper/references_todo.md).

**Licensing.** The code in this repository is released for research use. The clinical
datasets (MIMIC-IV, eICU) and medical ontologies (UMLS, SNOMED CT, RxNorm) are governed by
their own licenses and are **not** redistributed here. Obtaining them is the user's
responsibility via PhysioNet and the UMLS Terminology Services.
