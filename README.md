# Ontology-Calibrated Counterfactual Explanations for Anomaly Detection in EHR Sequences

This system detects clinically anomalous patterns in sequential electronic health record (EHR) data from ICU datasets (MIMIC-III, MIMIC-IV, eICU) and generates explanations grounded in medical ontologies. It combines a sequence-based anomaly detector with an ontology engine that encodes constraints from SNOMED CT and RxNorm, producing a calibrated anomaly score that distinguishes statistical rarity from ontology violations.

For each flagged record the system proposes minimal counterfactual edits — additions, removals, or replacements of clinical codes — that would resolve the detected anomaly while remaining clinically coherent. The counterfactual search is constrained by hierarchical and relational rules from medical ontologies, ensuring that proposed repairs respect known diagnostic, procedural, and pharmacological relationships.

## Directory Structure

```
src/
  preprocessing/          Sequence extraction and ontology mapping modules
    extract_mimic.py        MIMIC-III admission-level sequence builder
    extract_mimiciv.py      MIMIC-IV admission-level sequence builder
    extract_eicu.py         eICU (GOSSIS-1) ICU-stay-level sequence builder
    common.py               Shared normalization, I/O, and statistics helpers
    build_umls_maps.py      ICD-9/10 to SNOMED crosswalk from UMLS MRCONSO
    build_rxnorm_maps.py    Drug-name to RxCUI mapping from RXNCONSO
    map_sequences_to_ont.py Token-level ontology mapping for sequence files
  ontology/               Rule-based ontology validation engine
    index.py                Hierarchical code index (parents, children, terms)
    types.py                ClinicalRecord and OntologyViolation data classes
    rules.py                DemographicRule, RequiredCodesRule, MutualExclusionRule
    engine.py               Orchestrates rule checks and violation scoring
    loader.py               Loads real SNOMED/RxNorm data into the engine

scripts/                  CLI tools for ontology parsing and pipeline auditing
  parse_snomed.py           Parse SNOMED CT RF2 files into CSV + hierarchy JSON
  parse_rxnorm.py           Parse RxNorm RRF files into CSV
  build_rxnorm_graph.py     Build directed NetworkX graph from parsed RxNorm
  audit_pipeline.py         End-to-end pipeline audit (Day 13)
  check_edge_cases.py       Edge-case summary report generator
  day11_build_hospital_splits_auto.py   Train/val/test split builder (MIMIC-IV)
  day11_build_mimiciii_splits.py        Train/val/test split builder (MIMIC-III)

tests/                    Pytest test suite
  test_extract_mimic.py     MIMIC-III extraction tests
  test_extract_mimiciv.py   MIMIC-IV extraction tests
  test_extract_eicu.py      eICU extraction tests
  test_ontology.py          Ontology engine and loader tests
  test_build_maps.py        UMLS/RxNorm mapping and token conversion tests
  test_pipeline_audit.py    Pipeline audit tests

config/                   Configuration files
  data_semantics_policy.yaml  Token namespace and normalization policy

data/processed/           Extracted sequences, split manifests, summary stats
ontologies/               Placeholder directories for raw and processed ontology files
  processed/                Parsed SNOMED/RxNorm CSVs, hierarchy JSON, graph pickle
  umls_maps/                ICD-to-SNOMED and drug-to-RxCUI JSON dictionaries
docs/                     Architecture specs, literature reviews, audit reports
artifacts/                Pipeline audit outputs (JSON, Markdown)
```

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA (see `docs/setup/requirements-torch-cu128.txt` for GPU-specific versions)
- PhysioNet credentialed access (requires CITI human-subjects training and signed DUAs)
- UMLS Metathesaurus license (free UTS account at https://uts.nlm.nih.gov)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -r docs/setup/requirements-torch-cu128.txt
pip install pytest
```

Verify GPU access:

```bash
python ontology_cf_anomaly/scripts/check_torch_gpu.py
```

Configure dataset paths in `dataset_roots.yaml` to point to your local copies of each dataset.

## Data Acquisition

### Clinical Datasets (PhysioNet)

All three datasets require credentialed access via PhysioNet. Complete CITI training, sign the data use agreement for each, and download to local directories.

| Dataset | URL | Local directory key in `dataset_roots.yaml` |
|---------|-----|---------------------------------------------|
| MIMIC-III | https://physionet.org/content/mimiciii/ | `mimic3_root` |
| MIMIC-IV | https://physionet.org/content/mimiciv/ | `mimic4_root` |
| eICU (GOSSIS-1) | https://physionet.org/content/gossis/ | `eicu_root` |

See `docs/day4_6_data_access.md` for the specific tables and columns consumed by each extraction script.

### Ontology Resources (UMLS)

1. Register for a UTS account at https://uts.nlm.nih.gov and accept the UMLS license.
2. Download the UMLS Metathesaurus (specifically `MRCONSO.RRF`). Place under `ontology_cf_anomaly/ontologies/umls_maps/`.
3. Download SNOMED CT RF2 Snapshot (US Edition). Place under `ontology_cf_anomaly/ontologies/snomed/`.
4. Download RxNorm Full Release (`RXNCONSO.RRF`, `RXNREL.RRF`). Place under `ontology_cf_anomaly/ontologies/rxnorm/`.

Raw ontology files are excluded from git due to licensing restrictions and file size.

## Pipeline

### 1. Parse ontologies

```bash
python scripts/parse_snomed.py \
    --snomed-dir ontology_cf_anomaly/ontologies/snomed/Snapshot \
    --output-dir ontologies/processed

python scripts/parse_rxnorm.py \
    --rxnorm-dir ontology_cf_anomaly/ontologies/rxnorm \
    --output-dir ontologies/processed
```

### 2. Build mapping dictionaries

```bash
python -m src.preprocessing.build_umls_maps \
    --mrconso ontology_cf_anomaly/ontologies/umls_maps/MRCONSO.RRF \
    --output-dir ontologies/umls_maps

python -m src.preprocessing.build_rxnorm_maps \
    --rxnconso ontology_cf_anomaly/ontologies/rxnorm/RXNCONSO.RRF \
    --output-dir ontologies/umls_maps

python scripts/build_rxnorm_graph.py \
    --input-dir ontologies/processed \
    --output-path ontologies/processed/rxnorm_graph.gpickle
```

### 3. Extract sequences

```bash
python -m src.preprocessing.extract_mimic \
    --mimic-dir <mimic3_root> \
    --out-dir data/processed

python -m src.preprocessing.extract_mimiciv \
    --input-dir <mimic4_root> \
    --output-path data/processed/mimiciv_sequences.parquet \
    --stats-path data/processed/mimiciv_stats.json

python -m src.preprocessing.extract_eicu \
    --input-dir <eicu_root> \
    --output-path data/processed/eicu_sequences.parquet \
    --stats-path data/processed/eicu_stats.json
```

### 4. Map tokens to ontology space

```bash
python -m src.preprocessing.map_sequences_to_ont \
    --sequences-dir data/processed \
    --maps-dir ontologies/umls_maps \
    --output-dir data/processed
```

### 5. Build train/val/test splits

```bash
python scripts/day11_build_hospital_splits_auto.py
python scripts/day11_build_mimiciii_splits.py
```

### 6. Train detector (planned)

Detector implementation is in progress (Days 15-24 of the roadmap).

### 7. Run tests

```bash
python -m pytest tests/ -v
```

## Current Status

Days 1-14 of the 90-day roadmap are implemented:

- Literature review and system architecture (Days 1-2)
- Project scaffold and environment setup (Day 3)
- Data access for MIMIC-III, MIMIC-IV, eICU (Days 4-5)
- UMLS/SNOMED CT/RxNorm acquisition (Day 6)
- Ontology parsing: SNOMED RF2 and RxNorm RRF to CSV/JSON (Day 7)
- ICD-to-SNOMED crosswalk, drug-to-RxCUI mapping, sequence ontology mapping (Day 8)
- MIMIC-III sequence extraction with demographics (Day 9)
- MIMIC-IV and eICU sequence extraction (Day 10)
- Dataset integration and train/val/test splits (Day 11)
- Ontology engine with demographic, required-code, and mutual-exclusion rules; loader for real SNOMED/RxNorm data (Day 12)
- End-to-end pipeline audit and edge-case analysis (Day 13)
- Documentation and reproducibility (Day 14)

Next phase: baseline anomaly detector development (Days 15-28).

## Project Plan

The full project proposal and 90-day roadmap are documented in the project report by Alireza Yegane, supervised by Professor Xuyun Zhang (Macquarie University, Faculty of Science and Engineering). The roadmap is in Section 4.7 of the proposal.
