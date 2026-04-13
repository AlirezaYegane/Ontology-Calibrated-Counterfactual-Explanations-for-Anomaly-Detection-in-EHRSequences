# Days 4-6: Data and Ontology Acquisition

This document records the data-access and ontology-acquisition steps
corresponding to Days 4, 5, and 6 of the project roadmap (Section 4.7.1).

> **Note:** All raw dataset and ontology files are excluded from version
> control due to licensing restrictions and file size. Only directory
> placeholders (`.gitkeep`) are committed.

---

## 1. Clinical Datasets

### 1.1 MIMIC-III (Day 4)

| Field | Value |
|-------|-------|
| Source | PhysioNet https://physionet.org/content/mimiciii/ |
| License | PhysioNet Credentialed Health Data License 1.5.0 (requires CITI training + signed DUA) |
| Local path | `D:\Article\_local\Datasets\mimic3-carevue` (see `dataset_roots.yaml` key `mimic3_root`) |
| Git placeholder | `ontology_cf_anomaly/data/mimiciii/.gitkeep` |
| Extraction script | `src/preprocessing/extract_mimic.py` |

**Key tables consumed by the extraction script:**

| Table stem | Columns used |
|------------|-------------|
| `ADMISSIONS` | `subject_id`, `hadm_id`, `admittime`, `dischtime` |
| `PATIENTS` | `subject_id`, `gender`, `dob` |
| `ICUSTAYS` | `subject_id`, `hadm_id`, `icustay_id`, `intime`, `outtime` |
| `DIAGNOSES_ICD` | `subject_id`, `hadm_id`, `seq_num`, `icd9_code` |
| `PROCEDURES_ICD` | `subject_id`, `hadm_id`, `seq_num`, `icd9_code` |
| `PRESCRIPTIONS` | `subject_id`, `hadm_id`, `icustay_id`, `startdate`, `enddate`, `drug`, `drug_name_poe`, `drug_name_generic`, `ndc` |

Files are expected as `.csv.gz` or `.csv` in the root of the MIMIC-III directory.

### 1.2 MIMIC-IV (Day 5)

| Field | Value |
|-------|-------|
| Source | PhysioNet https://physionet.org/content/mimiciv/ |
| License | PhysioNet Credentialed Health Data License 1.5.0 |
| Local path (v3.1) | `D:\Article\_local\Datasets\physionet.org\files\mimiciv\3.1` (key `mimic4_root`) |
| Local path (v2.2) | `D:\Article\_local\Datasets\physionet.org\files\mimiciv\2.2` (key `mimic4_v22_root`) |
| Git placeholder | `ontology_cf_anomaly/data/mimiciv/.gitkeep` |
| Extraction script | `src/preprocessing/extract_mimiciv.py` |

The extraction script expects a MIMIC-IV root containing `hosp/` and `icu/`
subdirectories. It resolves these recursively if they are nested.

**Key tables consumed (under `hosp/`):**

| Table stem | Columns used |
|------------|-------------|
| `admissions` | `subject_id`, `hadm_id`, `admittime`, `dischtime` |
| `patients` | `subject_id`, `gender`, `anchor_age`, `anchor_year` |
| `diagnoses_icd` | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version` |
| `procedures_icd` | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version` |
| `prescriptions` | `subject_id`, `hadm_id`, `starttime`, `stoptime`, `drug`, `formulary_drug_cd`, `gsn`, `ndc` |

**Key tables consumed (under `icu/`):**

| Table stem | Columns used |
|------------|-------------|
| `icustays` | `subject_id`, `hadm_id`, `stay_id`, `intime`, `outtime` |

MIMIC-IV uses mixed ICD-9/ICD-10 coding; the script prefixes tokens
accordingly (`ICD9_DX:`, `ICD10_DX:`, `ICD9_PROC:`, `ICD10_PROC:`).
Age is derived from `anchor_age` + (`admit_year` - `anchor_year`).

### 1.3 eICU / GOSSIS-1 (Day 5)

| Field | Value |
|-------|-------|
| Source | PhysioNet https://physionet.org/content/gossis/ |
| License | PhysioNet Credentialed Health Data License 1.5.0 |
| Local path | `D:\Article\_local\Datasets\gossis-1-eicu` (key `eicu_root`) |
| Git placeholder | `ontology_cf_anomaly/data/eicu/.gitkeep` |
| Extraction script | `src/preprocessing/extract_eicu.py` |

The GOSSIS-1 release packages eICU data as a single flat CSV
(`gossis-1-eicu-only.csv.gz`) keyed by `patientunitstayid`, rather than the
standard multi-table eICU schema.

**Columns consumed:**

| Column group | Columns |
|-------------|---------|
| Demographics | `patientunitstayid`, `patient_id`, `age`, `gender`, `ethnicity`, `hospital_los_days`, `icu_los_days`, `hospital_death`, `icu_death` |
| Diagnosis | `apache_2_diagnosis`, `apache_3j_diagnosis`, `apache_3j_bodysystem`, `apache_2_bodysystem` |
| Comorbidity flags | `aids`, `cirrhosis`, `diabetes_mellitus`, `hepatic_failure`, `immunosuppression`, `leukemia`, `lymphoma`, `solid_tumor_with_metastasis` |

Token prefixes: `EICU_APACHE2_DX:`, `EICU_APACHE3_DX:`, `EICU_BODYSYS:`,
`EICU_COMORB:`.

---

## 2. Ontology Resources (Day 6)

All ontology resources require a UMLS Terminology Services (UTS) account
and acceptance of the individual licence terms for each vocabulary.

### 2.1 UMLS Metathesaurus

| Field | Value |
|-------|-------|
| Source | NLM UMLS https://www.nlm.nih.gov/research/umls/ |
| License | UMLS Metathesaurus License Agreement (free, requires UTS account) |
| Target directory | `ontology_cf_anomaly/ontologies/umls_maps/` |

**Required RRF files** (from the full or subset release):

| File | Purpose |
|------|---------|
| `MRCONSO.RRF` | Concept names and identifiers; maps CUI to source-vocabulary codes (ICD-9-CM, ICD-10-CM, SNOMED CT, RxNorm) |
| `MRREL.RRF` | Inter-concept relationships (parent/child, sibling, mapped-to) |
| `MRSTY.RRF` | Semantic types per CUI (used to filter clinical vs. non-clinical concepts) |
| `MRSAT.RRF` | Concept attributes (NDC mappings via RXNSAT data) |

These files enable the ICD-9 -> CUI -> SNOMED CT and NDC -> RxCUI mapping
chains used in the ontology engine (`src/ontology/`).

### 2.2 SNOMED CT

| Field | Value |
|-------|-------|
| Source | NLM (US Edition) via UMLS, or SNOMED International https://www.snomed.org/ |
| License | SNOMED CT Affiliate License (included in the UMLS licence for US users) |
| Target directory | `ontology_cf_anomaly/ontologies/snomed/` |

**Required RF2 files** (from the Snapshot release):

| File | Purpose |
|------|---------|
| `sct2_Concept_Snapshot_*.txt` | Active concept identifiers and status |
| `sct2_Description_Snapshot_*.txt` | Preferred terms and synonyms per concept |
| `sct2_Relationship_Snapshot_*.txt` | IS-A hierarchy and other defining relationships |

The ontology engine (`src/ontology/index.py`) uses these to build
parent-child hierarchies, preferred-term lookups, and domain classification.

### 2.3 RxNorm

| Field | Value |
|-------|-------|
| Source | NLM RxNorm https://www.nlm.nih.gov/research/umls/rxnorm/ |
| License | UMLS Metathesaurus License (same UTS account) |
| Target directory | `ontology_cf_anomaly/ontologies/rxnorm/` |

**Required files:**

| File | Purpose |
|------|---------|
| `RXNCONSO.RRF` | RxCUI concept names, term types, and source vocabularies |
| `RXNREL.RRF` | Inter-concept relationships (ingredient-of, has-tradename, etc.) |
| `RXNSAT.RRF` | Concept attributes including NDC-to-RxCUI mappings |

Day 8 work built a directed RxNorm knowledge graph from processed versions
of these files (see `ontology_cf_anomaly/docs/day8_rxnorm_graph.md`):
406,601 nodes, 1,661,222 edges.

### 2.4 cui2vec Embeddings (optional)

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/1804.01486 (embeddings distributed by authors) |
| Target directory | `ontology_cf_anomaly/ontologies/embeddings/` |

Pre-trained CUI embeddings used for ontology-aware embedding initialization
in the generative model (planned for Day 33).

---

## 3. Shared Infrastructure

| Item | Path |
|------|------|
| Dataset root configuration | `dataset_roots.yaml` |
| Processed output directory | `data/processed/` (key `processed_data_root`) |
| Shared extraction utilities | `src/preprocessing/common.py` |
| Token namespace policy | `config/data_semantics_policy.yaml` |

The extraction scripts share normalization, I/O, and statistics helpers from
`src/preprocessing/common.py`. All three scripts produce Parquet output and
JSON statistics summaries via `common.save_parquet()` and
`common.save_stats()`.
