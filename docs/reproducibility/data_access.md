# Data Access

All inputs to this project are **restricted** and cannot be redistributed here. This page
lists what to obtain and where. Obtaining them is the user's responsibility under the
respective licenses.

## Clinical data (PhysioNet, credentialed)

Both datasets require a PhysioNet credentialed account: complete CITI "Data or Specimens Only
Research" training and sign the data use agreement for each.

| Dataset | Purpose | URL | `dataset_roots.yaml` key |
|---|---|---|---|
| MIMIC-IV | EHR sequences, benchmark-v2 (final evidence) | https://physionet.org/content/mimiciv/ | `mimic4_root` |
| eICU (GOSSIS) | External-validation attempt (blocked) | https://physionet.org/content/gossis/ | `eicu_root` |
| MIMIC-III | (historical extraction path only) | https://physionet.org/content/mimiciii/ | `mimic3_root` |

The final paper uses **MIMIC-IV only**. eICU is used solely for the external-validation
schema check, which is documented as blocked (APACHE/body-system tokens do not map to the
ICD/SNOMED/RxNorm ontology).

## Ontology resources (UMLS Terminology Services)

1. Register for a UTS account at https://uts.nlm.nih.gov and accept the UMLS license.
2. Download the **UMLS Metathesaurus** (release 2026AA was used; `MRCONSO.RRF`, `MRMAP`,
   and related files).
3. Download **SNOMED CT** (US edition RF2 Snapshot), obtained through the UMLS/UTS affiliate
   license.
4. Download **RxNorm** (`RXNCONSO.RRF`, `RXNREL.RRF`) from the NLM.

Place raw downloads under the (git-ignored) `ontologies/raw/` area and parse them into
`ontologies/processed/` with the parsing scripts (see the runbook). Parsed outputs and
derived maps are git-ignored because they embed licensed content.

## What you do NOT need

You do not need any restricted data to:

- run the test suite (`python -m pytest`);
- read the committed aggregate results under `artifacts/`;
- run `scripts/run_phase8_final_checks.py`.

## Licensing reminder

MIMIC-IV / eICU are governed by the PhysioNet DUA; UMLS/SNOMED CT/RxNorm by the UMLS license
(with additional SNOMED affiliate terms). Do not commit any raw or derived patient/ontology
files. The `.gitignore` enforces this; see
[`artifact_manifest.md`](artifact_manifest.md) for the exclusion list.
