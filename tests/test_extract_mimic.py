from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "preprocessing" / "extract_mimic.py"
spec = importlib.util.spec_from_file_location("extract_mimic", MODULE_PATH)
extract_mimic = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extract_mimic)


@pytest.fixture()
def fixture_dir(tmp_path: Path) -> Path:
    base = tmp_path / "mimiciii"
    base.mkdir(parents=True, exist_ok=True)

    admissions = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 1001, "admittime": "2020-01-10 00:00:00", "dischtime": "2020-01-15 00:00:00"},
            {"subject_id": 2, "hadm_id": 1002, "admittime": "2020-02-10 00:00:00", "dischtime": "2020-02-15 00:00:00"},
            {"subject_id": 3, "hadm_id": 1003, "admittime": "2020-03-10 00:00:00", "dischtime": "2020-03-15 00:00:00"},
        ]
    )

    patients = pd.DataFrame(
        [
            {"subject_id": 1, "gender": "M", "dob": "1980-01-01 00:00:00"},
            {"subject_id": 2, "gender": "F", "dob": "1930-01-01 00:00:00"},
            {"subject_id": 3, "gender": "M", "dob": "2005-01-01 00:00:00"},
        ]
    )

    icustays = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 1001, "icustay_id": 9001, "intime": "2020-01-10 01:00:00", "outtime": "2020-01-12 01:00:00"},
            {"subject_id": 1, "hadm_id": 1001, "icustay_id": 9002, "intime": "2020-01-12 02:00:00", "outtime": "2020-01-14 01:00:00"},
        ]
    )

    diagnoses = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 1001, "seq_num": 2, "icd9_code": "4019"},
            {"subject_id": 1, "hadm_id": 1001, "seq_num": 1, "icd9_code": "25000"},
        ]
    )

    procedures = pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 1001, "seq_num": 2, "icd9_code": "5123"},
            {"subject_id": 1, "hadm_id": 1001, "seq_num": 1, "icd9_code": "3893"},
        ]
    )

    prescriptions = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 1001,
                "icustay_id": 9001,
                "startdate": "2020-01-10 03:00:00",
                "enddate": "2020-01-11 03:00:00",
                "drug": "Acetaminophen",
                "drug_name_poe": "Acetaminophen",
                "drug_name_generic": "Acetaminophen",
                "ndc": "00045050130",
            },
            {
                "subject_id": 2,
                "hadm_id": 1002,
                "icustay_id": None,
                "startdate": "2020-02-10 03:00:00",
                "enddate": "2020-02-11 03:00:00",
                "drug": "Metformin",
                "drug_name_poe": None,
                "drug_name_generic": "Metformin",
                "ndc": None,
            },
        ]
    )

    admissions.to_csv(base / "ADMISSIONS.csv", index=False)
    patients.to_csv(base / "PATIENTS.csv", index=False)
    icustays.to_csv(base / "ICUSTAYS.csv", index=False)
    diagnoses.to_csv(base / "DIAGNOSES_ICD.csv", index=False)
    procedures.to_csv(base / "PROCEDURES_ICD.csv", index=False)
    prescriptions.to_csv(base / "PRESCRIPTIONS.csv", index=False)

    return base


def test_grouping_and_ordering(fixture_dir: Path) -> None:
    df = extract_mimic.build_mimiciii_sequences(fixture_dir)
    row = df[df["hadm_id"] == 1001].iloc[0]

    assert row["diagnosis_tokens"] == ["DX_ICD9:25000", "DX_ICD9:4019"]
    assert row["procedure_tokens"] == ["PROC_ICD9:3893", "PROC_ICD9:5123"]
    assert row["medication_tokens"] == ["MED_NDC:00045050130"]
    assert row["icustay_ids"] == [9001, 9002]
    assert row["sequence_length"] == 5


def test_medication_fallback_token_logic(fixture_dir: Path) -> None:
    df = extract_mimic.build_mimiciii_sequences(fixture_dir)
    row = df[df["hadm_id"] == 1002].iloc[0]
    assert row["medication_tokens"] == ["MED_NAME:METFORMIN"]
    assert row["n_medications"] == 1


def test_empty_admissions_are_excluded(fixture_dir: Path) -> None:
    df = extract_mimic.build_mimiciii_sequences(fixture_dir)
    assert 1003 not in set(df["hadm_id"].tolist())


def test_age_group_derivation(fixture_dir: Path) -> None:
    df = extract_mimic.build_mimiciii_sequences(fixture_dir)
    age_groups = dict(zip(df["hadm_id"], df["age_group"]))
    assert age_groups[1001] == "30-44"
    assert age_groups[1002] == "80+"


def test_summary_generation(fixture_dir: Path) -> None:
    df = extract_mimic.build_mimiciii_sequences(fixture_dir)
    summary = extract_mimic.build_summary(df)

    assert summary["n_records"] == 2
    assert summary["n_subjects"] == 2
    assert summary["n_hadm"] == 2
    assert pytest.approx(summary["pct_with_icu_stay"], 0.01) == 0.5
    assert "sequence_length" in summary
    assert "gender_counts" in summary
    assert "age_group_counts" in summary


def test_cli_smoke_run(fixture_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--mimic-dir",
            str(fixture_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert (out_dir / "mimiciii_sequences.pkl").exists()
    assert (out_dir / "mimiciii_sequences_flat.csv").exists()
    assert (out_dir / "mimiciii_sequence_summary.json").exists()

    summary = json.loads((out_dir / "mimiciii_sequence_summary.json").read_text(encoding="utf-8"))
    assert summary["n_records"] == 2
    assert "Saved:" in result.stdout
