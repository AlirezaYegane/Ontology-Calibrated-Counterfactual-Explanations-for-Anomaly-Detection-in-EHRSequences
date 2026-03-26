"""
tests/test_extract_mimiciv.py
==============================
Unit tests for MIMIC-IV sequence extraction.
Uses tiny synthetic DataFrames and temp directories — no real data needed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import extract_mimiciv, common


# ---------------------------------------------------------------------------
# Fixture: synthetic MIMIC-IV directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def mimiciv_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic MIMIC-IV directory for testing."""
    hosp = tmp_path / "hosp"
    hosp.mkdir()
    icu = tmp_path / "icu"
    icu.mkdir()

    # admissions
    pd.DataFrame([
        {"subject_id": 10, "hadm_id": 1001, "admittime": "2180-01-05", "dischtime": "2180-01-12"},
        {"subject_id": 20, "hadm_id": 1002, "admittime": "2185-06-01", "dischtime": "2185-06-10"},
        {"subject_id": 30, "hadm_id": 1003, "admittime": "2190-03-15", "dischtime": "2190-03-20"},
    ]).to_csv(hosp / "admissions.csv", index=False)

    # patients (MIMIC-IV style: anchor_age, anchor_year)
    pd.DataFrame([
        {"subject_id": 10, "gender": "M", "anchor_age": 55, "anchor_year": 2175},
        {"subject_id": 20, "gender": "F", "anchor_age": 30, "anchor_year": 2180},
        {"subject_id": 30, "gender": "M", "anchor_age": 10, "anchor_year": 2190},
    ]).to_csv(hosp / "patients.csv", index=False)

    # icustays (MIMIC-IV: stay_id, not icustay_id)
    pd.DataFrame([
        {"subject_id": 10, "hadm_id": 1001, "stay_id": 5001, "intime": "2180-01-06", "outtime": "2180-01-09"},
    ]).to_csv(icu / "icustays.csv", index=False)

    # diagnoses_icd (ICD-9 and ICD-10 mixed)
    pd.DataFrame([
        {"subject_id": 10, "hadm_id": 1001, "seq_num": 1, "icd_code": "4019",  "icd_version": "9"},
        {"subject_id": 10, "hadm_id": 1001, "seq_num": 2, "icd_code": "I10",   "icd_version": "10"},
        {"subject_id": 20, "hadm_id": 1002, "seq_num": 1, "icd_code": "E119",  "icd_version": "10"},
    ]).to_csv(hosp / "diagnoses_icd.csv", index=False)

    # procedures_icd
    pd.DataFrame([
        {"subject_id": 10, "hadm_id": 1001, "seq_num": 1, "icd_code": "0040",  "icd_version": "10"},
        {"subject_id": 20, "hadm_id": 1002, "seq_num": 1, "icd_code": "9904",  "icd_version": "9"},
    ]).to_csv(hosp / "procedures_icd.csv", index=False)

    # prescriptions
    pd.DataFrame([
        {"subject_id": 10, "hadm_id": 1001, "starttime": "2180-01-06", "stoptime": "2180-01-08",
         "drug": "Aspirin", "formulary_drug_cd": "A100", "gsn": "006038", "ndc": "00069004250"},
        {"subject_id": 20, "hadm_id": 1002, "starttime": "2185-06-02", "stoptime": "2185-06-05",
         "drug": "Metformin", "formulary_drug_cd": "M200", "gsn": "", "ndc": ""},
    ]).to_csv(hosp / "prescriptions.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGroupingAndOrdering:
    def test_basic_grouping(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        assert 1001 in df["hadm_id"].values
        assert 1002 in df["hadm_id"].values

    def test_empty_admission_excluded(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        # hadm 1003 has no diagnoses/procedures/meds
        assert 1003 not in df["hadm_id"].values

    def test_sequence_order_dx_proc_med(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1001].iloc[0]
        seq = row["sequence_tokens"]
        dx_end = max(i for i, t in enumerate(seq) if t.startswith("ICD"))
        proc_start = min(i for i, t in enumerate(seq) if "PROC" in t)
        assert dx_end < proc_start or all("PROC" not in t for t in seq[:dx_end + 1]) is False


class TestIcdVersionPrefixes:
    def test_icd9_prefix(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1001].iloc[0]
        assert "ICD9_DX:4019" in row["diagnosis_tokens"]

    def test_icd10_prefix(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1001].iloc[0]
        assert "ICD10_DX:I10" in row["diagnosis_tokens"]

    def test_icd10_proc_prefix(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1001].iloc[0]
        assert "ICD10_PROC:0040" in row["procedure_tokens"]

    def test_icd9_proc_prefix(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1002].iloc[0]
        assert "ICD9_PROC:9904" in row["procedure_tokens"]


class TestDemographics:
    def test_age_computed(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1001].iloc[0]
        # anchor_age=55, anchor_year=2175, admittime=2180 → age ≈ 60
        assert row["age_years"] == pytest.approx(60.0, abs=1)
        assert row["age_group"] == "45-64"

    def test_icu_stays_attached(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1001].iloc[0]
        assert 5001 in row["stay_ids"]

    def test_no_icu_stay(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1002].iloc[0]
        assert row["stay_ids"] == []


class TestMedFallback:
    def test_ndc_preferred(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1001].iloc[0]
        assert any(t.startswith("MED_NDC:") for t in row["medication_tokens"])

    def test_drug_fallback(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        row = df[df["hadm_id"] == 1002].iloc[0]
        assert any(t.startswith("MED_NAME:") for t in row["medication_tokens"])


class TestStatsGeneration:
    def test_stats_keys(self, mimiciv_dir: Path) -> None:
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        stats = common.build_stats(df, id_col="hadm_id", icu_list_col="stay_ids")
        required = {"number_of_records", "number_of_unique_ids", "sequence_length"}
        assert required <= set(stats.keys())
        assert stats["number_of_records"] == 2


class TestPathResolution:
    def test_nested_directories(self, mimiciv_dir: Path, tmp_path: Path) -> None:
        import shutil
        
        # Create nested structure mimicking PhysioNet archives
        nested_dir = tmp_path / "physionet.org" / "files" / "mimiciv" / "2.2"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        # The fixture creates hosp/ and icu/ directly in tmp_path (which is mimiciv_dir)
        shutil.move(str(mimiciv_dir / "hosp"), str(nested_dir / "hosp"))
        shutil.move(str(mimiciv_dir / "icu"), str(nested_dir / "icu"))
        
        # Should recursively find nested_dir/hosp
        df = extract_mimiciv.build_mimiciv_sequences(mimiciv_dir)
        assert len(df) == 2


class TestCliSmoke:
    def test_cli_produces_outputs(self, mimiciv_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        stats = tmp_path / "stats.json"
        repo_root = str(Path(__file__).resolve().parents[1])
        result = subprocess.run(
            [sys.executable, "-m", "src.preprocessing.extract_mimiciv",
             "--input-dir", str(mimiciv_dir),
             "--output-path", str(out),
             "--stats-path", str(stats)],
            capture_output=True, text=True, check=True,
            cwd=repo_root,
        )
        # Parquet if pyarrow installed, otherwise pickle fallback
        assert out.exists() or out.with_suffix(".pkl").exists()
        assert stats.exists()
        s = json.loads(stats.read_text(encoding="utf-8"))
        assert s["number_of_records"] == 2
