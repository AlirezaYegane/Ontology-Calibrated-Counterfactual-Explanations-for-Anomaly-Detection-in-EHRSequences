"""
tests/test_extract_eicu.py
============================
Unit tests for eICU (GOSSIS-1) sequence extraction.
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

from src.preprocessing import extract_eicu, common


# ---------------------------------------------------------------------------
# Fixture: synthetic GOSSIS-1 CSV
# ---------------------------------------------------------------------------

@pytest.fixture()
def eicu_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic GOSSIS-1 eICU directory."""
    df = pd.DataFrame([
        {
            "patientunitstayid": 3001,
            "patient_id": 100,
            "age": 65,
            "gender": "Male",
            "ethnicity": "Caucasian",
            "hospital_los_days": 8.5,
            "icu_los_days": 3.2,
            "hospital_death": 0,
            "icu_death": 0,
            "apache_2_diagnosis": 113,
            "apache_3j_diagnosis": 502.01,
            "apache_3j_bodysystem": "Cardiovascular",
            "apache_2_bodysystem": "Cardiovascular",
            "aids": 0,
            "cirrhosis": 0,
            "diabetes_mellitus": 1,
            "hepatic_failure": 0,
            "immunosuppression": 0,
            "leukemia": 0,
            "lymphoma": 0,
            "solid_tumor_with_metastasis": 0,
        },
        {
            "patientunitstayid": 3002,
            "patient_id": 200,
            "age": 45,
            "gender": "Female",
            "ethnicity": "African American",
            "hospital_los_days": 12.0,
            "icu_los_days": 5.0,
            "hospital_death": 1,
            "icu_death": 1,
            "apache_2_diagnosis": 114,
            "apache_3j_diagnosis": 301.02,
            "apache_3j_bodysystem": "Respiratory",
            "apache_2_bodysystem": "Respiratory",
            "aids": 1,
            "cirrhosis": 0,
            "diabetes_mellitus": 0,
            "hepatic_failure": 0,
            "immunosuppression": 1,
            "leukemia": 0,
            "lymphoma": 0,
            "solid_tumor_with_metastasis": 0,
        },
        {
            # Row with string/alphanumeric IDs
            "patientunitstayid": "eicu_002-5276",
            "patient_id": "P-999_X",
            "age": 45,
            "gender": "Female",
            "ethnicity": "Caucasian",
            "hospital_los_days": 1.0,
            "icu_los_days": 1.0,
            "hospital_death": 0,
            "icu_death": 0,
            "apache_2_diagnosis": 113,
            "apache_3j_diagnosis": 502.01,
            "apache_3j_bodysystem": "Cardiovascular",
            "apache_2_bodysystem": "Cardiovascular",
            "aids": 0,
            "cirrhosis": 0,
            "diabetes_mellitus": 0,
            "hepatic_failure": 0,
            "immunosuppression": 0,
            "leukemia": 0,
            "lymphoma": 0,
            "solid_tumor_with_metastasis": 0,
        },
        {
            # Row with no diagnosis info → should be excluded
            "patientunitstayid": 3003,
            "patient_id": 300,
            "age": 30,
            "gender": "Male",
            "ethnicity": "",
            "hospital_los_days": 2.0,
            "icu_los_days": 1.0,
            "hospital_death": 0,
            "icu_death": 0,
            "apache_2_diagnosis": None,
            "apache_3j_diagnosis": None,
            "apache_3j_bodysystem": None,
            "apache_2_bodysystem": None,
            "aids": 0,
            "cirrhosis": 0,
            "diabetes_mellitus": 0,
            "hepatic_failure": 0,
            "immunosuppression": 0,
            "leukemia": 0,
            "lymphoma": 0,
            "solid_tumor_with_metastasis": 0,
        },
    ])
    df.to_csv(tmp_path / "gossis-1-eicu-only.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGrouping:
    def test_records_created(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        assert len(df) == 3  # 3003 excluded (no tokens)
        # Verify numeric and string IDs correctly preserved as strings
        ids = set(df["patientunitstayid"].astype(str).tolist())
        assert ids == {"3001", "3002", "eicu_002-5276"}

    def test_empty_excluded(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        assert 3003 not in df["patientunitstayid"].values
        assert "3003" not in set(df["patientunitstayid"].astype(str))

    def test_alphanumeric_ids(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        row = df[df["patientunitstayid"].astype(str) == "eicu_002-5276"].iloc[0]
        assert row["patient_id"] == "P-999_X"


class TestTextNormalization:
    def test_bodysystem_normalized(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        row = df[df["patientunitstayid"].astype(str) == "3001"].iloc[0]
        assert "EICU_BODYSYS:cardiovascular" in row["sequence_tokens"]

    def test_apache_diagnosis_code(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        row = df[df["patientunitstayid"].astype(str) == "3001"].iloc[0]
        assert "EICU_APACHE2_DX:113" in row["sequence_tokens"]
        assert any("EICU_APACHE3_DX:" in t for t in row["sequence_tokens"])


class TestComorbidityFlags:
    def test_positive_flag_creates_token(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        row = df[df["patientunitstayid"].astype(str) == "3001"].iloc[0]
        assert "EICU_COMORB:diabetes_mellitus" in row["comorbidity_tokens"]
        assert row["n_comorbidities"] == 1

    def test_multiple_comorbidities(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        row = df[df["patientunitstayid"].astype(str) == "3002"].iloc[0]
        assert "EICU_COMORB:aids" in row["comorbidity_tokens"]
        assert "EICU_COMORB:immunosuppression" in row["comorbidity_tokens"]
        assert row["n_comorbidities"] == 2


class TestDemographics:
    def test_age_group(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        row = df[df["patientunitstayid"].astype(str) == "3001"].iloc[0]
        assert row["age_years"] == 65.0
        assert row["age_group"] == "65-79"

    def test_gender(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        row = df[df["patientunitstayid"].astype(str) == "3002"].iloc[0]
        assert row["gender"] == "Female"


class TestSerializationShape:
    def test_output_columns(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        required = {
            "patientunitstayid", "patient_id", "gender", "age_years",
            "age_group", "diagnosis_tokens", "comorbidity_tokens",
            "sequence_tokens", "n_diagnoses", "sequence_length",
        }
        assert required <= set(df.columns)


class TestStatsGeneration:
    def test_stats_keys(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        stats = common.build_stats(
            df, id_col="patientunitstayid", subject_col="patient_id",
            n_diagnoses_col="n_diagnoses", n_procedures_col=None,
            n_medications_col=None, n_treatments_col="n_comorbidities",
        )
        assert stats["number_of_records"] == 3
        assert "sequence_length" in stats
        assert "gender_counts" in stats

    def test_unique_tokens_counted(self, eicu_dir: Path) -> None:
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        stats = common.build_stats(
            df, id_col="patientunitstayid", subject_col="patient_id",
            n_diagnoses_col="n_diagnoses",
        )
        assert stats["number_of_unique_tokens"] > 0


class TestPathResolution:
    def test_nested_file(self, eicu_dir: Path, tmp_path: Path) -> None:
        import shutil
        
        # Create nested structure mimicking PhysioNet archives
        nested_dir = tmp_path / "physionet.org" / "files" / "gossis-1-eicu" / "1.0.0"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        # The fixture creates the file directly in tmp_path (which is eicu_dir)
        src_file = eicu_dir / "gossis-1-eicu-only.csv"
        dst_file = nested_dir / "gossis-1-eicu-only.csv"
        shutil.move(str(src_file), str(dst_file))
        
        # Should recursively find nested_dir/gossis-1-eicu-only.csv
        df = extract_eicu.build_eicu_sequences(eicu_dir)
        assert len(df) == 3


class TestCliSmoke:
    def test_cli_produces_outputs(self, eicu_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        stats_p = tmp_path / "stats.json"
        repo_root = str(Path(__file__).resolve().parents[1])
        subprocess.run(
            [sys.executable, "-m", "src.preprocessing.extract_eicu",
             "--input-dir", str(eicu_dir),
             "--output-path", str(out),
             "--stats-path", str(stats_p)],
            capture_output=True, text=True, check=True,
            cwd=repo_root,
        )
        assert out.exists() or out.with_suffix(".pkl").exists()
        assert stats_p.exists()
        s = json.loads(stats_p.read_text(encoding="utf-8"))
        assert s["number_of_records"] == 3
