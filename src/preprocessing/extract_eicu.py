"""
src/preprocessing/extract_eicu.py
==================================
eICU (GOSSIS-1) ICU-stay-level sequence extraction.

The GOSSIS-1 dataset exposes eICU data as a *single flat CSV* per ICU stay
(keyed by ``patientunitstayid``), rather than the standard multi-table eICU
schema.  Diagnosis information is encoded via:

  * APACHE-2 / APACHE-3j diagnosis codes
  * APACHE body-system categories
  * Comorbidity flag columns (aids, cirrhosis, diabetes, …)

Token prefixes:

  * ``EICU_APACHE2_DX:<code>``
  * ``EICU_APACHE3_DX:<code>``
  * ``EICU_BODYSYS:<slug>``
  * ``EICU_COMORB:<name>``

CLI usage::

    python -m src.preprocessing.extract_eicu \\
        --input-dir  D:/data/gossis-1-eicu/physionet.org/files/gossis-1-eicu/1.0.0 \\
        --output-path data/processed/eicu_sequences.parquet \\
        --stats-path  data/processed/eicu_stats.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.common import (
    age_group,
    build_stats,
    clean_str,
    normalize_text_token,
    save_parquet,
    save_stats,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GOSSIS-1 column sets
# ---------------------------------------------------------------------------

_COMORBIDITY_FLAGS = (
    "aids",
    "cirrhosis",
    "diabetes_mellitus",
    "hepatic_failure",
    "immunosuppression",
    "leukemia",
    "lymphoma",
    "solid_tumor_with_metastasis",
)

_DEMOGRAPHIC_COLS = [
    "patientunitstayid",
    "patient_id",
    "age",
    "gender",
    "ethnicity",
    "hospital_los_days",
    "icu_los_days",
    "hospital_death",
    "icu_death",
]

_DIAGNOSIS_COLS = [
    "apache_2_diagnosis",
    "apache_3j_diagnosis",
    "apache_3j_bodysystem",
    "apache_2_bodysystem",
]


# ---------------------------------------------------------------------------
# Token builders
# ---------------------------------------------------------------------------


def _clean_apache_code(value: Any) -> str | None:
    """Clean APACHE diagnosis codes, stripping .0 from float representations."""
    raw = clean_str(value)
    if raw is None:
        return None
    # pandas reads integer codes as float (113 → 113.0); strip .0
    if raw.endswith(".0"):
        stripped = raw[:-2]
        if stripped.isdigit():
            return stripped
    return raw


def _build_tokens_for_stay(row: pd.Series) -> list[str]:
    """Build all tokens for a single ICU stay row."""
    tokens: list[str] = []

    # APACHE diagnosis codes
    a2dx = _clean_apache_code(row.get("apache_2_diagnosis"))
    if a2dx is not None:
        tokens.append(f"EICU_APACHE2_DX:{a2dx}")

    a3dx = _clean_apache_code(row.get("apache_3j_diagnosis"))
    if a3dx is not None:
        tokens.append(f"EICU_APACHE3_DX:{a3dx}")

    # Body system categories
    for col in ("apache_3j_bodysystem", "apache_2_bodysystem"):
        slug = normalize_text_token(row.get(col))
        if slug is not None:
            tokens.append(f"EICU_BODYSYS:{slug}")

    # Comorbidity flags
    for flag in _COMORBIDITY_FLAGS:
        val = row.get(flag)
        if pd.notna(val):
            try:
                if int(float(val)) == 1:
                    tokens.append(f"EICU_COMORB:{flag}")
            except (ValueError, TypeError):
                pass

    return tokens


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def _resolve_gossis_csv(input_dir: Path) -> Path:
    """Find the GOSSIS-1 flat CSV (gz or plain) in the input directory, searching recursively."""
    candidates = [
        "gossis-1-eicu-only.csv.gz",
        "gossis-1-eicu-only.csv",
    ]
    # Check directly first
    for name in candidates:
        p = input_dir / name
        if p.is_file():
            return p

    # Recursive search
    found_gz = list(input_dir.rglob("gossis-1-eicu-only.csv.gz"))
    if found_gz:
        found_gz.sort(key=lambda p: len(p.parts))
        log.info("Resolved eICU CSV to: %s", found_gz[0])
        return found_gz[0]

    found_csv = list(input_dir.rglob("gossis-1-eicu-only.csv"))
    if found_csv:
        found_csv.sort(key=lambda p: len(p.parts))
        log.info("Resolved eICU CSV to: %s", found_csv[0])
        return found_csv[0]

    raise FileNotFoundError(
        f"GOSSIS-1 eICU CSV not found in: {input_dir}\n"
        f"Searched recursively for: {candidates}"
    )


def build_eicu_sequences(
    input_dir: Path,
    *,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Build ICU-stay-level eICU sequences from GOSSIS-1 data.

    Parameters
    ----------
    input_dir:
        Directory containing the ``gossis-1-eicu-only.csv.gz`` file.
    max_records:
        If set, limit rows for smoke testing.
    """
    csv_path = _resolve_gossis_csv(input_dir)
    log.info("Loading %s …", csv_path.name)

    # Only read the columns we actually need
    needed_cols = set(_DEMOGRAPHIC_COLS + _DIAGNOSIS_COLS + list(_COMORBIDITY_FLAGS))
    df = pd.read_csv(csv_path, low_memory=False, nrows=max_records)
    df.columns = df.columns.str.strip().str.lower()

    # Keep only available columns
    available = [c for c in needed_cols if c in df.columns]
    missing = needed_cols - set(available)
    if missing:
        log.warning("Missing columns in GOSSIS CSV: %s", sorted(missing))
    df = df[available].copy()

    log.info("Loaded %d rows, %d usable columns", len(df), len(available))

    # --- build records ------------------------------------------------------
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        stay_id_raw = row.get("patientunitstayid")
        stay_id = _safe_identifier(stay_id_raw)
        if not stay_id:
            continue

        tokens = _build_tokens_for_stay(row)
        if not tokens:
            continue

        # Age handling
        age_raw = row.get("age")
        age_val: float | None = None
        if pd.notna(age_raw):
            try:
                age_val = float(age_raw)
            except (ValueError, TypeError):
                # eICU encodes 89+ as "> 89" sometimes
                if isinstance(age_raw, str) and "89" in age_raw:
                    age_val = 90.0

        patient_id_raw = row.get("patient_id")
        patient_id_val = _safe_identifier(patient_id_raw)

        records.append({
            "patientunitstayid": stay_id,
            "patient_id": patient_id_val,
            "gender": clean_str(row.get("gender")),
            "ethnicity": clean_str(row.get("ethnicity")),
            "age_years": age_val,
            "age_group": age_group(age_val),
            "hospital_los_days": _safe_float(row.get("hospital_los_days")),
            "icu_los_days": _safe_float(row.get("icu_los_days")),
            "hospital_death": _safe_int(row.get("hospital_death")),
            "icu_death": _safe_int(row.get("icu_death")),
            "diagnosis_tokens": [t for t in tokens if "DX:" in t or "BODYSYS:" in t],
            "comorbidity_tokens": [t for t in tokens if "COMORB:" in t],
            "sequence_tokens": tokens,
            "n_diagnoses": sum(1 for t in tokens if "DX:" in t or "BODYSYS:" in t),
            "n_comorbidities": sum(1 for t in tokens if "COMORB:" in t),
            "sequence_length": len(tokens),
        })

    result = pd.DataFrame(records)
    if result.empty:
        log.warning("No non-empty stay records produced.")
        return result

    result = result.sort_values("patientunitstayid", kind="mergesort").reset_index(drop=True)
    log.info("Assembled %d non-empty ICU-stay records", len(result))
    return result


def _safe_identifier(val: Any) -> str | None:
    """Safely convert a patient or stay identifier to a string without arbitrary casting."""
    if pd.isna(val):
        return None
    
    # Handle pandas reading integer IDs as float (e.g. 1234.0)
    if isinstance(val, float) and val.is_integer():
        s = str(int(val))
    else:
        s = str(val).strip()
        
    if not s or s.lower() == "nan":
        return None
    return s


def _safe_float(val: Any) -> float | None:
    if pd.isna(val):
        return None
    try:
        return round(float(val), 3)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 10 – eICU (GOSSIS-1) sequence extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, type=Path,
                    help="Directory containing gossis-1-eicu-only.csv.gz")
    p.add_argument("--output-path", required=True, type=Path,
                    help="Output Parquet path")
    p.add_argument("--stats-path", required=True, type=Path,
                    help="Output stats JSON path")
    p.add_argument("--max-records", type=int, default=None,
                    help="Limit rows (for smoke tests)")
    p.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    df = build_eicu_sequences(args.input_dir, max_records=args.max_records)

    if df.empty:
        log.error("Empty result – nothing to save.")
        sys.exit(1)

    save_parquet(df, args.output_path)
    stats = build_stats(
        df,
        id_col="patientunitstayid",
        subject_col="patient_id",
        n_diagnoses_col="n_diagnoses",
        n_procedures_col=None,
        n_medications_col=None,
        n_treatments_col="n_comorbidities",
    )
    save_stats(stats, args.stats_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
