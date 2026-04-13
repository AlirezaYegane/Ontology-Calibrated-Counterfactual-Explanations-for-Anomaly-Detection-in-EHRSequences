"""
src/preprocessing/common.py
============================
Shared helpers for multi-dataset clinical sequence extraction.

Provides reusable utilities for:
- Safe CSV loading (gz / plain)
- Text / code normalization
- Age bucketing
- Token grouping
- Dataset statistics computation
- Parquet serialization
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def resolve_csv(base_dir: Path, stem: str) -> Path:
    """Resolve a CSV or gzipped CSV path for a table *stem*.

    Checks lowercase then uppercase variants so that both MIMIC-III
    (ADMISSIONS.csv.gz) and MIMIC-IV (admissions.csv.gz) are found.
    """
    for name in (stem, stem.upper(), stem.lower()):
        for ext in (".csv.gz", ".csv"):
            path = base_dir / f"{name}{ext}"
            if path.exists():
                return path
    raise FileNotFoundError(
        f"Missing required file for table '{stem}' under: {base_dir}"
    )


def read_table(
    base_dir: Path,
    stem: str,
    usecols: list[str] | None = None,
    parse_dates: list[str] | None = None,
    dtype: dict[str, type] | None = None,
) -> pd.DataFrame:
    """Read a CSV/CSV.GZ table with lowercase column names."""
    path = resolve_csv(base_dir, stem)
    log.info("Loading %s …", path.name)
    df = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
    )
    df.columns = df.columns.str.strip().str.lower()
    return df


# ---------------------------------------------------------------------------
# String / code normalization
# ---------------------------------------------------------------------------

_BLANK_VALS = frozenset({"", "nan", "none", "null"})


def clean_str(value: Any) -> str | None:
    """Return a stripped non-empty string or ``None``."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in _BLANK_VALS:
        return None
    return text


def normalize_code(code: Any) -> str | None:
    """Conservatively normalize a clinical code (ICD, NDC, etc.)."""
    text = clean_str(code)
    if text is None:
        return None
    return text.replace(" ", "").upper()


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def normalize_text_token(text: Any) -> str | None:
    """Normalize free text into a deterministic, token-safe slug.

    - Lowercases
    - Strips whitespace
    - Replaces non-alphanumeric runs with ``_``
    - Collapses repeated underscores
    - Strips leading/trailing underscores
    """
    raw = clean_str(text)
    if raw is None:
        return None
    slug = _SLUG_RE.sub("_", raw.lower()).strip("_")
    if not slug:
        return None
    return slug


def normalize_drug_name(name: Any) -> str | None:
    """Normalize a drug name to a stable uppercase token."""
    text = clean_str(name)
    if text is None:
        return None
    return "_".join(text.upper().split())


def normalize_ndc(code: Any) -> str | None:
    """Normalize an NDC identifier, filtering known null sentinels."""
    text = clean_str(code)
    if text is None:
        return None
    if text in {"0", "0000", "00000000000"}:
        return None
    return text


# ---------------------------------------------------------------------------
# Age helpers
# ---------------------------------------------------------------------------


def age_group(age: float | None) -> str:
    """Bucket a numeric age (years) into a human-readable label."""
    if age is None or pd.isna(age):
        return "unknown"
    if age < 0:
        return "unknown"
    if age < 18:
        return "0-17"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 65:
        return "45-64"
    if age < 80:
        return "65-79"
    return "80+"


# ---------------------------------------------------------------------------
# Token grouping
# ---------------------------------------------------------------------------


def ordered_group_list(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    sort_cols: list[str],
) -> dict[int, list[str]]:
    """Group *value_col* tokens by *group_col*, preserving *sort_cols* order."""
    if df.empty:
        return {}

    cols = [group_col] + [c for c in sort_cols if c in df.columns] + [value_col]
    temp = df[cols].copy()
    temp = temp.dropna(subset=[group_col, value_col])

    present = [c for c in sort_cols if c in temp.columns]
    if present:
        temp = temp.sort_values([group_col] + present, kind="mergesort")

    grouped = temp.groupby(group_col)[value_col].apply(list).to_dict()
    return {int(k): v for k, v in grouped.items()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def build_stats(
    df: pd.DataFrame,
    *,
    id_col: str = "hadm_id",
    subject_col: str = "subject_id",
    seq_col: str = "sequence_length",
    gender_col: str = "gender",
    age_group_col: str = "age_group",
    n_diagnoses_col: str = "n_diagnoses",
    n_procedures_col: str | None = "n_procedures",
    n_medications_col: str | None = "n_medications",
    n_treatments_col: str | None = None,
    icu_list_col: str | None = None,
) -> dict[str, Any]:
    """Build dataset-level statistics for a processed sequence DataFrame.

    Returns a JSON-serialisable dict suitable for saving as a stats file.
    """
    if df.empty:
        return {
            "number_of_records": 0,
            "number_of_unique_subjects": 0,
            "number_of_unique_ids": 0,
        }

    sl = df[seq_col]
    stats: dict[str, Any] = {
        "number_of_records": int(len(df)),
        "number_of_unique_subjects": int(df[subject_col].nunique()) if subject_col in df.columns else 0,
        "number_of_unique_ids": int(df[id_col].nunique()) if id_col in df.columns else 0,
    }

    # Diagnosis / procedure / medication / treatment counts
    for label, col in [
        ("number_with_diagnoses", n_diagnoses_col),
        ("number_with_procedures", n_procedures_col),
        ("number_with_medications", n_medications_col),
        ("number_with_treatments", n_treatments_col),
    ]:
        if col and col in df.columns:
            stats[label] = int((df[col] > 0).sum())

    # ICU stay percentage
    if icu_list_col and icu_list_col in df.columns:
        stats["pct_with_icu_stay"] = round(
            float((df[icu_list_col].map(len) > 0).mean() * 100), 2
        )

    # Sequence length distribution
    stats["sequence_length"] = {
        "min": int(sl.min()),
        "p25": float(sl.quantile(0.25)),
        "median": float(sl.median()),
        "mean": round(float(sl.mean()), 3),
        "p75": float(sl.quantile(0.75)),
        "p95": float(sl.quantile(0.95)),
        "max": int(sl.max()),
    }

    # Per-type means
    for label, col in [
        ("diagnosis_count_mean", n_diagnoses_col),
        ("procedure_count_mean", n_procedures_col),
        ("medication_count_mean", n_medications_col),
        ("treatment_count_mean", n_treatments_col),
    ]:
        if col and col in df.columns:
            stats[label] = round(float(df[col].mean()), 3)

    # Unique tokens
    if "sequence_tokens" in df.columns:
        all_tokens: set[str] = set()
        for toks in df["sequence_tokens"]:
            if isinstance(toks, list):
                all_tokens.update(toks)
        stats["number_of_unique_tokens"] = len(all_tokens)

    # Demographics
    if gender_col in df.columns:
        stats["gender_counts"] = (
            df[gender_col].fillna("unknown").value_counts(dropna=False).to_dict()
        )
        stats["missing_gender"] = int(df[gender_col].isna().sum())
    if age_group_col in df.columns:
        stats["age_group_counts"] = (
            df[age_group_col].fillna("unknown").value_counts(dropna=False).to_dict()
        )

    return stats


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

_LIST_COLS = (
    "icustay_ids",
    "stay_ids",
    "diagnosis_tokens",
    "procedure_tokens",
    "medication_tokens",
    "treatment_tokens",
    "comorbidity_tokens",
    "sequence_tokens",
)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as Parquet (list columns serialised as JSON strings).

    Falls back to pickle if ``pyarrow`` is not installed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in _LIST_COLS:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
    try:
        out.to_parquet(path, index=False, engine="pyarrow")
        log.info("Saved parquet → %s (%d rows)", path, len(out))
    except ImportError:
        pkl_path = path.with_suffix(".pkl")
        out.to_pickle(pkl_path)
        log.warning(
            "pyarrow not installed – saved pickle instead → %s (%d rows)",
            pkl_path, len(out),
        )


def save_stats(stats: dict[str, Any], path: Path) -> None:
    """Write statistics dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("Saved stats → %s", path)
