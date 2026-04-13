"""
src/preprocessing/extract_mimiciv.py
=====================================
MIMIC-IV admission-level sequence extraction.

Loads standard MIMIC-IV v2.x hospital + ICU tables, builds per-admission
token sequences from diagnoses (ICD-9/10), procedures (ICD-9/10) and
prescriptions, attaches demographic metadata, and writes:

  * ``<output-path>``  – Parquet with one row per admission
  * ``<stats-path>``   – JSON summary statistics

CLI usage::

    python -m src.preprocessing.extract_mimiciv \\
        --input-dir  D:/data/mimiciv/2.2 \\
        --output-path data/processed/mimiciv_sequences.parquet \\
        --stats-path  data/processed/mimiciv_stats.json
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
    normalize_code,
    normalize_drug_name,
    normalize_ndc,
    ordered_group_list,
    read_table,
    save_parquet,
    save_stats,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token builders
# ---------------------------------------------------------------------------


def _build_diagnosis_tokens(hosp_dir: Path) -> dict[int, list[str]]:
    """Build diagnosis tokens with ICD-9/10 version prefixes."""
    df = read_table(
        hosp_dir,
        "diagnoses_icd",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": str},
    )
    df["icd_code"] = df["icd_code"].map(normalize_code)
    df = df.dropna(subset=["icd_code", "hadm_id"])

    def _prefix(row: pd.Series) -> str:
        ver = clean_str(row.get("icd_version"))
        if ver == "10":
            return f"ICD10_DX:{row['icd_code']}"
        return f"ICD9_DX:{row['icd_code']}"

    df["token"] = df.apply(_prefix, axis=1)
    return ordered_group_list(df, "hadm_id", "token", ["seq_num"])


def _build_procedure_tokens(hosp_dir: Path) -> dict[int, list[str]]:
    """Build procedure tokens with ICD-9/10 version prefixes."""
    df = read_table(
        hosp_dir,
        "procedures_icd",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": str},
    )
    df["icd_code"] = df["icd_code"].map(normalize_code)
    df = df.dropna(subset=["icd_code", "hadm_id"])

    def _prefix(row: pd.Series) -> str:
        ver = clean_str(row.get("icd_version"))
        if ver == "10":
            return f"ICD10_PROC:{row['icd_code']}"
        return f"ICD9_PROC:{row['icd_code']}"

    df["token"] = df.apply(_prefix, axis=1)
    return ordered_group_list(df, "hadm_id", "token", ["seq_num"])


def _build_medication_tokens(hosp_dir: Path) -> dict[int, list[str]]:
    """Build medication tokens from prescriptions."""
    df = read_table(
        hosp_dir,
        "prescriptions",
        usecols=[
            "subject_id", "hadm_id", "starttime", "stoptime",
            "drug", "formulary_drug_cd", "gsn", "ndc",
        ],
        parse_dates=["starttime", "stoptime"],
        dtype={"ndc": str, "gsn": str},
    )
    df["ndc"] = df["ndc"].map(normalize_ndc)
    df["drug"] = df["drug"].map(normalize_drug_name)

    def _med_token(row: pd.Series) -> str | None:
        ndc = row.get("ndc")
        if pd.notna(ndc) and ndc is not None:
            return f"MED_NDC:{ndc}"
        drug = row.get("drug")
        if pd.notna(drug) and drug is not None:
            return f"MED_NAME:{drug}"
        return None

    df["token"] = df.apply(_med_token, axis=1)
    df = df.dropna(subset=["token", "hadm_id"])
    return ordered_group_list(df, "hadm_id", "token", ["starttime", "stoptime"])


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def _resolve_mimiciv_dirs(input_dir: Path) -> tuple[Path, Path]:
    """Find ``hosp`` and ``icu`` directories, even if nested."""
    hosp_dir = input_dir / "hosp"
    icu_dir = input_dir / "icu"

    if hosp_dir.is_dir():
        return hosp_dir, icu_dir

    hosp_cands = [p for p in input_dir.rglob("hosp") if p.is_dir()]
    if not hosp_cands:
        raise FileNotFoundError(
            f"hosp/ subdirectory not found in: {input_dir}\n"
            f"Searched recursively but found no 'hosp' directories."
        )

    # Pick the shortest path (closest to root)
    hosp_cands.sort(key=lambda p: len(p.parts))
    best_hosp = hosp_cands[0]
    best_icu = best_hosp.parent / "icu"

    if not best_icu.is_dir():
        icu_cands = [p for p in input_dir.rglob("icu") if p.is_dir()]
        if icu_cands:
            icu_cands.sort(key=lambda p: len(p.parts))
            best_icu = icu_cands[0]

    log.info("Resolved MIMIC-IV hosp dir to: %s", best_hosp)
    if best_icu.is_dir():
        log.info("Resolved MIMIC-IV icu dir to: %s", best_icu)

    return best_hosp, best_icu


def build_mimiciv_sequences(
    input_dir: Path,
    *,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Build admission-level MIMIC-IV sequences with demographics.

    Parameters
    ----------
    input_dir:
        Root MIMIC-IV directory containing ``hosp/`` and ``icu/`` subdirs.
    max_records:
        If set, limit admissions for smoke testing.
    """
    hosp_dir, icu_dir = _resolve_mimiciv_dirs(input_dir)


    # --- admissions ---------------------------------------------------------
    admissions = read_table(
        hosp_dir,
        "admissions",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )
    if max_records:
        admissions = admissions.head(max_records)

    # --- patients (MIMIC-IV uses anchor_age, not DOB) -----------------------
    patients = read_table(
        hosp_dir,
        "patients",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )

    # --- ICU stays (MIMIC-IV uses stay_id, not icustay_id) ------------------
    stay_ids_by_hadm: dict[int, list[int]] = {}
    if icu_dir.is_dir():
        try:
            icu = read_table(
                icu_dir,
                "icustays",
                usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                parse_dates=["intime", "outtime"],
            )
            grouped = (
                icu.sort_values(["hadm_id", "intime"], kind="mergesort")
                .groupby("hadm_id")["stay_id"]
                .apply(lambda s: [int(x) for x in s.dropna().tolist()])
                .to_dict()
            )
            stay_ids_by_hadm = {int(k): v for k, v in grouped.items()}
        except FileNotFoundError:
            log.warning("icustays not found – continuing without ICU linkage")

    # --- tokens -------------------------------------------------------------
    dx_tokens = _build_diagnosis_tokens(hosp_dir)
    proc_tokens = _build_procedure_tokens(hosp_dir)
    med_tokens = _build_medication_tokens(hosp_dir)

    log.info(
        "Token maps: dx=%d hadm, proc=%d hadm, med=%d hadm",
        len(dx_tokens), len(proc_tokens), len(med_tokens),
    )

    # --- merge demographics -------------------------------------------------
    base = admissions.merge(patients, on="subject_id", how="left")

    # Compute approximate age at admission from anchor_age + anchor_year
    if "anchor_age" in base.columns and "anchor_year" in base.columns:
        base["age_years"] = base.apply(
            lambda r: _compute_age_iv(r["anchor_age"], r["anchor_year"], r["admittime"]),
            axis=1,
        )
    else:
        base["age_years"] = None

    base["age_group"] = base["age_years"].map(age_group)

    # --- assemble records ---------------------------------------------------
    records: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        hadm_id = int(row.hadm_id)

        dx_toks = dx_tokens.get(hadm_id, [])
        pr_toks = proc_tokens.get(hadm_id, [])
        md_toks = med_tokens.get(hadm_id, [])
        seq = dx_toks + pr_toks + md_toks

        if not seq:
            continue

        age_raw = row.age_years
        age_val = None if age_raw is None or (isinstance(age_raw, float) and pd.isna(age_raw)) else float(age_raw)

        records.append({
            "subject_id": int(row.subject_id),
            "hadm_id": hadm_id,
            "stay_ids": stay_ids_by_hadm.get(hadm_id, []),
            "admittime": row.admittime,
            "dischtime": row.dischtime,
            "gender": clean_str(getattr(row, "gender", None)),
            "age_years": age_val,
            "age_group": row.age_group,
            "diagnosis_tokens": dx_toks,
            "procedure_tokens": pr_toks,
            "medication_tokens": md_toks,
            "sequence_tokens": seq,
            "n_diagnoses": len(dx_toks),
            "n_procedures": len(pr_toks),
            "n_medications": len(md_toks),
            "sequence_length": len(seq),
        })

    result = pd.DataFrame(records)
    if result.empty:
        log.warning("No non-empty admission records produced.")
        return result

    result = result.sort_values(["subject_id", "hadm_id"], kind="mergesort").reset_index(drop=True)
    log.info("Assembled %d non-empty admission records", len(result))
    return result


def _compute_age_iv(
    anchor_age: Any,
    anchor_year: Any,
    admittime: pd.Timestamp,
) -> float | None:
    """Approximate age at admission from MIMIC-IV anchor variables.

    MIMIC-IV provides ``anchor_age`` (age at ``anchor_year``) instead of
    a date of birth.  Age ≈ anchor_age + (admit_year − anchor_year).
    """
    if pd.isna(anchor_age) or pd.isna(anchor_year) or pd.isna(admittime):
        return None
    try:
        age = float(anchor_age) + (admittime.year - float(anchor_year))
    except (TypeError, ValueError):
        return None
    if age < 0:
        return None
    return round(age, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 10 – MIMIC-IV sequence extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, type=Path,
                    help="Root MIMIC-IV directory (contains hosp/ and icu/)")
    p.add_argument("--output-path", required=True, type=Path,
                    help="Output Parquet path")
    p.add_argument("--stats-path", required=True, type=Path,
                    help="Output stats JSON path")
    p.add_argument("--max-records", type=int, default=None,
                    help="Limit admissions (for smoke tests)")
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

    df = build_mimiciv_sequences(args.input_dir, max_records=args.max_records)

    if df.empty:
        log.error("Empty result – nothing to save.")
        sys.exit(1)

    save_parquet(df, args.output_path)
    stats = build_stats(
        df,
        id_col="hadm_id",
        icu_list_col="stay_ids",
        n_treatments_col=None,
    )
    save_stats(stats, args.stats_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
