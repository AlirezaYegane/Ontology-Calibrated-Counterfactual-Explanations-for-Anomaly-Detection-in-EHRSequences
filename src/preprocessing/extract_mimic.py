from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_csv(base_dir: Path, stem: str) -> Path:
    """Resolve a CSV or gzipped CSV path for a MIMIC table stem."""
    for ext in (".csv.gz", ".csv"):
        path = base_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required file for table '{stem}' under: {base_dir}")


def _read_csv_any(
    base_dir: Path,
    stem: str,
    usecols: list[str] | None = None,
    parse_dates: list[str] | None = None,
    dtype: dict[str, type] | None = None,
) -> pd.DataFrame:
    """Read either .csv.gz or .csv for a MIMIC table."""
    path = _resolve_csv(base_dir, stem)
    return pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
    )


def _clean_str(value: Any) -> str | None:
    """Normalize blank / null strings to None."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _normalize_icd9(code: Any) -> str | None:
    """Conservatively normalize ICD-9 codes."""
    text = _clean_str(code)
    if text is None:
        return None
    return text.replace(" ", "").upper()


def _normalize_ndc(code: Any) -> str | None:
    """Normalize medication NDC identifiers."""
    text = _clean_str(code)
    if text is None:
        return None
    if text in {"0", "0000", "00000000000"}:
        return None
    return text


def _normalize_drug_name(name: Any) -> str | None:
    """Normalize drug strings into stable token-safe values."""
    text = _clean_str(name)
    if text is None:
        return None
    return "_".join(text.upper().split())


def _compute_age_years(dob: pd.Timestamp, admittime: pd.Timestamp) -> float | None:
    """Compute age in years from DOB and admission time."""
    if pd.isna(dob) or pd.isna(admittime):
        return None
    age = (admittime - dob).days / 365.2425
    if age < 0:
        return None
    # Practical guard for de-identification artifacts
    if age > 120:
        return 90.0
    return round(float(age), 3)


def _age_group(age: float | None) -> str:
    """Bucket age into coarse groups."""
    if age is None or pd.isna(age):
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


def _ordered_group_list(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    sort_cols: list[str],
) -> dict[int, list[str]]:
    """Group ordered tokens into lists keyed by group_col."""
    if df.empty:
        return {}

    cols = [group_col] + [c for c in sort_cols if c in df.columns] + [value_col]
    temp = df[cols].copy()
    temp = temp.dropna(subset=[group_col, value_col])

    present_sort_cols = [c for c in sort_cols if c in temp.columns]
    if present_sort_cols:
        temp = temp.sort_values([group_col] + present_sort_cols, kind="mergesort")

    grouped = temp.groupby(group_col)[value_col].apply(list).to_dict()
    return {int(k): v for k, v in grouped.items()}


def build_mimiciii_sequences(mimic_dir: Path) -> pd.DataFrame:
    """Build admission-level MIMIC-III sequences with demographics."""
    admissions = _read_csv_any(
        mimic_dir,
        "ADMISSIONS",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )

    patients = _read_csv_any(
        mimic_dir,
        "PATIENTS",
        usecols=["subject_id", "gender", "dob"],
        parse_dates=["dob"],
    )

    icustays = _read_csv_any(
        mimic_dir,
        "ICUSTAYS",
        usecols=["subject_id", "hadm_id", "icustay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )

    diagnoses = _read_csv_any(
        mimic_dir,
        "DIAGNOSES_ICD",
        usecols=["subject_id", "hadm_id", "seq_num", "icd9_code"],
        dtype={"icd9_code": str},
    )
    diagnoses["icd9_code"] = diagnoses["icd9_code"].map(_normalize_icd9)
    diagnoses = diagnoses.dropna(subset=["icd9_code"])
    diagnoses["token"] = diagnoses["icd9_code"].map(lambda x: f"DX_ICD9:{x}")

    procedures = _read_csv_any(
        mimic_dir,
        "PROCEDURES_ICD",
        usecols=["subject_id", "hadm_id", "seq_num", "icd9_code"],
        dtype={"icd9_code": str},
    )
    procedures["icd9_code"] = procedures["icd9_code"].map(_normalize_icd9)
    procedures = procedures.dropna(subset=["icd9_code"])
    procedures["token"] = procedures["icd9_code"].map(lambda x: f"PROC_ICD9:{x}")

    prescriptions = _read_csv_any(
        mimic_dir,
        "PRESCRIPTIONS",
        usecols=[
            "subject_id",
            "hadm_id",
            "icustay_id",
            "startdate",
            "enddate",
            "drug",
            "drug_name_poe",
            "drug_name_generic",
            "ndc",
        ],
        parse_dates=["startdate", "enddate"],
        dtype={"ndc": str},
    )

    prescriptions["ndc"] = prescriptions["ndc"].map(_normalize_ndc)
    prescriptions["drug_name_generic"] = prescriptions["drug_name_generic"].map(_normalize_drug_name)
    prescriptions["drug_name_poe"] = prescriptions["drug_name_poe"].map(_normalize_drug_name)
    prescriptions["drug"] = prescriptions["drug"].map(_normalize_drug_name)

    def _med_token(row: pd.Series) -> str | None:
        if pd.notna(row["ndc"]) and row["ndc"] is not None:
            return f"MED_NDC:{row['ndc']}"
        for col in ("drug_name_generic", "drug_name_poe", "drug"):
            val = row[col]
            if pd.notna(val) and val is not None:
                return f"MED_NAME:{val}"
        return None

    prescriptions["token"] = prescriptions.apply(_med_token, axis=1)
    prescriptions = prescriptions.dropna(subset=["token"])

    dx_by_hadm = _ordered_group_list(diagnoses, "hadm_id", "token", ["seq_num"])
    proc_by_hadm = _ordered_group_list(procedures, "hadm_id", "token", ["seq_num"])
    med_by_hadm = _ordered_group_list(prescriptions, "hadm_id", "token", ["startdate", "enddate"])

    icu_ids_by_hadm = (
        icustays.sort_values(["hadm_id", "intime"], kind="mergesort")
        .groupby("hadm_id")["icustay_id"]
        .apply(lambda s: [int(x) for x in s.dropna().tolist()])
        .to_dict()
    )
    icu_ids_by_hadm = {int(k): v for k, v in icu_ids_by_hadm.items()}

    base = admissions.merge(patients, on="subject_id", how="left")
    base["age_years"] = [
        _compute_age_years(dob, adm)
        for dob, adm in zip(base["dob"], base["admittime"])
    ]
    base["age_group"] = base["age_years"].map(_age_group)

    records: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        hadm_id = int(row.hadm_id)

        diagnosis_tokens = dx_by_hadm.get(hadm_id, [])
        procedure_tokens = proc_by_hadm.get(hadm_id, [])
        medication_tokens = med_by_hadm.get(hadm_id, [])
        icustay_ids = icu_ids_by_hadm.get(hadm_id, [])

        sequence_tokens = diagnosis_tokens + procedure_tokens + medication_tokens
        if not sequence_tokens:
            continue

        records.append(
            {
                "subject_id": int(row.subject_id),
                "hadm_id": hadm_id,
                "icustay_ids": icustay_ids,
                "admittime": row.admittime,
                "dischtime": row.dischtime,
                "gender": _clean_str(row.gender),
                "age_years": row.age_years,
                "age_group": row.age_group,
                "diagnosis_tokens": diagnosis_tokens,
                "procedure_tokens": procedure_tokens,
                "medication_tokens": medication_tokens,
                "sequence_tokens": sequence_tokens,
                "n_diagnoses": len(diagnosis_tokens),
                "n_procedures": len(procedure_tokens),
                "n_medications": len(medication_tokens),
                "sequence_length": len(sequence_tokens),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result = result.sort_values(["subject_id", "hadm_id"], kind="mergesort").reset_index(drop=True)
    return result


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build summary statistics for extracted sequences."""
    if df.empty:
        return {
            "n_records": 0,
            "n_subjects": 0,
            "n_hadm": 0,
            "pct_with_icu_stay": 0.0,
            "sequence_length": {"min": 0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "max": 0},
            "diagnosis_count_mean": 0.0,
            "procedure_count_mean": 0.0,
            "medication_count_mean": 0.0,
            "gender_counts": {},
            "age_group_counts": {},
        }

    seq = df["sequence_length"]
    return {
        "n_records": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_hadm": int(df["hadm_id"].nunique()),
        "pct_with_icu_stay": float((df["icustay_ids"].map(len) > 0).mean()),
        "sequence_length": {
            "min": int(seq.min()),
            "p25": float(seq.quantile(0.25)),
            "median": float(seq.median()),
            "mean": float(seq.mean()),
            "p75": float(seq.quantile(0.75)),
            "max": int(seq.max()),
        },
        "diagnosis_count_mean": float(df["n_diagnoses"].mean()),
        "procedure_count_mean": float(df["n_procedures"].mean()),
        "medication_count_mean": float(df["n_medications"].mean()),
        "gender_counts": df["gender"].fillna("unknown").value_counts(dropna=False).to_dict(),
        "age_group_counts": df["age_group"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build admission-level MIMIC-III sequences.")
    parser.add_argument("--mimic-dir", type=Path, required=True, help="Path to raw MIMIC-III CSV files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for processed artifacts")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_mimiciii_sequences(args.mimic_dir)

    pkl_path = args.out_dir / "mimiciii_sequences.pkl"
    csv_path = args.out_dir / "mimiciii_sequences_flat.csv"
    summary_path = args.out_dir / "mimiciii_sequence_summary.json"

    df.to_pickle(pkl_path)

    flat = df.copy()
    for col in ("icustay_ids", "diagnosis_tokens", "procedure_tokens", "medication_tokens", "sequence_tokens"):
        if col in flat.columns:
            flat[col] = flat[col].map(json.dumps)
    flat.to_csv(csv_path, index=False)

    summary = build_summary(df)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Saved: {pkl_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
