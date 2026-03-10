from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def find_project_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return Path(__file__).resolve().parents[1]


def get_data_root(project_root: Path) -> Path:
    env_root = os.environ.get("HOSPITAL_DATA_ROOT") or os.environ.get("MIMIC3_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p

    candidates = [
        project_root / "_local" / "Datasets" / "physionet.org" / "files" / "mimiciv" / "3.1" / "hosp",
        project_root / "_local" / "Datasets" / "mimic3-carevue",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("No hospital dataset root found. Set HOSPITAL_DATA_ROOT.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def clean_token(x: object) -> str:
    s = "" if x is None else str(x).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "UNK"


def age_bucket(age: float | None) -> str:
    if age is None or pd.isna(age):
        return "UNKNOWN"
    if age < 18:
        return "<18"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 65:
        return "45-64"
    if age < 80:
        return "65-79"
    return "80+"


def code_category(code: str) -> str:
    s = str(code).upper()
    if s.startswith("MED_"):
        return "medication"
    if s.startswith("PROC_"):
        return "procedure"
    if s.startswith("DX_"):
        return "diagnosis"
    return "other"


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(x) for x in obj]
    if isinstance(obj, tuple):
        return [json_ready(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)
    return normalize_columns(df)


def detect_dataset_kind(root: Path) -> str:
    patients = load_csv(root / "PATIENTS.csv.gz", nrows=5)
    diagnoses = load_csv(root / "DIAGNOSES_ICD.csv.gz", nrows=5)

    if "anchor_age" in patients.columns or "icd_version" in diagnoses.columns:
        return "mimiciv"
    return "mimiciii"


def build_sequences_from_raw(root: Path) -> tuple[str, pd.DataFrame]:
    dataset_kind = detect_dataset_kind(root)
    print(f"[INFO] Detected dataset kind: {dataset_kind}")

    admissions = load_csv(root / "ADMISSIONS.csv.gz")
    patients = load_csv(root / "PATIENTS.csv.gz")

    # admissions
    admissions = admissions.rename(
        columns={
            "subject_id": "subject_id",
            "hadm_id": "hadm_id",
            "admittime": "admittime",
        }
    )
    required_adm = {"subject_id", "hadm_id", "admittime"}
    missing_adm = required_adm - set(admissions.columns)
    if missing_adm:
        raise KeyError(f"Admissions missing columns: {missing_adm}")

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")

    # patients: mimiciii has dob, mimiciv has anchor_age
    if "dob" in patients.columns:
        patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")
        base = admissions.merge(
            patients[[c for c in ["subject_id", "gender", "dob"] if c in patients.columns]],
            on="subject_id",
            how="left",
        )
        age_years = (base["admittime"] - base["dob"]).dt.days / 365.2425
        age_years = age_years.clip(lower=0)
        age_years = age_years.where(age_years <= 120, 90)
        base["age_years"] = age_years
    elif "anchor_age" in patients.columns:
        keep_cols = [c for c in ["subject_id", "gender", "anchor_age"] if c in patients.columns]
        base = admissions.merge(patients[keep_cols], on="subject_id", how="left")
        base["age_years"] = pd.to_numeric(base["anchor_age"], errors="coerce")
    else:
        keep_cols = [c for c in ["subject_id", "gender"] if c in patients.columns]
        base = admissions.merge(patients[keep_cols], on="subject_id", how="left")
        base["age_years"] = np.nan

    base["age_group"] = base["age_years"].apply(age_bucket)

    # diagnoses
    dx = load_csv(root / "DIAGNOSES_ICD.csv.gz")
    dx = dx.dropna(subset=[c for c in ["hadm_id"] if c in dx.columns]).copy()
    if "icd9_code" in dx.columns:
        dx["code_raw"] = dx["icd9_code"]
        dx["code_version"] = 9
    else:
        dx["code_raw"] = dx["icd_code"]
        dx["code_version"] = dx["icd_version"] if "icd_version" in dx.columns else ""

    dx = dx.dropna(subset=["code_raw"]).copy()
    dx["hadm_id"] = pd.to_numeric(dx["hadm_id"], errors="coerce")
    dx = dx.dropna(subset=["hadm_id"]).copy()
    dx["hadm_id"] = dx["hadm_id"].astype("int64")
    dx["seq_num"] = pd.to_numeric(dx["seq_num"], errors="coerce").fillna(999999)
    dx["token"] = "DX_" + dx["code_version"].astype(str).map(clean_token) + "_" + dx["code_raw"].map(clean_token)
    dx = dx.sort_values(["hadm_id", "seq_num"])
    dx_map = dx.groupby("hadm_id")["token"].apply(list).to_dict()

    # procedures
    proc = load_csv(root / "PROCEDURES_ICD.csv.gz")
    proc = proc.dropna(subset=[c for c in ["hadm_id"] if c in proc.columns]).copy()
    if "icd9_code" in proc.columns:
        proc["code_raw"] = proc["icd9_code"]
        proc["code_version"] = 9
    else:
        proc["code_raw"] = proc["icd_code"]
        proc["code_version"] = proc["icd_version"] if "icd_version" in proc.columns else ""

    proc = proc.dropna(subset=["code_raw"]).copy()
    proc["hadm_id"] = pd.to_numeric(proc["hadm_id"], errors="coerce")
    proc = proc.dropna(subset=["hadm_id"]).copy()
    proc["hadm_id"] = proc["hadm_id"].astype("int64")
    proc["seq_num"] = pd.to_numeric(proc["seq_num"], errors="coerce").fillna(999999)
    proc["token"] = "PROC_" + proc["code_version"].astype(str).map(clean_token) + "_" + proc["code_raw"].map(clean_token)
    proc = proc.sort_values(["hadm_id", "seq_num"])
    proc_map = proc.groupby("hadm_id")["token"].apply(list).to_dict()

    # prescriptions
    med_map: dict[int, list[str]] = defaultdict(list)
    rx_path = root / "PRESCRIPTIONS.csv.gz"
    time_col = None

    rx_head = load_csv(rx_path, nrows=2)
    if "startdate" in rx_head.columns:
        time_col = "startdate"
    elif "starttime" in rx_head.columns:
        time_col = "starttime"

    usecols = ["hadm_id", "drug"]
    if time_col:
        usecols.append(time_col)

    for chunk in pd.read_csv(
        rx_path,
        usecols=usecols,
        compression="gzip",
        low_memory=False,
        chunksize=250000,
    ):
        chunk = normalize_columns(chunk)
        chunk = chunk.dropna(subset=["hadm_id", "drug"]).copy()
        if chunk.empty:
            continue

        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce")
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")

        if time_col and time_col in chunk.columns:
            chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
            chunk = chunk.sort_values(["hadm_id", time_col])
        else:
            chunk = chunk.sort_values(["hadm_id"])

        chunk["token"] = "MED_" + chunk["drug"].map(clean_token)
        grouped = chunk.groupby("hadm_id")["token"].apply(list)
        for hadm_id, toks in grouped.items():
            med_map[int(hadm_id)].extend(toks)

    hadm_ids = sorted(
        set(base["hadm_id"].dropna().astype("int64").tolist())
        | set(dx_map.keys())
        | set(proc_map.keys())
        | set(med_map.keys())
    )

    base_idx = base.set_index("hadm_id")

    rows = []
    for hadm_id in hadm_ids:
        subject_id = None
        gender = None
        age_grp = "UNKNOWN"

        if hadm_id in base_idx.index:
            row0 = base_idx.loc[hadm_id]
            if isinstance(row0, pd.DataFrame):
                row0 = row0.iloc[0]
            if "subject_id" in row0.index and not pd.isna(row0["subject_id"]):
                subject_id = int(row0["subject_id"])
            if "gender" in row0.index and not pd.isna(row0["gender"]):
                gender = str(row0["gender"])
            if "age_group" in row0.index and not pd.isna(row0["age_group"]):
                age_grp = str(row0["age_group"])

        codes = dx_map.get(hadm_id, []) + proc_map.get(hadm_id, []) + med_map.get(hadm_id, [])
        if not codes:
            continue

        rows.append(
            {
                "subject_id": subject_id,
                "hadm_id": int(hadm_id),
                "gender": gender,
                "age_group": age_grp,
                "codes": codes,
            }
        )

    if not rows:
        raise ValueError("No sequences were constructed from raw tables.")

    df = pd.DataFrame(rows)
    df["sequence_length"] = df["codes"].apply(len)
    df = df[df["sequence_length"] > 0].reset_index(drop=True)
    return dataset_kind, df


def build_summary(
    df: pd.DataFrame,
    *,
    code_col: str,
    patient_col: str | None,
    admission_col: str | None,
    gender_col: str | None,
    age_group_col: str | None,
    split_name: str,
) -> dict:
    seqs = df[code_col]
    lengths = seqs.apply(len)

    per_record_counts = {
        "diagnosis": [],
        "procedure": [],
        "medication": [],
        "other": [],
    }
    global_counts = Counter()

    for seq in seqs:
        c = Counter(code_category(code) for code in seq)
        global_counts.update(c)
        for key in per_record_counts:
            per_record_counts[key].append(int(c.get(key, 0)))

    summary = {
        "split": split_name,
        "num_records": int(len(df)),
        "avg_sequence_length": float(lengths.mean()) if len(lengths) else 0.0,
        "median_sequence_length": float(lengths.median()) if len(lengths) else 0.0,
        "p95_sequence_length": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
        "avg_diagnosis_codes_per_record": float(np.mean(per_record_counts["diagnosis"])) if len(df) else 0.0,
        "avg_procedure_codes_per_record": float(np.mean(per_record_counts["procedure"])) if len(df) else 0.0,
        "avg_medication_codes_per_record": float(np.mean(per_record_counts["medication"])) if len(df) else 0.0,
        "avg_other_codes_per_record": float(np.mean(per_record_counts["other"])) if len(df) else 0.0,
        "global_code_type_counts": dict(global_counts),
    }

    if patient_col is not None:
        summary["num_unique_patients"] = int(df[patient_col].astype(str).nunique())
    if admission_col is not None:
        summary["num_unique_admissions"] = int(df[admission_col].astype(str).nunique())
    if gender_col is not None:
        summary["gender_distribution"] = (
            df[gender_col].fillna("UNKNOWN").astype(str).value_counts(dropna=False).to_dict()
        )
    if age_group_col is not None:
        summary["age_group_distribution"] = (
            df[age_group_col].fillna("UNKNOWN").astype(str).value_counts(dropna=False).to_dict()
        )

    return summary


def main():
    project_root = find_project_root()
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    root = get_data_root(project_root)
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Data root: {root}")
    print(f"[INFO] Processed dir: {processed_dir}")

    dataset_kind, df = build_sequences_from_raw(root)

    prefix = dataset_kind
    base_sequences_out = processed_dir / f"{prefix}_sequences.pkl"
    full_out = processed_dir / f"{prefix}_experiment_dataset.pkl"
    train_out = processed_dir / f"{prefix}_train.pkl"
    val_out = processed_dir / f"{prefix}_val.pkl"
    test_out = processed_dir / f"{prefix}_test.pkl"
    ids_out = processed_dir / f"{prefix}_split_ids.json"
    manifest_out = processed_dir / f"{prefix}_split_manifest.json"
    summary_out = processed_dir / f"{prefix}_day11_summary.json"
    table_out = processed_dir / f"{prefix}_day11_split_stats.csv"
    report_out = processed_dir / f"{prefix}_day11_summary.md"

    df.to_pickle(base_sequences_out)

    code_col = "codes"
    patient_col = "subject_id"
    admission_col = "hadm_id"
    gender_col = "gender" if "gender" in df.columns else None
    age_group_col = "age_group" if "age_group" in df.columns else None

    split_unit_col = patient_col if patient_col in df.columns else admission_col
    split_level = "patient" if split_unit_col == patient_col else "admission"

    groups = sorted(df[split_unit_col].dropna().astype(str).unique().tolist())
    rng = random.Random(SEED)
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, math.floor(n * 0.70))
    n_val = max(1, math.floor(n * 0.10))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough groups to build train/val/test split.")

    train_ids = set(groups[:n_train])
    val_ids = set(groups[n_train:n_train + n_val])
    test_ids = set(groups[n_train + n_val:])

    def assign_split(v: object) -> str:
        s = str(v)
        if s in train_ids:
            return "train"
        if s in val_ids:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df[split_unit_col].map(assign_split)

    overall_summary = build_summary(
        df, code_col=code_col, patient_col=patient_col, admission_col=admission_col,
        gender_col=gender_col, age_group_col=age_group_col, split_name="overall"
    )
    train_summary = build_summary(
        df[df["split"] == "train"], code_col=code_col, patient_col=patient_col, admission_col=admission_col,
        gender_col=gender_col, age_group_col=age_group_col, split_name="train"
    )
    val_summary = build_summary(
        df[df["split"] == "val"], code_col=code_col, patient_col=patient_col, admission_col=admission_col,
        gender_col=gender_col, age_group_col=age_group_col, split_name="val"
    )
    test_summary = build_summary(
        df[df["split"] == "test"], code_col=code_col, patient_col=patient_col, admission_col=admission_col,
        gender_col=gender_col, age_group_col=age_group_col, split_name="test"
    )

    manifest = {
        "dataset_kind": dataset_kind,
        "input_root": str(root),
        "split_level": split_level,
        "split_unit_column": split_unit_col,
        "random_seed": SEED,
        "ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
        "split_counts": {
            "train_groups": len(train_ids),
            "val_groups": len(val_ids),
            "test_groups": len(test_ids),
        },
        "summaries": {
            "overall": overall_summary,
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        },
    }

    df.to_pickle(full_out)
    df[df["split"] == "train"].to_pickle(train_out)
    df[df["split"] == "val"].to_pickle(val_out)
    df[df["split"] == "test"].to_pickle(test_out)

    ids_payload = {
        "split_level": split_level,
        "split_unit_column": split_unit_col,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }
    ids_out.write_text(json.dumps(json_ready(ids_payload), indent=2), encoding="utf-8")
    manifest_out.write_text(json.dumps(json_ready(manifest), indent=2), encoding="utf-8")
    summary_out.write_text(json.dumps(json_ready(manifest["summaries"]), indent=2), encoding="utf-8")

    rows = []
    for name, summary in manifest["summaries"].items():
        rows.append(
            {
                "split": name,
                "num_records": summary.get("num_records", 0),
                "num_unique_patients": summary.get("num_unique_patients"),
                "num_unique_admissions": summary.get("num_unique_admissions"),
                "avg_sequence_length": summary.get("avg_sequence_length", 0.0),
                "median_sequence_length": summary.get("median_sequence_length", 0.0),
                "p95_sequence_length": summary.get("p95_sequence_length", 0.0),
                "avg_diagnosis_codes_per_record": summary.get("avg_diagnosis_codes_per_record", 0.0),
                "avg_procedure_codes_per_record": summary.get("avg_procedure_codes_per_record", 0.0),
                "avg_medication_codes_per_record": summary.get("avg_medication_codes_per_record", 0.0),
                "avg_other_codes_per_record": summary.get("avg_other_codes_per_record", 0.0),
            }
        )
    pd.DataFrame(rows).to_csv(table_out, index=False)

    report_lines = [
        "# Day 11 Dataset Integration Summary",
        "",
        f"- Dataset kind: **{dataset_kind}**",
        f"- Input root: `{root}`",
        f"- Split level: **{split_level}** (`{split_unit_col}`)",
        f"- Random seed: **{SEED}**",
        "",
        "## Split overview",
        "",
    ]
    for name in ["overall", "train", "val", "test"]:
        s = manifest["summaries"][name]
        report_lines.extend(
            [
                f"### {name}",
                f"- records: {s.get('num_records', 0)}",
                f"- unique patients: {s.get('num_unique_patients', 'N/A')}",
                f"- unique admissions: {s.get('num_unique_admissions', 'N/A')}",
                f"- avg seq len: {s.get('avg_sequence_length', 0.0):.2f}",
                f"- median seq len: {s.get('median_sequence_length', 0.0):.2f}",
                f"- p95 seq len: {s.get('p95_sequence_length', 0.0):.2f}",
                f"- avg diagnosis / procedure / medication per record: "
                f"{s.get('avg_diagnosis_codes_per_record', 0.0):.2f} / "
                f"{s.get('avg_procedure_codes_per_record', 0.0):.2f} / "
                f"{s.get('avg_medication_codes_per_record', 0.0):.2f}",
                "",
            ]
        )
    report_out.write_text("\n".join(report_lines), encoding="utf-8")

    print("[DONE] Wrote:")
    for p in [base_sequences_out, full_out, train_out, val_out, test_out, ids_out, manifest_out, summary_out, table_out, report_out]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
