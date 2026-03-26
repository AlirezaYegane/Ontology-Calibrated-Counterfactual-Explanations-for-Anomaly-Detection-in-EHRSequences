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
PRIMARY_DATASET = "mimiciii"


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

    cur = Path(__file__).resolve()
    for p in [cur.parent, *cur.parents]:
        if (p / ".git").exists():
            return p
    return Path(__file__).resolve().parents[1]


def looks_like_mimic3_root(path: Path) -> bool:
    required = [
        "ADMISSIONS.csv.gz",
        "PATIENTS.csv.gz",
        "DIAGNOSES_ICD.csv.gz",
        "PROCEDURES_ICD.csv.gz",
        "PRESCRIPTIONS.csv.gz",
    ]
    return path.exists() and all((path / name).exists() for name in required)


def find_mimic3_root(project_root: Path) -> Path:
    candidates = []

    env_root = os.environ.get("MIMIC3_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            project_root / "_local" / "Datasets" / "mimic3-carevue",
            project_root / "data" / "mimiciii",
            project_root / "mimiciii",
        ]
    )

    for c in candidates:
        if looks_like_mimic3_root(c):
            return c

    search_roots = [
        project_root / "_local" / "Datasets",
        project_root,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("ADMISSIONS.csv.gz"):
            cand = p.parent
            if looks_like_mimic3_root(cand):
                return cand

    raise FileNotFoundError(
        "Could not locate a valid MIMIC-III root. "
        "Set MIMIC3_ROOT to the folder containing ADMISSIONS.csv.gz, PATIENTS.csv.gz, "
        "DIAGNOSES_ICD.csv.gz, PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz."
    )


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


def build_sequences_from_raw(mimic_root: Path) -> pd.DataFrame:
    admissions = pd.read_csv(
        mimic_root / "ADMISSIONS.csv.gz",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
        compression="gzip",
        low_memory=False,
    )
    patients = pd.read_csv(
        mimic_root / "PATIENTS.csv.gz",
        usecols=["SUBJECT_ID", "GENDER", "DOB"],
        compression="gzip",
        low_memory=False,
    )

    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")

    base = admissions.merge(patients, on="SUBJECT_ID", how="left")
    age_years = (base["ADMITTIME"] - base["DOB"]).dt.days / 365.2425
    age_years = age_years.clip(lower=0)
    age_years = age_years.where(age_years <= 120, 90)
    base["age_years"] = age_years
    base["age_group"] = base["age_years"].apply(age_bucket)

    dx = pd.read_csv(
        mimic_root / "DIAGNOSES_ICD.csv.gz",
        usecols=["HADM_ID", "SEQ_NUM", "ICD9_CODE"],
        compression="gzip",
        low_memory=False,
    )
    dx = dx.dropna(subset=["HADM_ID", "ICD9_CODE"]).copy()
    dx["HADM_ID"] = dx["HADM_ID"].astype("int64")
    dx["SEQ_NUM"] = pd.to_numeric(dx["SEQ_NUM"], errors="coerce").fillna(999999)
    dx["token"] = "DX_" + dx["ICD9_CODE"].map(clean_token)
    dx = dx.sort_values(["HADM_ID", "SEQ_NUM"])
    dx_map = dx.groupby("HADM_ID")["token"].apply(list).to_dict()

    proc = pd.read_csv(
        mimic_root / "PROCEDURES_ICD.csv.gz",
        usecols=["HADM_ID", "SEQ_NUM", "ICD9_CODE"],
        compression="gzip",
        low_memory=False,
    )
    proc = proc.dropna(subset=["HADM_ID", "ICD9_CODE"]).copy()
    proc["HADM_ID"] = proc["HADM_ID"].astype("int64")
    proc["SEQ_NUM"] = pd.to_numeric(proc["SEQ_NUM"], errors="coerce").fillna(999999)
    proc["token"] = "PROC_" + proc["ICD9_CODE"].map(clean_token)
    proc = proc.sort_values(["HADM_ID", "SEQ_NUM"])
    proc_map = proc.groupby("HADM_ID")["token"].apply(list).to_dict()

    med_map: dict[int, list[str]] = defaultdict(list)
    rx_path = mimic_root / "PRESCRIPTIONS.csv.gz"
    for chunk in pd.read_csv(
        rx_path,
        usecols=["HADM_ID", "STARTDATE", "DRUG"],
        compression="gzip",
        low_memory=False,
        chunksize=250000,
    ):
        chunk = chunk.dropna(subset=["HADM_ID", "DRUG"]).copy()
        if chunk.empty:
            continue
        chunk["HADM_ID"] = chunk["HADM_ID"].astype("int64")
        chunk["STARTDATE"] = pd.to_datetime(chunk["STARTDATE"], errors="coerce")
        chunk["token"] = "MED_" + chunk["DRUG"].map(clean_token)
        chunk = chunk.sort_values(["HADM_ID", "STARTDATE"])
        grouped = chunk.groupby("HADM_ID")["token"].apply(list)
        for hadm_id, toks in grouped.items():
            med_map[int(hadm_id)].extend(toks)

    hadm_ids = sorted(
        set(base["HADM_ID"].astype("int64").tolist())
        | set(dx_map.keys())
        | set(proc_map.keys())
        | set(med_map.keys())
    )

    base_idx = base.set_index("HADM_ID")

    rows = []
    for hadm_id in hadm_ids:
        subject_id = None
        gender = None
        age_grp = "UNKNOWN"
        if hadm_id in base_idx.index:
            row0 = base_idx.loc[hadm_id]
            if isinstance(row0, pd.DataFrame):
                row0 = row0.iloc[0]
            subject_id = int(row0["SUBJECT_ID"]) if not pd.isna(row0["SUBJECT_ID"]) else None
            gender = None if pd.isna(row0["GENDER"]) else str(row0["GENDER"])
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
        raise ValueError("No sequences were constructed from raw MIMIC-III tables.")

    out = pd.DataFrame(rows)
    out["sequence_length"] = out["codes"].apply(len)
    out = out[out["sequence_length"] > 0].reset_index(drop=True)
    return out


def maybe_load_existing_sequences(processed_dir: Path) -> pd.DataFrame | None:
    candidates = [
        processed_dir / "mimiciii_sequences.pkl",
        processed_dir / "mimiciii_sequences.parquet",
        processed_dir / "mimiciii_sequences.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".pkl":
                return pd.read_pickle(p)
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.suffix == ".csv":
                return pd.read_csv(p)
    return None


def main():
    project_root = find_project_root()
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Processed dir: {processed_dir}")

    df = maybe_load_existing_sequences(processed_dir)
    input_desc = None

    if df is None:
        mimic_root = find_mimic3_root(project_root)
        print(f"[INFO] MIMIC-III root: {mimic_root}")
        df = build_sequences_from_raw(mimic_root)
        input_desc = str(mimic_root)
        df.to_pickle(processed_dir / "mimiciii_sequences.pkl")
        print(f"[INFO] Saved base sequence file: {processed_dir / 'mimiciii_sequences.pkl'}")
    else:
        input_desc = "existing_processed_sequences"

    code_col = "codes"
    patient_col = "subject_id" if "subject_id" in df.columns else None
    admission_col = "hadm_id" if "hadm_id" in df.columns else None
    gender_col = "gender" if "gender" in df.columns else None
    age_group_col = "age_group" if "age_group" in df.columns else None

    if patient_col is not None:
        split_unit_col = patient_col
        split_level = "patient"
    elif admission_col is not None:
        split_unit_col = admission_col
        split_level = "admission"
    else:
        raise KeyError("No patient/admission identifier column available for splitting.")

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
        df,
        code_col=code_col,
        patient_col=patient_col,
        admission_col=admission_col,
        gender_col=gender_col,
        age_group_col=age_group_col,
        split_name="overall",
    )
    train_summary = build_summary(
        df[df["split"] == "train"],
        code_col=code_col,
        patient_col=patient_col,
        admission_col=admission_col,
        gender_col=gender_col,
        age_group_col=age_group_col,
        split_name="train",
    )
    val_summary = build_summary(
        df[df["split"] == "val"],
        code_col=code_col,
        patient_col=patient_col,
        admission_col=admission_col,
        gender_col=gender_col,
        age_group_col=age_group_col,
        split_name="val",
    )
    test_summary = build_summary(
        df[df["split"] == "test"],
        code_col=code_col,
        patient_col=patient_col,
        admission_col=admission_col,
        gender_col=gender_col,
        age_group_col=age_group_col,
        split_name="test",
    )

    manifest = {
        "primary_dataset": PRIMARY_DATASET,
        "input_source": input_desc,
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

    full_out = processed_dir / "mimiciii_experiment_dataset.pkl"
    train_out = processed_dir / "mimiciii_train.pkl"
    val_out = processed_dir / "mimiciii_val.pkl"
    test_out = processed_dir / "mimiciii_test.pkl"
    ids_out = processed_dir / "mimiciii_split_ids.json"
    manifest_out = processed_dir / "mimiciii_split_manifest.json"
    summary_out = processed_dir / "mimiciii_day11_summary.json"
    table_out = processed_dir / "mimiciii_day11_split_stats.csv"
    report_out = processed_dir / "mimiciii_day11_summary.md"

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
        f"- Primary dataset: **{PRIMARY_DATASET}**",
        f"- Input source: `{input_desc}`",
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
    for p in [full_out, train_out, val_out, test_out, ids_out, manifest_out, summary_out, table_out, report_out]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
