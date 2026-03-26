from __future__ import annotations

import ast
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

SEED = 42
PRIMARY_DATASET = "mimiciii"


def find_input_file() -> Path:
    preferred = [
        PROCESSED / "mimiciii_sequences.pkl",
        PROCESSED / "mimiciii_sequences.parquet",
        PROCESSED / "mimiciii_sequences.csv",
        PROCESSED / "mimiciii_sequences_mapped.pkl",
        PROCESSED / "mimiciii_mapped_sequences.pkl",
    ]
    for path in preferred:
        if path.exists():
            return path

    patterns = [
        "*mimiciii*sequence*.pkl",
        "*mimiciii*sequence*.parquet",
        "*mimiciii*sequence*.csv",
        "*mimiciii*.pkl",
        "*mimiciii*.parquet",
        "*mimiciii*.csv",
    ]
    for pattern in patterns:
        matches = sorted(PROCESSED.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No processed MIMIC-III sequence file found under: {PROCESSED}"
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in existing:
            return existing[name.lower()]
    return None


def normalize_sequence(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist() if str(x).strip()]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
        if "|" in s:
            return [tok.strip() for tok in s.split("|") if tok.strip()]
        if "," in s and "[" not in s and "]" not in s:
            return [tok.strip() for tok in s.split(",") if tok.strip()]
        return [tok.strip() for tok in s.split() if tok.strip()]

    return [str(value)]


def code_category(code: str) -> str:
    s = str(code).upper()

    if s.startswith(("MED_", "RXNORM_", "NDC_")):
        return "medication"

    if s.startswith(("PROC_", "ICD9PROC_", "ICD10PROC_")):
        return "procedure"

    if s.startswith(("ICD9_", "ICD10_", "SNOMED_", "DX_", "DIAG_")):
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


def main():
    input_path = find_input_file()
    print(f"[INFO] Using input file: {input_path}")

    df = load_dataframe(input_path)
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    code_col = pick_column(df, ["codes", "mapped_codes", "sequence", "sequences", "events"])
    if code_col is None:
        raise KeyError(
            f"Could not find a sequence column in {list(df.columns)}. "
            "Expected one of: codes, mapped_codes, sequence, sequences, events"
        )

    patient_col = pick_column(df, ["SUBJECT_ID", "subject_id", "patient_id"])
    admission_col = pick_column(df, ["HADM_ID", "hadm_id", "admission_id", "stay_id"])
    gender_col = pick_column(df, ["gender", "GENDER", "sex", "SEX"])
    age_group_col = pick_column(df, ["age_group", "AGE_GROUP", "age_bucket"])

    df = df.copy()
    df[code_col] = df[code_col].apply(normalize_sequence)
    df["sequence_length"] = df[code_col].apply(len)

    before = len(df)
    df = df[df["sequence_length"] > 0].reset_index(drop=True)
    after = len(df)
    print(f"[INFO] Dropped empty sequences: {before - after}")

    if patient_col is not None:
        split_unit_col = patient_col
        split_level = "patient"
    elif admission_col is not None:
        split_unit_col = admission_col
        split_level = "admission"
    else:
        raise KeyError("Neither patient-level nor admission-level ID column found.")

    groups = sorted(df[split_unit_col].astype(str).unique().tolist())
    rng = random.Random(SEED)
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, math.floor(n * 0.70))
    n_val = max(1, math.floor(n * 0.10))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough groups to build a 70/10/20 split safely.")

    train_ids = set(groups[:n_train])
    val_ids = set(groups[n_train:n_train + n_val])
    test_ids = set(groups[n_train + n_val:])

    def assign_split(v: str) -> str:
        if v in train_ids:
            return "train"
        if v in val_ids:
            return "val"
        return "test"

    df["split"] = df[split_unit_col].astype(str).map(assign_split)

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
        "input_file": str(input_path),
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

    full_out = PROCESSED / "mimiciii_experiment_dataset.pkl"
    train_out = PROCESSED / "mimiciii_train.pkl"
    val_out = PROCESSED / "mimiciii_val.pkl"
    test_out = PROCESSED / "mimiciii_test.pkl"
    ids_out = PROCESSED / "mimiciii_split_ids.json"
    manifest_out = PROCESSED / "mimiciii_split_manifest.json"
    summary_out = PROCESSED / "mimiciii_day11_summary.json"
    table_out = PROCESSED / "mimiciii_day11_split_stats.csv"
    report_out = PROCESSED / "mimiciii_day11_summary.md"

    df.to_pickle(full_out)
    df[df["split"] == "train"].to_pickle(train_out)
    df[df["split"] == "val"].to_pickle(val_out)
    df[df["split"] == "test"].to_pickle(test_out)

    split_ids_payload = {
        "split_level": split_level,
        "split_unit_column": split_unit_col,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }
    ids_out.write_text(json.dumps(json_ready(split_ids_payload), indent=2), encoding="utf-8")
    manifest_out.write_text(json.dumps(json_ready(manifest), indent=2), encoding="utf-8")
    summary_out.write_text(
        json.dumps(json_ready(manifest["summaries"]), indent=2),
        encoding="utf-8",
    )

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
        f"- Input file: `{input_path.name}`",
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

    print("[DONE] Outputs written:")
    for p in [full_out, train_out, val_out, test_out, ids_out, manifest_out, summary_out, table_out, report_out]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
