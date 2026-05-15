from __future__ import annotations

import argparse
import ast
import json
import random
import re
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEQ_COL_CANDIDATES = [
    "sequence_tokens",
    "codes",
    "sequence",
    "tokens",
    "event_codes",
    "concepts",
]

BAD_VALUES = {"", "nan", "none", "null", "na", "n/a"}


CHEMO_MEDS = {
    "OXALIPLATIN",
    "CISPLATIN",
    "CARBOPLATIN",
    "DOXORUBICIN",
    "CYCLOPHOSPHAMIDE",
    "FLUOROURACIL",
    "PACLITAXEL",
    "DOCETAXEL",
    "ETOPOSIDE",
    "VINCRISTINE",
    "IRINOTECAN",
    "METHOTREXATE",
    "GEMCITABINE",
    "PEMETREXED",
    "LEUCOVORIN",
}

OB_MEDS = {
    "OXYTOCIN",
    "DINOPROSTONE",
    "MISOPROSTOL",
    "METHYLERGONOVINE",
    "CARBOPROST",
}

DIABETES_MEDS = {
    "INSULIN",
    "GLUCAGON",
    "GLUCOSE_GEL",
    "DEXTROSE_50",
}

ANTICOAG_MEDS = {
    "HEPARIN",
    "ENOXAPARIN",
    "WARFARIN",
    "APIXABAN",
    "RIVAROXABAN",
    "DABIGATRAN",
    "CLOPIDOGREL",
    "TICAGRELOR",
    "TIROFIBAN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="data/processed/mimiciv_val_detector_supervised.pkl")
    parser.add_argument("--detector_scores_csv", default="outputs/detector_eval/day20_supervised/run_luxury/scores.csv")
    parser.add_argument("--out_dir", default="artifacts/day35_6")
    parser.add_argument("--calibration_fraction", type=float, default=0.50)
    parser.add_argument("--grid_step", type=float, default=0.05)
    parser.add_argument("--max_sgen_weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            token = str(item).strip()
            if token.lower() not in BAD_VALUES:
                out.append(token)
        return out

    if isinstance(value, str):
        text = value.strip()
        if text.lower() in BAD_VALUES:
            return []

        if text.startswith("[") and text.endswith("]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                    if isinstance(parsed, list):
                        return normalize_tokens(parsed)
                except Exception:
                    pass

        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]

        return [part.strip() for part in text.split() if part.strip()]

    if value is None:
        return []

    token = str(value).strip()
    return [token] if token and token.lower() not in BAD_VALUES else []


def infer_sequence_column(df: pd.DataFrame) -> str:
    for col in SEQ_COL_CANDIDATES:
        if col in df.columns:
            return col

    for col in df.columns:
        sample = df[col].dropna().head(20)
        if len(sample) and any(isinstance(x, (list, tuple)) for x in sample):
            return col

    raise ValueError(f"Could not infer sequence column. Columns={list(df.columns)}")


def prepare_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(df)}")

    df = df.copy()
    seq_col = infer_sequence_column(df)
    df["sequence_tokens"] = df[seq_col].apply(normalize_tokens)
    df = df[df["sequence_tokens"].map(len) > 0].copy()

    if "label" not in df.columns:
        if "is_synthetic_anomaly" in df.columns:
            df["label"] = df["is_synthetic_anomaly"].astype(int)
        elif "source" in df.columns:
            df["label"] = df["source"].astype(str).str.contains("anomaly", case=False, na=False).astype(int)
        else:
            raise ValueError("Could not infer label.")

    df["label"] = df["label"].astype(int)

    if "source" not in df.columns:
        df["source"] = np.where(df["label"] == 1, "synthetic_anomaly", "normal")

    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = ""

    df["source"] = df["source"].fillna("").astype(str)
    df["anomaly_type"] = df["anomaly_type"].fillna("").astype(str)

    return df.reset_index(drop=True)


def dx9_code(token: str) -> str | None:
    up = token.upper()
    if up.startswith("DX_9_"):
        return up.replace("DX_9_", "", 1)
    return None


def dx10_code(token: str) -> str | None:
    up = token.upper()
    if up.startswith("DX_10_"):
        return up.replace("DX_10_", "", 1)
    return None


def proc9_code(token: str) -> str | None:
    up = token.upper()
    if up.startswith("PROC_9_"):
        return up.replace("PROC_9_", "", 1)
    return None


def proc10_code(token: str) -> str | None:
    up = token.upper()
    if up.startswith("PROC_10_"):
        return up.replace("PROC_10_", "", 1)
    return None


def med_name(token: str) -> str | None:
    up = token.upper()
    if up.startswith("MED_"):
        return up.replace("MED_", "", 1)
    return None


def starts_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(p) for p in prefixes)


def numeric_prefix(code: str, n: int = 3) -> int | None:
    m = re.match(r"^(\d+)", code)
    if not m:
        return None
    return int(m.group(1)[:n])


def is_pregnancy_dx(token: str) -> bool:
    c10 = dx10_code(token)
    if c10:
        return starts_any(c10, ("O", "Z3A", "Z33", "Z34", "Z36", "Z37", "Z39"))

    c9 = dx9_code(token)
    if c9:
        if starts_any(c9, ("V22", "V23", "V24", "V27", "V28")):
            return True
        num = numeric_prefix(c9, 3)
        return num is not None and 630 <= num <= 679

    return False


def is_obstetric_proc(token: str) -> bool:
    c9 = proc9_code(token)
    if c9 and starts_any(c9, ("72", "73", "74", "75")):
        return True

    c10 = proc10_code(token)
    if c10 and starts_any(c10, ("10D", "10E", "10A")):
        return True

    return False


def is_ob_med(token: str) -> bool:
    m = med_name(token)
    if not m:
        return False
    return any(name in m for name in OB_MEDS)


def is_male_specific_dx(token: str) -> bool:
    c10 = dx10_code(token)
    if c10:
        return starts_any(
            c10,
            (
                "C60",
                "C61",
                "C62",
                "C63",
                "N40",
                "N41",
                "N42",
                "N43",
                "N44",
                "N45",
                "N46",
                "N47",
                "N48",
                "N49",
                "Q53",
                "Q54",
                "Q55",
                "R86",
                "Z125",
                "Z8546",
            ),
        )

    c9 = dx9_code(token)
    if c9:
        if starts_any(c9, ("V1046",)):
            return True
        num = numeric_prefix(c9, 3)
        return num is not None and (
            num in {185}
            or 600 <= num <= 608
        )

    return False


def is_cancer_dx(token: str) -> bool:
    c10 = dx10_code(token)
    if c10:
        return starts_any(c10, ("C", "Z5111", "Z510", "Z85"))

    c9 = dx9_code(token)
    if c9:
        if starts_any(c9, ("V10", "V58")):
            return True
        num = numeric_prefix(c9, 3)
        return num is not None and 140 <= num <= 239

    return False


def is_diabetes_dx(token: str) -> bool:
    c10 = dx10_code(token)
    if c10:
        return starts_any(c10, ("E10", "E11", "E13", "E14", "O24", "R73"))

    c9 = dx9_code(token)
    if c9:
        return starts_any(c9, ("250",))

    return False


def is_ischemic_or_thrombotic_dx(token: str) -> bool:
    c10 = dx10_code(token)
    if c10:
        return starts_any(c10, ("I21", "I22", "I23", "I24", "I25", "I26", "I48", "I63", "I70", "I73", "I74", "I82", "Z7901", "Z7902"))

    c9 = dx9_code(token)
    if c9:
        return starts_any(c9, ("410", "411", "412", "413", "414", "415", "42731", "433", "434", "440", "443", "444", "453", "V5861", "V5863"))

    return False


def med_contains(token: str, names: set[str]) -> bool:
    m = med_name(token)
    if not m:
        return False
    return any(name in m for name in names)


def score_curated_sont(tokens: list[str]) -> tuple[float, list[str]]:
    pregnancy = [t for t in tokens if is_pregnancy_dx(t)]
    ob_proc = [t for t in tokens if is_obstetric_proc(t)]
    ob_med = [t for t in tokens if is_ob_med(t)]
    male_specific = [t for t in tokens if is_male_specific_dx(t)]

    cancer_dx = [t for t in tokens if is_cancer_dx(t)]
    chemo_med = [t for t in tokens if med_contains(t, CHEMO_MEDS)]

    diabetes_dx = [t for t in tokens if is_diabetes_dx(t)]
    diabetes_med = [t for t in tokens if med_contains(t, DIABETES_MEDS)]

    ischemic_dx = [t for t in tokens if is_ischemic_or_thrombotic_dx(t)]
    anticoag_med = [t for t in tokens if med_contains(t, ANTICOAG_MEDS)]

    score = 0.0
    hits: list[str] = []

    # Strong content contradiction: pregnancy-related evidence mixed with male-specific diagnosis.
    if pregnancy and male_specific:
        score += 4.0
        hits.append(f"pregnancy_male_specific_conflict::{pregnancy[0]}::{male_specific[0]}")

    # Strong-ish synthetic demographic signal: isolated pregnancy/delivery code inside non-obstetric context.
    # Normal obstetric records usually contain several OB signals, e.g., pregnancy diagnosis + delivery outcome/procedure + oxytocin.
    obstetric_context_count = len(set(pregnancy + ob_proc + ob_med))
    if pregnancy and obstetric_context_count <= 1:
        score += 2.5
        hits.append(f"isolated_pregnancy_signal::{pregnancy[0]}")

    # Missing indication: OB procedure or OB medication without pregnancy diagnosis.
    if (ob_proc or ob_med) and not pregnancy:
        score += 3.0
        trigger = (ob_proc + ob_med)[0]
        hits.append(f"obstetric_intervention_without_pregnancy_dx::{trigger}")

    # Medication-indication mismatch: chemotherapy without cancer diagnosis.
    if chemo_med and not cancer_dx:
        score += 3.0
        hits.append(f"chemotherapy_without_cancer_dx::{chemo_med[0]}")

    # Weak medication-indication signal: insulin/glucagon/dextrose pattern without diabetes diagnosis.
    # This is deliberately low weight because ICU hyperglycemia treatment can occur without coded diabetes.
    if diabetes_med and not diabetes_dx:
        score += 0.75
        hits.append(f"diabetes_med_without_diabetes_dx::{diabetes_med[0]}")

    # Weak anticoagulation signal. Low weight because prophylaxis is common.
    if anticoag_med and not ischemic_dx:
        score += 0.50
        hits.append(f"anticoag_or_antiplatelet_without_cardiovascular_dx::{anticoag_med[0]}")

    return float(score), hits


def minmax(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size == 0:
        return arr

    lo = float(np.min(arr))
    hi = float(np.max(arr))

    if np.isclose(lo, hi):
        return np.zeros_like(arr)

    return (arr - lo) / (hi - lo)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if len(thresholds) == 0:
        threshold = 0.5
        pred = (scores >= threshold).astype(int)
        return (
            threshold,
            float(precision_score(y_true, pred, zero_division=0)),
            float(recall_score(y_true, pred, zero_division=0)),
            float(f1_score(y_true, pred, zero_division=0)),
        )

    f1_values = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1],
        1e-12,
        None,
    )
    idx = int(np.argmax(f1_values))
    threshold = float(thresholds[idx])
    pred = (scores >= threshold).astype(int)

    return (
        threshold,
        float(precision_score(y_true, pred, zero_division=0)),
        float(recall_score(y_true, pred, zero_division=0)),
        float(f1_score(y_true, pred, zero_division=0)),
    )


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "threshold": 0.5,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "predicted_positive_rate": float("nan"),
        }

    threshold, precision, recall, f1 = best_f1_threshold(y_true, scores)
    pred = (scores >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(pred.mean()),
    }


def stratified_split(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    y = np.asarray(y).astype(int)

    calib_idx: list[int] = []
    eval_idx: list[int] = []

    for label in sorted(set(y.tolist())):
        idx = [i for i, value in enumerate(y) if int(value) == int(label)]
        rng.shuffle(idx)

        n_calib = int(round(len(idx) * fraction))
        calib_idx.extend(idx[:n_calib])
        eval_idx.extend(idx[n_calib:])

    return np.asarray(sorted(calib_idx), dtype=int), np.asarray(sorted(eval_idx), dtype=int)


def grid_search(
    y: np.ndarray,
    sdet: np.ndarray,
    sont: np.ndarray,
    sgen: np.ndarray,
    grid_step: float,
    max_sgen_weight: float,
) -> pd.DataFrame:
    values = np.round(np.arange(0.0, 1.0 + grid_step, grid_step), 6)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()

    for wd_raw, wo_raw, wg_raw in product(values, values, values):
        if wg_raw > max_sgen_weight:
            continue

        total = float(wd_raw + wo_raw + wg_raw)
        if total <= 0:
            continue

        wd = float(wd_raw / total)
        wo = float(wo_raw / total)
        wg = float(wg_raw / total)

        key = (round(wd, 4), round(wo, 4), round(wg, 4))
        if key in seen:
            continue
        seen.add(key)

        score = wd * sdet + wo * sont + wg * sgen
        rows.append(
            {
                "w_det": wd,
                "w_ont": wo,
                "w_gen": wg,
                **metrics(y, score),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["average_precision", "roc_auc", "f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def oracle_reference(df: pd.DataFrame) -> np.ndarray:
    severity = {
        "": 0.0,
        "normal": 0.0,
        "none": 0.0,
        "demographic_conflict": 1.0,
        "medication_mismatch": 0.80,
        "missing_diagnosis": 0.65,
        "missing_indication": 0.65,
    }
    values = []
    for raw in df["anomaly_type"].fillna("").astype(str):
        key = raw.strip().lower()
        values.append(float(severity.get(key, 0.50 if key else 0.0)))
    return minmax(np.asarray(values, dtype=float))


def flag_metrics(df: pd.DataFrame) -> dict[str, Any]:
    flagged = df[df["Sont_curated_raw"] > 0].copy()

    if len(flagged) == 0:
        return {
            "flagged_count": 0,
            "flagged_rate": 0.0,
            "flagged_precision": float("nan"),
            "flagged_recall": 0.0,
            "flagged_anomaly_type_counts": {},
            "rule_hit_counts": {},
        }

    rule_counter: dict[str, int] = {}
    for text in flagged["Sont_curated_hits"].fillna("").astype(str):
        for hit in text.split(" | "):
            if not hit:
                continue
            rule = hit.split("::")[0]
            rule_counter[rule] = rule_counter.get(rule, 0) + 1

    return {
        "flagged_count": int(len(flagged)),
        "flagged_rate": float(len(flagged) / len(df)),
        "flagged_precision": float(flagged["label"].mean()),
        "flagged_recall": float(flagged["label"].sum() / max(float(df["label"].sum()), 1.0)),
        "flagged_anomaly_type_counts": {
            str(k): int(v) for k, v in flagged["anomaly_type"].fillna("").value_counts().to_dict().items()
        },
        "rule_hit_counts": rule_counter,
    }


def render_readme(summary: dict[str, Any]) -> str:
    h = summary["heldout_evaluation"]
    b = summary["best_weights_from_calibration"]
    f = summary["heldout_flag_metrics"]

    return f"""# Day 35.6 — Curated Independent Ontology/Rule Scorer

## Status
Complete.

## Scientific purpose
Day 35.5 showed that purely data-mined co-occurrence rules were too weak. Day 35.6 introduces curated content-only clinical rules.

## No-leakage policy
Scoring uses only `sequence_tokens`.

The scorer does not use:

- `label`
- `source`
- `anomaly_type`

Those fields are used only after scoring for evaluation and breakdown.

## Curated rule families
- isolated pregnancy/delivery signal
- pregnancy + male-specific code contradiction
- obstetric intervention without pregnancy diagnosis
- chemotherapy without cancer diagnosis
- weak insulin/diabetes indication mismatch
- weak anticoagulation/cardiovascular indication mismatch

## Best calibrated weights
- `w_det`: {b["w_det"]:.4f}
- `w_ont`: {b["w_ont"]:.4f}
- `w_gen`: {b["w_gen"]:.4f}

## Held-out metrics

| Signal | ROC-AUC | AP | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Detector only | {h["Sdet_only"]["roc_auc"]:.4f} | {h["Sdet_only"]["average_precision"]:.4f} | {h["Sdet_only"]["f1"]:.4f} | {h["Sdet_only"]["precision"]:.4f} | {h["Sdet_only"]["recall"]:.4f} |
| Curated Sont only | {h["Sont_curated_only"]["roc_auc"]:.4f} | {h["Sont_curated_only"]["average_precision"]:.4f} | {h["Sont_curated_only"]["f1"]:.4f} | {h["Sont_curated_only"]["precision"]:.4f} | {h["Sont_curated_only"]["recall"]:.4f} |
| Sdet + curated Sont | {h["Scal_curated"]["roc_auc"]:.4f} | {h["Scal_curated"]["average_precision"]:.4f} | {h["Scal_curated"]["f1"]:.4f} | {h["Scal_curated"]["precision"]:.4f} | {h["Scal_curated"]["recall"]:.4f} |
| Oracle proxy reference | {h["Sont_oracle_proxy_reference"]["roc_auc"]:.4f} | {h["Sont_oracle_proxy_reference"]["average_precision"]:.4f} | {h["Sont_oracle_proxy_reference"]["f1"]:.4f} | {h["Sont_oracle_proxy_reference"]["precision"]:.4f} | {h["Sont_oracle_proxy_reference"]["recall"]:.4f} |

## Rule-flag quality on held-out split
- flagged count: {f["flagged_count"]}
- flagged precision: {f["flagged_precision"]:.4f}
- flagged recall: {f["flagged_recall"]:.4f}

## Interpretation
For publication, the main comparison is detector-only vs `Sdet + curated independent Sont`.

The oracle proxy remains only an upper-bound reference.
"""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df = prepare_frame(args.val_path)
    det_df = pd.read_csv(args.detector_scores_csv)

    if len(det_df) != len(val_df):
        raise ValueError(f"Detector scores mismatch: det={len(det_df)} val={len(val_df)}")
    if "prob_anomaly" not in det_df.columns:
        raise ValueError("Expected prob_anomaly in detector scores.")

    raw_scores: list[float] = []
    hits_list: list[str] = []

    for tokens in val_df["sequence_tokens"]:
        score, hits = score_curated_sont(tokens)
        raw_scores.append(score)
        hits_list.append(" | ".join(hits))

    result = val_df[["label", "source", "anomaly_type", "sequence_tokens"]].copy()
    result["Sdet_raw"] = det_df["prob_anomaly"].astype(float).to_numpy()
    result["Sdet"] = minmax(result["Sdet_raw"])
    result["Sont_curated_raw"] = np.asarray(raw_scores, dtype=float)
    result["Sont_curated"] = minmax(result["Sont_curated_raw"])
    result["Sont_curated_hits"] = hits_list
    result["Sgen"] = 0.0
    result["Sont_oracle_proxy_reference"] = oracle_reference(result)

    y = result["label"].to_numpy(dtype=int)
    calib_idx, eval_idx = stratified_split(y, args.calibration_fraction, args.seed)

    calib = result.iloc[calib_idx].reset_index(drop=True)
    holdout = result.iloc[eval_idx].reset_index(drop=True)

    weight_table = grid_search(
        y=calib["label"].to_numpy(dtype=int),
        sdet=calib["Sdet"].to_numpy(dtype=float),
        sont=calib["Sont_curated"].to_numpy(dtype=float),
        sgen=calib["Sgen"].to_numpy(dtype=float),
        grid_step=args.grid_step,
        max_sgen_weight=args.max_sgen_weight,
    )

    best = weight_table.iloc[0].to_dict()
    wd = float(best["w_det"])
    wo = float(best["w_ont"])
    wg = float(best["w_gen"])

    result["Scal_curated"] = wd * result["Sdet"] + wo * result["Sont_curated"] + wg * result["Sgen"]
    calib["Scal_curated"] = wd * calib["Sdet"] + wo * calib["Sont_curated"] + wg * calib["Sgen"]
    holdout["Scal_curated"] = wd * holdout["Sdet"] + wo * holdout["Sont_curated"] + wg * holdout["Sgen"]

    heldout_metrics = {
        "Sdet_only": metrics(holdout["label"].to_numpy(dtype=int), holdout["Sdet"].to_numpy(dtype=float)),
        "Sont_curated_only": metrics(holdout["label"].to_numpy(dtype=int), holdout["Sont_curated"].to_numpy(dtype=float)),
        "Scal_curated": metrics(holdout["label"].to_numpy(dtype=int), holdout["Scal_curated"].to_numpy(dtype=float)),
        "Sgen_only": metrics(holdout["label"].to_numpy(dtype=int), holdout["Sgen"].to_numpy(dtype=float)),
        "Sont_oracle_proxy_reference": metrics(
            holdout["label"].to_numpy(dtype=int),
            holdout["Sont_oracle_proxy_reference"].to_numpy(dtype=float),
        ),
    }

    calibration_metrics = {
        "Sdet_only": metrics(calib["label"].to_numpy(dtype=int), calib["Sdet"].to_numpy(dtype=float)),
        "Sont_curated_only": metrics(calib["label"].to_numpy(dtype=int), calib["Sont_curated"].to_numpy(dtype=float)),
        "Scal_curated": metrics(calib["label"].to_numpy(dtype=int), calib["Scal_curated"].to_numpy(dtype=float)),
        "Sont_oracle_proxy_reference": metrics(
            calib["label"].to_numpy(dtype=int),
            calib["Sont_oracle_proxy_reference"].to_numpy(dtype=float),
        ),
    }

    breakdown = (
        result.groupby("anomaly_type", dropna=False)
        .agg(
            count=("label", "size"),
            label_rate=("label", "mean"),
            mean_Sdet=("Sdet", "mean"),
            mean_Sont_curated=("Sont_curated", "mean"),
            mean_Scal_curated=("Scal_curated", "mean"),
            flagged_rate=("Sont_curated_raw", lambda x: float((x > 0).mean())),
        )
        .reset_index()
        .sort_values("mean_Sont_curated", ascending=False)
    )

    rule_hit_rows: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        hit_text = str(row["Sont_curated_hits"])
        for hit in hit_text.split(" | "):
            if not hit:
                continue
            rule_hit_rows.append(
                {
                    "label": int(row["label"]),
                    "anomaly_type": str(row["anomaly_type"]),
                    "rule": hit.split("::")[0],
                    "hit": hit,
                }
            )
    rule_hits = pd.DataFrame(rule_hit_rows)
    if len(rule_hits):
        rule_hit_summary = (
            rule_hits.groupby("rule")
            .agg(
                count=("label", "size"),
                precision=("label", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )
    else:
        rule_hit_summary = pd.DataFrame(columns=["rule", "count", "precision"])

    paper_table = pd.DataFrame(
        [
            {"split": "heldout_evaluation", "signal": name, **metric_row}
            for name, metric_row in heldout_metrics.items()
        ]
    )

    result.to_csv(out_dir / "val_curated_sont_scores.csv", index=False)
    weight_table.to_csv(out_dir / "day35_6_weight_search_calibration.csv", index=False)
    breakdown.to_csv(out_dir / "anomaly_type_breakdown_curated_sont.csv", index=False)
    rule_hit_summary.to_csv(out_dir / "curated_rule_hit_summary.csv", index=False)
    paper_table.to_csv(out_dir / "paper_ready_metrics.csv", index=False)

    summary = {
        "day": "35.6",
        "status": "complete",
        "scientific_goal": "Evaluate a curated content-only independent ontology/rule score without anomaly_type leakage.",
        "no_leakage_policy": {
            "used_for_scoring": ["sequence_tokens"],
            "not_used_for_scoring": ["label", "source", "anomaly_type"],
            "used_only_for_evaluation": ["label", "source", "anomaly_type"],
        },
        "val_rows": int(len(result)),
        "calibration_rows": int(len(calib)),
        "heldout_evaluation_rows": int(len(holdout)),
        "best_weights_from_calibration": {
            "w_det": wd,
            "w_ont": wo,
            "w_gen": wg,
        },
        "calibration_metrics": calibration_metrics,
        "heldout_evaluation": heldout_metrics,
        "heldout_flag_metrics": flag_metrics(holdout),
        "full_validation_flag_metrics": flag_metrics(result),
        "interpretation_for_paper": (
            "Use detector-only vs Sdet+curated-independent-Sont as the publishable comparison. "
            "Report oracle proxy only as an upper-bound reference. "
            "If global AP does not improve, report curated Sont as a high-precision/low-recall explanatory signal if flag precision is strong."
        ),
        "outputs": {
            "scores": str(out_dir / "val_curated_sont_scores.csv"),
            "weight_search": str(out_dir / "day35_6_weight_search_calibration.csv"),
            "breakdown": str(out_dir / "anomaly_type_breakdown_curated_sont.csv"),
            "rule_hit_summary": str(out_dir / "curated_rule_hit_summary.csv"),
            "paper_ready_metrics": str(out_dir / "paper_ready_metrics.csv"),
            "summary": str(out_dir / "day35_6_scientific_summary.json"),
            "readme": str(out_dir / "README.md"),
        },
    }

    save_json(out_dir / "day35_6_scientific_summary.json", summary)
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
