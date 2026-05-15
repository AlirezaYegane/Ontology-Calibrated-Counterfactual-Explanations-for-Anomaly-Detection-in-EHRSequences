from __future__ import annotations

import argparse
import ast
import json
import math
import random
from collections import Counter, defaultdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/mimiciv_train_detector_supervised.pkl")
    parser.add_argument("--val_path", default="data/processed/mimiciv_val_detector_supervised.pkl")
    parser.add_argument("--detector_scores_csv", default="outputs/detector_eval/day20_supervised/run_luxury/scores.csv")
    parser.add_argument("--out_dir", default="artifacts/day35_5")

    parser.add_argument("--min_trigger_support", type=int, default=50)
    parser.add_argument("--min_expected_support", type=int, default=20)
    parser.add_argument("--min_confidence", type=float, default=0.08)
    parser.add_argument("--max_expected_per_trigger", type=int, default=25)

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
            raise ValueError("Could not infer label column.")

    df["label"] = df["label"].astype(int)

    if "source" not in df.columns:
        df["source"] = np.where(df["label"] == 1, "synthetic_anomaly", "normal")

    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = ""

    df["source"] = df["source"].fillna("").astype(str)
    df["anomaly_type"] = df["anomaly_type"].fillna("").astype(str)

    return df.reset_index(drop=True)


def token_kind(token: str) -> str:
    up = token.upper()

    if up.startswith(("SNOMED", "ICD", "DIAG", "DX")):
        return "diagnosis"

    if up.startswith(("RXNORM", "NDC", "MED", "DRUG", "RAW_DRUG")):
        return "medication"

    if up.startswith(("PROC", "CPT", "PROCEDURE")):
        return "procedure"

    return "other"


def split_by_kind(tokens: list[str]) -> dict[str, set[str]]:
    buckets: dict[str, set[str]] = {
        "diagnosis": set(),
        "medication": set(),
        "procedure": set(),
        "other": set(),
    }

    for tok in tokens:
        buckets[token_kind(tok)].add(tok)

    return buckets


def minmax(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0

    if np.isclose(lo, hi):
        return np.zeros_like(arr)

    return (arr - lo) / (hi - lo)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if len(thresholds) == 0:
        thr = 0.5
        pred = (scores >= thr).astype(int)
        return (
            thr,
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
    thr = float(thresholds[idx])
    pred = (scores >= thr).astype(int)

    return (
        thr,
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

    thr, precision, recall, f1 = best_f1_threshold(y_true, scores)
    pred = (scores >= thr).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "threshold": float(thr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(pred.mean()),
    }


def learn_indication_rules(
    train_df: pd.DataFrame,
    min_trigger_support: int,
    min_expected_support: int,
    min_confidence: float,
    max_expected_per_trigger: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Learn content-only rules from normal training records.

    The learned rule shape is:
    medication/procedure token -> expected diagnosis tokens

    Labels/anomaly_type/source are not used for validation scoring.
    """
    normal_df = train_df[train_df["label"] == 0].copy()

    trigger_counts: Counter[str] = Counter()
    trigger_diag_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for tokens in normal_df["sequence_tokens"]:
        buckets = split_by_kind(tokens)
        diagnoses = buckets["diagnosis"]
        triggers = buckets["medication"] | buckets["procedure"]

        if not diagnoses or not triggers:
            continue

        for trigger in triggers:
            trigger_counts[trigger] += 1
            trigger_diag_counts[trigger].update(diagnoses)

    rules: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for trigger, trigger_support in trigger_counts.items():
        if trigger_support < min_trigger_support:
            continue

        expected: list[dict[str, Any]] = []
        for diag, pair_support in trigger_diag_counts[trigger].most_common():
            if pair_support < min_expected_support:
                continue

            confidence = pair_support / trigger_support
            if confidence < min_confidence:
                continue

            expected.append(
                {
                    "diagnosis": diag,
                    "pair_support": int(pair_support),
                    "confidence": float(confidence),
                }
            )

            if len(expected) >= max_expected_per_trigger:
                break

        if expected:
            rules[trigger] = {
                "trigger_support": int(trigger_support),
                "trigger_kind": token_kind(trigger),
                "expected_diagnoses": expected,
            }

            rows.append(
                {
                    "trigger": trigger,
                    "trigger_kind": token_kind(trigger),
                    "trigger_support": int(trigger_support),
                    "n_expected_diagnoses": int(len(expected)),
                    "top_expected_diagnosis": expected[0]["diagnosis"],
                    "top_confidence": float(expected[0]["confidence"]),
                }
            )

    rule_table = pd.DataFrame(rows).sort_values(
        ["trigger_support", "top_confidence"],
        ascending=[False, False],
    )

    meta = {
        "normal_train_rows": int(len(normal_df)),
        "learned_rule_count": int(len(rules)),
        "min_trigger_support": min_trigger_support,
        "min_expected_support": min_expected_support,
        "min_confidence": min_confidence,
        "max_expected_per_trigger": max_expected_per_trigger,
        "rules": rules,
    }

    return meta, rule_table


def score_record(tokens: list[str], rules: dict[str, Any]) -> tuple[float, list[str]]:
    buckets = split_by_kind(tokens)
    diagnoses = buckets["diagnosis"]
    triggers = buckets["medication"] | buckets["procedure"]

    score = 0.0
    violations: list[str] = []

    for trigger in sorted(triggers):
        rule = rules.get(trigger)
        if not rule:
            continue

        expected = [item["diagnosis"] for item in rule["expected_diagnoses"]]
        if not expected:
            continue

        if diagnoses.isdisjoint(expected):
            confidence_weight = max(float(item["confidence"]) for item in rule["expected_diagnoses"])
            support_weight = math.log1p(float(rule["trigger_support"]))
            contribution = confidence_weight * support_weight

            score += contribution
            violations.append(
                f"missing_expected_diagnosis_for::{trigger}::top_expected::{expected[0]}"
            )

    return float(score), violations[:20]


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

    calib_idx = np.asarray(sorted(calib_idx), dtype=int)
    eval_idx = np.asarray(sorted(eval_idx), dtype=int)
    return calib_idx, eval_idx


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
        row = {
            "w_det": wd,
            "w_ont": wo,
            "w_gen": wg,
            **metrics(y, score),
        }
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["average_precision", "roc_auc", "f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def make_oracle_proxy(df: pd.DataFrame) -> np.ndarray:
    severity = {
        "": 0.0,
        "normal": 0.0,
        "none": 0.0,
        "demographic_conflict": 1.0,
        "medication_mismatch": 0.80,
        "missing_diagnosis": 0.65,
        "missing_indication": 0.65,
        "forbidden_cooccurrence": 0.95,
        "forbidden_co_occurrence": 0.95,
        "temporal_inconsistency": 0.70,
    }

    vals: list[float] = []
    for raw in df["anomaly_type"].fillna("").astype(str):
        key = raw.strip().lower()
        vals.append(float(severity.get(key, 0.50 if key else 0.0)))

    return minmax(np.asarray(vals, dtype=float))


def render_readme(summary: dict[str, Any]) -> str:
    eval_metrics = summary["heldout_evaluation"]
    best = summary["best_weights_from_calibration"]

    return f"""# Day 35.5 — Independent Ontology/Rule Scorer

## Status
Complete.

## Scientific purpose
Day 35 used an ontology proxy derived from `anomaly_type`, which is useful as an oracle-style integration check but not a publishable independent ontology score.

Day 35.5 replaces that with an independent content-only `Sont_independent` scorer.

## No-leakage rule
The scorer is learned from normal training records only and scores validation records using:

- `sequence_tokens`
- learned medication/procedure → expected diagnosis rules

The scorer does **not** use the following during scoring:

- `label`
- `anomaly_type`
- `source`

Those fields are used only after scoring for evaluation and breakdown.

## Best calibrated weights from calibration split
- `w_det`: {best["w_det"]:.4f}
- `w_ont`: {best["w_ont"]:.4f}
- `w_gen`: {best["w_gen"]:.4f}

## Held-out evaluation metrics

| Signal | ROC-AUC | Average Precision | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Detector only | {eval_metrics["Sdet_only"]["roc_auc"]:.4f} | {eval_metrics["Sdet_only"]["average_precision"]:.4f} | {eval_metrics["Sdet_only"]["f1"]:.4f} | {eval_metrics["Sdet_only"]["precision"]:.4f} | {eval_metrics["Sdet_only"]["recall"]:.4f} |
| Independent Sont only | {eval_metrics["Sont_independent_only"]["roc_auc"]:.4f} | {eval_metrics["Sont_independent_only"]["average_precision"]:.4f} | {eval_metrics["Sont_independent_only"]["f1"]:.4f} | {eval_metrics["Sont_independent_only"]["precision"]:.4f} | {eval_metrics["Sont_independent_only"]["recall"]:.4f} |
| Sdet + independent Sont | {eval_metrics["Scal_independent"]["roc_auc"]:.4f} | {eval_metrics["Scal_independent"]["average_precision"]:.4f} | {eval_metrics["Scal_independent"]["f1"]:.4f} | {eval_metrics["Scal_independent"]["precision"]:.4f} | {eval_metrics["Scal_independent"]["recall"]:.4f} |
| Oracle Sont proxy reference | {eval_metrics["Sont_oracle_proxy_reference"]["roc_auc"]:.4f} | {eval_metrics["Sont_oracle_proxy_reference"]["average_precision"]:.4f} | {eval_metrics["Sont_oracle_proxy_reference"]["f1"]:.4f} | {eval_metrics["Sont_oracle_proxy_reference"]["precision"]:.4f} | {eval_metrics["Sont_oracle_proxy_reference"]["recall"]:.4f} |

## Interpretation
The publishable comparison is `Detector only` versus `Sdet + independent Sont`.

The oracle proxy is retained only as an upper-bound/sanity-check reference and must not be presented as independent ontology performance.

## Output files
- `artifacts/day35_5/val_independent_sont_scores.csv`
- `artifacts/day35_5/learned_indication_rules.json`
- `artifacts/day35_5/learned_indication_rules.csv`
- `artifacts/day35_5/day35_5_scientific_summary.json`
- `artifacts/day35_5/paper_ready_metrics.csv`
"""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = prepare_frame(args.train_path)
    val_df = prepare_frame(args.val_path)

    detector_df = pd.read_csv(args.detector_scores_csv)
    if len(detector_df) != len(val_df):
        raise ValueError(
            f"Detector score row count mismatch: detector={len(detector_df)} val={len(val_df)}"
        )

    if "prob_anomaly" not in detector_df.columns:
        raise ValueError(f"Expected prob_anomaly in detector scores. Columns={list(detector_df.columns)}")

    rules_payload, rules_table = learn_indication_rules(
        train_df=train_df,
        min_trigger_support=args.min_trigger_support,
        min_expected_support=args.min_expected_support,
        min_confidence=args.min_confidence,
        max_expected_per_trigger=args.max_expected_per_trigger,
    )

    rules = rules_payload["rules"]

    raw_scores: list[float] = []
    violations_list: list[str] = []
    violation_counts: list[int] = []

    for tokens in val_df["sequence_tokens"]:
        score, violations = score_record(tokens, rules)
        raw_scores.append(score)
        violation_counts.append(len(violations))
        violations_list.append(" | ".join(violations))

    result = val_df[["label", "source", "anomaly_type", "sequence_tokens"]].copy()
    result["Sdet_raw"] = detector_df["prob_anomaly"].astype(float).to_numpy()
    result["Sdet"] = minmax(result["Sdet_raw"])
    result["Sont_independent_raw"] = np.asarray(raw_scores, dtype=float)
    result["Sont_independent"] = minmax(result["Sont_independent_raw"])
    result["Sont_independent_violation_count"] = violation_counts
    result["Sont_independent_violations"] = violations_list
    result["Sgen"] = 0.0
    result["Sont_oracle_proxy_reference"] = make_oracle_proxy(result)

    y = result["label"].to_numpy(dtype=int)
    calib_idx, eval_idx = stratified_split(y, args.calibration_fraction, args.seed)

    calib = result.iloc[calib_idx].reset_index(drop=True)
    holdout = result.iloc[eval_idx].reset_index(drop=True)

    weight_table = grid_search(
        y=calib["label"].to_numpy(dtype=int),
        sdet=calib["Sdet"].to_numpy(dtype=float),
        sont=calib["Sont_independent"].to_numpy(dtype=float),
        sgen=calib["Sgen"].to_numpy(dtype=float),
        grid_step=args.grid_step,
        max_sgen_weight=args.max_sgen_weight,
    )

    best = weight_table.iloc[0].to_dict()
    wd = float(best["w_det"])
    wo = float(best["w_ont"])
    wg = float(best["w_gen"])

    result["Scal_independent"] = wd * result["Sdet"] + wo * result["Sont_independent"] + wg * result["Sgen"]
    calib["Scal_independent"] = wd * calib["Sdet"] + wo * calib["Sont_independent"] + wg * calib["Sgen"]
    holdout["Scal_independent"] = wd * holdout["Sdet"] + wo * holdout["Sont_independent"] + wg * holdout["Sgen"]

    heldout_metrics = {
        "Sdet_only": metrics(
            holdout["label"].to_numpy(dtype=int),
            holdout["Sdet"].to_numpy(dtype=float),
        ),
        "Sont_independent_only": metrics(
            holdout["label"].to_numpy(dtype=int),
            holdout["Sont_independent"].to_numpy(dtype=float),
        ),
        "Scal_independent": metrics(
            holdout["label"].to_numpy(dtype=int),
            holdout["Scal_independent"].to_numpy(dtype=float),
        ),
        "Sgen_only": metrics(
            holdout["label"].to_numpy(dtype=int),
            holdout["Sgen"].to_numpy(dtype=float),
        ),
        "Sont_oracle_proxy_reference": metrics(
            holdout["label"].to_numpy(dtype=int),
            holdout["Sont_oracle_proxy_reference"].to_numpy(dtype=float),
        ),
    }

    calibration_metrics = {
        "Sdet_only": metrics(
            calib["label"].to_numpy(dtype=int),
            calib["Sdet"].to_numpy(dtype=float),
        ),
        "Sont_independent_only": metrics(
            calib["label"].to_numpy(dtype=int),
            calib["Sont_independent"].to_numpy(dtype=float),
        ),
        "Scal_independent": metrics(
            calib["label"].to_numpy(dtype=int),
            calib["Scal_independent"].to_numpy(dtype=float),
        ),
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
            mean_Sont_independent=("Sont_independent", "mean"),
            mean_Scal_independent=("Scal_independent", "mean"),
            mean_violation_count=("Sont_independent_violation_count", "mean"),
        )
        .reset_index()
        .sort_values("mean_Sont_independent", ascending=False)
    )

    paper_rows = []
    for name, metric_row in heldout_metrics.items():
        paper_rows.append(
            {
                "split": "heldout_evaluation",
                "signal": name,
                **metric_row,
            }
        )
    paper_table = pd.DataFrame(paper_rows)

    result.to_csv(out_dir / "val_independent_sont_scores.csv", index=False)
    weight_table.to_csv(out_dir / "day35_5_weight_search_calibration.csv", index=False)
    rules_table.to_csv(out_dir / "learned_indication_rules.csv", index=False)
    breakdown.to_csv(out_dir / "anomaly_type_breakdown_independent_sont.csv", index=False)
    paper_table.to_csv(out_dir / "paper_ready_metrics.csv", index=False)
    save_json(out_dir / "learned_indication_rules.json", rules_payload)

    summary = {
        "day": "35.5",
        "status": "complete",
        "scientific_goal": "Replace anomaly_type-derived oracle Sont with an independent content-only ontology/rule score.",
        "no_leakage_policy": {
            "used_for_rule_learning": ["train records with label=0 only", "sequence_tokens"],
            "used_for_validation_scoring": ["sequence_tokens only", "learned rules"],
            "not_used_for_validation_scoring": ["label", "source", "anomaly_type"],
            "used_only_for_evaluation": ["label", "source", "anomaly_type"],
        },
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "calibration_rows": int(len(calib)),
        "heldout_evaluation_rows": int(len(holdout)),
        "rule_learning": {
            "normal_train_rows": int(rules_payload["normal_train_rows"]),
            "learned_rule_count": int(rules_payload["learned_rule_count"]),
            "min_trigger_support": args.min_trigger_support,
            "min_expected_support": args.min_expected_support,
            "min_confidence": args.min_confidence,
            "max_expected_per_trigger": args.max_expected_per_trigger,
        },
        "best_weights_from_calibration": {
            "w_det": wd,
            "w_ont": wo,
            "w_gen": wg,
        },
        "calibration_metrics": calibration_metrics,
        "heldout_evaluation": heldout_metrics,
        "interpretation_for_paper": (
            "Report detector-only vs Sdet+independent-Sont on the heldout evaluation split. "
            "Treat anomaly_type-derived Sont_oracle_proxy_reference only as an upper-bound/sanity-check, not as an independent ontology result. "
            "Sgen remains diagnostic because Day 34 found the current diffusion Sgen proxy weak."
        ),
        "outputs": {
            "scores": str(out_dir / "val_independent_sont_scores.csv"),
            "rules_json": str(out_dir / "learned_indication_rules.json"),
            "rules_csv": str(out_dir / "learned_indication_rules.csv"),
            "weight_search": str(out_dir / "day35_5_weight_search_calibration.csv"),
            "breakdown": str(out_dir / "anomaly_type_breakdown_independent_sont.csv"),
            "paper_ready_metrics": str(out_dir / "paper_ready_metrics.csv"),
            "summary": str(out_dir / "day35_5_scientific_summary.json"),
            "readme": str(out_dir / "README.md"),
        },
    }

    save_json(out_dir / "day35_5_scientific_summary.json", summary)
    (out_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
