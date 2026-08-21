from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.baselines.common import demographic_tokens, load_dataframe
from src.baselines.cooccurrence_pmi import StatisticalRelationalBaseline
from src.baselines.rarity import TokenRarityBaseline


SCORE_COLUMNS = (
    "b0_max_token_surprisal",
    "b0_mean_negative_log_frequency",
    "b0_rare_code_fraction",
    "b1_worst_relation",
    "b1_mean_relation",
    "b1_q90_relation",
    "b1_topk_relation",
    "b1_npmi_anomaly",
    "b1_confidence_anomaly",
    "b1_lift_anomaly",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alpha-b0", type=float, default=1.0)
    parser.add_argument("--rare-quantile", type=float, default=0.05)

    parser.add_argument("--alpha-b1", type=float, default=0.5)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--bottom-k", type=int, default=5)

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
        ).strip()
        return bool(output)
    except Exception:
        return True


def package_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
    )

    result: dict[str, str] = {}

    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except Exception:
            result[package] = "not-installed"

    return result


def discrimination_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "pr_auc": float(
            average_precision_score(labels, scores)
        ),
        "roc_auc": float(
            roc_auc_score(labels, scores)
        ),
    }


def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> float | None:
    if x.nunique(dropna=True) < 2:
        return None

    if y.nunique(dropna=True) < 2:
        return None

    value = x.corr(y, method="spearman")

    if pd.isna(value):
        return None

    return float(value)


def family_breakdown(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for family, subset in frame.groupby("anomaly_type"):
        family_payload: dict[str, Any] = {
            "n": int(len(subset)),
            "n_anomaly": int(subset["label"].sum()),
        }

        for score_name in SCORE_COLUMNS:
            family_payload[score_name] = {
                "mean": float(subset[score_name].mean()),
                "median": float(subset[score_name].median()),
            }

        output[str(family)] = family_payload

    return output


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)

    train_path = Path(args.train)
    val_path = Path(args.val)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Hard guard: this script is validation-only.
    forbidden_test_names = {"test.pkl", "benchmark_test.pkl"}

    if train_path.name.lower() in forbidden_test_names:
        raise ValueError("Test data cannot be used as training data.")

    if val_path.name.lower() in forbidden_test_names:
        raise ValueError(
            "Paper Phase 2 Day 1 is validation-only. "
            "Do not pass test.pkl."
        )

    print("[1/6] Loading canonical benchmark-v2 train/validation...")

    train = load_dataframe(train_path)
    val = load_dataframe(val_path)

    if len(train) != 20570:
        raise ValueError(
            f"Unexpected benchmark-v2 train size: {len(train)}"
        )

    if len(val) != 3123:
        raise ValueError(
            f"Unexpected benchmark-v2 validation size: {len(val)}"
        )

    if set(train["_label"].dropna().astype(int).unique()) != {0}:
        raise ValueError(
            "Canonical training split must contain clean normals only."
        )

    if int(val["_label"].sum()) != 610:
        raise ValueError(
            "Unexpected benchmark-v2 validation anomaly count."
        )

    print("[2/6] Fitting B0 token-rarity statistics from clean train...")

    train_sequences = train["_sequence_tokens"].tolist()

    b0 = TokenRarityBaseline(
        alpha=args.alpha_b0,
        rare_quantile=args.rare_quantile,
    ).fit(train_sequences)

    print(
        f"      B0 vocab={b0.vocab_size:,} "
        f"train_tokens={b0.total_tokens:,}"
    )

    print("[3/6] Fitting B1 statistical-relational statistics...")

    train_relational = [
        (
            row["_sequence_tokens"],
            demographic_tokens(row),
        )
        for _, row in train.iterrows()
    ]

    b1 = StatisticalRelationalBaseline(
        alpha=args.alpha_b1,
        min_support=args.min_support,
        quantile=args.quantile,
        bottom_k=args.bottom_k,
    ).fit(train_relational)

    print(
        f"      B1 records={b1.n_records:,} "
        f"items={len(b1.item_support):,} "
        f"pairs={len(b1.pair_support):,}"
    )

    print("[4/6] Scoring validation records...")

    scored_rows: list[dict[str, Any]] = []

    for _, row in val.iterrows():
        sequence = row["_sequence_tokens"]
        demographics = demographic_tokens(row)

        b0_scores = b0.score(sequence)
        b1_scores = b1.score(sequence, demographics)

        scored_rows.append(
            {
                "label": int(row["_label"]),
                "anomaly_type": str(row["_anomaly_type"]),
                "sequence_length": int(len(sequence)),
                "b0_max_token_surprisal": (
                    b0_scores["max_token_surprisal"]
                ),
                "b0_mean_negative_log_frequency": (
                    b0_scores["mean_negative_log_frequency"]
                ),
                "b0_rare_code_fraction": (
                    b0_scores["rare_code_fraction"]
                ),
                "b1_worst_relation": (
                    b1_scores["worst_relation"]
                ),
                "b1_mean_relation": (
                    b1_scores["mean_relation"]
                ),
                "b1_q90_relation": (
                    b1_scores["q90_relation"]
                ),
                "b1_topk_relation": (
                    b1_scores["topk_relation"]
                ),
                "b1_npmi_anomaly": (
                    b1_scores["npmi_anomaly"]
                ),
                "b1_confidence_anomaly": (
                    b1_scores["confidence_anomaly"]
                ),
                "b1_lift_anomaly": (
                    b1_scores["lift_anomaly"]
                ),
                "candidate_pair_count": int(
                    b1_scores["candidate_pair_count"]
                ),
                "supported_pair_fraction": float(
                    b1_scores["supported_pair_fraction"]
                ),
            }
        )

    scored = pd.DataFrame(scored_rows)

    print("[5/6] Computing validation metrics and bias diagnostics...")

    labels = scored["label"].to_numpy(dtype=int)

    metrics: dict[str, Any] = {}

    for score_name in SCORE_COLUMNS:
        metrics[score_name] = discrimination_metrics(
            labels,
            scored[score_name].to_numpy(dtype=float),
        )

    correlations: dict[str, Any] = {}

    for score_name in SCORE_COLUMNS:
        correlations[score_name] = {
            "spearman_vs_sequence_length": safe_spearman(
                scored[score_name],
                scored["sequence_length"],
            ),
            "spearman_vs_candidate_pair_count": safe_spearman(
                scored[score_name],
                scored["candidate_pair_count"],
            ),
        }

    train_lengths = train["_sequence_length"].astype(float)

    q33 = float(train_lengths.quantile(1.0 / 3.0))
    q67 = float(train_lengths.quantile(2.0 / 3.0))

    # Validation discrimination by train-derived length bins.
    scored["length_bin"] = pd.cut(
        scored["sequence_length"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["short", "medium", "long"],
        include_lowest=True,
    )

    length_bin_metrics: dict[str, Any] = {}

    for bin_name, subset in scored.groupby(
        "length_bin",
        observed=True,
    ):
        if subset["label"].nunique() != 2:
            continue

        y_bin = subset["label"].to_numpy(dtype=int)

        length_bin_metrics[str(bin_name)] = {
            "n": int(len(subset)),
            "anomaly_rate": float(subset["label"].mean()),
            "scores": {
                score_name: discrimination_metrics(
                    y_bin,
                    subset[score_name].to_numpy(dtype=float),
                )
                for score_name in SCORE_COLUMNS
            },
        }

    summary = {
        "status": "complete",
        "experiment": "paper_phase2_day1_b0_b1",
        "scientific_role": (
            "untuned benchmark-v2 validation diagnostic only"
        ),
        "benchmark": "v2",
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "validation_normal": int((val["_label"] == 0).sum()),
        "validation_anomaly": int((val["_label"] == 1).sum()),
        "validation_anomaly_types": {
            str(key): int(value)
            for key, value in val[
                "_anomaly_type"
            ].value_counts().to_dict().items()
        },
        "b0_fit": {
            "vocab_size": int(b0.vocab_size),
            "total_train_tokens": int(b0.total_tokens),
            "alpha": args.alpha_b0,
            "rare_quantile": args.rare_quantile,
            "rare_count_threshold": (
                b0.rare_count_threshold
            ),
        },
        "b1_fit": {
            "train_records": int(b1.n_records),
            "unique_items": int(len(b1.item_support)),
            "unique_pairs": int(len(b1.pair_support)),
            "alpha": args.alpha_b1,
            "min_support": args.min_support,
            "quantile": args.quantile,
            "bottom_k": args.bottom_k,
        },
        "validation_metrics": metrics,
        "length_pair_bias_correlations": correlations,
        "train_derived_length_bins": {
            "short_max": q33,
            "medium_max": q67,
            "long_above": q67,
        },
        "length_bin_metrics": length_bin_metrics,
        "anomaly_family_score_breakdown": family_breakdown(scored),
        "test_data_accessed": False,
        "ontology_used": False,
        "hidden_eval_metadata_used_as_feature": False,
        "audit_metadata_used_as_feature": False,
    }

    manifest = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).replace(microsecond=0).isoformat(),
        "git_commit": git_commit(),
        "git_dirty_at_run": git_dirty(),
        "benchmark_version": "benchmark_v2",
        "seed": args.seed,
        "split_hashes": {
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(val_path),
        },
        "configuration": {
            "b0": {
                "alpha": args.alpha_b0,
                "rare_quantile": args.rare_quantile,
            },
            "b1": {
                "alpha": args.alpha_b1,
                "min_support": args.min_support,
                "quantile": args.quantile,
                "bottom_k": args.bottom_k,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "data_policy": {
            "statistics_from_clean_train_only": True,
            "validation_only": True,
            "test_accessed": False,
            "ontology_used": False,
            "hidden_answer_keys_used": False,
        },
        "per_record_scores_saved": False,
        "per_record_scores_committed": False,
    }

    print("[6/6] Writing aggregate-only artifacts...")

    (out_dir / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    ranking = sorted(
        (
            {
                "score": score_name,
                **metric,
            }
            for score_name, metric in metrics.items()
        ),
        key=lambda item: item["pr_auc"],
        reverse=True,
    )

    print("")
    print("=== Validation ranking by PR-AUC ===")

    for item in ranking:
        print(
            f"{item['score']:<38} "
            f"PR-AUC={item['pr_auc']:.4f} "
            f"ROC-AUC={item['roc_auc']:.4f}"
        )

    print("")
    print("Artifacts:")
    print(out_dir / "summary.json")
    print(out_dir / "manifest.json")


if __name__ == "__main__":
    main()
