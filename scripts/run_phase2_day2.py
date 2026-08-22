from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.baselines.common import demographic_tokens, load_dataframe
from src.baselines.cooccurrence_pmi import StatisticalRelationalBaseline
from src.baselines.rarity import TokenRarityBaseline
from src.baselines.selection import (
    aggregate_upper_scores,
    compare_selected_to_extreme,
    ensure_validation_only_paths,
    evaluate_candidate,
    select_best_extreme,
    select_best_non_extreme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/paper_phase2/day2_b0_b1_selection.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/paper_phase2/day2",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Day 2 config must contain a YAML mapping.")

    return payload


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
        "pyyaml",
    )

    result: dict[str, str] = {}

    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except Exception:
            result[package] = "not-installed"

    return result


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def token_surprisals(
    model: TokenRarityBaseline,
    sequence: list[str],
) -> np.ndarray:
    if not sequence:
        return np.asarray([], dtype=float)

    return np.asarray(
        [
            -math.log(
                max(model.token_probability(token), 1e-12)
            )
            for token in sequence
        ],
        dtype=float,
    )


def b1_score_components(
    model: StatisticalRelationalBaseline,
    pairs: list[tuple[str, str]],
    quantiles: list[float],
    top_ks: list[int],
) -> dict[str, float]:
    if not pairs:
        output = aggregate_upper_scores(
            np.asarray([], dtype=float),
            quantiles,
            top_ks,
        )

        output.update(
            {
                "npmi_mean": 0.0,
                "confidence_mean": 0.0,
                "lift_mean": 0.0,
            }
        )

        return output

    relation_scores: list[float] = []
    npmi_scores: list[float] = []
    confidence_scores: list[float] = []
    lift_scores: list[float] = []

    for a, b in pairs:
        stats = model.relation_statistics(a, b)

        relation_scores.append(
            -math.log(
                max(stats["conditional"], 1e-12)
            )
        )

        npmi_scores.append(
            (1.0 - stats["npmi"]) / 2.0
        )

        confidence_scores.append(
            -math.log(
                max(stats["confidence"], 1e-12)
            )
        )

        lift_scores.append(
            -math.log(
                max(stats["lift"], 1e-12)
            )
        )

    output = aggregate_upper_scores(
        np.asarray(relation_scores, dtype=float),
        quantiles,
        top_ks,
    )

    output.update(
        {
            "npmi_mean": float(np.mean(npmi_scores)),
            "confidence_mean": float(
                np.mean(confidence_scores)
            ),
            "lift_mean": float(np.mean(lift_scores)),
        }
    )

    return output


def render_readme(
    selection: dict[str, Any],
) -> str:
    b0 = selection["selected_b0"]
    b1 = selection["selected_b1"]

    b0_cmp = selection["b0_robust_vs_extreme"]
    b1_cmp = selection["b1_robust_vs_extreme"]

    lines = [
        "# Paper Phase 2 — Day 2 Validation Selection",
        "",
        "## Status",
        "",
        "Complete — validation-only B0/B1 tuning and length/pair-count bias audit.",
        "",
        "This remains a preliminary benchmark-v2 thesis-diagnostic experiment until benchmark-v3 generator-independent strata are ready.",
        "",
        "## Data policy",
        "",
        "- Statistics fitted from clean training data only.",
        "- Hyperparameter selection used validation only.",
        "- Test data was not accessed.",
        "- Ontology information was not used.",
        "- Hidden/audit answer keys were not used as model features.",
        "- No per-record scores were saved.",
        "",
        "## Frozen B0",
        "",
        f"- Candidate: `{b0['candidate_id']}`",
        f"- PR-AUC: `{b0['overall']['pr_auc']:.6f}`",
        f"- ROC-AUC: `{b0['overall']['roc_auc']:.6f}`",
        f"- Max |Spearman bias|: `{b0['bias']['max_abs_spearman']}`",
        "",
        "## Frozen B1",
        "",
        f"- Candidate: `{b1['candidate_id']}`",
        f"- PR-AUC: `{b1['overall']['pr_auc']:.6f}`",
        f"- ROC-AUC: `{b1['overall']['roc_auc']:.6f}`",
        f"- Max |Spearman bias|: `{b1['bias']['max_abs_spearman']}`",
        "",
        "## Robust vs extreme audit",
        "",
        "### B0",
        "",
        f"- PR-AUC delta selected - extreme: `{b0_cmp['pr_auc_delta_selected_minus_extreme']:.6f}`",
        f"- Bias delta selected - extreme: `{b0_cmp['max_abs_bias_delta_selected_minus_extreme']}`",
        f"- Bias reduced: `{b0_cmp['bias_reduced']}`",
        "",
        "### B1",
        "",
        f"- PR-AUC delta selected - extreme: `{b1_cmp['pr_auc_delta_selected_minus_extreme']:.6f}`",
        f"- Bias delta selected - extreme: `{b1_cmp['max_abs_bias_delta_selected_minus_extreme']}`",
        f"- Bias reduced: `{b1_cmp['bias_reduced']}`",
        "",
        "## Selection policy",
        "",
        "Extreme max/worst aggregations were audited but were not eligible for the final frozen scorer.",
        "",
        "Among non-extreme candidates, selection was:",
        "",
        "1. highest validation PR-AUC;",
        "2. lower maximum absolute Spearman correlation with sequence length / candidate-pair count as the first tie-breaker;",
        "3. higher ROC-AUC as the second tie-breaker;",
        "4. lexical candidate ID as the final deterministic tie-breaker.",
        "",
        "No post-hoc numerical correlation threshold was introduced.",
        "",
        "## Next step",
        "",
        "Day 3 may open the test split once, using only these frozen B0/B1 configurations.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)

    train_path = Path(config["data"]["train"])
    val_path = Path(config["data"]["validation"])

    ensure_validation_only_paths(
        str(train_path),
        str(val_path),
    )

    print("[1/8] Loading canonical benchmark-v2 train/validation...")

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

    train_labels = set(
        train["_label"].dropna().astype(int).unique()
    )

    if train_labels != {0}:
        raise ValueError(
            "Canonical training split must contain clean normals only."
        )

    if int(val["_label"].sum()) != 610:
        raise ValueError(
            "Unexpected benchmark-v2 validation anomaly count."
        )

    labels = val["_label"].to_numpy(dtype=int)

    sequence_lengths = val[
        "_sequence_length"
    ].to_numpy(dtype=int)

    anomaly_types = val[
        "_anomaly_type"
    ].astype(str).to_numpy()

    train_lengths = train[
        "_sequence_length"
    ].astype(float)

    q33 = float(
        train_lengths.quantile(1.0 / 3.0)
    )
    q67 = float(
        train_lengths.quantile(2.0 / 3.0)
    )

    length_bins = np.where(
        sequence_lengths <= q33,
        "short",
        np.where(
            sequence_lengths <= q67,
            "medium",
            "long",
        ),
    )

    train_sequences = train[
        "_sequence_tokens"
    ].tolist()

    val_sequences = val[
        "_sequence_tokens"
    ].tolist()

    train_demographics = [
        demographic_tokens(row)
        for _, row in train.iterrows()
    ]

    val_demographics = [
        demographic_tokens(row)
        for _, row in val.iterrows()
    ]

    print(
        f"      train={len(train):,} val={len(val):,} "
        f"anomalies={int(labels.sum()):,}"
    )
    print(
        f"      train-derived bins: "
        f"short<={q33:.1f}, medium<={q67:.1f}, long>{q67:.1f}"
    )

    print("[2/8] Preparing candidate relation-pair counts...")

    val_pairs = [
        StatisticalRelationalBaseline.relation_pairs(
            sequence,
            demographics,
        )
        for sequence, demographics
        in zip(
            val_sequences,
            val_demographics,
            strict=True,
        )
    ]

    candidate_pair_counts = np.asarray(
        [len(pairs) for pairs in val_pairs],
        dtype=int,
    )

    candidates: list[dict[str, Any]] = []

    b0_grid = config["b0_grid"]

    b0_alphas = [
        float(value)
        for value in b0_grid["alpha"]
    ]

    b0_quantiles = [
        float(value)
        for value in b0_grid["quantiles"]
    ]

    b0_top_ks = [
        int(value)
        for value in b0_grid["top_k"]
    ]

    print("[3/8] Running B0 validation grid...")

    for alpha in b0_alphas:
        model = TokenRarityBaseline(
            alpha=alpha,
            rare_quantile=0.05,
        ).fit(train_sequences)

        score_lists: dict[str, list[float]] = {}

        for sequence in val_sequences:
            values = token_surprisals(
                model,
                sequence,
            )

            aggregated = aggregate_upper_scores(
                values,
                b0_quantiles,
                b0_top_ks,
            )

            for aggregation, value in aggregated.items():
                score_lists.setdefault(
                    aggregation,
                    [],
                ).append(value)

        for aggregation, values in score_lists.items():
            candidate_id = (
                f"b0|surprisal|alpha={alpha:g}|agg={aggregation}"
            )

            candidates.append(
                evaluate_candidate(
                    candidate_id=candidate_id,
                    model="B0",
                    score_family="token_surprisal",
                    aggregation=aggregation,
                    parameters={
                        "alpha": alpha,
                    },
                    is_extreme=(
                        aggregation == "worst"
                    ),
                    labels=labels,
                    scores=np.asarray(
                        values,
                        dtype=float,
                    ),
                    sequence_lengths=sequence_lengths,
                    candidate_pair_counts=candidate_pair_counts,
                    length_bins=length_bins,
                    anomaly_types=anomaly_types,
                )
            )

    for rare_quantile in [
        float(value)
        for value in b0_grid["rare_quantile"]
    ]:
        model = TokenRarityBaseline(
            alpha=1.0,
            rare_quantile=rare_quantile,
        ).fit(train_sequences)

        values = np.asarray(
            [
                model.score(sequence)[
                    "rare_code_fraction"
                ]
                for sequence in val_sequences
            ],
            dtype=float,
        )

        candidate_id = (
            f"b0|rare_fraction|rare_q={rare_quantile:g}"
        )

        candidates.append(
            evaluate_candidate(
                candidate_id=candidate_id,
                model="B0",
                score_family="rare_code_fraction",
                aggregation="fraction",
                parameters={
                    "rare_quantile": rare_quantile,
                },
                is_extreme=False,
                labels=labels,
                scores=values,
                sequence_lengths=sequence_lengths,
                candidate_pair_counts=candidate_pair_counts,
                length_bins=length_bins,
                anomaly_types=anomaly_types,
            )
        )

    print(
        "      B0 candidates:",
        sum(
            candidate["model"] == "B0"
            for candidate in candidates
        ),
    )

    print("[4/8] Fitting B1 clean-training statistics once...")

    train_relational = list(
        zip(
            train_sequences,
            train_demographics,
            strict=True,
        )
    )

    b1 = StatisticalRelationalBaseline(
        alpha=0.5,
        min_support=5,
        quantile=0.90,
        bottom_k=5,
    ).fit(train_relational)

    print(
        f"      records={b1.n_records:,} "
        f"items={len(b1.item_support):,} "
        f"pairs={len(b1.pair_support):,}"
    )

    b1_grid = config["b1_grid"]

    b1_alphas = [
        float(value)
        for value in b1_grid["alpha"]
    ]

    min_support_values = [
        int(value)
        for value in b1_grid["min_support"]
    ]

    b1_quantiles = [
        float(value)
        for value in b1_grid["quantiles"]
    ]

    b1_top_ks = [
        int(value)
        for value in b1_grid["top_k"]
    ]

    print("[5/8] Running B1 validation grid...")

    for alpha in b1_alphas:
        for min_support in min_support_values:
            b1.alpha = alpha
            b1.min_support = min_support

            score_lists: dict[str, list[float]] = {}

            for pairs in val_pairs:
                components = b1_score_components(
                    b1,
                    pairs,
                    b1_quantiles,
                    b1_top_ks,
                )

                for name, value in components.items():
                    score_lists.setdefault(
                        name,
                        [],
                    ).append(value)

            for aggregation, values in score_lists.items():
                if aggregation in {
                    "npmi_mean",
                    "confidence_mean",
                    "lift_mean",
                }:
                    score_family = aggregation.replace(
                        "_mean",
                        "",
                    )
                else:
                    score_family = "conditional_relation"

                candidate_id = (
                    f"b1|{score_family}|"
                    f"alpha={alpha:g}|"
                    f"minsup={min_support}|"
                    f"agg={aggregation}"
                )

                candidates.append(
                    evaluate_candidate(
                        candidate_id=candidate_id,
                        model="B1",
                        score_family=score_family,
                        aggregation=aggregation,
                        parameters={
                            "alpha": alpha,
                            "min_support": min_support,
                        },
                        is_extreme=(
                            score_family
                            == "conditional_relation"
                            and aggregation == "worst"
                        ),
                        labels=labels,
                        scores=np.asarray(
                            values,
                            dtype=float,
                        ),
                        sequence_lengths=sequence_lengths,
                        candidate_pair_counts=candidate_pair_counts,
                        length_bins=length_bins,
                        anomaly_types=anomaly_types,
                    )
                )

            print(
                f"      alpha={alpha:g} "
                f"min_support={min_support} done"
            )

    print(
        "      B1 candidates:",
        sum(
            candidate["model"] == "B1"
            for candidate in candidates
        ),
    )

    print("[6/8] Selecting frozen non-extreme B0/B1 configurations...")

    selected_b0 = select_best_non_extreme(
        candidates,
        "B0",
    )

    selected_b1 = select_best_non_extreme(
        candidates,
        "B1",
    )

    best_extreme_b0 = select_best_extreme(
        candidates,
        "B0",
    )

    best_extreme_b1 = select_best_extreme(
        candidates,
        "B1",
    )

    b0_comparison = compare_selected_to_extreme(
        selected_b0,
        best_extreme_b0,
    )

    b1_comparison = compare_selected_to_extreme(
        selected_b1,
        best_extreme_b1,
    )

    selection = {
        "status": "complete",
        "experiment": (
            "paper_phase2_day2_b0_b1_selection"
        ),
        "scientific_role": (
            "preliminary benchmark-v2 thesis diagnostic"
        ),
        "selection_policy": {
            "eligible": (
                "non-extreme aggregations only"
            ),
            "primary": (
                "highest validation PR-AUC"
            ),
            "tie_breaker_1": (
                "lowest max absolute Spearman correlation "
                "with sequence length / candidate-pair count"
            ),
            "tie_breaker_2": (
                "highest validation ROC-AUC"
            ),
            "final_tie_breaker": (
                "lexical candidate ID"
            ),
            "numeric_bias_threshold_used": False,
        },
        "train_derived_length_bins": {
            "short_max": q33,
            "medium_max": q67,
            "long_above": q67,
        },
        "selected_b0": selected_b0,
        "selected_b1": selected_b1,
        "best_extreme_b0": best_extreme_b0,
        "best_extreme_b1": best_extreme_b1,
        "b0_robust_vs_extreme": b0_comparison,
        "b1_robust_vs_extreme": b1_comparison,
        "b1_minus_b0": {
            "pr_auc": float(
                selected_b1["overall"]["pr_auc"]
                - selected_b0["overall"]["pr_auc"]
            ),
            "roc_auc": float(
                selected_b1["overall"]["roc_auc"]
                - selected_b0["overall"]["roc_auc"]
            ),
        },
        "test_data_accessed": False,
        "ontology_used": False,
        "hidden_eval_metadata_used_as_feature": False,
        "audit_metadata_used_as_feature": False,
        "per_record_scores_saved": False,
    }

    grid_results = {
        "experiment": (
            "paper_phase2_day2_b0_b1_selection"
        ),
        "candidate_count": len(candidates),
        "b0_candidate_count": sum(
            candidate["model"] == "B0"
            for candidate in candidates
        ),
        "b1_candidate_count": sum(
            candidate["model"] == "B1"
            for candidate in candidates
        ),
        "search_space": {
            "b0": b0_grid,
            "b1": b1_grid,
        },
        "candidates": candidates,
    }

    manifest = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).replace(
            microsecond=0
        ).isoformat(),
        "git_commit": git_commit(),
        "git_dirty_at_run": git_dirty(),
        "benchmark_version": "benchmark_v2",
        "seed": seed,
        "split_hashes": {
            "train_sha256": sha256_file(
                train_path
            ),
            "validation_sha256": sha256_file(
                val_path
            ),
        },
        "config": {
            "path": config_path.as_posix(),
            "sha256": sha256_file(config_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "data_policy": {
            "statistics_from_clean_train_only": True,
            "validation_only_tuning": True,
            "test_accessed": False,
            "ontology_used": False,
            "hidden_answer_keys_used_as_features": False,
        },
        "artifact_policy": {
            "aggregate_only": True,
            "per_record_scores_saved": False,
            "per_record_scores_committed": False,
        },
    }

    print("[7/8] Writing aggregate-only Day 2 artifacts...")

    write_json(
        out_dir / "grid_results.json",
        grid_results,
    )

    write_json(
        out_dir / "day2_selection.json",
        selection,
    )

    write_json(
        out_dir / "manifest.json",
        manifest,
    )

    (out_dir / "README.md").write_text(
        render_readme(selection),
        encoding="utf-8",
    )

    print("[8/8] Final validation selection")
    print("")
    print("=== FROZEN B0 ===")
    print(selected_b0["candidate_id"])
    print(
        f"PR-AUC={selected_b0['overall']['pr_auc']:.4f} "
        f"ROC-AUC={selected_b0['overall']['roc_auc']:.4f} "
        f"bias={selected_b0['bias']['max_abs_spearman']}"
    )

    print("")
    print("=== FROZEN B1 ===")
    print(selected_b1["candidate_id"])
    print(
        f"PR-AUC={selected_b1['overall']['pr_auc']:.4f} "
        f"ROC-AUC={selected_b1['overall']['roc_auc']:.4f} "
        f"bias={selected_b1['bias']['max_abs_spearman']}"
    )

    print("")
    print("=== B1 - B0 ===")
    print(
        "PR-AUC delta="
        f"{selection['b1_minus_b0']['pr_auc']:+.4f}"
    )
    print(
        "ROC-AUC delta="
        f"{selection['b1_minus_b0']['roc_auc']:+.4f}"
    )

    print("")
    print("Artifacts:")
    print(out_dir / "day2_selection.json")
    print(out_dir / "grid_results.json")
    print(out_dir / "manifest.json")
    print(out_dir / "README.md")


if __name__ == "__main__":
    main()
