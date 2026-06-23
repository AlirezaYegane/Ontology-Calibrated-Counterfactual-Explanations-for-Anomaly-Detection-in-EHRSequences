"""
src/evaluation/protocol.py
==========================
Phase 1b -- Reusable utilities for *non-circular* evaluation.

This module provides defensive helpers so that later phases (detector rebuild,
counterfactual repair, ablations) cannot accidentally:

  * leak patients across train/validation/test splits (subject leakage),
  * feed injection answer-key columns into a model (metadata leakage),
  * train and "generalization-test" on the same anomaly type (leave-type-out).

The functions operate on pandas DataFrames and raise clear, actionable errors.
They are intentionally model-agnostic and import nothing heavy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Forbidden leakage columns
# ---------------------------------------------------------------------------
# Any column that encodes the corruption "answer key" or links a record back to
# its uncorrupted source must never be visible to a model. This list is a
# conservative superset; extend it rather than narrow it.
FORBIDDEN_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        # explicit repair-answer columns
        "bad_code",
        "expected_code",
        "replacement_code",
        "original_code",
        "injected_code",
        "removed_code",
        "repair_code",
        "ground_truth_repair",
        "repair_target",
        # source / identity links back to the clean record
        "source_anomaly_id",
        "source_row_id",
        # paired clean/corrupted sequences and human-readable corruption notes
        "codes_original",
        "codes_corrupted",
        "corruption_note",
    }
)


def _normalize(name: str) -> str:
    return str(name).strip().lower()


# ---------------------------------------------------------------------------
# Subject / split leakage
# ---------------------------------------------------------------------------


def _subject_set(frame: pd.DataFrame, subject_col: str) -> set[Any]:
    if subject_col not in frame.columns:
        raise KeyError(
            f"Subject column '{subject_col}' not in DataFrame columns: {list(frame.columns)}"
        )
    return set(frame[subject_col].dropna().unique().tolist())


def summarize_split_overlap(
    splits: Mapping[str, pd.DataFrame],
    subject_col: str = "subject_id",
) -> dict[str, Any]:
    """Report pairwise subject overlap between named splits.

    Parameters
    ----------
    splits:
        Mapping from split name (e.g. ``"train"``) to its DataFrame.
    subject_col:
        Column holding the patient/subject identifier.

    Returns
    -------
    A dict with per-split subject counts and pairwise overlap counts/examples.
    """
    if len(splits) < 2:
        raise ValueError("Need at least two splits to summarize overlap.")

    subjects = {
        name: _subject_set(frame, subject_col) for name, frame in splits.items()
    }
    names = list(splits.keys())

    pairwise: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = subjects[a] & subjects[b]
            pairwise.append(
                {
                    "split_a": a,
                    "split_b": b,
                    "overlap_count": len(inter),
                    "overlap_examples": sorted(map(str, list(inter)))[:10],
                }
            )

    return {
        "subject_column": subject_col,
        "per_split_subject_counts": {name: len(s) for name, s in subjects.items()},
        "pairwise_overlap": pairwise,
        "total_overlap": sum(p["overlap_count"] for p in pairwise),
    }


def assert_no_subject_overlap(
    splits: Mapping[str, pd.DataFrame],
    subject_col: str = "subject_id",
) -> None:
    """Raise ``ValueError`` if any subject appears in more than one split.

    Use this before training/evaluation to guarantee patient-level separation.
    """
    summary = summarize_split_overlap(splits, subject_col=subject_col)
    offending = [p for p in summary["pairwise_overlap"] if p["overlap_count"] > 0]
    if offending:
        details = "; ".join(
            f"{p['split_a']}∩{p['split_b']}={p['overlap_count']} (e.g. {p['overlap_examples']})"
            for p in offending
        )
        raise ValueError(f"Subject leakage detected across splits: {details}")


# ---------------------------------------------------------------------------
# Forbidden-column leakage
# ---------------------------------------------------------------------------


def find_forbidden_columns(
    columns: Iterable[str],
    forbidden: Iterable[str] | None = None,
) -> list[str]:
    """Return the subset of *columns* that are forbidden leakage columns."""
    forbidden_norm = {_normalize(c) for c in (forbidden or FORBIDDEN_LEAKAGE_COLUMNS)}
    present = []
    for col in columns:
        if _normalize(col) in forbidden_norm:
            present.append(col)
    return present


def validate_no_forbidden_columns(
    data: pd.DataFrame | Iterable[str],
    forbidden: Iterable[str] | None = None,
) -> None:
    """Raise ``ValueError`` if any forbidden leakage column is present.

    Accepts either a DataFrame or an iterable of column names. Intended to wrap
    model inputs: ``validate_no_forbidden_columns(model_features)``.
    """
    columns = list(data.columns) if isinstance(data, pd.DataFrame) else list(data)
    offending = find_forbidden_columns(columns, forbidden=forbidden)
    if offending:
        raise ValueError(
            "Forbidden injection/answer-key columns present in model-visible data: "
            f"{sorted(offending)}. Drop them before training/scoring."
        )


# ---------------------------------------------------------------------------
# Leave-anomaly-type-out splits
# ---------------------------------------------------------------------------


def make_leave_anomaly_type_out_splits(
    df: pd.DataFrame,
    held_out_type: str,
    type_col: str = "anomaly_type",
    label_col: str = "is_synthetic_anomaly",
    normal_type_values: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a leave-anomaly-type-out (train, eval) partition.

    The training partition contains all normal records plus anomalies of every
    type *except* ``held_out_type``. The evaluation partition contains all normal
    records plus only the held-out anomaly type. This tests generalization to an
    anomaly family the model never saw during training.

    Parameters
    ----------
    df:
        Labeled DataFrame with a type column and a binary label column.
    held_out_type:
        The anomaly type to exclude from training and isolate for evaluation.
    type_col:
        Column holding the anomaly type (normals usually NaN / "normal").
    label_col:
        Binary anomaly label column (1 = anomaly, 0 = normal).
    normal_type_values:
        Optional explicit set of values that denote "normal" in *type_col*.
        Defaults to recognizing NaN and the strings "", "normal", "nan", "none".

    Returns
    -------
    ``{"train": train_df, "eval": eval_df, "meta": {...}}``
    """
    if type_col not in df.columns:
        raise KeyError(f"Type column '{type_col}' not in DataFrame: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not in DataFrame: {list(df.columns)}"
        )

    normal_tokens = {
        v.lower() for v in (normal_type_values or ("", "normal", "nan", "none"))
    }

    type_str = df[type_col].astype("string")
    is_normal_type = type_str.isna() | type_str.str.lower().isin(normal_tokens)
    label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    available_types = sorted(
        {t for t in type_str[~is_normal_type].dropna().unique().tolist()}
    )
    if held_out_type not in available_types:
        raise ValueError(
            f"held_out_type '{held_out_type}' not among anomaly types {available_types}."
        )

    is_held = (~is_normal_type) & (type_str == held_out_type)
    is_other_anom = (~is_normal_type) & (type_str != held_out_type)
    is_normal = is_normal_type | (label == 0)

    train_df = df[is_normal | is_other_anom].copy()
    eval_df = df[is_normal | is_held].copy()

    # Guarantee: the held-out type must not appear in training.
    train_types = train_df[type_col].astype("string")
    leaked = (train_types == held_out_type).sum()
    if leaked > 0:
        raise RuntimeError(
            f"Internal error: {leaked} held-out '{held_out_type}' rows leaked into train."
        )

    meta = {
        "held_out_type": held_out_type,
        "available_types": available_types,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_anomaly_rows": int(
            (pd.to_numeric(train_df[label_col], errors="coerce").fillna(0) == 1).sum()
        ),
        "eval_anomaly_rows": int(
            (pd.to_numeric(eval_df[label_col], errors="coerce").fillna(0) == 1).sum()
        ),
    }
    return {"train": train_df, "eval": eval_df, "meta": meta}


# ---------------------------------------------------------------------------
# Label distribution
# ---------------------------------------------------------------------------


def summarize_label_distribution(
    df: pd.DataFrame,
    label_col: str = "is_synthetic_anomaly",
    type_col: str | None = "anomaly_type",
) -> dict[str, Any]:
    """Summarize positive/negative balance and per-type counts."""
    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not in DataFrame: {list(df.columns)}"
        )
    label = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    n = int(len(df))
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    out: dict[str, Any] = {
        "n_records": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": (n_pos / n) if n else 0.0,
    }
    if type_col and type_col in df.columns:
        counts = (
            df[type_col].astype("string").fillna("normal").value_counts(dropna=False)
        )
        out["by_type"] = {str(k): int(v) for k, v in counts.items()}
    return out


__all__ = [
    "FORBIDDEN_LEAKAGE_COLUMNS",
    "assert_no_subject_overlap",
    "summarize_split_overlap",
    "find_forbidden_columns",
    "validate_no_forbidden_columns",
    "make_leave_anomaly_type_out_splits",
    "summarize_label_distribution",
]
