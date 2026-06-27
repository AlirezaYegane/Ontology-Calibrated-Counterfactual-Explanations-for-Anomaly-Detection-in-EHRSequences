"""
scripts/diagnose_anomaly_triviality.py
=======================================
Phase 1a -- Benchmark triviality diagnostic.

Question answered: does the current synthetic anomaly benchmark measure real
anomaly reasoning, or can trivially computable signals (sequence length, token
counts, rare-token presence, ...) recover the anomaly labels almost as well as
the trained supervised detector?

If a trivial baseline rivals the detector's ROC-AUC (~0.80), the benchmark is
artifact-driven / circular and must be redesigned (Phase 1b anomaly v2).

This script does NOT train or modify any model. It only reads the labeled
synthetic anomaly file and computes label-free trivial signals.

Usage
-----
    python scripts/diagnose_anomaly_triviality.py
    python scripts/diagnose_anomaly_triviality.py --input data/processed/mimiciv_val_synth_anomaly.pkl

Outputs
-------
    artifacts/diagnostics/rf2_triviality.json
    artifacts/diagnostics/rf2_triviality.md
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Prefer sklearn when available; fall back to a NumPy implementation.
    from sklearn.metrics import average_precision_score as _sk_ap
    from sklearn.metrics import roc_auc_score as _sk_auc

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - environment guard
    _sk_ap = None
    _sk_auc = None
    _HAVE_SKLEARN = False


def roc_auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties handled by average rank)."""
    if _HAVE_SKLEARN:
        return float(_sk_auc(y_true, score))
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes required for AUC.")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (area under precision-recall, step interpolation)."""
    if _HAVE_SKLEARN:
        return float(_sk_ap(y_true, score))
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    total_pos = cum_tp[-1] if len(cum_tp) else 0
    if total_pos == 0:
        return 0.0
    precision = cum_tp / (np.arange(len(y_sorted)) + 1.0)
    recall = cum_tp / total_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "mimiciv_val_synth_anomaly.pkl"
OUT_DIR = PROJECT_ROOT / "artifacts" / "diagnostics"
DETECTOR_REF_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "day45"
    / "detector_only"
    / "day45_test_set_metrics.json"
)

# Candidate column names (schema-tolerant).
LABEL_CANDIDATES = ("is_synthetic_anomaly", "label", "y_true", "is_anomaly")
TYPE_CANDIDATES = ("anomaly_type", "type", "violation_type")
SEQ_CANDIDATES = (
    "model_visible_sequence",
    "codes",
    "sequence_tokens",
    "tokens",
    "sequence",
    "event_codes",
)
GENDER_CANDIDATES = ("gender", "sex", "patient_sex")

DIAG_PREFIXES = ("DX_", "DIAG_", "ICD", "SNOMED")
PROC_PREFIXES = ("PROC_", "CPT_")
MED_PREFIXES = ("MED_", "RX", "NDC_", "DRUG_", "RAW_DRUG")

# Pregnancy / sex-specific token patterns (string-level; diagnostic heuristic only).
PREGNANCY_PATTERNS = ("PREG", "GESTAT", "OBSTET", "ANTENATAL", "PUERPER", "DELIVERY")
# ICD-9/10 prefixes commonly used for sex-specific conditions in this dataset.
SEX_SPECIFIC_PREFIXES = (
    "DX_9_63",
    "DX_9_64",
    "DX_9_65",
    "DX_9_66",
    "DX_9_67",
    "DX_9_V22",
    "DX_9_V23",
    "DX_9_V27",
    "DX_9_V28",
    "DX_9_185",
    "DX_9_600",
    "DX_9_601",
    "DX_9_602",
    "DX_10_O",
    "DX_10_Z33",
    "DX_10_Z34",
    "DX_10_C61",
    "DX_10_N40",
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def resolve_input(explicit: str | None) -> Path:
    """Resolve the input file, with a clear error if ambiguous or missing."""
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Specified --input does not exist: {path}")
        return path

    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT

    processed = PROJECT_ROOT / "data" / "processed"
    if not processed.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed}")

    candidates = sorted(
        p
        for p in processed.glob("*.pkl")
        if "synth" in p.name.lower() and "anomaly" in p.name.lower()
    )
    val_candidates = [p for p in candidates if "val" in p.name.lower()]

    if len(val_candidates) == 1:
        return val_candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No synthetic-anomaly pickle found under data/processed/ "
            "(expected something like 'mimiciv_val_synth_anomaly.pkl')."
        )
    raise RuntimeError(
        "Ambiguous validation synthetic-anomaly file. Pass --input explicitly. "
        f"Candidates: {[str(c.relative_to(PROJECT_ROOT)) for c in candidates]}"
    )


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_tokens(value: Any) -> list[str]:
    """Robustly parse a code sequence cell into a list of string tokens."""
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        for loader in (ast.literal_eval, json.loads):
            try:
                obj = loader(text)
                if isinstance(obj, (list, tuple)):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                continue
    for sep in ("|", ";", ","):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text]


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def has_prefix(token: str, prefixes: tuple[str, ...]) -> bool:
    up = token.upper()
    return any(up.startswith(p) for p in prefixes)


def namespace_of(token: str) -> str:
    if has_prefix(token, DIAG_PREFIXES):
        return "diagnosis"
    if has_prefix(token, PROC_PREFIXES):
        return "procedure"
    if has_prefix(token, MED_PREFIXES):
        return "medication"
    return "other"


def namespace_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(namespace_of(t) for t in tokens)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12, 2)
    return ent


def is_sex_specific(token: str) -> bool:
    up = token.upper()
    if any(pat in up for pat in PREGNANCY_PATTERNS):
        return True
    return any(up.startswith(p) for p in SEX_SPECIFIC_PREFIXES)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    if not np.all(np.isfinite(score)):
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return None


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    if not np.all(np.isfinite(score)):
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return float(average_precision_score(y_true, score))
    except Exception:
        return None


def distribution(score: np.ndarray) -> dict[str, float]:
    if score.size == 0:
        return {}
    return {
        "mean": float(np.mean(score)),
        "std": float(np.std(score)),
        "min": float(np.min(score)),
        "p25": float(np.percentile(score, 25)),
        "median": float(np.median(score)),
        "p75": float(np.percentile(score, 75)),
        "max": float(np.max(score)),
    }


def evaluate_signal(
    y_true: np.ndarray,
    score: np.ndarray,
) -> dict[str, Any]:
    auc = safe_auc(y_true, score)
    ap = safe_ap(y_true, score)
    discriminative_power = None
    direction = None
    if auc is not None:
        discriminative_power = float(max(auc, 1.0 - auc))
        direction = "higher_is_anomalous" if auc >= 0.5 else "lower_is_anomalous"
    return {
        "roc_auc": auc,
        "average_precision": ap,
        "discriminative_power": discriminative_power,
        "direction": direction,
        "dist_normal": distribution(score[y_true == 0]),
        "dist_anomaly": distribution(score[y_true == 1]),
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def compute_signals(
    token_lists: list[list[str]],
    genders: list[str | None],
    rare_threshold: int,
) -> dict[str, np.ndarray]:
    """Compute all trivial label-free signals.

    Rare/unknown signals are defined relative to a token frequency table built
    from this file's own records (document frequency). This is label-free: it
    does not look at the anomaly label.
    """
    # Document frequency across all records (label-free).
    doc_freq: Counter[str] = Counter()
    for toks in token_lists:
        for t in set(toks):
            doc_freq[t] += 1

    known_vocab = set(doc_freq.keys())
    rare_vocab = {t for t, c in doc_freq.items() if c <= rare_threshold}

    seq_len = np.array([len(t) for t in token_lists], dtype=float)
    diag_ct = np.array(
        [sum(namespace_of(x) == "diagnosis" for x in t) for t in token_lists],
        dtype=float,
    )
    proc_ct = np.array(
        [sum(namespace_of(x) == "procedure" for x in t) for t in token_lists],
        dtype=float,
    )
    med_ct = np.array(
        [sum(namespace_of(x) == "medication" for x in t) for t in token_lists],
        dtype=float,
    )
    rare_ct = np.array(
        [sum(x in rare_vocab for x in t) for t in token_lists], dtype=float
    )
    rare_present = (rare_ct > 0).astype(float)
    # "Unknown/unmapped" here = tokens whose namespace is 'other' (no recognized
    # clinical prefix). This is the closest label-free proxy without a real
    # ontology; it is NOT a true ontology-coverage signal (that arrives in Phase 2).
    unknown_ct = np.array(
        [sum(namespace_of(x) == "other" for x in t) for t in token_lists], dtype=float
    )
    ns_entropy = np.array([namespace_entropy(t) for t in token_lists], dtype=float)
    sex_specific = np.array(
        [1.0 if any(is_sex_specific(x) for x in t) else 0.0 for t in token_lists],
        dtype=float,
    )

    return {
        "sequence_length": seq_len,
        "diagnosis_token_count": diag_ct,
        "procedure_token_count": proc_ct,
        "medication_token_count": med_ct,
        "rare_token_count": rare_ct,
        "rare_token_presence": rare_present,
        "unknown_or_unmapped_token_count": unknown_ct,
        "token_namespace_entropy": ns_entropy,
        "contains_pregnancy_or_sex_specific_token": sex_specific,
        "_meta_known_vocab_size": len(known_vocab),
        "_meta_rare_vocab_size": len(rare_vocab),
    }


def load_detector_reference() -> dict[str, Any] | None:
    if not DETECTOR_REF_PATH.exists():
        return None
    try:
        data = json.loads(DETECTOR_REF_PATH.read_text(encoding="utf-8"))
        gm = data.get("global_metrics", {})
        return {
            "source": str(DETECTOR_REF_PATH.relative_to(PROJECT_ROOT)),
            "roc_auc": gm.get("roc_auc"),
            "average_precision": gm.get("average_precision"),
            "note": "Reference only; not recomputed here. Detector was trained on this same synthetic scheme.",
        }
    except Exception:
        return None


def build_verdict(
    overall: dict[str, dict[str, Any]],
    by_type: dict[str, dict[str, dict[str, Any]]],
    detector_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    signal_names = [k for k in overall if not k.startswith("_meta")]

    ranked = sorted(
        (
            (name, overall[name]["discriminative_power"])
            for name in signal_names
            if overall[name].get("discriminative_power") is not None
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    strongest_name, strongest_power = ranked[0] if ranked else (None, None)

    # Most suspicious anomaly type: highest trivial discriminative power achievable.
    suspicious = []
    for atype, sigs in by_type.items():
        best = max(
            (
                (n, sigs[n]["discriminative_power"])
                for n in sigs
                if not n.startswith("_meta")
                and sigs[n].get("discriminative_power") is not None
            ),
            key=lambda x: x[1],
            default=(None, None),
        )
        if best[0] is not None:
            suspicious.append(
                {
                    "anomaly_type": atype,
                    "best_signal": best[0],
                    "discriminative_power": best[1],
                }
            )
    suspicious.sort(key=lambda d: d["discriminative_power"], reverse=True)

    det_auc = (detector_ref or {}).get("roc_auc")
    rivals_detector = (
        strongest_power is not None
        and det_auc is not None
        and strongest_power >= det_auc - 0.05
    )

    if strongest_power is None:
        severity = "undetermined"
    elif strongest_power >= 0.80:
        severity = "severe"
    elif strongest_power >= 0.70:
        severity = "high"
    elif strongest_power >= 0.60:
        severity = "moderate"
    else:
        severity = "low"

    artifact_driven = severity in {"severe", "high"} or rivals_detector
    redesign_mandatory = artifact_driven

    return {
        "strongest_trivial_signal": strongest_name,
        "strongest_discriminative_power": strongest_power,
        "signal_ranking": ranked,
        "most_suspicious_anomaly_types": suspicious,
        "detector_reference_roc_auc": det_auc,
        "trivial_rivals_detector": rivals_detector,
        "artifact_severity": severity,
        "labels_likely_artifact_driven": artifact_driven,
        "benchmark_redesign_mandatory": redesign_mandatory,
    }


def render_markdown(report: dict[str, Any]) -> str:
    v = report["verdict"]
    lines: list[str] = []
    lines.append("# RF2 — Benchmark Triviality Diagnostic\n")
    lines.append(f"**Input:** `{report['input_file']}`  ")
    lines.append(
        f"**Records:** {report['n_records']} "
        f"(normal={report['n_normal']}, anomaly={report['n_anomaly']})  "
    )
    lines.append(
        f"**Label column:** `{report['label_column']}` | "
        f"**Type column:** `{report['type_column']}` | "
        f"**Sequence column:** `{report['sequence_column']}`\n"
    )

    det = report.get("detector_reference")
    if det and det.get("roc_auc") is not None:
        lines.append(
            f"**Detector reference ROC-AUC:** {det['roc_auc']:.4f} "
            f"(`{det['source']}`, trained on the same synthetic scheme)\n"
        )

    lines.append("## Overall trivial-baseline results\n")
    lines.append(
        "| Signal | ROC-AUC | Avg Precision | Discriminative power | Direction |"
    )
    lines.append("|---|---:|---:|---:|---|")
    overall = report["overall"]
    ordered = sorted(
        (n for n in overall if not n.startswith("_meta")),
        key=lambda n: overall[n].get("discriminative_power") or 0.0,
        reverse=True,
    )
    for name in ordered:
        m = overall[name]
        auc = m["roc_auc"]
        ap = m["average_precision"]
        dp = m["discriminative_power"]
        lines.append(
            f"| {name} | {auc:.4f} | {ap:.4f} | {dp:.4f} | {m['direction']} |"
            if auc is not None
            else f"| {name} | n/a | n/a | n/a | n/a |"
        )
    lines.append("")

    lines.append("## Per-anomaly-type best trivial signal\n")
    lines.append("| Anomaly type | Best trivial signal | Discriminative power |")
    lines.append("|---|---|---:|")
    for row in v["most_suspicious_anomaly_types"]:
        lines.append(
            f"| {row['anomaly_type']} | {row['best_signal']} | {row['discriminative_power']:.4f} |"
        )
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(
        f"- **Are current labels likely artifact-driven?** "
        f"{'YES' if v['labels_likely_artifact_driven'] else 'No (not strongly)'} "
        f"(severity: {v['artifact_severity']})."
    )
    lines.append(
        f"- **Strongest trivial baseline:** `{v['strongest_trivial_signal']}` "
        f"(discriminative power {v['strongest_discriminative_power']:.4f})."
    )
    if v["most_suspicious_anomaly_types"]:
        worst = v["most_suspicious_anomaly_types"][0]
        lines.append(
            f"- **Most suspicious anomaly type:** `{worst['anomaly_type']}` — "
            f"recoverable by `{worst['best_signal']}` at discriminative power "
            f"{worst['discriminative_power']:.4f}."
        )
    lines.append(
        f"- **Do trivial signals rival the detector (~{v.get('detector_reference_roc_auc')})?** "
        f"{'YES' if v['trivial_rivals_detector'] else 'No'}."
    )
    lines.append(
        f"- **Is benchmark redesign mandatory?** "
        f"{'YES — proceed to Phase 1b anomaly v2 before any further modeling.' if v['benchmark_redesign_mandatory'] else 'Not strictly, but recommended.'}"
    )
    lines.append("")
    lines.append("## What this means for A* readiness\n")
    if v["benchmark_redesign_mandatory"]:
        lines.append(
            "The current synthetic anomalies are recoverable by label-free trivial "
            "signals at a level comparable to the trained detector. Any detection or "
            "ontology-calibration result on this benchmark is therefore **not A*-defensible**: "
            "a reviewer can attribute the performance to injection artifacts (e.g., changes in "
            "sequence length or token-type counts) rather than clinical anomaly reasoning. "
            "Benchmark redesign (Phase 1b: anomaly v2 + leave-type-out protocol) is a prerequisite "
            "for any headline claim."
        )
    else:
        lines.append(
            "Trivial baselines do not fully explain the labels, which is encouraging, but the "
            "supervised-on-synthetic training still creates circularity. A non-circular protocol "
            "(Phase 1b/3) remains required before A* claims."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1a benchmark triviality diagnostic."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a labeled synthetic-anomaly pickle (default: data/processed/mimiciv_val_synth_anomaly.pkl).",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=1,
        help="A token is 'rare' if it appears in <= this many records (document frequency). Default 1.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: artifacts/diagnostics).",
    )
    parser.add_argument(
        "--out-prefix",
        default="rf2_triviality",
        help="Output filename prefix (e.g. 'rf2_triviality_v2').",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input(args.input)
    print(f"[rf2] Loading: {input_path.relative_to(PROJECT_ROOT)}")
    df = pd.read_pickle(input_path)

    label_col = pick_column(df, LABEL_CANDIDATES)
    if label_col is None:
        raise RuntimeError(
            f"No label column found. Looked for {LABEL_CANDIDATES}. Have: {list(df.columns)}"
        )
    seq_col = pick_column(df, SEQ_CANDIDATES)
    if seq_col is None:
        raise RuntimeError(
            f"No sequence column found. Looked for {SEQ_CANDIDATES}. Have: {list(df.columns)}"
        )
    type_col = pick_column(df, TYPE_CANDIDATES)
    gender_col = pick_column(df, GENDER_CANDIDATES)

    y_true = (
        pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    )
    token_lists = [parse_tokens(v) for v in df[seq_col].tolist()]
    genders = df[gender_col].tolist() if gender_col else [None] * len(df)

    n_records = int(len(df))
    n_anomaly = int((y_true == 1).sum())
    n_normal = int((y_true == 0).sum())
    print(f"[rf2] records={n_records} normal={n_normal} anomaly={n_anomaly}")
    if n_anomaly == 0 or n_normal == 0:
        raise RuntimeError(
            "Need both normal and anomalous records to compute discrimination metrics."
        )

    signals = compute_signals(token_lists, genders, rare_threshold=args.rare_threshold)
    meta = {k: signals.pop(k) for k in list(signals) if k.startswith("_meta")}

    overall = {name: evaluate_signal(y_true, score) for name, score in signals.items()}

    by_type: dict[str, dict[str, dict[str, Any]]] = {}
    if type_col is not None:
        types = df[type_col].astype(str).fillna("nan").to_numpy()
        anomaly_types = sorted(
            {t for t, y in zip(types, y_true) if y == 1 and t.lower() != "nan"}
        )
        normal_mask = y_true == 0
        for atype in anomaly_types:
            sub_mask = normal_mask | ((y_true == 1) & (types == atype))
            sub_y = y_true[sub_mask]
            res = {}
            for name, score in signals.items():
                res[name] = evaluate_signal(sub_y, score[sub_mask])
            by_type[atype] = res

    detector_ref = load_detector_reference()
    verdict = build_verdict(overall, by_type, detector_ref)

    report = {
        "phase": "1a",
        "diagnostic": "benchmark_triviality",
        "input_file": str(input_path.relative_to(PROJECT_ROOT)),
        "n_records": n_records,
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "label_column": label_col,
        "type_column": type_col,
        "sequence_column": seq_col,
        "gender_column": gender_col,
        "rare_threshold": args.rare_threshold,
        "vocab_meta": meta,
        "detector_reference": detector_ref,
        "overall": overall,
        "by_anomaly_type": by_type,
        "verdict": verdict,
    }

    json_path = out_dir / f"{args.out_prefix}.json"
    md_path = out_dir / f"{args.out_prefix}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"[rf2] Wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[rf2] Wrote {md_path.relative_to(PROJECT_ROOT)}")
    print(
        f"[rf2] Strongest trivial signal: {verdict['strongest_trivial_signal']} "
        f"(discriminative power {verdict['strongest_discriminative_power']})"
    )
    print(
        f"[rf2] Artifact severity: {verdict['artifact_severity']} | "
        f"redesign mandatory: {verdict['benchmark_redesign_mandatory']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
