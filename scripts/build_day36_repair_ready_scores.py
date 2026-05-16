from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


BAD_VALUES = {"", "nan", "none", "null", "na", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_path", required=True)
    parser.add_argument(
        "--supervised_path",
        default=r"data\processed\mimiciv_val_detector_supervised.pkl",
    )
    parser.add_argument(
        "--synth_path", default=r"data\processed\mimiciv_val_synth_anomaly.pkl"
    )
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_path", required=True)
    return parser.parse_args()


def normalize_token(value: Any) -> str:
    return str(value).strip()


def is_bad_value(value: Any) -> bool:
    return normalize_token(value).lower() in BAD_VALUES


def parse_list_like(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, float) and str(value).lower() == "nan":
        return []

    if isinstance(value, (list, tuple, set)):
        return [normalize_token(x) for x in value if not is_bad_value(x)]

    text = normalize_token(value)
    if text.lower() in BAD_VALUES:
        return []

    if text.startswith("[") and text.endswith("]"):
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(text)
                if isinstance(obj, (list, tuple, set)):
                    return parse_list_like(obj)
            except Exception:
                pass

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    if "," in text and not text.startswith("ICD"):
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def multiset_minus(left: list[str], right: list[str]) -> list[str]:
    right_counts = Counter(right)
    out: list[str] = []
    for token in left:
        if right_counts[token] > 0:
            right_counts[token] -= 1
        else:
            out.append(token)
    return out


def infer_repair(original: list[str], corrupted: list[str]) -> dict[str, Any]:
    bad_codes = multiset_minus(corrupted, original)
    expected_codes = multiset_minus(original, corrupted)

    if bad_codes and expected_codes:
        edit_hint = "replace_or_remove_add"
    elif bad_codes:
        edit_hint = "remove_bad_code"
    elif expected_codes:
        edit_hint = "add_expected_code"
    else:
        edit_hint = "no_observed_token_delta"

    return {
        "bad_codes_list": bad_codes,
        "expected_codes_list": expected_codes,
        "bad_code": bad_codes[0] if bad_codes else "",
        "expected_code": expected_codes[0] if expected_codes else "",
        "bad_codes": compact_json(bad_codes),
        "expected_codes": compact_json(expected_codes),
        "repair_edit_hint": edit_hint,
        "repair_delta_size": int(len(bad_codes) + len(expected_codes)),
    }


def find_sequence_column(df: pd.DataFrame) -> str:
    candidates = [
        "sequence_tokens",
        "codes",
        "tokens",
        "sequence",
        "event_codes",
        "concepts",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        sample = df[col].dropna().head(20)
        if len(sample) and any(isinstance(x, (list, tuple)) for x in sample):
            return col
    raise ValueError(f"Could not infer sequence column from: {list(df.columns)}")


def label_to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def anomaly_type(value: Any) -> str:
    text = normalize_token(value)
    if text.lower() in BAD_VALUES:
        return ""
    return text


def main() -> None:
    args = parse_args()

    scores_path = Path(args.scores_path)
    supervised_path = Path(args.supervised_path)
    synth_path = Path(args.synth_path)
    out_path = Path(args.out_path)
    summary_path = Path(args.summary_path)

    scores_df = pd.read_csv(scores_path, low_memory=False)
    supervised_df = pd.read_pickle(supervised_path)
    synth_df = pd.read_pickle(synth_path)

    if not isinstance(supervised_df, pd.DataFrame):
        raise TypeError(f"supervised_path is not a DataFrame: {type(supervised_df)}")
    if not isinstance(synth_df, pd.DataFrame):
        raise TypeError(f"synth_path is not a DataFrame: {type(synth_df)}")

    if len(scores_df) != len(supervised_df):
        raise ValueError(
            f"Scores and supervised rows are not aligned by length: "
            f"{len(scores_df)} vs {len(supervised_df)}"
        )

    synth_seq_col = find_sequence_column(synth_df)
    synth_df = synth_df.copy()
    synth_df["_codes_corrupted_norm"] = synth_df[synth_seq_col].apply(parse_list_like)
    synth_df["_codes_original_norm"] = synth_df["codes_original"].apply(parse_list_like)

    if "is_synthetic_anomaly" in synth_df.columns:
        synth_anom = synth_df[
            synth_df["is_synthetic_anomaly"]
            .astype(str)
            .isin(["1", "1.0", "True", "true"])
        ].copy()
    else:
        synth_anom = synth_df.copy()

    synth_anom = synth_anom[synth_anom["_codes_corrupted_norm"].map(len) > 0].copy()

    exact_queues: dict[tuple[tuple[str, ...], str], deque[dict[str, Any]]] = (
        defaultdict(deque)
    )
    sequence_only_queues: dict[tuple[str, ...], deque[dict[str, Any]]] = defaultdict(
        deque
    )

    for synth_idx, row in synth_anom.iterrows():
        corrupted = list(row["_codes_corrupted_norm"])
        original = list(row["_codes_original_norm"])
        repair = infer_repair(original, corrupted)

        payload = {
            "synth_row_index": int(synth_idx),
            "source_row_id": row.get("source_row_id", ""),
            "gender": row.get("gender", ""),
            "age_group": row.get("age_group", ""),
            "corruption_note": row.get("corruption_note", ""),
            "codes_original": compact_json(original),
            "codes_corrupted": compact_json(corrupted),
            **{k: v for k, v in repair.items() if not k.endswith("_list")},
        }

        key_type = anomaly_type(row.get("anomaly_type", ""))
        key_seq = tuple(corrupted)
        exact_queues[(key_seq, key_type)].append(payload)
        sequence_only_queues[key_seq].append(payload)

    enriched = scores_df.copy()

    new_cols = {
        "expected_code": [],
        "expected_codes": [],
        "bad_code": [],
        "bad_codes": [],
        "repair_edit_hint": [],
        "repair_delta_size": [],
        "codes_original": [],
        "codes_corrupted": [],
        "corruption_note": [],
        "source_row_id": [],
        "gender": [],
        "age_group": [],
        "repair_match_status": [],
    }

    matched_exact = 0
    matched_sequence_only = 0
    unmatched_anomalies = 0
    normal_rows = 0

    sup_seq_col = find_sequence_column(supervised_df)

    for _, sup_row in supervised_df.iterrows():
        label = label_to_int(sup_row.get("label", 0))
        seq = tuple(parse_list_like(sup_row.get(sup_seq_col)))
        atype = anomaly_type(sup_row.get("anomaly_type", ""))

        payload: dict[str, Any] | None = None
        match_status = ""

        if label == 0:
            normal_rows += 1
            match_status = "normal_no_repair_target"
        else:
            exact_key = (seq, atype)
            if exact_queues.get(exact_key) and len(exact_queues[exact_key]) > 0:
                payload = exact_queues[exact_key].popleft()
                matched_exact += 1
                match_status = "matched_exact_sequence_and_type"
            elif sequence_only_queues.get(seq) and len(sequence_only_queues[seq]) > 0:
                payload = sequence_only_queues[seq].popleft()
                matched_sequence_only += 1
                match_status = "matched_sequence_only"
            else:
                unmatched_anomalies += 1
                match_status = "unmatched_anomaly"

        if payload is None:
            payload = {
                "expected_code": "",
                "expected_codes": "[]",
                "bad_code": "",
                "bad_codes": "[]",
                "repair_edit_hint": "",
                "repair_delta_size": 0,
                "codes_original": "[]",
                "codes_corrupted": "[]",
                "corruption_note": "",
                "source_row_id": "",
                "gender": sup_row.get("gender", ""),
                "age_group": "",
            }

        for col in new_cols:
            if col == "repair_match_status":
                new_cols[col].append(match_status)
            else:
                new_cols[col].append(payload.get(col, ""))

    for col, values in new_cols.items():
        enriched[col] = values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    enriched.to_csv(out_path, index=False)

    summary = {
        "scores_path": str(scores_path),
        "supervised_path": str(supervised_path),
        "synth_path": str(synth_path),
        "out_path": str(out_path),
        "rows_scores": int(len(scores_df)),
        "rows_supervised": int(len(supervised_df)),
        "rows_synth": int(len(synth_df)),
        "rows_synth_anomaly_filtered": int(len(synth_anom)),
        "normal_rows": int(normal_rows),
        "matched_exact": int(matched_exact),
        "matched_sequence_only": int(matched_sequence_only),
        "unmatched_anomalies": int(unmatched_anomalies),
        "repair_ready_rows": int(len(enriched)),
        "rows_with_expected_code": int(
            (enriched["expected_code"].astype(str).str.len() > 0).sum()
        ),
        "rows_with_bad_code": int(
            (enriched["bad_code"].astype(str).str.len() > 0).sum()
        ),
        "repair_edit_hint_counts": {
            str(k): int(v)
            for k, v in enriched["repair_edit_hint"]
            .fillna("")
            .value_counts()
            .to_dict()
            .items()
        },
        "repair_match_status_counts": {
            str(k): int(v)
            for k, v in enriched["repair_match_status"]
            .fillna("")
            .value_counts()
            .to_dict()
            .items()
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
