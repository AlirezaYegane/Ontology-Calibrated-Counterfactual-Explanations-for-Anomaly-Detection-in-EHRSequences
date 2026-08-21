from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


SEQUENCE_COLUMNS = (
    "model_visible_sequence",
    "sequence_tokens",
    "codes",
    "sequence",
    "tokens",
    "event_codes",
    "concepts",
)

LABEL_COLUMNS = (
    "label",
    "is_anomaly",
    "anomaly_label",
    "is_synthetic_anomaly",
    "target",
)

ANOMALY_TYPE_COLUMNS = (
    "anomaly_type",
    "anomaly_family",
    "type",
)

BAD_VALUES = {"", "nan", "none", "null", "na", "n/a"}


def normalize_tokens(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        output: list[str] = []
        for item in value:
            token = str(item).strip()
            if token.lower() not in BAD_VALUES:
                output.append(token)
        return output

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
                    continue

        if "," in text:
            return [
                part.strip()
                for part in text.split(",")
                if part.strip()
            ]

        return [
            part.strip()
            for part in text.split()
            if part.strip()
        ]

    token = str(value).strip()
    return [token] if token.lower() not in BAD_VALUES else []


def infer_sequence_column(df: pd.DataFrame) -> str:
    for column in SEQUENCE_COLUMNS:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify a canonical model-visible sequence column. "
        f"Available columns: {list(df.columns)}"
    )


def infer_label_column(df: pd.DataFrame) -> str | None:
    for column in LABEL_COLUMNS:
        if column in df.columns:
            return column
    return None


def infer_anomaly_type_column(df: pd.DataFrame) -> str | None:
    for column in ANOMALY_TYPE_COLUMNS:
        if column in df.columns:
            return column
    return None


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pkl":
        obj = pd.read_pickle(path)
    elif suffix == ".parquet":
        obj = pd.read_parquet(path)
    elif suffix == ".csv":
        obj = pd.read_csv(path)
    elif suffix == ".jsonl":
        obj = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported input: {path}")

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"Expected DataFrame; got {type(obj)}"
        )

    df = obj.copy()

    sequence_column = infer_sequence_column(df)

    df["_sequence_tokens"] = df[
        sequence_column
    ].apply(normalize_tokens)

    df = df[
        df["_sequence_tokens"].map(len) > 0
    ].copy()

    df["_sequence_length"] = df[
        "_sequence_tokens"
    ].map(len)

    label_column = infer_label_column(df)

    if label_column is not None:
        df["_label"] = pd.to_numeric(
            df[label_column],
            errors="coerce",
        )

    anomaly_type_column = infer_anomaly_type_column(df)

    if anomaly_type_column is not None:
        df["_anomaly_type"] = (
            df[anomaly_type_column]
            .fillna("")
            .astype(str)
        )

    df.attrs["source_sequence_column"] = sequence_column
    df.attrs["source_label_column"] = label_column
    df.attrs["source_anomaly_type_column"] = anomaly_type_column

    return df.reset_index(drop=True)


def demographic_tokens(row: pd.Series) -> list[str]:
    output: list[str] = []

    for key in ("gender", "sex"):
        if key in row.index and pd.notna(row[key]):
            value = str(row[key]).strip().upper()

            if value:
                output.append(
                    f"__DEMOG__SEX={value}"
                )
            break

    if "age_group" in row.index:
        value = row["age_group"]

        if pd.notna(value):
            value_s = str(value).strip().upper()

            if value_s:
                output.append(
                    f"__DEMOG__AGE={value_s}"
                )

    return output
