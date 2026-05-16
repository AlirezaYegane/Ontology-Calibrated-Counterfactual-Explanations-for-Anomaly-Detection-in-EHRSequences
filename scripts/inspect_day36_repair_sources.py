from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SEARCH_DIRS = [
    Path("artifacts"),
    Path("data/processed"),
    Path("outputs"),
]

TARGET_COLUMNS = {
    "sequence_tokens",
    "codes",
    "tokens",
    "sequence",
    "event_codes",
    "concepts",
    "label",
    "is_synthetic_anomaly",
    "anomaly_type",
    "expected_code",
    "expected_codes",
    "missing_code",
    "missing_codes",
    "removed_code",
    "removed_codes",
    "bad_code",
    "bad_codes",
    "injected_code",
    "injected_codes",
    "added_code",
    "added_codes",
    "conflict_code",
    "conflict_codes",
    "flagged_code",
    "flagged_codes",
    "replacement_code",
    "replacement_codes",
    "gender",
    "sex",
}


def read_columns(path: Path) -> tuple[tuple[int, int] | None, list[str]]:
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=20, low_memory=False)
        elif path.suffix.lower() == ".pkl":
            df = pd.read_pickle(path)
            if not isinstance(df, pd.DataFrame):
                return None, []
            df = df.head(20)
        elif path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, list):
                df = pd.DataFrame(obj[:20])
            elif isinstance(obj, dict):
                for key in ("records", "rows", "data", "items", "examples"):
                    if isinstance(obj.get(key), list):
                        df = pd.DataFrame(obj[key][:20])
                        break
                else:
                    df = pd.DataFrame([obj])
            else:
                return None, []
        elif path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if i >= 20:
                        break
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            df = pd.DataFrame(rows)
        else:
            return None, []

        try:
            full_shape = (
                pd.read_pickle(path).shape if path.suffix.lower() == ".pkl" else None
            )
        except Exception:
            full_shape = None

        return full_shape, list(df.columns)
    except Exception:
        return None, []


def main() -> None:
    rows = []

    for root in SEARCH_DIRS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.suffix.lower() not in {".csv", ".pkl", ".json", ".jsonl"}:
                continue

            full_l = str(path).lower()

            if not any(
                x in full_l
                for x in [
                    "anom",
                    "synthetic",
                    "day17",
                    "day20",
                    "day35",
                    "supervised",
                    "val",
                ]
            ):
                continue

            shape, cols = read_columns(path)
            if not cols:
                continue

            matched = sorted(set(cols) & TARGET_COLUMNS)
            repair_cols = [
                c
                for c in matched
                if c.lower()
                in {
                    "expected_code",
                    "expected_codes",
                    "missing_code",
                    "missing_codes",
                    "removed_code",
                    "removed_codes",
                    "bad_code",
                    "bad_codes",
                    "injected_code",
                    "injected_codes",
                    "added_code",
                    "added_codes",
                    "conflict_code",
                    "conflict_codes",
                    "flagged_code",
                    "flagged_codes",
                    "replacement_code",
                    "replacement_codes",
                }
            ]

            if matched:
                rows.append(
                    {
                        "path": str(path),
                        "shape_if_known": shape,
                        "matched_columns": matched,
                        "repair_target_columns": repair_cols,
                        "all_columns": cols,
                    }
                )

    rows = sorted(
        rows,
        key=lambda r: (
            -len(r["repair_target_columns"]),
            -len(r["matched_columns"]),
            r["path"],
        ),
    )

    out_path = Path("artifacts/day36/repair_source_candidates.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(rows[:20], indent=2, ensure_ascii=False))
    print(f"\nSaved full candidate list to: {out_path}")


if __name__ == "__main__":
    main()
