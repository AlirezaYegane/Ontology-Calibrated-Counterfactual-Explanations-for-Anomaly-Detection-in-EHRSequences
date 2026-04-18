from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ontology.rules import infer_sequence_column
from src.scoring.ontology_aware import Day25OntologyAwareScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--max_examples_per_type", type=int, default=5)
    parser.add_argument("--only_synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.data_path)

    if args.only_synthetic and "is_synthetic_anomaly" in df.columns:
        df = df[df["is_synthetic_anomaly"] == 1].copy()

    seq_col = infer_sequence_column(df)
    df = df.head(args.max_rows).copy()

    scorer = Day25OntologyAwareScorer(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab_path,
        device=args.device,
    )

    rows: list[dict[str, object]] = []

    for idx, row in df.iterrows():
        raw = row.to_dict()
        scored = scorer.score_record(raw)

        rows.append(
            {
                "row_index": int(idx),
                "anomaly_type": str(raw.get("anomaly_type", "")),
                "gender": str(raw.get("gender", raw.get("sex", ""))),
                "is_synthetic_anomaly": int(raw.get("is_synthetic_anomaly", 0)) if "is_synthetic_anomaly" in raw else "",
                "sequence_length": int(len(scored["tokens"])),
                "sdet": scored["sdet"],
                "sont": scored["sont"],
                "scal": scored["scal"],
                "n_violations": int(len(scored["violations"])),
                "violations": json.dumps([v["name"] for v in scored["violations"]], ensure_ascii=False),
                "top_implicated_tokens": json.dumps(scored["top_implicated_tokens"], ensure_ascii=False),
                "tokens_preview": json.dumps(scored["tokens"][:20], ensure_ascii=False),
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "scored_rows.csv", index=False)

    by_type_df = (
        results_df.groupby("anomaly_type", dropna=False)
        .agg(
            count=("row_index", "size"),
            mean_sdet=("sdet", "mean"),
            mean_sont=("sont", "mean"),
            mean_scal=("scal", "mean"),
            mean_violations=("n_violations", "mean"),
        )
        .reset_index()
        .sort_values(["mean_scal", "count"], ascending=[False, False])
    )
    by_type_df.to_csv(out_dir / "by_anomaly_type.csv", index=False)

    top_df = (
        results_df.sort_values("scal", ascending=False)
        .groupby("anomaly_type", dropna=False)
        .head(args.max_examples_per_type)
        .reset_index(drop=True)
    )
    top_df.to_csv(out_dir / "top_examples_per_type.csv", index=False)

    summary = {
        "data_path": args.data_path,
        "checkpoint": args.checkpoint,
        "vocab_path": args.vocab_path,
        "sequence_column": seq_col,
        "rows_scored": int(len(results_df)),
        "global_mean_scores": {
            "sdet": float(results_df["sdet"].mean()),
            "sont": float(results_df["sont"].mean()),
            "scal": float(results_df["scal"].mean()),
        },
        "max_scal_row_index": int(results_df.sort_values("scal", ascending=False).iloc[0]["row_index"]) if len(results_df) else None,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
