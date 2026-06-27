"""
scripts/build_benchmark_v2.py
=============================
Phase 1b -- Build the ontology-backed, non-circular benchmark-v2.

Loads clean normal MIMIC-IV records, generates v2 anomalies (gender-flip /
indication-removal / mutually-exclusive co-occurrence), preserves normals, and
writes subject-level train/val/test splits with strict field separation:

  model_visible_sequence, gender, age_group  -> detector/scorer input
  label, anomaly_type, subject_id, split      -> bookkeeping
  hidden_eval_metadata                         -> answer key (eval only)
  audit_metadata                               -> provenance (never a model input)

Usage
-----
    python scripts/build_benchmark_v2.py --max-records 30000 --seed 42
    python scripts/build_benchmark_v2.py --max-records 3000 --seed 42 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.anomaly_injection_v2 import (  # noqa: E402
    INJECTOR_REGISTRY,
    AnomalyGenConfig,
    AnomalyTypeV2,
)

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_v2"
ONT_MANIFEST = (
    PROJECT_ROOT / "ontologies" / "processed" / "ontology_asset_manifest.json"
)
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "mimiciv_val_synth_anomaly.pkl"


def _subject_splits(subjects: list[str], rng_seed: int) -> dict[str, str]:
    import random

    rng = random.Random(rng_seed)
    uniq = sorted(set(subjects))
    rng.shuffle(uniq)
    n = len(uniq)
    n_train, n_val = int(0.7 * n), int(0.1 * n)
    assignment: dict[str, str] = {}
    for i, s in enumerate(uniq):
        assignment[s] = (
            "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        )
    return assignment


def build(source: Path, max_records: int, seed: int) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_pickle(source)
    # Use only clean normal records as the base population.
    if "is_synthetic_anomaly" in df.columns:
        df = df[df["is_synthetic_anomaly"].astype(int) == 0].copy()
    if max_records and len(df) > max_records:
        df = df.head(max_records).copy()

    config = AnomalyGenConfig(seed=seed)
    rng_pick = config.derive_rng("typepick")
    rows = df.to_dict(orient="records")

    subjects = [str(r.get("subject_id")) for r in rows]
    split_of = _subject_splits(subjects, seed)

    injectors = [(t, INJECTOR_REGISTRY[t]) for t in config.enabled_types]
    type_counts: dict[str, int] = {t.value: 0 for t in AnomalyTypeV2}

    out_records: list[dict[str, Any]] = []
    audit_lines: list[dict[str, Any]] = []

    for r in rows:
        subj = str(r.get("subject_id"))
        split = split_of.get(subj, "train")
        base_seq = r.get(config.sequence_column, [])
        base_visible = {
            "model_visible_sequence": list(base_seq)
            if isinstance(base_seq, (list, tuple))
            else [],
            "gender": r.get("gender"),
            "age_group": r.get("age_group"),
        }

        # Decide whether to corrupt this record (never corrupt train -> train is
        # the clean pool for unsupervised detector training).
        corrupt = split != "train" and rng_pick.random() < 0.5
        event = None
        if corrupt:
            order = list(injectors)
            rng_pick.shuffle(order)
            for _t, fn in order:
                event = fn(r, config)
                if event is not None:
                    break

        if event is None:
            out_records.append(
                {
                    **base_visible,
                    "label": 0,
                    "anomaly_type": "normal",
                    "subject_id": subj,
                    "split": split,
                    "hidden_eval_metadata": {},
                    "audit_metadata": {"source": "clean_normal"},
                }
            )
        else:
            mv = event.to_model_visible_row()  # runs leakage guard
            type_counts[event.anomaly_type.value] += 1
            rec = {
                "model_visible_sequence": mv.get(config.sequence_column, []),
                "gender": mv.get("gender"),
                "age_group": mv.get("age_group"),
                "label": 1,
                "anomaly_type": event.anomaly_type.value,
                "subject_id": subj,
                "split": split,
                "hidden_eval_metadata": event.hidden_eval,
                "audit_metadata": event.audit,
            }
            out_records.append(rec)
            audit_lines.append(
                {
                    "subject_id": subj,
                    "split": split,
                    "anomaly_type": event.anomaly_type.value,
                    "audit": event.audit,
                    "hidden_eval": event.hidden_eval,
                }
            )

    out_df = pd.DataFrame(out_records)
    return {
        "df": out_df,
        "type_counts": type_counts,
        "audit_lines": audit_lines,
        "split_of": split_of,
    }


def _write_outputs(
    result: dict[str, Any], out_dir: Path, source: Path, seed: int, max_records: int
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = result["df"]

    split_sizes = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        part.to_pickle(out_dir / f"{split}.pkl")
        split_sizes[split] = {
            "n": int(len(part)),
            "n_anomaly": int((part["label"] == 1).sum()),
            "n_normal": int((part["label"] == 0).sum()),
            "subjects": int(part["subject_id"].nunique()),
        }

    # subject overlap check
    subj = {
        s: set(df[df["split"] == s]["subject_id"]) for s in ("train", "val", "test")
    }
    overlap = {
        "train_val": len(subj["train"] & subj["val"]),
        "train_test": len(subj["train"] & subj["test"]),
        "val_test": len(subj["val"] & subj["test"]),
    }

    with (out_dir / "anomaly_v2_audit.jsonl").open("w", encoding="utf-8") as fh:
        for line in result["audit_lines"]:
            fh.write(json.dumps(line, default=str) + "\n")

    ont_version = None
    if ONT_MANIFEST.exists():
        try:
            ont_version = json.loads(ONT_MANIFEST.read_text(encoding="utf-8")).get(
                "generated_from"
            )
        except Exception:
            ont_version = "unavailable"

    manifest = {
        "benchmark": "v2",
        "source_dataset": str(source.relative_to(PROJECT_ROOT))
        if source.is_relative_to(PROJECT_ROOT)
        else str(source),
        "seed": seed,
        "max_records": max_records,
        "n_records": int(len(df)),
        "n_normal": int((df["label"] == 0).sum()),
        "n_anomaly": int((df["label"] == 1).sum()),
        "anomaly_type_counts": result["type_counts"],
        "split_sizes": split_sizes,
        "subject_overlap": overlap,
        "subject_level_split_ok": all(v == 0 for v in overlap.values()),
        "ontology_asset_version": ont_version,
        "leakage_guard": {
            "model_visible_fields": ["model_visible_sequence", "gender", "age_group"],
            "answer_keys_in_hidden_only": True,
            "train_split_is_clean_normal_only": int(
                (df[(df["split"] == "train")]["label"] == 1).sum()
            )
            == 0,
        },
        "design_note": "Ontology-backed SYNTHETIC benchmark (not real-world external validation). Non-circular by construction: gender-flip injects no token; indication-removal/partner-add keep individual tokens common.",
    }
    (out_dir / "benchmark_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build non-circular benchmark-v2.")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--max-records", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source EHR data not found: {source}")
    max_records = 3000 if args.smoke else args.max_records

    result = build(source, max_records=max_records, seed=args.seed)
    manifest = _write_outputs(
        result, Path(args.out_dir), source, args.seed, max_records
    )

    print(
        f"[benchmark_v2] records={manifest['n_records']} normal={manifest['n_normal']} anomaly={manifest['n_anomaly']}"
    )
    print(f"[benchmark_v2] anomaly_type_counts={manifest['anomaly_type_counts']}")
    print(
        f"[benchmark_v2] subject_overlap={manifest['subject_overlap']} (ok={manifest['subject_level_split_ok']})"
    )
    print(f"[benchmark_v2] wrote {Path(args.out_dir) / 'benchmark_v2_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
