"""
scripts/diagnose_ontology_rule_coverage.py
==========================================
Phase 3b -- Rule-coverage diagnostic for the real ontology scorer.

Runs the real-mode OntologyAwareScorer (with the Phase 3b curated rule packs)
over benchmark-v2 and reports, by record class / anomaly family:
  * violation rate and mean S_ont,
  * which rules fire (top firing rule_ids / kinds),
  * false-positive patterns on NORMAL records,
  * anomaly families still not covered,
  * a few example firings (model-visible content only -- NEVER hidden/audit).

Outputs:
  artifacts/phase3b/ontology_rule_coverage_v2_after_rules.json
  artifacts/phase3b/ontology_rule_coverage_v2_after_rules.md

Leakage note: the scorer is fed ONLY model-visible fields
(model_visible_sequence -> codes, gender, age_group). label / anomaly_type are
used here purely to BUCKET the diagnostic output; they are never passed to the
scorer.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_ONT = PROJECT_ROOT / "ontologies" / "processed"
V2_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_v2"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase3b"

ANOMALY_FAMILIES = (
    "demographic_incompatibility",
    "medication_indication_mismatch",
    "forbidden_cooccurrence",
)


def _load(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_pickle(path).to_dict(orient="records")


def _seq(rec: dict[str, Any]) -> list[str]:
    s = rec.get("model_visible_sequence", rec.get("codes", []))
    return [str(t) for t in s] if isinstance(s, (list, tuple)) else []


def _scorer_row(rec: dict[str, Any]) -> dict[str, Any]:
    # ONLY model-visible fields reach the scorer.
    return {
        "codes": _seq(rec),
        "gender": rec.get("gender"),
        "age_group": rec.get("age_group"),
    }


def _bucket(rec: dict[str, Any]) -> str:
    if int(rec.get("label", 0)) == 0:
        return "normal"
    return str(rec.get("anomaly_type") or "anomaly")


def run(split: str, ontology_dir: Path) -> dict[str, Any]:
    from src.ontology.rule_loader import build_rule_manifest
    from src.scoring.ontology_aware import (
        OntologyAwareScorer,
        ScoreWeights,
    )

    records = _load(V2_DIR / f"{split}.pkl")
    scorer = OntologyAwareScorer.from_processed_dir(
        ontology_dir, ontology_mode="real", weights=ScoreWeights()
    )
    manifest = build_rule_manifest(scorer.engine.index)

    n = collections.Counter()
    n_violation = collections.Counter()
    sum_sont = collections.defaultdict(float)
    # rule firing: (bucket, rule_id) -> count
    rule_fires: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for rec in records:
        bucket = _bucket(rec)
        n[bucket] += 1
        res = scorer.score(_scorer_row(rec), s_det=0.0)
        sont = float(res["s_ont"])
        sum_sont[bucket] += sont
        viols = res.get("violations", [])
        if sont > 0 and viols:
            n_violation[bucket] += 1
            for v in viols:
                rid = v.get("rule_id", "?")
                rule_fires[bucket][rid] += 1
                # collect a few model-visible examples per rule_id
                if len(examples[rid]) < 3:
                    examples[rid].append(
                        {
                            "bucket": bucket,
                            "gender": rec.get("gender"),
                            "age_group": rec.get("age_group"),
                            "evidence_codes": v.get("codes", [])[:5],
                            "kind": v.get("kind"),
                            "n_tokens": len(_seq(rec)),
                        }
                    )

    per_class = {}
    for bucket in sorted(n):
        cnt = n[bucket]
        per_class[bucket] = {
            "n": cnt,
            "violation_rate": round(n_violation[bucket] / cnt, 4) if cnt else 0.0,
            "mean_s_ont": round(sum_sont[bucket] / cnt, 4) if cnt else 0.0,
            "rule_fire_counts": dict(rule_fires[bucket].most_common()),
        }

    uncovered = [
        fam
        for fam in ANOMALY_FAMILIES
        if per_class.get(fam, {}).get("violation_rate", 0.0) < 0.05
    ]

    normal_fp = per_class.get("normal", {}).get("violation_rate", 0.0)

    return {
        "split": split,
        "n_records": len(records),
        "rule_manifest": manifest,
        "per_class": per_class,
        "normal_false_positive_rate": normal_fp,
        "anomaly_families_still_uncovered_lt_5pct": uncovered,
        "examples_model_visible_only": {k: v for k, v in examples.items()},
    }


def write_md(result: dict[str, Any], path: Path) -> None:
    pc = result["per_class"]
    md = [
        "# Phase 3b -- Real Ontology Rule Coverage (after rule packs)\n",
        f"**Split:** `{result['split']}`  |  **records:** {result['n_records']}  ",
        f"**Normal false-positive rate:** {result['normal_false_positive_rate']}  ",
        f"**Families still uncovered (<5%):** "
        f"{result['anomaly_families_still_uncovered_lt_5pct'] or 'none'}\n",
        "## Violation rate / mean S_ont by class\n",
        "| class | n | violation_rate | mean_S_ont | rules fired |",
        "|---|---:|---:|---:|---|",
    ]
    for bucket in ["normal", *ANOMALY_FAMILIES]:
        if bucket not in pc:
            continue
        row = pc[bucket]
        fired = ", ".join(f"{k}={v}" for k, v in row["rule_fire_counts"].items()) or "-"
        md.append(
            f"| {bucket} | {row['n']} | {row['violation_rate']} | {row['mean_s_ont']} | {fired} |"
        )
    md.append("\n## Rule manifest\n")
    md.append("| rule_id | type | severity | sizes | limitations |")
    md.append("|---|---|---:|---|---|")
    for m in result["rule_manifest"]:
        sizes = ", ".join(f"{k}={v}" for k, v in m.items() if k.startswith("n_"))
        md.append(
            f"| {m['rule_id']} | {m['rule_type']} | {m['severity']} | {sizes} | "
            f"{m['limitations'][:90]}... |"
        )
    path.write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3b ontology rule-coverage diagnostic."
    )
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--ontology-dir", default=str(PROCESSED_ONT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run(args.split, Path(args.ontology_dir))

    json_path = out_dir / "ontology_rule_coverage_v2_after_rules.json"
    md_path = out_dir / "ontology_rule_coverage_v2_after_rules.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result, md_path)

    print(
        f"[phase3b] split={args.split} normal_fp={result['normal_false_positive_rate']}"
    )
    for bucket in ["normal", *ANOMALY_FAMILIES]:
        if bucket in result["per_class"]:
            row = result["per_class"][bucket]
            print(
                f"[phase3b] {bucket}: viol_rate={row['violation_rate']} "
                f"mean_s_ont={row['mean_s_ont']} rules={row['rule_fire_counts']}"
            )
    print(
        f"[phase3b] uncovered(<5%): {result['anomaly_families_still_uncovered_lt_5pct']}"
    )
    print(f"[phase3b] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
