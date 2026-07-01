"""
scripts/run_phase7_ablations.py
===============================
Phase 7 -- Final ablation suite on benchmark-v2.

1. Ontology-rule ablations: decompose S_ont by rule family in a SINGLE real pass
   (each rule contributes its own violation severities), plus a legacy pass and an
   "ontology_disabled" reference. Reports ROC-AUC / AP / CI, per-anomaly-family
   ROC-AUC, normal false-positive rate, and rule-firing rates.
2. Score-component ablations: S_ont only / S_det only / S_ont+S_det. Sgen stays
   EXCLUDED (referenced as a Phase-5 diagnostic row, never in the core).
3. Anomaly-family breakdown for the main variant.

Only aggregate outputs are written.

  python scripts/run_phase7_ablations.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_ONT = PROJECT_ROOT / "ontologies" / "processed"
OUT = PROJECT_ROOT / "artifacts" / "phase7"
DEFAULT_CONFIG = "configs/phase6_detector_full.yaml"
DEFAULT_RUN = "phase6_detector_full_gpu"

RULE_FAMILIES = {
    "demographic_rules_only": "sex_restricted_concepts",
    "medication_rules_only": "medication_required_context",
    "forbidden_cooccurrence_rules_only": "diabetes_type_exclusion",
}
ANOMALY_FAMILIES = (
    "demographic_incompatibility",
    "medication_indication_mismatch",
    "forbidden_cooccurrence",
)


def _auc_ap_ci(y, scores, seed):
    import numpy as np

    from src.evaluation.stats import average_precision, bootstrap_auc_ap, roc_auc

    s = np.array(scores, dtype=float)
    if len(set(scores)) <= 1:
        return {
            "roc_auc": None,
            "roc_auc_ci": [None, None],
            "average_precision": round(float(np.mean(y)), 4),
            "note": "degenerate (all-equal scores)",
        }
    boot = bootstrap_auc_ap(y, s, n_boot=1000, seed=seed)
    return {
        "roc_auc": round(roc_auc(y, s), 4),
        "roc_auc_ci": [
            round(boot["roc_auc"]["ci_low"], 4),
            round(boot["roc_auc"]["ci_high"], 4),
        ],
        "average_precision": round(average_precision(y, s), 4),
    }


def run(config_path: str = DEFAULT_CONFIG, run_id: str = DEFAULT_RUN) -> dict[str, Any]:
    import numpy as np

    from src.experiments.config import ExperimentConfig
    from src.experiments.eval_common import (
        combined_scores,
        detector_scores,
        load_records,
        minmax_apply,
        minmax_fit,
    )
    from src.models.detector_unsup import UnsupervisedSequenceDetector
    from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

    cfg = ExperimentConfig.from_file(config_path)
    real = OntologyAwareScorer.from_processed_dir(
        PROCESSED_ONT, ontology_mode="real", weights=ScoreWeights()
    )
    legacy = OntologyAwareScorer(ontology_mode="legacy", weights=ScoreWeights())
    test = load_records(cfg.split_path("test"), cfg.sequence_key, cfg.label_key)
    y = np.array([r["label"] for r in test])

    # single real pass: capture per-record severity by rule family
    fam_scores = {k: [] for k in ["real_ontology_rules_full", *RULE_FAMILIES]}
    firing = {k: 0 for k in RULE_FAMILIES}
    normal_fired = {k: 0 for k in ["real_ontology_rules_full", *RULE_FAMILIES]}
    n_normal = int((y == 0).sum())
    for r in test:
        row = {"codes": r["seq"], "gender": r["gender"], "age_group": r["age_group"]}
        res = real.score(row, s_det=0.0)
        by_rule: dict[str, float] = {}
        for v in res.get("violations", []):
            by_rule[v.get("rule_id", "?")] = by_rule.get(
                v.get("rule_id", "?"), 0.0
            ) + float(v.get("severity", 0.0))
        full = float(res["s_ont"])
        fam_scores["real_ontology_rules_full"].append(full)
        if full > 0 and r["label"] == 0:
            normal_fired["real_ontology_rules_full"] += 1
        for fam, rid in RULE_FAMILIES.items():
            sc = by_rule.get(rid, 0.0)
            fam_scores[fam].append(sc)
            if sc > 0:
                firing[fam] += 1
                if r["label"] == 0:
                    normal_fired[fam] += 1

    legacy_scores = [
        float(
            legacy.score(
                {"codes": r["seq"], "gender": r["gender"], "age_group": r["age_group"]},
                s_det=0.0,
            )["s_ont"]
        )
        for r in test
    ]

    # ---- ontology-rule ablation table ----
    onto_rows = []
    variants = {
        "real_ontology_rules_full": fam_scores["real_ontology_rules_full"],
        "legacy_icd_prefix_rules": legacy_scores,
        **{k: fam_scores[k] for k in RULE_FAMILIES},
        "ontology_disabled": [0.0] * len(test),
    }
    for name, scores in variants.items():
        m = _auc_ap_ci(y, scores, cfg.seed)
        nfired = normal_fired.get(name)
        row = {
            "variant": name,
            **m,
            "normal_fp_rate": (
                round(nfired / n_normal, 4) if nfired is not None and n_normal else None
            ),
            "n_records_fired": int(sum(1 for s in scores if s > 0)),
        }
        # per-anomaly-family AUC
        fam_auc = {}
        for fam in ANOMALY_FAMILIES:
            idx = [
                i
                for i, r in enumerate(test)
                if r["label"] == 0 or r["anomaly_type"] == fam
            ]
            yy = y[idx]
            ss = [scores[i] for i in idx]
            if len(set(int(v) for v in yy)) == 2 and len(set(ss)) > 1:
                from src.evaluation.stats import roc_auc as _ra

                fam_auc[fam] = round(_ra(yy, np.array(ss, dtype=float)), 4)
            else:
                fam_auc[fam] = None
        row["per_anomaly_family_roc_auc"] = fam_auc
        onto_rows.append(row)

    # ---- score-component ablation (reuse detector) ----
    ckpt = PROJECT_ROOT / cfg.output_dir / run_id / "checkpoints"
    component_rows = []
    if (ckpt / "detector_unsup.pt").exists():
        val = load_records(cfg.split_path("val"), cfg.sequence_key, cfg.label_key)
        det = UnsupervisedSequenceDetector.load(ckpt, device=cfg.resolved_device())
        val_det = detector_scores(det, val, cfg.batch_size)
        test_det = detector_scores(det, test, cfg.batch_size)
        lo, hi = minmax_fit(val_det)
        test_det_n = minmax_apply(test_det, lo, hi)
        comb = combined_scores(
            test_det_n, fam_scores["real_ontology_rules_full"], cfg.scoring_weights
        )
        for name, scores in [
            ("S_ont_only", fam_scores["real_ontology_rules_full"]),
            ("S_det_only", test_det_n),
            ("S_ont_plus_S_det", comb),
        ]:
            component_rows.append({"variant": name, **_auc_ap_ci(y, scores, cfg.seed)})
    component_rows.append(
        {
            "variant": "S_ont_plus_S_det_plus_Sgen",
            "roc_auc": None,
            "status": "EXCLUDED_FROM_CORE",
            "note": "Sgen removed in Phase 5 (diagnostic ROC-AUC 0.4868, harms combined). w_gen=0.",
        }
    )

    result = {
        "phase": 7,
        "benchmark": "benchmark-v2",
        "n_test": len(test),
        "ontology_rule_ablation": onto_rows,
        "score_component_ablation": component_rows,
        "rule_firing_rates_over_test": {
            k: round(v / len(test), 4) for k, v in firing.items()
        },
        "counterfactual_ablation_ref": "artifacts/phase7/counterfactual_final.json (edit-strategy ablation)",
        "note": "Sgen excluded from all core variants (w_gen=0).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_md(result, OUT / "ablation_results.md")
    _write_tables(result, OUT)

    print(f"[phase7][ablation] n_test={len(test)}")
    for r in onto_rows:
        print(
            f"[phase7][ablation]   {r['variant']}: ROC-AUC={r.get('roc_auc')} AP={r.get('average_precision')} normal_fp={r.get('normal_fp_rate')}"
        )
    return result


def _write_md(r: dict[str, Any], path: Path) -> None:
    md = [
        "# Phase 7 -- Ablation Results (benchmark-v2)\n",
        "## 1. Ontology-rule ablation\n",
        "| variant | ROC-AUC | 95% CI | AP | normal FP | demo | med | forbidden |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for v in r["ontology_rule_ablation"]:
        fa = v.get("per_anomaly_family_roc_auc", {})
        md.append(
            f"| {v['variant']} | {v.get('roc_auc')} | {v.get('roc_auc_ci')} | {v.get('average_precision')} | {v.get('normal_fp_rate')} | {fa.get('demographic_incompatibility')} | {fa.get('medication_indication_mismatch')} | {fa.get('forbidden_cooccurrence')} |"
        )
    md.append("\n## 2. Score-component ablation (Sgen excluded)\n")
    md.append("| variant | ROC-AUC | AP | note |")
    md.append("|---|---:|---:|---|")
    for v in r["score_component_ablation"]:
        md.append(
            f"| {v['variant']} | {v.get('roc_auc')} | {v.get('average_precision', '-')} | {v.get('note', v.get('status', ''))} |"
        )
    md.append(f"\n> {r['note']}")
    path.write_text("\n".join(md), encoding="utf-8")


def _write_tables(r: dict[str, Any], out: Path) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    with (out / "rule_family_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "variant",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "average_precision",
                "normal_fp_rate",
                "n_records_fired",
            ]
        )
        for v in r["ontology_rule_ablation"]:
            ci = v.get("roc_auc_ci", [None, None])
            w.writerow(
                [
                    v["variant"],
                    v.get("roc_auc"),
                    ci[0],
                    ci[1],
                    v.get("average_precision"),
                    v.get("normal_fp_rate"),
                    v.get("n_records_fired"),
                ]
            )
    with (out / "anomaly_family_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["ontology_variant", "demographic", "medication", "forbidden"])
        for v in r["ontology_rule_ablation"]:
            fa = v.get("per_anomaly_family_roc_auc", {})
            w.writerow(
                [
                    v["variant"],
                    fa.get("demographic_incompatibility"),
                    fa.get("medication_indication_mismatch"),
                    fa.get("forbidden_cooccurrence"),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 ablations.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--run-id", default=DEFAULT_RUN)
    args = ap.parse_args(argv)
    run(args.config, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
