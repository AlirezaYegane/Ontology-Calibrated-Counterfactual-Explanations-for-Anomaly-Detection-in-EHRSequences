"""
scripts/run_phase5_generative_eval.py
=====================================
Phase 5 -- Generative / diffusion Sgen evaluation on benchmark-v2.

Loads the existing diffusion checkpoint via a minimal compatibility shim (key
remap + skip the architecture-drifted time-embedding MLP; see
artifacts/phase5/generative_audit_before.md), computes the per-record Sgen
(midpoint denoising surprise) over benchmark-v2, and reports whether Sgen carries
any anomaly signal and whether it improves the combined score.

The checkpoint was trained on OLD circular-era data and is mode-collapsed, so ALL
results here are DIAGNOSTIC-ONLY (never paper SOTA). If no usable checkpoint
exists, a structured BLOCKED report is written instead of fabricated numbers.

Leakage note: label/anomaly_type are used ONLY to score/bucket the diagnostic;
the generative surprise is unsupervised and never sees labels.

Outputs: artifacts/phase5/generative_eval.{json,md}, generative_score_table.csv,
failure_cases.jsonl, sgen_decision.{json,md}, per_record_scores.csv (gitignored).
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
V2_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark_v2"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase5"
DETECTOR_DIR = PROJECT_ROOT / "artifacts" / "phase3" / "detector_unsup_v2"
CKPT_DIR = PROJECT_ROOT / "outputs" / "diffusion" / "day33_ontology_regularized_fixed"
VOCAB_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "detector"
    / "day20_supervised"
    / "run_luxury"
    / "vocab.json"
)

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


# ---------------------------------------------------------------------------
# Minimal compatibility-shim loader (Phase 5 safe repair: load only; no retrain)
# ---------------------------------------------------------------------------


def _remap_key(k: str) -> str:
    if k.startswith("encoder."):
        return "denoiser." + k[len("encoder.") :]
    if k.startswith("pos_embedding."):
        return "position_embedding." + k[len("pos_embedding.") :]
    if k.startswith("norm."):
        return "output_head.0." + k[len("norm.") :]
    if k.startswith("out."):
        return "output_head.1." + k[len("out.") :]
    return k


def load_diffusion_checkpoint(
    ckpt_dir: Path, vocab_path: Path
) -> tuple[Any, dict[str, int], str]:
    """Return (model, vocab, status). model/vocab are None when unavailable."""
    ckpt = ckpt_dir / "last.pt"
    cfg_path = ckpt_dir / "config.json"
    if not ckpt.exists() or not cfg_path.exists() or not vocab_path.exists():
        return None, None, "unavailable (missing checkpoint/config/vocab)"
    import torch

    from src.models.diffusion import DiffusionModel

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    state = torch.load(ckpt, map_location="cpu")["model_state"]
    # remap renamed modules; drop the architecture-drifted time-embedding MLP.
    remapped = {
        _remap_key(k): v
        for k, v in state.items()
        if not k.startswith("time_embedding.")
    }
    model = DiffusionModel(
        vocab_size=cfg["vocab_size"],
        max_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ff_dim=cfg["ff_dim"],
        num_diffusion_steps=cfg["diffusion_steps"],
        pad_idx=cfg["pad_idx"],
    )
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    learned_missing = [
        k
        for k in missing
        if not any(b in k for b in ("betas", "alpha", "sqrt"))
        and "time_embedding" not in k
    ]
    model.eval()
    status = (
        "loaded_via_compat_shim (real token/transformer/output weights; time-MLP "
        f"NOT loaded due to arch drift; learned_missing={len(learned_missing)}, "
        f"unexpected={len(list(unexpected))}). DIAGNOSTIC-ONLY: old-data, mode-collapsed."
    )
    return model, vocab, status


def compute_sgen(
    model: Any,
    vocab: dict[str, int],
    records: list[dict[str, Any]],
    seq_len: int,
    n_noise: int,
    seed: int,
) -> list[float]:
    import torch

    def enc(seq: list[str]) -> list[int]:
        ids = [vocab.get(str(t), 1) for t in seq][:seq_len]
        return ids + [0] * (seq_len - len(ids))

    torch.manual_seed(seed)
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(records), 128):
            batch = records[i : i + 128]
            x = torch.tensor([enc(_seq(r)) for r in batch], dtype=torch.long)
            mask = (x != 0).long()
            acc = torch.zeros(len(batch))
            for _ in range(n_noise):  # average noise draws for stability
                acc += model.surprise_score(x, attention_mask=mask)
            out.extend((acc / n_noise).tolist())
    return out


def _maybe_detector():
    if not (DETECTOR_DIR / "detector_unsup.pt").exists():
        return None
    try:
        from src.models.detector_unsup import UnsupervisedSequenceDetector

        return UnsupervisedSequenceDetector.load(DETECTOR_DIR)
    except Exception:
        return None


def _minmax(values, lo=None, hi=None):
    import numpy as np

    v = np.asarray(values, dtype=float)
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    if hi - lo <= 0:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def run(
    split: str, max_records: int, seed: int, n_noise: int, out_dir: Path
) -> dict[str, Any]:
    import numpy as np

    from src.evaluation.generative_gate import GenerativeGateInputs, decide_sgen_gate
    from src.evaluation.stats import average_precision, bootstrap_auc_ap, roc_auc
    from src.scoring.ontology_aware import (
        OntologyAwareScorer,
        ScoreWeights,
        compute_calibrated_score,
        normalize_sont,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model, vocab, ckpt_status = load_diffusion_checkpoint(CKPT_DIR, VOCAB_PATH)

    if model is None:
        report = {
            "phase": 5,
            "status": "blocked_no_valid_generative_model",
            "checkpoint_status": ckpt_status,
            "reason": "No usable diffusion checkpoint/vocab; cannot evaluate Sgen without large retraining.",
            "sgen_roc_auc": None,
            "final_paper_evidence_claimable": False,
        }
        _write_blocked(report, out_dir)
        decision = decide_sgen_gate(GenerativeGateInputs(model_available=False))
        _write_decision(decision.to_dict(), report, out_dir)
        return report

    records = _load(V2_DIR / f"{split}.pkl")
    if max_records and len(records) > max_records:
        records = records[:max_records]
    labels = np.array([int(r.get("label", 0)) for r in records])
    types = [str(r.get("anomaly_type") or "normal") for r in records]

    # --- three scores per record ---
    s_gen = np.array(compute_sgen(model, vocab, records, CKPT_seq_len(), n_noise, seed))

    scorer = OntologyAwareScorer.from_processed_dir(
        PROCESSED_ONT, ontology_mode="real", weights=ScoreWeights()
    )
    s_ont = np.array(
        [
            float(
                scorer.score(
                    {
                        "codes": _seq(r),
                        "gender": r.get("gender"),
                        "age_group": r.get("age_group"),
                    },
                    s_det=0.0,
                )["s_ont"]
            )
            for r in records
        ]
    )
    detector = _maybe_detector()
    if detector is not None:
        raw_det = detector.anomaly_scores([_seq(r) for r in records])
        s_det = _minmax(raw_det)
    else:
        s_det = np.zeros(len(records))

    s_gen_n = _minmax(s_gen)
    s_ont_n = np.array([normalize_sont(x) for x in s_ont])

    # --- Sgen metrics ---
    boot = bootstrap_auc_ap(labels, list(map(float, s_gen)), n_boot=500, seed=seed)
    sgen_auc = roc_auc(labels, s_gen)
    sgen_ci = (
        round(boot["roc_auc"]["ci_low"], 4),
        round(boot["roc_auc"]["ci_high"], 4),
    )

    # --- combined with / without Sgen (w_gen=0.3 test weight for the 'with' variant) ---
    w_no = ScoreWeights()  # w_gen=0
    w_yes = ScoreWeights(w_det=0.7, w_ont=0.3, w_gen=0.3)
    comb_no = np.array(
        [compute_calibrated_score(d, o, weights=w_no) for d, o in zip(s_det, s_ont)]
    )
    comb_yes = np.array(
        [
            compute_calibrated_score(d, o, g, weights=w_yes)
            for d, o, g in zip(s_det, s_ont, s_gen_n)
        ]
    )
    auc_comb_no = roc_auc(labels, comb_no)
    auc_comb_yes = roc_auc(labels, comb_yes)
    auc_ont = roc_auc(labels, s_ont_n)
    auc_det = roc_auc(labels, s_det)

    # --- correlations + per-family ---
    def _corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    per_family = {}
    for fam in ANOMALY_FAMILIES:
        idx = np.array([t == fam for t in types])
        mask = idx | (labels == 0)
        y = labels[mask]
        if y.sum() > 0 and (y == 0).sum() > 0:
            per_family[fam] = {
                "n_anom": int(idx.sum()),
                "sgen_roc_auc": round(roc_auc(y, s_gen[mask]), 4),
            }

    adds_signal = bool(auc_comb_yes > auc_comb_no + 0.005) and bool(sgen_auc > 0.55)
    combined_improves = bool(auc_comb_yes > auc_comb_no)

    # --- save per-record scores (gitignored) for the ablation ---
    with (out_dir / "per_record_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(
            ["idx", "label", "anomaly_type", "s_gen", "s_gen_norm", "s_ont", "s_det"]
        )
        for i in range(len(records)):
            w.writerow(
                [
                    i,
                    int(labels[i]),
                    types[i],
                    round(float(s_gen[i]), 6),
                    round(float(s_gen_n[i]), 6),
                    round(float(s_ont[i]), 6),
                    round(float(s_det[i]), 6),
                ]
            )

    report = {
        "phase": 5,
        "status": "evaluated_diagnostic_only",
        "split": split,
        "n": int(len(records)),
        "n_anomaly": int(labels.sum()),
        "checkpoint_status": ckpt_status,
        "evidence_level": "diagnostic_only_old_data_modecollapsed",
        "final_paper_evidence_claimable": False,
        "sgen_roc_auc": round(float(sgen_auc), 4),
        "sgen_roc_auc_ci": list(sgen_ci),
        "sgen_average_precision": round(float(average_precision(labels, s_gen)), 4),
        "sgen_score_std": round(float(np.std(s_gen)), 4),
        "sgen_mean_normal": round(float(s_gen[labels == 0].mean()), 4),
        "sgen_mean_anomaly": round(float(s_gen[labels == 1].mean()), 4),
        "sgen_per_family": per_family,
        "corr_sgen_sont": _corr(s_gen, s_ont),
        "corr_sgen_sdet": _corr(s_gen, s_det),
        "auc_detector_only": round(float(auc_det), 4),
        "auc_ontology_only_real": round(float(auc_ont), 4),
        "auc_combined_without_sgen": round(float(auc_comb_no), 4),
        "auc_combined_with_sgen": round(float(auc_comb_yes), 4),
        "sgen_adds_signal_beyond_ont_det": adds_signal,
        "combined_with_sgen_improves": combined_improves,
        "warnings": [
            "DIAGNOSTIC-ONLY: checkpoint trained on OLD circular-era data + mode-collapsed; "
            "time-embedding MLP not loaded (architecture drift). NOT paper evidence.",
        ],
    }

    gate = decide_sgen_gate(
        GenerativeGateInputs(
            model_available=True,
            sgen_roc_auc=float(sgen_auc),
            sgen_roc_auc_ci=sgen_ci,
            adds_signal_beyond_ont_det=adds_signal,
            combined_with_sgen_improves=combined_improves,
            leakage_detected=False,
            protocol_valid=False,  # old-data checkpoint
            mode_collapse=True,  # documented in the audit
        )
    )
    report["gate_decision"] = gate.decision

    _write_eval(report, out_dir)
    _write_decision(gate.to_dict(), report, out_dir)
    return report


def CKPT_seq_len() -> int:
    cfg = json.loads((CKPT_DIR / "config.json").read_text(encoding="utf-8"))
    return int(cfg.get("seq_len", 256))


def _write_eval(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "generative_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = [
        "# Phase 5 -- Generative / Diffusion Sgen Evaluation (benchmark-v2)\n",
        f"**Status:** `{report['status']}` | **gate:** `{report.get('gate_decision')}` | "
        f"evidence: {report['evidence_level']}\n",
        f"> {report['warnings'][0]}\n",
        f"- Sgen ROC-AUC **{report['sgen_roc_auc']}** CI {report['sgen_roc_auc_ci']} | "
        f"AP {report['sgen_average_precision']} | score std {report['sgen_score_std']}",
        f"- mean Sgen normal {report['sgen_mean_normal']} vs anomaly {report['sgen_mean_anomaly']}",
        f"- corr(Sgen,S_ont) {report['corr_sgen_sont']} | corr(Sgen,S_det) {report['corr_sgen_sdet']}",
        f"- AUC detector_only {report['auc_detector_only']} | ontology_only_real {report['auc_ontology_only_real']}",
        f"- AUC combined **without** Sgen {report['auc_combined_without_sgen']} | "
        f"**with** Sgen {report['auc_combined_with_sgen']}",
        f"- Sgen adds signal beyond ont+det: **{report['sgen_adds_signal_beyond_ont_det']}** | "
        f"combined improves: **{report['combined_with_sgen_improves']}**\n",
        "## Sgen ROC-AUC by anomaly family",
        "| family | n | Sgen ROC-AUC |",
        "|---|---:|---:|",
    ]
    for fam, v in report["sgen_per_family"].items():
        md.append(f"| {fam} | {v['n_anom']} | {v['sgen_roc_auc']} |")
    (out_dir / "generative_eval.md").write_text("\n".join(md), encoding="utf-8")
    with (out_dir / "generative_score_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "roc_auc"])
        w.writerow(["sgen_only", report["sgen_roc_auc"]])
        w.writerow(["detector_only", report["auc_detector_only"]])
        w.writerow(["ontology_only_real", report["auc_ontology_only_real"]])
        w.writerow(["combined_without_sgen", report["auc_combined_without_sgen"]])
        w.writerow(["combined_with_sgen", report["auc_combined_with_sgen"]])
    (out_dir / "failure_cases.jsonl").write_text(
        json.dumps(
            {
                "failure": "sgen_below_or_near_chance_modecollapsed",
                "sgen_roc_auc": report["sgen_roc_auc"],
                "mean_normal": report["sgen_mean_normal"],
                "mean_anomaly": report["sgen_mean_anomaly"],
                "note": "anomalies do not have higher surprise than normals; mode-collapsed model.",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_blocked(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "generative_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "generative_eval.md").write_text(
        f"# Phase 5 -- Generative Eval: BLOCKED\n\n`{report['status']}`\n\n{report['reason']}\n",
        encoding="utf-8",
    )
    with (out_dir / "generative_score_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        csv.writer(fh).writerow(["variant", "roc_auc"])
    (out_dir / "failure_cases.jsonl").write_text(
        json.dumps({"failure": "no_valid_generative_model"}) + "\n", encoding="utf-8"
    )


def _write_decision(
    decision: dict[str, Any], report: dict[str, Any], out_dir: Path
) -> None:
    payload = {
        "phase": 5,
        "decision": decision["decision"],
        "reasons": decision["reasons"],
        "criteria": decision["criteria"],
        "evidence": {
            "sgen_roc_auc": report.get("sgen_roc_auc"),
            "sgen_roc_auc_ci": report.get("sgen_roc_auc_ci"),
            "auc_combined_without_sgen": report.get("auc_combined_without_sgen"),
            "auc_combined_with_sgen": report.get("auc_combined_with_sgen"),
        },
        "w_gen_default": 0.0,
        "sgen_in_core_score": decision["decision"] == "keep_main",
    }
    (out_dir / "sgen_decision.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (out_dir / "sgen_decision.md").write_text(
        "# Phase 5 -- Sgen Decision\n\n"
        f"## Decision: `{payload['decision']}`\n\n"
        + "\n".join(f"- {r}" for r in payload["reasons"])
        + f"\n\n`w_gen` default stays **0.0**; Sgen in core score: "
        f"**{payload['sgen_in_core_score']}**.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 5 generative Sgen evaluation.")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--max-records", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-noise", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    max_records = min(args.max_records, 300) if args.smoke else args.max_records
    report = run(args.split, max_records, args.seed, args.n_noise, Path(args.out_dir))
    print(
        f"[phase5] status={report['status']} gate={report.get('gate_decision', 'blocked')}"
    )
    if report.get("sgen_roc_auc") is not None:
        print(
            f"[phase5] Sgen ROC-AUC={report['sgen_roc_auc']} CI={report['sgen_roc_auc_ci']} "
            f"combined_without={report['auc_combined_without_sgen']} "
            f"combined_with={report['auc_combined_with_sgen']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
