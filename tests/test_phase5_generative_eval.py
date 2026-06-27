"""Phase 5 -- tests for the generative eval harness (blocked path + leakage)."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_phase5_generative_eval as ev


def test_load_returns_none_when_checkpoint_absent(tmp_path: Path) -> None:
    model, vocab, status = ev.load_diffusion_checkpoint(
        tmp_path / "nope", tmp_path / "v.json"
    )
    assert model is None and vocab is None
    assert "unavailable" in status.lower()


def test_blocked_report_when_model_unavailable(tmp_path: Path, monkeypatch) -> None:
    # Point the checkpoint dir at a nonexistent path -> structured BLOCKED report,
    # NOT fabricated metrics. (Returns before touching benchmark data / ontology.)
    monkeypatch.setattr(ev, "CKPT_DIR", tmp_path / "missing_ckpt")
    monkeypatch.setattr(ev, "VOCAB_PATH", tmp_path / "missing_vocab.json")
    report = ev.run(split="test", max_records=5, seed=42, n_noise=1, out_dir=tmp_path)

    assert report["status"] == "blocked_no_valid_generative_model"
    assert report["sgen_roc_auc"] is None
    assert report["final_paper_evidence_claimable"] is False
    assert (tmp_path / "generative_eval.json").exists()

    decision = json.loads((tmp_path / "sgen_decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "blocked_no_valid_model"
    assert decision["w_gen_default"] == 0.0
    assert decision["sgen_in_core_score"] is False


def test_sgen_input_uses_only_model_visible_sequence() -> None:
    # The diffusion input is built from model_visible_sequence only; hidden/audit
    # metadata on the record is ignored.
    rec = {
        "model_visible_sequence": ["DX_10_O80", "MED_ASPIRIN"],
        "gender": "M",
        "label": 1,
        "anomaly_type": "demographic_incompatibility",
        "hidden_eval_metadata": {"original_gender": "F"},
        "audit_metadata": {"method": "gender_flip"},
    }
    assert ev._seq(rec) == ["DX_10_O80", "MED_ASPIRIN"]


def test_blocked_score_table_has_no_fabricated_rows(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ev, "CKPT_DIR", tmp_path / "missing")
    monkeypatch.setattr(ev, "VOCAB_PATH", tmp_path / "missing.json")
    ev.run(split="test", max_records=5, seed=42, n_noise=1, out_dir=tmp_path)
    rows = (
        (tmp_path / "generative_score_table.csv")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert rows == ["variant,roc_auc"]  # header only, no fake numbers
