"""Phase 4 -- LEAKAGE guards for the counterfactual generator (most important).

The generator must produce identical results whether or not the record carries
benchmark answer keys (label / anomaly_type / hidden_eval_metadata /
audit_metadata / bad_code / expected_code / replacement_code). It must read ONLY
model-visible fields.
"""

from __future__ import annotations

from src.explanations.counterfactual import (
    extract_model_visible,
    generate_counterfactual,
)
from src.ontology.engine import OntologyEngine
from src.ontology.index import OntologyIndex
from src.ontology.rule_loader import build_real_ontology_rules
from src.scoring.ontology_aware import OntologyAwareScorer, ScoreWeights

ANSWER_KEYS = {
    "label": 1,
    "anomaly_type": "demographic_incompatibility",
    "hidden_eval_metadata": {
        "original_gender": "F",
        "removed_indication_codes": ["DX_10_E11"],
    },
    "audit_metadata": {"method": "gender_flip", "source_normal_record_id": "xyz"},
    "bad_code": "DX_10_O80",
    "expected_code": "DX_10_N40",
    "replacement_code": "DX_10_E11",
    "injected_token": "DX_10_O80",
    "repair_key": "remove DX_10_O80",
    "source_normal_record_id": "abc123",
}


def _scorer() -> OntologyAwareScorer:
    index = OntologyIndex()
    rules, _ = build_real_ontology_rules(index)
    engine = OntologyEngine(index=index, rules=rules)
    return OntologyAwareScorer(
        ontology_mode="real",
        engine=engine,
        weights=ScoreWeights(),
        require_assets=False,
    )


def _gen(record):
    scorer = _scorer()
    return generate_counterfactual(
        record, scorer, scorer.engine.index, max_edits=3, seed=42
    )


def test_extract_model_visible_ignores_answer_keys() -> None:
    rec = {
        "model_visible_sequence": ["DX_10_O80", "MED_ASPIRIN"],
        "gender": "M",
        "age_group": "30-44",
        **ANSWER_KEYS,
    }
    tokens, gender, age = extract_model_visible(rec)
    assert tokens == ["DX_10_O80", "MED_ASPIRIN"]
    assert gender == "M"
    assert age == "30-44"


def test_answer_keys_do_not_change_repair() -> None:
    clean = {
        "model_visible_sequence": ["DX_10_O80", "DX_10_I10"],
        "gender": "M",
        "age_group": "30-44",
    }
    leaky = {**clean, **ANSWER_KEYS}
    a = _gen(clean).to_dict()
    b = _gen(leaky).to_dict()
    # Identical repair regardless of answer keys.
    assert a["edits"] == b["edits"]
    assert a["status"] == b["status"]
    assert a["s_ont_before"] == b["s_ont_before"]
    assert a["s_ont_after"] == b["s_ont_after"]


def test_repair_works_without_any_hidden_metadata() -> None:
    minimal = {"model_visible_sequence": ["DX_10_O80", "DX_10_I10"], "gender": "M"}
    res = _gen(minimal)
    assert res.validity is True
    assert res.num_edits >= 1
    assert res.delta_s_ont > 0


def test_misleading_answer_keys_are_ignored() -> None:
    # Answer keys point at the WRONG token; the generator must still target the
    # real (ontology-detected) one, proving it ignores the keys.
    rec = {
        "model_visible_sequence": ["DX_10_O80", "DX_10_I10"],
        "gender": "M",
        "bad_code": "DX_10_I10",  # wrong: I10 is fine; O80 is the conflict
        "expected_code": "DX_10_I10",
        "replacement_code": "DX_10_I10",
    }
    res = _gen(rec)
    removed = {e.token_before for e in res.edits if e.operation == "remove"}
    assert "DX_10_O80" in removed  # the real conflict, not the misleading key
    assert "DX_10_I10" not in removed


def test_generator_output_has_no_hidden_metadata_fields() -> None:
    rec = {"model_visible_sequence": ["DX_10_O80"], "gender": "M", **ANSWER_KEYS}
    out = _gen(rec).to_dict()
    flat = str(out).lower()
    for forbidden in (
        "hidden_eval",
        "audit_metadata",
        "bad_code",
        "expected_code",
        "replacement_code",
        "source_normal_record_id",
        "repair_key",
        "injected_token",
    ):
        assert forbidden not in flat


def test_counterfactual_module_does_not_access_answer_key_columns() -> None:
    # Static guard: the rewritten module must not READ any injection answer key
    # (docstrings may *name* them when explaining what is avoided, so we look for
    # actual dict-access / .get() patterns, not bare mentions).
    import pathlib

    import src.explanations.counterfactual as cf

    src = pathlib.Path(cf.__file__).read_text(encoding="utf-8").lower()
    for key in (
        "bad_code",
        "expected_code",
        "replacement_code",
        "hidden_eval_metadata",
        "audit_metadata",
        "injected_code",
        "anomaly_type",
        "repair_key",
        "label",
    ):
        for pattern in (f'["{key}"]', f"['{key}']", f'.get("{key}"', f".get('{key}'"):
            assert pattern not in src, f"counterfactual.py must not access {key}"
