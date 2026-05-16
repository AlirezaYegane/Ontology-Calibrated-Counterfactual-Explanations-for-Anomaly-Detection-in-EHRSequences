from src.explanation.text_generator import build_explanation, build_explanation_batch, summarize_explanations


def test_build_explanation_mentions_scores_and_action() -> None:
    row = {
        "record_id": "case_001",
        "anomaly_type": "medication_mismatch",
        "s_det": 0.8,
        "s_gen": 0.1,
        "s_ont": 1.0,
        "scal_before": 0.9,
        "scal_after": 0.2,
        "violations": ["medication appears without compatible indication"],
        "action": ["add compatible diagnosis"],
        "edit_count": 1,
    }

    explanation = build_explanation(row)

    assert explanation["record_id"] == "case_001"
    assert explanation["delta_scal"] > 0
    assert "add compatible diagnosis" in explanation["explanation_short"]
    assert "Sdet=" in explanation["explanation_research"]
    assert "diagnostic auxiliary" in explanation["explanation_research"]


def test_batch_summary_counts_cases() -> None:
    rows = [
        {
            "record_id": "a",
            "anomaly_type": "demographic_conflict",
            "s_det": 0.9,
            "s_ont": 1.0,
            "scal_before": 0.9,
            "scal_after": 0.1,
            "action": ["remove incompatible code"],
            "edit_count": 1,
        },
        {
            "record_id": "b",
            "anomaly_type": "missing_diagnosis",
            "s_det": 0.4,
            "s_ont": 0.3,
            "scal_before": 0.4,
            "scal_after": 0.35,
            "action": ["add expected diagnosis"],
            "edit_count": 1,
        },
    ]

    explanations = build_explanation_batch(rows)
    summary = summarize_explanations(explanations)

    assert summary["n_cases"] == 2
    assert summary["pct_positive_reduction"] == 1.0
    assert summary["pct_one_or_two_edits"] == 1.0
