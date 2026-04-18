from src.ontology.rules import compute_s_ont
from src.scoring.ontology_aware import combine_scores


def test_missing_diagnosis_for_medication():
    record = {
        "sequence_tokens": ["MED_12345", "RXNORM_999"],
        "gender": "male",
    }
    result = compute_s_ont(record)
    names = {v["name"] for v in result["violations"]}

    assert result["sont"] > 0
    assert "missing_diagnosis_for_medication" in names


def test_male_female_specific_conflict():
    record = {
        "sequence_tokens": ["DX_10_Z3400", "MED_123"],
        "gender": "male",
    }
    result = compute_s_ont(record)
    names = {v["name"] for v in result["violations"]}

    assert result["sont"] > 0
    assert "male_female_specific_conflict" in names


def test_combined_score_monotonicity():
    base = combine_scores(sdet=0.40, sont=0.0)
    raised = combine_scores(sdet=0.40, sont=2.0)

    assert raised > base
