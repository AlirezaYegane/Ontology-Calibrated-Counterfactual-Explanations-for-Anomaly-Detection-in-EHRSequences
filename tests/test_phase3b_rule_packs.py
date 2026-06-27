"""Phase 3b -- tests for the curated ontology rule packs.

These exercise the rule CLASSES directly with hand-built groups + synthetic
records (no real ontology files needed), plus the loader-built rule set on a
small constructed index.
"""

from __future__ import annotations

from src.ontology.index import OntologyIndex
from src.ontology.records import ClinicalRecord
from src.ontology.rule_loader import build_real_ontology_rules
from src.ontology.rule_packs import (
    GroupConflict,
    GroupMutualExclusionRule,
    RequiredContextRule,
    SexRestrictedRule,
    build_concept_group,
)

EMPTY = OntologyIndex()  # the rule check() paths below do not call index methods


# --------------------------------------------------------------------------
# SexRestrictedRule
# --------------------------------------------------------------------------


def test_sex_rule_fires_on_source_token_family() -> None:
    rule = SexRestrictedRule(
        rule_id="sex",
        sex_to_forbidden_icd_prefixes={"M": ("O", "Z3A"), "F": ("N40", "C61")},
    )
    male_preg = ClinicalRecord(
        codes=(), sex="M", source_tokens=("DX_10_O80", "DX_9_4019")
    )
    v = rule.check(male_preg, EMPTY)
    assert len(v) == 1
    assert v[0].kind == "demographic_mismatch"
    assert v[0].severity == 1.0


def test_sex_rule_does_not_fire_on_compatible_sex() -> None:
    rule = SexRestrictedRule(
        rule_id="sex",
        sex_to_forbidden_icd_prefixes={"M": ("O",), "F": ("N40",)},
    )
    # Female with a pregnancy code is NOT an anomaly.
    female_preg = ClinicalRecord(codes=(), sex="F", source_tokens=("DX_10_O80",))
    assert rule.check(female_preg, EMPTY) == []
    # Male with a prostate code IS flagged (prostate forbidden for... female);
    # male with prostate is fine.
    male_prostate = ClinicalRecord(codes=(), sex="M", source_tokens=("DX_10_N40",))
    assert rule.check(male_prostate, EMPTY) == []


def test_sex_rule_fires_on_clean_snomed_subtree() -> None:
    rule = SexRestrictedRule(
        rule_id="sex", sex_to_forbidden_group={"M": {"SNOMED:72892002"}}
    )
    male = ClinicalRecord(codes=("SNOMED:72892002",), sex="M")
    assert rule.check(male, EMPTY)[0].kind == "demographic_mismatch"
    female = ClinicalRecord(codes=("SNOMED:72892002",), sex="F")
    assert rule.check(female, EMPTY) == []


def test_sex_rule_unknown_sex_is_silent() -> None:
    rule = SexRestrictedRule(rule_id="sex", sex_to_forbidden_icd_prefixes={"M": ("O",)})
    rec = ClinicalRecord(codes=(), sex=None, source_tokens=("DX_10_O80",))
    assert rule.check(rec, EMPTY) == []


# --------------------------------------------------------------------------
# RequiredContextRule
# --------------------------------------------------------------------------


def test_required_context_fires_when_context_absent() -> None:
    rule = RequiredContextRule(
        rule_id="med",
        drug_to_context={"RXNORM:11289": frozenset({"SNOMED:49436004"})},
    )
    rec = ClinicalRecord(codes=("RXNORM:11289", "SNOMED:38341003"), sex="F")
    v = rule.check(rec, EMPTY)
    assert len(v) == 1
    assert v[0].kind == "missing_required_code"
    assert v[0].severity == 0.5


def test_required_context_satisfied_when_context_present() -> None:
    rule = RequiredContextRule(
        rule_id="med",
        drug_to_context={"RXNORM:11289": frozenset({"SNOMED:49436004"})},
    )
    rec = ClinicalRecord(codes=("RXNORM:11289", "SNOMED:49436004"), sex="F")
    assert rule.check(rec, EMPTY) == []


def test_required_context_insulin_by_name_unmapped_drug() -> None:
    rule = RequiredContextRule(
        rule_id="med",
        drug_name_to_context={"INSULIN": frozenset({"SNOMED:73211009"})},
    )
    # insulin present, no diabetes context -> fires
    no_dx = ClinicalRecord(
        codes=(), sex="F", source_tokens=("MED_INSULIN_GLARGINE", "DX_10_I10")
    )
    assert rule.check(no_dx, EMPTY)[0].kind == "missing_required_code"
    # insulin present WITH diabetes context -> satisfied
    with_dx = ClinicalRecord(
        codes=("SNOMED:73211009",), sex="F", source_tokens=("MED_INSULIN",)
    )
    assert rule.check(with_dx, EMPTY) == []


# --------------------------------------------------------------------------
# GroupMutualExclusionRule
# --------------------------------------------------------------------------


def test_group_mutual_exclusion_fires_on_source_token_families() -> None:
    rule = GroupMutualExclusionRule(
        rule_id="dm",
        groups=[
            GroupConflict(
                label="t1_vs_t2",
                left=set(),
                right=set(),
                left_icd_prefixes=("E10",),
                right_icd_prefixes=("E11",),
            )
        ],
    )
    both = ClinicalRecord(
        codes=(), sex="M", source_tokens=("DX_10_E119", "DX_10_E10_9")
    )
    v = rule.check(both, EMPTY)
    assert len(v) == 1
    assert v[0].kind == "mutual_exclusion"
    only_t2 = ClinicalRecord(codes=(), sex="M", source_tokens=("DX_10_E119",))
    assert rule.check(only_t2, EMPTY) == []


def test_group_mutual_exclusion_fires_on_concept_groups() -> None:
    rule = GroupMutualExclusionRule(
        rule_id="dm",
        groups=[GroupConflict(label="t1_vs_t2", left={"SNOMED:A"}, right={"SNOMED:B"})],
    )
    both = ClinicalRecord(codes=("SNOMED:A", "SNOMED:B"), sex="F")
    assert rule.check(both, EMPTY)[0].kind == "mutual_exclusion"
    one = ClinicalRecord(codes=("SNOMED:A",), sex="F")
    assert rule.check(one, EMPTY) == []


# --------------------------------------------------------------------------
# build_concept_group + loader manifest
# --------------------------------------------------------------------------


def test_build_concept_group_keyword_filter_drops_generic_crosswalk() -> None:
    index = OntologyIndex(
        preferred_terms={
            "SNOMED:P1": "Trauma to vulva during delivery",
            "SNOMED:G1": "Benign essential hypertension",
            "SNOMED:S1": "Gastroesophageal reflux disease in pregnancy",
        },
        icd_to_snomed={"O80": ["SNOMED:P1", "SNOMED:G1", "SNOMED:S1"]},
    )
    grp = build_concept_group(
        index,
        icd_prefixes=("O",),
        crosswalk_term_keywords=("vulva", "pregnancy"),
        crosswalk_exclude_substrings=(" in pregnancy",),
    )
    assert "SNOMED:P1" in grp  # has 'vulva'
    assert "SNOMED:G1" not in grp  # generic, no keyword
    assert "SNOMED:S1" not in grp  # excluded ' in pregnancy'


def test_loader_builds_three_rule_packs_and_manifest() -> None:
    index = OntologyIndex()
    rules, manifest = build_real_ontology_rules(index)
    rule_ids = {r.rule_id for r in rules}
    assert rule_ids == {
        "sex_restricted_concepts",
        "medication_required_context",
        "diabetes_type_exclusion",
    }
    assert {m["rule_id"] for m in manifest} == rule_ids
    for m in manifest:
        assert "rationale" in m and "source" in m and "limitations" in m
        assert m["severity"] > 0


def test_score_normalization_bounded() -> None:
    from src.scoring.ontology_aware import normalize_sont

    assert normalize_sont(0.0) == 0.0
    assert 0.0 <= normalize_sont(0.5) < 1.0
    assert normalize_sont(5.0) < 1.0  # still strictly below 1 for realistic sums
    assert normalize_sont(1000.0) <= 1.0  # asymptotes to 1, never exceeds it
    assert normalize_sont(-5.0) == 0.0  # negative clamped


def test_legacy_mode_does_not_use_rule_packs() -> None:
    from src.scoring.ontology_aware import OntologyAwareScorer

    legacy = OntologyAwareScorer(ontology_mode="legacy")
    out = legacy.score({"codes": ["DX_10_O80"], "gender": "M"}, s_det=0.0)
    # legacy path is the ICD-prefix compute_s_ont, not the canonical engine
    assert out["ontology_mode"] == "legacy"
    assert out["ontology_status"] == "ok_legacy"
