"""
src/ontology/rule_loader.py
===========================
Phase 3b -- Build concrete ontology rules from the curated rule packs.

This turns the curated clinical tables in :mod:`src.ontology.rule_packs` into
concrete :class:`OntologyRule` objects bound to a loaded :class:`OntologyIndex`,
and emits an auditable manifest (rule_id / rule_type / severity / rationale /
source / limitations / group sizes).

The rule packs are CODE-defined (always importable); the data they bind to (the
ICD->SNOMED crosswalk + SNOMED hierarchy in the index) may be absent, in which
case each concept group degrades gracefully to its seeded SNOMED roots only --
the same safe behavior the Phase 2 loader/fixtures rely on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import rule_packs as rp
from .rule_engine import OntologyRule

if TYPE_CHECKING:
    from .index import OntologyIndex


def build_real_ontology_rules(
    index: "OntologyIndex",
) -> tuple[list[OntologyRule], list[dict[str, Any]]]:
    """Construct the Phase 3b real-mode rules + their audit manifest."""
    # --- concept groups (built from the same crosswalk the scorer uses) ---
    # Sex rule: CLEAN hierarchy-only SNOMED groups (NO crosswalk -> no
    # many-to-many noise). Real-data coverage comes from the source-token ICD
    # families below; the clean group serves the fixture/hierarchy path.
    female_group = rp.build_concept_group(
        index, snomed_roots=rp.FEMALE_ONLY_SNOMED_ROOTS
    )
    male_group = rp.build_concept_group(index, snomed_roots=rp.MALE_ONLY_SNOMED_ROOTS)
    female_icd = rp.FEMALE_ONLY_ICD10_PREFIXES + rp.FEMALE_ONLY_ICD9_PREFIXES
    male_icd = rp.MALE_ONLY_ICD10_PREFIXES + rp.MALE_ONLY_ICD9_PREFIXES

    dm_type1 = rp.build_concept_group(
        index,
        snomed_roots=rp.DM_TYPE1_SNOMED_ROOTS,
        icd_prefixes=rp.DM_TYPE1_ICD10_PREFIXES,
        crosswalk_term_keywords=rp.DM_TYPE1_TERM_KEYWORDS,
        crosswalk_exclude_substrings=rp.DM_EXCLUDE_SUBSTRINGS,
    )
    dm_type2 = rp.build_concept_group(
        index,
        snomed_roots=rp.DM_TYPE2_SNOMED_ROOTS,
        icd_prefixes=rp.DM_TYPE2_ICD10_PREFIXES,
        crosswalk_term_keywords=rp.DM_TYPE2_TERM_KEYWORDS,
        crosswalk_exclude_substrings=rp.DM_EXCLUDE_SUBSTRINGS,
    )
    # Force the diabetes groups disjoint: a concept shared by both (e.g. a generic
    # "diabetes mellitus" parent, or any seed-subtree overlap) must not satisfy
    # both sides of the mutual-exclusion test on a single-type patient.
    shared_dm = dm_type1 & dm_type2
    dm_type1 -= shared_dm
    dm_type2 -= shared_dm
    # Also keep sex groups disjoint (defensive; anatomy should not overlap).
    shared_sex = female_group & male_group
    female_group -= shared_sex
    male_group -= shared_sex
    anticoag_context = frozenset(
        rp.build_concept_group(
            index,
            snomed_roots=rp.ANTICOAG_CONTEXT_SNOMED_ROOTS,
            icd_prefixes=rp.ANTICOAG_CONTEXT_ICD10_PREFIXES
            + rp.ANTICOAG_CONTEXT_ICD9_PREFIXES,
        )
    )
    hypothyroid_context = frozenset(
        rp.build_concept_group(
            index,
            snomed_roots=rp.HYPOTHYROID_CONTEXT_SNOMED_ROOTS,
            icd_prefixes=rp.HYPOTHYROID_CONTEXT_ICD10_PREFIXES
            + rp.HYPOTHYROID_CONTEXT_ICD9_PREFIXES,
        )
    )
    diabetes_context = frozenset(
        rp.build_concept_group(
            index,
            snomed_roots=rp.DIABETES_CONTEXT_SNOMED_ROOTS,
            icd_prefixes=rp.DIABETES_CONTEXT_ICD10_PREFIXES
            + rp.DIABETES_CONTEXT_ICD9_PREFIXES,
        )
    )

    # --- drug -> required-context map (mapped RxCUIs) ---
    drug_to_context: dict[str, frozenset[str]] = {}
    for cui in rp.ANTICOAGULANT_RXCUIS:
        drug_to_context[cui] = anticoag_context
    for cui in rp.LEVOTHYROXINE_RXCUIS:
        drug_to_context[cui] = hypothyroid_context
    for cui in rp.INSULIN_RXCUIS:
        drug_to_context[cui] = diabetes_context
    # Source-token ingredient names for drugs the RxCUI map leaves UNMAPPED.
    # Insulin only: bare 'MED_INSULIN' has no RxCUI but unambiguously requires a
    # diabetes/hyperglycaemia context. (Prophylactic heparin/enoxaparin are NOT
    # added by name -- prophylaxis legitimately lacks an active-clot diagnosis.)
    drug_name_to_context: dict[str, frozenset[str]] = {"INSULIN": diabetes_context}

    # --- rule objects ---
    sex_rule = rp.SexRestrictedRule(
        rule_id="sex_restricted_concepts",
        sex_to_forbidden_group={"M": female_group, "F": male_group},
        sex_to_forbidden_icd_prefixes={"M": female_icd, "F": male_icd},
        severity=1.0,
    )
    med_rule = rp.RequiredContextRule(
        rule_id="medication_required_context",
        drug_to_context=drug_to_context,
        drug_name_to_context=drug_name_to_context,
        severity=0.5,
    )
    dm_rule = rp.GroupMutualExclusionRule(
        rule_id="diabetes_type_exclusion",
        groups=[
            rp.GroupConflict(
                label="type1_vs_type2_diabetes",
                left=dm_type1,
                right=dm_type2,
                left_icd_prefixes=rp.DM_TYPE1_ICD10_PREFIXES,
                right_icd_prefixes=rp.DM_TYPE2_ICD10_PREFIXES,
            )
        ],
        severity=0.5,
    )

    rules: list[OntologyRule] = [sex_rule, med_rule, dm_rule]

    manifest: list[dict[str, Any]] = [
        {
            "rule_id": "sex_restricted_concepts",
            "rule_type": "sex_restriction",
            "kind": "demographic_mismatch",
            "severity": 1.0,
            "n_female_icd_families": len(female_icd),
            "n_male_icd_families": len(male_icd),
            "n_female_clean_snomed_concepts": len(female_group),
            "n_male_clean_snomed_concepts": len(male_group),
            "rationale": (
                "Pregnancy/childbirth/puerperium and female reproductive-organ "
                "diagnoses are anatomically impossible in male patients; prostate "
                "and male-genital diagnoses are impossible in female patients."
            ),
            "source": (
                "Anchored on the SOURCE ICD diagnosis-code family (ICD-10 chapter "
                "XV O*, Z33/Z34/Z3A/Z37, female/male genital neoplasms; ICD-9 "
                "630-679, V22-V28, 179-184, 218-221 / 600-608, 185-187, 222) -- "
                "standard clinical knowledge -- plus a clean SNOMED pregnancy "
                "subtree (root 77386006) for hierarchy-expressed records."
            ),
            "limitations": (
                "Fires on the presence of an opposite-sex diagnosis-code FAMILY in "
                "the source tokens; this deliberately avoids the lossy many-to-many "
                "ICD->SNOMED crosswalk (generic codes cross-map to obstetric "
                "variants). Sex is read from the model-visible gender field only."
            ),
        },
        {
            "rule_id": "medication_required_context",
            "rule_type": "required_context",
            "kind": "missing_required_code",
            "severity": 0.5,
            "n_drugs": len(drug_to_context),
            "n_anticoag_context_concepts": len(anticoag_context),
            "n_hypothyroid_context_concepts": len(hypothyroid_context),
            "n_diabetes_context_concepts": len(diabetes_context),
            "rationale": (
                "Anticoagulants imply a thromboembolic/atrial-fibrillation "
                "indication; levothyroxine implies hypothyroidism; insulin "
                "implies a diabetes/hyperglycaemia context. Absence of any "
                "supporting diagnosis is a (weak) implausibility signal."
            ),
            "source": (
                "RxCUIs verified against benchmark-v2 token->RxNorm mappings; "
                "context diagnosis families mapped via the ICD->SNOMED crosswalk."
            ),
            "limitations": (
                "WEAK signal (severity 0.5): drugs have secondary/off-label uses "
                "and coding is incomplete, so a missing diagnosis is not proof of "
                "mismatch. Bare 'MED_INSULIN' is UNMAPPED by the drug->RxCUI map, "
                "so insulin-indication anomalies on benchmark-v2 cannot fire "
                "(token->RxNorm mapping gap)."
            ),
        },
        {
            "rule_id": "diabetes_type_exclusion",
            "rule_type": "group_mutual_exclusion",
            "kind": "mutual_exclusion",
            "severity": 0.5,
            "n_type1_concepts": len(dm_type1),
            "n_type2_concepts": len(dm_type2),
            "rationale": (
                "Type 1 and type 2 diabetes are distinct etiologies; concurrent "
                "coding of both is implausible for a single encounter."
            ),
            "source": (
                "ICD-10 E10* (type 1) vs E11* (type 2) mapped via crosswalk; "
                "seeded with SNOMED 46635009 / 44054006. ICD-9 250 (type-"
                "ambiguous) deliberately excluded."
            ),
            "limitations": (
                "MEDIUM severity (0.5): real EHRs sometimes co-code both types "
                "during diagnostic transition or uncertainty, so this is not a "
                "hard contradiction."
            ),
        },
    ]

    return rules, manifest


def build_rule_manifest(index: "OntologyIndex") -> list[dict[str, Any]]:
    """Convenience: just the manifest (for audit/diagnostic scripts)."""
    _, manifest = build_real_ontology_rules(index)
    return manifest


__all__ = ["build_real_ontology_rules", "build_rule_manifest"]
