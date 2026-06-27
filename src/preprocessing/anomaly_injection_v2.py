"""
src/preprocessing/anomaly_injection_v2.py
==========================================
Phase 1b -- Ontology-backed, NON-CIRCULAR anomaly generation.

The v1 benchmark was circular: it injected a rare cross-sex token, so a
label-free "token present?" feature recovered the label (~0.94). v2 fixes this by
making each anomaly a violation of a *relationship* while keeping every individual
model-visible token common in normal records:

  1. demographic_incompatibility -- FLIP the gender field on a record that already
     contains sex-specific codes. No token is injected; the sequence is unchanged
     and clinically plausible. The conflict is (sex-specific code, wrong gender),
     detectable only by joint reasoning, never by an injected-token artifact.
  2. medication_indication_mismatch -- REMOVE the indication diagnosis required by
     a drug already present (e.g. insulin without any diabetes diagnosis). Only a
     diagnosis count changes; the drug token stays common.
  3. forbidden_cooccurrence -- ADD a curated mutually-exclusive partner that is
     itself mapped and non-rare (e.g. type-1 diabetes alongside type-2), so the
     added token is not a trivial giveaway.

Field separation is strict:
  * ``model_visible`` -- what a detector/scorer may see (sequence + demographics);
  * ``audit``         -- human provenance, never a model input;
  * ``hidden_eval``   -- the answer key (original gender, removed/added codes),
    used by evaluation only and never written into model-visible output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Columns that must never appear in model-visible output.
FORBIDDEN_MODEL_VISIBLE_COLUMNS: frozenset[str] = frozenset(
    {
        "bad_code",
        "expected_code",
        "replacement_code",
        "original_code",
        "injected_code",
        "removed_code",
        "repair_code",
        "ground_truth_repair",
        "repair_target",
        "source_anomaly_id",
        "source_row_id",
        "codes_original",
        "codes_corrupted",
        "corruption_note",
        "original_gender",
        "conflict_reason",
    }
)


class AnomalyTypeV2(str, Enum):
    """Anomaly families targeted by the v2 generator."""

    DEMOGRAPHIC_INCOMPATIBILITY = "demographic_incompatibility"
    MEDICATION_INDICATION_MISMATCH = "medication_indication_mismatch"
    FORBIDDEN_COOCCURRENCE = "forbidden_cooccurrence"


# ---------------------------------------------------------------------------
# Curated, ontology-informed rule sets (auditable; ICD families that map to
# SNOMED sex-restricted / mutually-exclusive concepts).
# ---------------------------------------------------------------------------

# Sex-specific diagnosis families (prefixes). These map (via Phase-2 ICD->SNOMED)
# to SNOMED sex-restricted concepts (pregnancy/obstetric vs prostate/male-genital).
FEMALE_ONLY_PREFIXES: tuple[str, ...] = (
    "DX_10_O",  # pregnancy, childbirth, puerperium
    "DX_10_Z33",
    "DX_10_Z34",
    "DX_10_Z3A",  # pregnancy state / supervision / weeks
    "DX_9_63",
    "DX_9_64",
    "DX_9_65",
    "DX_9_66",  # complications of pregnancy/childbirth
    "DX_9_V22",
    "DX_9_V23",
    "DX_9_V24",
    "DX_9_V27",
    "DX_9_V28",
)
MALE_ONLY_PREFIXES: tuple[str, ...] = (
    "DX_10_N40",  # benign prostatic hyperplasia
    "DX_10_C61",  # malignant neoplasm of prostate
    "DX_10_N41",
    "DX_10_N42",
    "DX_10_N43",
    "DX_10_N44",
    "DX_10_N45",
    "DX_9_600",
    "DX_9_601",
    "DX_9_602",  # prostate disorders
    "DX_9_185",  # malignant neoplasm of prostate
)

# Medication -> required indication diagnosis families (ontology-informed).
MEDICATION_INDICATION_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "insulin_requires_diabetes",
        "drug_terms": ("INSULIN", "GLUCAGON"),
        "required_dx_prefixes": ("DX_10_E10", "DX_10_E11", "DX_10_E13", "DX_9_250"),
    },
    {
        "name": "anticoagulant_requires_af_or_thrombosis",
        "drug_terms": (
            "WARFARIN",
            "APIXABAN",
            "RIVAROXABAN",
            "DABIGATRAN",
            "ENOXAPARIN",
        ),
        "required_dx_prefixes": (
            "DX_10_I48",
            "DX_10_I26",
            "DX_10_I80",
            "DX_10_I82",
            "DX_10_Z79_01",
            "DX_9_4273",
            "DX_9_4151",
            "DX_9_453",
        ),
    },
    {
        "name": "levothyroxine_requires_hypothyroid",
        "drug_terms": ("LEVOTHYROXINE", "LIOTHYRONINE"),
        "required_dx_prefixes": ("DX_10_E03", "DX_10_E02", "DX_10_E890", "DX_9_244"),
    },
)

# Curated mutually-exclusive pairs. ``present_prefix`` must already be in the
# record; ``add_code`` (a concrete, mapped, non-rare token) is added to create the
# impossible co-occurrence. ``conflict_prefix`` documents the contradiction.
MUTUALLY_EXCLUSIVE_PAIRS: tuple[dict[str, Any], ...] = (
    {
        "name": "type2_with_type1_diabetes",
        "present_prefix": "DX_10_E11",  # type 2 diabetes (common)
        "add_code": "DX_10_E10_9",  # type 1 diabetes, concrete code
        "conflict_prefix": "DX_10_E10",
    },
    {
        "name": "type1_with_type2_diabetes",
        "present_prefix": "DX_10_E10",  # type 1 diabetes
        "add_code": "DX_10_E11_9",  # type 2 diabetes (common)
        "conflict_prefix": "DX_10_E11",
    },
)


@dataclass(frozen=True)
class AnomalyGenConfig:
    """Deterministic anomaly-generation configuration."""

    seed: int = 42
    rate_per_type: float = 0.1
    enabled_types: tuple[AnomalyTypeV2, ...] = tuple(AnomalyTypeV2)
    sequence_column: str = "codes"
    subject_column: str = "subject_id"
    gender_column: str = "gender"
    age_column: str = "age_group"

    def derive_rng(self, salt: str = "") -> random.Random:
        return random.Random(f"{self.seed}:{salt}")


@dataclass
class AnomalyEvent:
    """A single anomaly with strict model_visible / audit / hidden_eval separation."""

    anomaly_type: AnomalyTypeV2
    model_visible: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    hidden_eval: dict[str, Any] = field(default_factory=dict)

    def to_model_visible_row(self) -> dict[str, Any]:
        validate_model_visible_fields(self.model_visible)
        return dict(self.model_visible)


def validate_model_visible_fields(fields: dict[str, Any]) -> None:
    """Raise if model-visible fields include any answer-key/leakage column."""
    offending = sorted(
        k for k in fields if str(k).strip().lower() in FORBIDDEN_MODEL_VISIBLE_COLUMNS
    )
    if offending:
        raise ValueError(
            f"Model-visible output exposes forbidden repair-answer columns: {offending}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _codes(record: dict[str, Any], config: AnomalyGenConfig) -> list[str]:
    seq = record.get(config.sequence_column, [])
    if isinstance(seq, (list, tuple)):
        return [str(c) for c in seq]
    return []


def _has_prefix(codes: list[str], prefixes: tuple[str, ...]) -> list[str]:
    up = [(c, c.upper()) for c in codes]
    return [c for c, u in up if any(u.startswith(p) for p in prefixes)]


def _has_term(codes: list[str], terms: tuple[str, ...]) -> list[str]:
    up = [(c, c.upper()) for c in codes]
    return [c for c, u in up if any(t in u for t in terms)]


def _norm_gender(value: Any) -> str | None:
    if value is None:
        return None
    t = str(value).strip().lower()
    if t in {"m", "male"}:
        return "M"
    if t in {"f", "female"}:
        return "F"
    return None


def _base_visible(
    record: dict[str, Any], config: AnomalyGenConfig, codes: list[str]
) -> dict[str, Any]:
    return {
        config.sequence_column: list(codes),
        config.gender_column: record.get(config.gender_column),
        config.age_column: record.get(config.age_column),
    }


# ---------------------------------------------------------------------------
# Injectors (real, non-circular)
# ---------------------------------------------------------------------------


def inject_demographic_incompatibility(
    record: dict[str, Any],
    config: AnomalyGenConfig,
) -> AnomalyEvent | None:
    """Flip the gender field on a record that already holds sex-specific codes.

    No token is injected: the model-visible sequence is identical to a normal
    record; only the gender field now conflicts with the (unchanged) codes.
    """
    codes = _codes(record, config)
    gender = _norm_gender(record.get(config.gender_column))
    if gender is None:
        return None

    if gender == "F":
        conflicting = _has_prefix(codes, FEMALE_ONLY_PREFIXES)
        flipped = "M"
    else:
        conflicting = _has_prefix(codes, MALE_ONLY_PREFIXES)
        flipped = "F"
    if not conflicting:
        return None

    visible = _base_visible(record, config, codes)
    visible[config.gender_column] = flipped  # the only change
    return AnomalyEvent(
        anomaly_type=AnomalyTypeV2.DEMOGRAPHIC_INCOMPATIBILITY,
        model_visible=visible,
        audit={"method": "gender_flip", "n_conflicting_codes": len(conflicting)},
        hidden_eval={"original_gender": gender, "conflicting_codes": conflicting},
    )


def inject_medication_indication_mismatch(
    record: dict[str, Any],
    config: AnomalyGenConfig,
) -> AnomalyEvent | None:
    """Remove the indication diagnosis required by a drug already present."""
    codes = _codes(record, config)
    rng = config.derive_rng(f"med:{record.get(config.subject_column)}")
    rules = list(MEDICATION_INDICATION_RULES)
    rng.shuffle(rules)
    for rule in rules:
        drugs = _has_term(codes, rule["drug_terms"])
        indications = _has_prefix(codes, rule["required_dx_prefixes"])
        if drugs and indications:
            remaining = [c for c in codes if c not in set(indications)]
            if not remaining:
                continue
            visible = _base_visible(record, config, remaining)
            return AnomalyEvent(
                anomaly_type=AnomalyTypeV2.MEDICATION_INDICATION_MISMATCH,
                model_visible=visible,
                audit={
                    "method": "remove_indication",
                    "rule": rule["name"],
                    "n_removed": len(indications),
                },
                hidden_eval={
                    "drug_codes": drugs,
                    "removed_indication_codes": indications,
                    "rule": rule["name"],
                },
            )
    return None


def inject_forbidden_cooccurrence(
    record: dict[str, Any],
    config: AnomalyGenConfig,
) -> AnomalyEvent | None:
    """Add a curated mutually-exclusive partner code (mapped, non-rare)."""
    codes = _codes(record, config)
    rng = config.derive_rng(f"cooc:{record.get(config.subject_column)}")
    rules = list(MUTUALLY_EXCLUSIVE_PAIRS)
    rng.shuffle(rules)
    for rule in rules:
        has_present = _has_prefix(codes, (rule["present_prefix"],))
        has_conflict = _has_prefix(codes, (rule["conflict_prefix"],))
        if has_present and not has_conflict:
            new_codes = list(codes) + [rule["add_code"]]
            visible = _base_visible(record, config, new_codes)
            return AnomalyEvent(
                anomaly_type=AnomalyTypeV2.FORBIDDEN_COOCCURRENCE,
                model_visible=visible,
                audit={"method": "add_exclusive_partner", "rule": rule["name"]},
                hidden_eval={
                    "present_codes": has_present,
                    "added_code": rule["add_code"],
                    "rule": rule["name"],
                },
            )
    return None


INJECTOR_REGISTRY: dict[AnomalyTypeV2, Any] = {
    AnomalyTypeV2.DEMOGRAPHIC_INCOMPATIBILITY: inject_demographic_incompatibility,
    AnomalyTypeV2.MEDICATION_INDICATION_MISMATCH: inject_medication_indication_mismatch,
    AnomalyTypeV2.FORBIDDEN_COOCCURRENCE: inject_forbidden_cooccurrence,
}


__all__ = [
    "AnomalyTypeV2",
    "AnomalyGenConfig",
    "AnomalyEvent",
    "FORBIDDEN_MODEL_VISIBLE_COLUMNS",
    "FEMALE_ONLY_PREFIXES",
    "MALE_ONLY_PREFIXES",
    "MEDICATION_INDICATION_RULES",
    "MUTUALLY_EXCLUSIVE_PAIRS",
    "validate_model_visible_fields",
    "inject_demographic_incompatibility",
    "inject_medication_indication_mismatch",
    "inject_forbidden_cooccurrence",
    "INJECTOR_REGISTRY",
]
