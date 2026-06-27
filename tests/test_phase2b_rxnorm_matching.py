"""Phase 2b-fix -- tests for conservative RxNorm ingredient matching."""

from __future__ import annotations

from src.preprocessing.code_normalization import (
    match_medication_ingredient,
    strip_medication_descriptors,
)

# Small synthetic ingredient set (uppercase, underscore-joined), test-only.
INGREDIENTS = frozenset(
    {
        "ACETAMINOPHEN",
        "HYDROMORPHONE",
        "OXYCODONE",
        "SODIUM_CHLORIDE",
        "NOREPINEPHRINE",
        "HEPARIN",
        "TRAMADOL",
        "SODIUM",  # generic single fragment -> must not match alone
    }
)


def test_strip_descriptors() -> None:
    assert strip_medication_descriptors(["ACETAMINOPHEN", "IV"]) == ["ACETAMINOPHEN"]
    assert strip_medication_descriptors(["0", "9", "SODIUM", "CHLORIDE"]) == [
        "SODIUM",
        "CHLORIDE",
    ]
    assert strip_medication_descriptors(["OXYCODONE", "IMMEDIATE", "RELEASE"]) == [
        "OXYCODONE"
    ]


def test_maps_safe_examples() -> None:
    assert (
        match_medication_ingredient("MED_ACETAMINOPHEN", INGREDIENTS) == "ACETAMINOPHEN"
    )
    assert (
        match_medication_ingredient("MED_ACETAMINOPHEN_IV", INGREDIENTS)
        == "ACETAMINOPHEN"
    )
    assert (
        match_medication_ingredient("MED_HYDROMORPHONE_DILAUDID", INGREDIENTS)
        == "HYDROMORPHONE"
    )
    assert (
        match_medication_ingredient("MED_OXYCODONE_IMMEDIATE_RELEASE", INGREDIENTS)
        == "OXYCODONE"
    )
    assert match_medication_ingredient("MED_TRAMADOL_ULTRAM", INGREDIENTS) == "TRAMADOL"


def test_longest_span_prefers_full_phrase_over_fragment() -> None:
    # Must map to SODIUM_CHLORIDE, never the bare SODIUM fragment.
    assert (
        match_medication_ingredient("MED_0_9_SODIUM_CHLORIDE", INGREDIENTS)
        == "SODIUM_CHLORIDE"
    )
    assert (
        match_medication_ingredient("MED_SODIUM_CHLORIDE_0_9_FLUSH", INGREDIENTS)
        == "SODIUM_CHLORIDE"
    )


def test_ambiguous_examples_are_skipped() -> None:
    # Bare insulin has no clean ingredient -> must not be force-mapped.
    assert match_medication_ingredient("MED_INSULIN", INGREDIENTS) is None
    # A bare generic electrolyte fragment alone must not map.
    assert match_medication_ingredient("MED_SODIUM", INGREDIENTS) is None
    # Unknown drug -> None.
    assert match_medication_ingredient("MED_NOTADRUG", INGREDIENTS) is None


def test_empty_ingredient_set_maps_nothing() -> None:
    assert match_medication_ingredient("MED_ACETAMINOPHEN", frozenset()) is None
