"""Phase 2b-fix -- tests for MRMAP parsing, ICD merge, and ICD-9 bridge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.build_umls_maps import (
    bridge_icd9_via_icd10,
    build_icd9_to_icd10_via_cui,
    merge_icd_maps,
    parse_mrmap_icd10_to_snomed,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ontology" / "mrmap_sample.rrf"


def test_parse_mrmap_filters_correctly() -> None:
    m = parse_mrmap_icd10_to_snomed(FIXTURE)
    # E11.9 <- 44054006, E10.9 <- 46635009
    assert m["E11.9"] == {"44054006"}
    assert m["E10.9"] == {"46635009"}
    # trailing '?' map flag stripped
    assert m["I50.9"] == {"84114007"}
    # REL=XR row excluded
    assert "Z99.9" not in m
    # non-SNOMEDCT_US mapset excluded
    assert "A00.0" not in m


def test_merge_icd_maps_preserves_existing_and_adds() -> None:
    base = {"E11.9": ["SNOMED_OLD"], "X00": ["KEEP"]}
    extra = {"E11.9": {"44054006"}, "E10.9": {"46635009"}}
    merged, prov = merge_icd_maps(base, extra)
    # existing mapping not deleted, new one unioned
    assert "SNOMED_OLD" in merged["E11.9"]
    assert "44054006" in merged["E11.9"]
    assert merged["X00"] == ["KEEP"]  # untouched
    assert merged["E10.9"] == ["46635009"]  # added
    assert prov["E11.9"] == "both"
    assert prov["E10.9"] == "mrmap"
    assert prov["X00"] == "shared_cui"


def test_build_icd9_to_icd10_via_cui() -> None:
    mrconso = pd.DataFrame(
        {
            "CUI": ["C1", "C1", "C2", "C2"],
            "SAB": ["ICD9CM", "ICD10CM", "ICD9CM", "SNOMEDCT_US"],
            "CODE": ["250.00", "E11.9", "999", "111"],
            "STR": ["", "", "", ""],
            "TTY": ["", "", "", ""],
        }
    )
    bridge = build_icd9_to_icd10_via_cui(mrconso)
    assert bridge["250.00"] == {"E11.9"}
    # C2 has ICD9 but no ICD10 sibling -> not bridged
    assert "999" not in bridge


def test_bridge_icd9_via_icd10_adds_without_deleting() -> None:
    icd9_base = {"401.9": ["EXISTING"]}
    icd9_to_icd10 = {"250.00": {"E11.9"}, "401.9": {"I10"}}
    icd10_map = {"E11.9": ["44054006"], "I10": ["38341003"]}
    merged, prov = bridge_icd9_via_icd10(icd9_base, icd9_to_icd10, icd10_map)
    # new code bridged
    assert merged["250.00"] == ["44054006"]
    assert prov["250.00"] == "bridge_icd10_mrmap"
    # existing preserved + augmented
    assert "EXISTING" in merged["401.9"]
    assert "38341003" in merged["401.9"]
    assert prov["401.9"] == "both"
