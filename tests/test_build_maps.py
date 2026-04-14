"""
tests/test_build_maps.py
=========================
Unit tests for Day 8 mapping builders and sequence-to-ontology conversion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.build_umls_maps import build_crosswalks, load_mrconso
from src.preprocessing.build_rxnorm_maps import build_drugname_map
from src.preprocessing.map_sequences_to_ont import map_token, map_token_list, load_maps


# ---------------------------------------------------------------------------
# Fixtures: synthetic MRCONSO data
# ---------------------------------------------------------------------------

_MRCONSO_HEADER = "CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF\n"


@pytest.fixture()
def mock_mrconso(tmp_path: Path) -> Path:
    """Write a minimal MRCONSO.RRF with known ICD-9, ICD-10, and SNOMED rows."""
    lines = [
        # CUI C0018802: ICD-9 428.0 and ICD-10 I50.9 both map to SNOMED 42343007 (Heart failure)
        "C0018802|ENG|P|L0018802|PF|S0000001|Y|A0000001||42343007||SNOMEDCT_US|PT|42343007|Heart failure|0|N|\n",
        "C0018802|ENG|P|L0018802|PF|S0000002|Y|A0000002||428.0||ICD9CM|PT|428.0|Congestive heart failure|0|N|\n",
        "C0018802|ENG|P|L0018802|PF|S0000003|Y|A0000003||I50.9||ICD10CM|PT|I50.9|Heart failure, unspecified|0|N|\n",
        # CUI C0011849: ICD-9 250.00 maps to SNOMED 44054006 (Diabetes mellitus type 2)
        "C0011849|ENG|P|L0011849|PF|S0000004|Y|A0000004||44054006||SNOMEDCT_US|PT|44054006|Type 2 diabetes mellitus|0|N|\n",
        "C0011849|ENG|P|L0011849|PF|S0000005|Y|A0000005||250.00||ICD9CM|PT|250.00|Diabetes mellitus type II|0|N|\n",
        # CUI C9999999: ICD-10 only, no SNOMED mapping (should produce no crosswalk)
        "C9999999|ENG|P|L9999999|PF|S0000006|Y|A0000006||Z99.9||ICD10CM|PT|Z99.9|Unknown condition|0|N|\n",
        # CUI C0000001: SNOMED-only concept (no ICD mapping, but should appear in snomed_terms)
        "C0000001|ENG|P|L0000001|PF|S0000007|Y|A0000007||12345678||SNOMEDCT_US|PT|12345678|Some disorder|0|N|\n",
        # Suppressed row (should be filtered out)
        "C0018802|ENG|P|L0018802|PF|S0000008|Y|A0000008||428.0||ICD9CM|PT|428.0|CHF suppressed|0|O|\n",
    ]
    path = tmp_path / "MRCONSO.RRF"
    path.write_text("".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def mock_rxnconso(tmp_path: Path) -> Path:
    """Write a minimal RXNCONSO.RRF with known drug entries."""
    # RRF layout: RXCUI|LAT|TS|LUI|STT|SUI|ISPREF|RXAUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF|
    lines = [
        "1049630|ENG|P|L001|PF|S001|Y|A001||1049630||RXNORM|IN|1049630|Acetaminophen|0|N|\n",
        "6809|ENG|P|L002|PF|S002|Y|A002||6809||RXNORM|IN|6809|Metformin|0|N|\n",
        "6809|ENG|P|L002|PF|S003|Y|A003||6809||RXNORM|SCD|6809|Metformin 500 MG Oral Tablet|0|N|\n",
        # Non-English (should be excluded)
        "9999|SPA|P|L003|PF|S004|Y|A004||9999||RXNORM|IN|9999|Aspirina|0|N|\n",
        # Suppressed
        "8888|ENG|P|L004|PF|S005|Y|A005||8888||RXNORM|IN|8888|Removed drug|0|O|\n",
    ]
    path = tmp_path / "RXNCONSO.RRF"
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests: build_umls_maps
# ---------------------------------------------------------------------------


class TestBuildUmlsMaps:
    def test_load_mrconso_filters_correctly(self, mock_mrconso: Path) -> None:
        df = load_mrconso(mock_mrconso)
        # Should exclude the suppressed row and non-English rows
        assert len(df) == 7
        assert set(df["SAB"].unique()) == {"SNOMEDCT_US", "ICD9CM", "ICD10CM"}

    def test_icd9_to_snomed_mapping(self, mock_mrconso: Path) -> None:
        df = load_mrconso(mock_mrconso)
        icd9_map, _, _ = build_crosswalks(df)
        assert "428.0" in icd9_map
        assert "42343007" in icd9_map["428.0"]
        assert "250.00" in icd9_map
        assert "44054006" in icd9_map["250.00"]

    def test_icd10_to_snomed_mapping(self, mock_mrconso: Path) -> None:
        df = load_mrconso(mock_mrconso)
        _, icd10_map, _ = build_crosswalks(df)
        assert "I50.9" in icd10_map
        assert "42343007" in icd10_map["I50.9"]
        # Z99.9 has no SNOMED mapping
        assert "Z99.9" not in icd10_map

    def test_snomed_terms_populated(self, mock_mrconso: Path) -> None:
        df = load_mrconso(mock_mrconso)
        _, _, snomed_terms = build_crosswalks(df)
        assert snomed_terms["42343007"] == "Heart failure"
        assert snomed_terms["44054006"] == "Type 2 diabetes mellitus"
        assert "12345678" in snomed_terms


# ---------------------------------------------------------------------------
# Tests: build_rxnorm_maps
# ---------------------------------------------------------------------------


class TestBuildRxnormMaps:
    def test_drugname_map_basic(self, mock_rxnconso: Path) -> None:
        from src.preprocessing.build_rxnorm_maps import load_rxnconso

        df = load_rxnconso(mock_rxnconso)
        drugname_map = build_drugname_map(df)

        assert "ACETAMINOPHEN" in drugname_map
        assert drugname_map["ACETAMINOPHEN"] == "1049630"
        assert "METFORMIN" in drugname_map
        assert drugname_map["METFORMIN"] == "6809"

    def test_suppressed_excluded(self, mock_rxnconso: Path) -> None:
        from src.preprocessing.build_rxnorm_maps import load_rxnconso

        df = load_rxnconso(mock_rxnconso)
        drugname_map = build_drugname_map(df)
        assert "REMOVED_DRUG" not in drugname_map

    def test_non_english_excluded(self, mock_rxnconso: Path) -> None:
        from src.preprocessing.build_rxnorm_maps import load_rxnconso

        df = load_rxnconso(mock_rxnconso)
        drugname_map = build_drugname_map(df)
        assert "ASPIRINA" not in drugname_map


# ---------------------------------------------------------------------------
# Tests: map_sequences_to_ont (token conversion)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_maps(tmp_path: Path) -> dict:
    """Create mock mapping dictionaries."""
    return {
        "icd9_to_snomed": {"4280": ["42343007"], "25000": ["44054006"]},
        "icd10_to_snomed": {"E119": ["44054006"], "I509": ["42343007"]},
        "drugname_to_rxcui": {"ACETAMINOPHEN": "1049630", "METFORMIN": "6809"},
        "ndc_to_rxcui": {"00045050130": "1049630"},
    }


class TestMapToken:
    def test_icd9_dx_maps_to_snomed(self, mock_maps: dict) -> None:
        assert map_token("DX_ICD9:4280", mock_maps) == "SNOMED:42343007"

    def test_icd9_dx_alt_prefix(self, mock_maps: dict) -> None:
        assert map_token("ICD9_DX:4280", mock_maps) == "SNOMED:42343007"

    def test_icd10_dx_maps_to_snomed(self, mock_maps: dict) -> None:
        assert map_token("ICD10_DX:E119", mock_maps) == "SNOMED:44054006"

    def test_unmapped_icd9_gets_unk_prefix(self, mock_maps: dict) -> None:
        assert map_token("DX_ICD9:99999", mock_maps) == "UNK_DX_ICD9:99999"

    def test_unmapped_icd10_gets_unk_prefix(self, mock_maps: dict) -> None:
        assert map_token("ICD10_DX:ZZZZZ", mock_maps) == "UNK_ICD10_DX:ZZZZZ"

    def test_ndc_maps_to_rxnorm(self, mock_maps: dict) -> None:
        assert map_token("MED_NDC:00045050130", mock_maps) == "RXNORM:1049630"

    def test_drug_name_maps_to_rxnorm(self, mock_maps: dict) -> None:
        assert map_token("MED_NAME:ACETAMINOPHEN", mock_maps) == "RXNORM:1049630"

    def test_unmapped_ndc_gets_unk_prefix(self, mock_maps: dict) -> None:
        assert map_token("MED_NDC:00000000000", mock_maps) == "UNK_MED_NDC:00000000000"

    def test_unmapped_drug_gets_unk_prefix(self, mock_maps: dict) -> None:
        assert map_token("MED_NAME:UNKNOWNDRUG", mock_maps) == "UNK_MED_NAME:UNKNOWNDRUG"

    def test_procedure_tokens_pass_through(self, mock_maps: dict) -> None:
        assert map_token("PROC_ICD9:3893", mock_maps) == "PROC_ICD9:3893"
        assert map_token("ICD9_PROC:3893", mock_maps) == "ICD9_PROC:3893"
        assert map_token("ICD10_PROC:0210", mock_maps) == "ICD10_PROC:0210"

    def test_eicu_tokens_pass_through(self, mock_maps: dict) -> None:
        assert map_token("EICU_APACHE2_DX:113", mock_maps) == "EICU_APACHE2_DX:113"
        assert map_token("EICU_COMORB:diabetes_mellitus", mock_maps) == "EICU_COMORB:diabetes_mellitus"

    def test_no_colon_gets_unk_prefix(self, mock_maps: dict) -> None:
        assert map_token("MALFORMED", mock_maps) == "UNK_MALFORMED"


class TestMapTokenList:
    def test_mixed_list(self, mock_maps: dict) -> None:
        tokens = ["DX_ICD9:4280", "MED_NAME:METFORMIN", "PROC_ICD9:3893", "DX_ICD9:99999"]
        result = map_token_list(tokens, mock_maps)
        assert result == [
            "SNOMED:42343007",
            "RXNORM:6809",
            "PROC_ICD9:3893",
            "UNK_DX_ICD9:99999",
        ]


class TestLoadMaps:
    def test_load_maps_from_dir(self, tmp_path: Path) -> None:
        (tmp_path / "icd9_to_snomed.json").write_text(
            json.dumps({"428.0": ["42343007"]}), encoding="utf-8",
        )
        (tmp_path / "drugname_to_rxcui.json").write_text(
            json.dumps({"ASPIRIN": "1191"}), encoding="utf-8",
        )
        maps = load_maps(tmp_path)
        assert "icd9_to_snomed" in maps
        assert "drugname_to_rxcui" in maps
        assert maps["icd9_to_snomed"]["428.0"] == ["42343007"]

    def test_load_maps_missing_files_ok(self, tmp_path: Path) -> None:
        maps = load_maps(tmp_path)
        assert len(maps) == 0
