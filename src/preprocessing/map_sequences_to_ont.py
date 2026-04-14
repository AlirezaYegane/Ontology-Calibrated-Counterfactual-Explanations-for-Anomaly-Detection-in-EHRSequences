"""
src/preprocessing/map_sequences_to_ont.py
==========================================
Day 8 -- Map token sequences from extraction-time codes to ontology codes.

Loads processed sequence Parquet/pickle files and mapping JSONs, then
converts tokens:

  * ``DX_ICD9:4280``     -> ``SNOMED:<concept_id>``
  * ``ICD9_DX:4280``     -> ``SNOMED:<concept_id>``
  * ``ICD10_DX:E119``    -> ``SNOMED:<concept_id>``
  * ``MED_NDC:xxx``      -> ``RXNORM:<rxcui>``  (via NDC->RxCUI, if available)
  * ``MED_NAME:DRUG``    -> ``RXNORM:<rxcui>``

Unmapped tokens are kept with a ``UNK_`` prefix (e.g. ``UNK_DX_ICD9:4280``).
Mapped files are saved with an ``_ont`` suffix.

CLI::

    python -m src.preprocessing.map_sequences_to_ont \\
        --sequences-dir data/processed \\
        --maps-dir ontologies/umls_maps \\
        --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    """Load a JSON mapping file."""
    log.info("Loading %s", path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def load_maps(maps_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all mapping dictionaries from the maps directory."""
    maps: dict[str, dict[str, Any]] = {}

    icd9_path = maps_dir / "icd9_to_snomed.json"
    if icd9_path.exists():
        maps["icd9_to_snomed"] = _load_json(icd9_path)

    icd10_path = maps_dir / "icd10_to_snomed.json"
    if icd10_path.exists():
        maps["icd10_to_snomed"] = _load_json(icd10_path)

    drugname_path = maps_dir / "drugname_to_rxcui.json"
    if drugname_path.exists():
        maps["drugname_to_rxcui"] = _load_json(drugname_path)

    ndc_path = maps_dir / "ndc_to_rxcui.json"
    if ndc_path.exists():
        maps["ndc_to_rxcui"] = _load_json(ndc_path)

    log.info("Loaded %d mapping dictionaries", len(maps))
    return maps


# ---------------------------------------------------------------------------
# Token mapping
# ---------------------------------------------------------------------------


def map_token(token: str, maps: dict[str, dict[str, Any]]) -> str:
    """Map a single token to its ontology-normalized form.

    Returns the mapped token or a ``UNK_``-prefixed version if unmapped.
    """
    if ":" not in token:
        return f"UNK_{token}"

    prefix, code = token.split(":", 1)

    # ICD-9 diagnosis -> SNOMED
    if prefix in ("DX_ICD9", "ICD9_DX"):
        icd9_map = maps.get("icd9_to_snomed", {})
        snomed_ids = icd9_map.get(code, [])
        if snomed_ids:
            return f"SNOMED:{snomed_ids[0]}"
        return f"UNK_{token}"

    # ICD-10 diagnosis -> SNOMED
    if prefix in ("ICD10_DX",):
        icd10_map = maps.get("icd10_to_snomed", {})
        snomed_ids = icd10_map.get(code, [])
        if snomed_ids:
            return f"SNOMED:{snomed_ids[0]}"
        return f"UNK_{token}"

    # ICD-9 procedure -- keep as-is (no SNOMED procedure crosswalk)
    if prefix in ("PROC_ICD9", "ICD9_PROC"):
        return token

    # ICD-10 procedure -- keep as-is
    if prefix in ("ICD10_PROC",):
        return token

    # NDC medication -> RxNorm
    if prefix == "MED_NDC":
        ndc_map = maps.get("ndc_to_rxcui", {})
        rxcui = ndc_map.get(code)
        if rxcui:
            return f"RXNORM:{rxcui}"
        return f"UNK_{token}"

    # Drug-name medication -> RxNorm
    if prefix == "MED_NAME":
        drugname_map = maps.get("drugname_to_rxcui", {})
        rxcui = drugname_map.get(code)
        if rxcui:
            return f"RXNORM:{rxcui}"
        return f"UNK_{token}"

    # eICU tokens and others -- pass through unchanged
    return token


def map_token_list(tokens: list[str], maps: dict[str, dict[str, Any]]) -> list[str]:
    """Map a list of tokens."""
    return [map_token(t, maps) for t in tokens]


# ---------------------------------------------------------------------------
# Sequence file processing
# ---------------------------------------------------------------------------

_TOKEN_LIST_COLS = (
    "diagnosis_tokens",
    "procedure_tokens",
    "medication_tokens",
    "comorbidity_tokens",
    "sequence_tokens",
)


def _load_sequences(path: Path) -> pd.DataFrame:
    """Load a sequence file (Parquet or pickle)."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Deserialize JSON-encoded list columns
    for col in _TOKEN_LIST_COLS:
        if col in df.columns:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample, str):
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
    return df


def process_file(
    path: Path,
    maps: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Map tokens in a single sequence file and save with _ont suffix."""
    log.info("Processing %s", path.name)
    df = _load_sequences(path)

    total_tokens = 0
    mapped_tokens = 0

    for col in _TOKEN_LIST_COLS:
        if col not in df.columns:
            continue
        new_col: list[list[str]] = []
        for tokens in df[col]:
            if not isinstance(tokens, list):
                new_col.append([])
                continue
            mapped = map_token_list(tokens, maps)
            total_tokens += len(mapped)
            mapped_tokens += sum(1 for t in mapped if not t.startswith("UNK_"))
            new_col.append(mapped)
        df[col] = new_col

    # Rebuild sequence_tokens from component columns
    if "sequence_tokens" in df.columns:
        rebuilt: list[list[str]] = []
        for _, row in df.iterrows():
            seq: list[str] = []
            for col in ("diagnosis_tokens", "procedure_tokens", "medication_tokens", "comorbidity_tokens"):
                if col in df.columns and isinstance(row[col], list):
                    seq.extend(row[col])
            rebuilt.append(seq)
        df["sequence_tokens"] = rebuilt
        df["sequence_length"] = df["sequence_tokens"].apply(len)

    # Save with _ont suffix
    stem = path.stem.replace("_ont", "")
    out_path = output_dir / f"{stem}_ont{path.suffix}"

    # Serialize list columns to JSON strings for Parquet
    out = df.copy()
    for col in _TOKEN_LIST_COLS:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

    if path.suffix == ".parquet":
        out.to_parquet(out_path, index=False)
    else:
        df.to_pickle(out_path)

    coverage = (mapped_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
    stats = {
        "file": path.name,
        "total_tokens": total_tokens,
        "mapped_tokens": mapped_tokens,
        "unmapped_tokens": total_tokens - mapped_tokens,
        "coverage_pct": round(coverage, 2),
    }
    log.info(
        "  %s: %d/%d tokens mapped (%.1f%% coverage)",
        path.name, mapped_tokens, total_tokens, coverage,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Day 8 -- Map sequence tokens to ontology codes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sequences-dir", required=True, type=Path,
        help="Directory with processed sequence files (Parquet/pickle)",
    )
    p.add_argument(
        "--maps-dir", required=True, type=Path,
        help="Directory with JSON mapping files",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for mapped sequence files",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps = load_maps(args.maps_dir)

    # Find sequence files (Parquet and pickle, exclude already-mapped _ont files)
    seq_files = sorted(
        p for p in args.sequences_dir.iterdir()
        if p.suffix in (".parquet", ".pkl") and "_ont" not in p.stem
    )

    if not seq_files:
        log.warning("No sequence files found in %s", args.sequences_dir)
        return

    all_stats: list[dict[str, Any]] = []
    for path in seq_files:
        stats = process_file(path, maps, args.output_dir)
        all_stats.append(stats)

    # Summary
    total = sum(s["total_tokens"] for s in all_stats)
    mapped = sum(s["mapped_tokens"] for s in all_stats)
    overall = (mapped / total * 100) if total > 0 else 0.0
    log.info("Overall: %d/%d tokens mapped (%.1f%% coverage)", mapped, total, overall)

    # Save coverage report
    report_path = args.output_dir / "ontology_mapping_coverage.json"
    report = {"files": all_stats, "overall_coverage_pct": round(overall, 2)}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Coverage report saved to %s", report_path.name)


if __name__ == "__main__":
    main()
