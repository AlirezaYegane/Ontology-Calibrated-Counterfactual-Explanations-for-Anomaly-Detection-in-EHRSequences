"""
scripts/bridge_icd9_via_icd10.py
================================
Phase 2b-fix -- Enrich ICD-9 -> SNOMED by bridging through UMLS CUI-linked ICD-10
siblings + the MRMAP ICD-10 -> SNOMED map.

Rationale: UMLS 2026AA has no SNOMED -> ICD-9-CM map (retired), so ICD-9 codes
only had shared-CUI SNOMED links (~0.48 coverage). Many ICD-9 codes share a UMLS
CUI with an ICD-10 code that *is* mapped to SNOMED via MRMAP. Bridging through
that authoritative CUI synonymy recovers real disease mappings without any fuzzy
matching.

Usage
-----
    python scripts/bridge_icd9_via_icd10.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.build_umls_maps import (  # noqa: E402
    bridge_icd9_via_icd10,
    build_icd9_to_icd10_via_cui,
    load_mrconso,
)

PROC = PROJECT_ROOT / "ontologies" / "processed"
MRCONSO = (
    PROJECT_ROOT
    / "ontologies"
    / "raw"
    / "umls"
    / "2026AA"
    / "umls_extract"
    / "MRCONSO.RRF"
)
OUT = PROJECT_ROOT / "artifacts" / "phase2b_fix"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2b-fix ICD-9->ICD-10->SNOMED bridge."
    )
    ap.add_argument("--mrconso", default=str(MRCONSO))
    ap.add_argument("--processed-dir", default=str(PROC))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)

    mrconso_path = Path(args.mrconso)
    proc = Path(args.processed_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not mrconso_path.exists():
        raise FileNotFoundError(f"MRCONSO not found: {mrconso_path}")

    icd9_base = json.loads((proc / "icd9_to_snomed.json").read_text(encoding="utf-8"))
    icd10_map = json.loads((proc / "icd10_to_snomed.json").read_text(encoding="utf-8"))

    print(f"[bridge] loading MRCONSO (for ICD-9<->ICD-10 CUI links): {mrconso_path}")
    mrconso = load_mrconso(mrconso_path)
    icd9_to_icd10 = build_icd9_to_icd10_via_cui(mrconso)
    print(f"[bridge] ICD-9 codes with CUI-linked ICD-10 siblings: {len(icd9_to_icd10)}")

    merged_icd9, prov9 = bridge_icd9_via_icd10(icd9_base, icd9_to_icd10, icd10_map)
    added = sorted(set(merged_icd9) - set(icd9_base))

    (proc / "icd9_to_snomed.json").write_text(
        json.dumps(merged_icd9, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    delta = {
        "phase": "2b-fix",
        "step": "icd9_bridge",
        "method": "ICD-9 -> CUI-linked ICD-10 (MRCONSO synonymy) -> SNOMED (MRMAP)",
        "icd9_count_before": len(icd9_base),
        "icd9_count_after": len(merged_icd9),
        "icd9_added": len(added),
        "provenance_breakdown": {
            "both": sum(1 for v in prov9.values() if v == "both"),
            "bridge_only": sum(1 for v in prov9.values() if v == "bridge_icd10_mrmap"),
            "shared_cui_only": sum(1 for v in prov9.values() if v == "shared_cui"),
        },
        "sample_added": {k: merged_icd9[k][:3] for k in added[:15]},
    }
    (out / "icd9_bridge_delta.json").write_text(
        json.dumps(delta, indent=2), encoding="utf-8"
    )

    # update combined provenance file if present
    prov_path = proc / "icd_to_snomed_provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        prov["icd9"] = prov9
        prov_path.write_text(
            json.dumps(prov, indent=0, sort_keys=True), encoding="utf-8"
        )

    print(
        f"[bridge] ICD-9 codes: {len(icd9_base)} -> {len(merged_icd9)} (+{len(added)})"
    )
    print(f"[bridge] wrote {out / 'icd9_bridge_delta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
