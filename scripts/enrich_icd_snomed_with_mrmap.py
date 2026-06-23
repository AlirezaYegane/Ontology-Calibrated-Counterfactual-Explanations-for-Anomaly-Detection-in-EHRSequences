"""
scripts/enrich_icd_snomed_with_mrmap.py
=======================================
Phase 2b-fix -- Enrich ICD-10 -> SNOMED maps using the authoritative UMLS
SNOMED CT -> ICD-10-CM map (MRMAP.RRF), without deleting existing shared-CUI
mappings.

NOTE: UMLS 2026AA contains a SNOMED -> ICD-10-CM map but NOT a SNOMED -> ICD-9-CM
map (that map was retired). ICD-9 therefore keeps its shared-CUI mappings only.

Usage
-----
    python scripts/enrich_icd_snomed_with_mrmap.py
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
    merge_icd_maps,
    parse_mrmap_icd10_to_snomed,
)

PROC = PROJECT_ROOT / "ontologies" / "processed"
MRMAP = (
    PROJECT_ROOT
    / "ontologies"
    / "raw"
    / "umls"
    / "2026AA"
    / "umls_extract"
    / "MRMAP.RRF"
)
OUT = PROJECT_ROOT / "artifacts" / "phase2b_fix"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2b-fix MRMAP ICD enrichment.")
    ap.add_argument("--mrmap", default=str(MRMAP))
    ap.add_argument("--processed-dir", default=str(PROC))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)

    mrmap_path = Path(args.mrmap)
    proc = Path(args.processed_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not mrmap_path.exists():
        raise FileNotFoundError(f"MRMAP not found: {mrmap_path}")

    icd10_base = json.loads((proc / "icd10_to_snomed.json").read_text(encoding="utf-8"))
    icd9_base = json.loads((proc / "icd9_to_snomed.json").read_text(encoding="utf-8"))

    print(f"[enrich] parsing MRMAP: {mrmap_path}")
    mrmap_icd10 = parse_mrmap_icd10_to_snomed(mrmap_path)

    merged_icd10, prov10 = merge_icd_maps(icd10_base, mrmap_icd10)

    added_icd10 = sorted(set(merged_icd10) - set(icd10_base))
    sample_added = {k: merged_icd10[k][:3] for k in added_icd10[:15]}

    # Write merged ICD-10 map back (ICD-9 unchanged: no MRMAP source).
    (proc / "icd10_to_snomed.json").write_text(
        json.dumps(merged_icd10, indent=1, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    # Provenance (ICD-9 all shared_cui; ICD-10 from merge).
    provenance = {"icd9": {k: "shared_cui" for k in icd9_base}, "icd10": prov10}
    (proc / "icd_to_snomed_provenance.json").write_text(
        json.dumps(provenance, indent=0, sort_keys=True), encoding="utf-8"
    )

    delta = {
        "phase": "2b-fix",
        "source": "UMLS MRMAP.RRF SNOMEDCT_US -> ICD-10-CM (TOTYPE=SDUI, excl REL=XR)",
        "icd9": {
            "note": "No SNOMED->ICD-9 map in UMLS 2026AA; ICD-9 unchanged (shared-CUI only).",
            "count": len(icd9_base),
        },
        "icd10": {
            "count_before": len(icd10_base),
            "count_after": len(merged_icd10),
            "added": len(added_icd10),
            "provenance_breakdown": {
                "both": sum(1 for v in prov10.values() if v == "both"),
                "mrmap_only": sum(1 for v in prov10.values() if v == "mrmap"),
                "shared_cui_only": sum(1 for v in prov10.values() if v == "shared_cui"),
            },
            "sample_added_mappings": sample_added,
        },
    }
    (out / "icd_map_delta.json").write_text(
        json.dumps(delta, indent=2), encoding="utf-8"
    )

    print(
        f"[enrich] ICD-10 codes: {len(icd10_base)} -> {len(merged_icd10)} (+{len(added_icd10)})"
    )
    print(f"[enrich] ICD-9 unchanged: {len(icd9_base)} (no SNOMED->ICD-9 map in UMLS)")
    print(f"[enrich] wrote {out / 'icd_map_delta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
