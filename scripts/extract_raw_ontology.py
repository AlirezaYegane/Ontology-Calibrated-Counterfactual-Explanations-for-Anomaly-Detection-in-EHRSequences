"""
scripts/extract_raw_ontology.py
===============================
Phase 2b -- Selective extraction of the specific RF2/RRF files needed by the
parsers, without unpacking the huge unused members (e.g. UMLS MRREL ~6.4GB,
RxNorm RXNREL/RXNSAT). Streaming copy keeps memory bounded.

Usage
-----
    python scripts/extract_raw_ontology.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "ontologies" / "raw"

# (zip glob, [(regex of entries to extract, output subdir)])
JOBS = [
    (
        RAW / "snomed" / "us_edition",
        "SnomedCT_*.zip",
        [
            (
                re.compile(r"Snapshot/Terminology/sct2_Concept_Snapshot", re.I),
                "snomed_extract",
            ),
            (
                re.compile(r"Snapshot/Terminology/sct2_Description_Snapshot", re.I),
                "snomed_extract",
            ),
            (
                re.compile(r"Snapshot/Terminology/sct2_Relationship_Snapshot", re.I),
                "snomed_extract",
            ),
        ],
    ),
    (
        RAW / "umls" / "2026AA",
        "umls-*-metathesaurus-full.zip",
        [(re.compile(r"META/MRCONSO\.RRF$", re.I), "umls_extract")],
    ),
    (
        RAW / "rxnorm",
        "RxNorm_full_current.zip",
        [(re.compile(r"(^|/)rrf/RXNCONSO\.RRF$", re.I), "rxnorm_extract")],
    ),
]


def extract_entry(zf: zipfile.ZipFile, entry: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(entry) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    return out_path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2b selective raw-ontology extraction."
    )
    ap.add_argument(
        "--force", action="store_true", help="Re-extract even if target exists."
    )
    args = ap.parse_args(argv)

    extracted: list[str] = []
    for base_dir, zip_glob, patterns in JOBS:
        zips = sorted(base_dir.glob(zip_glob)) if base_dir.exists() else []
        if not zips:
            print(f"[extract] WARN: no zip matching {zip_glob} under {base_dir}")
            continue
        zip_path = zips[0]
        try:
            zf = zipfile.ZipFile(zip_path)
        except Exception as exc:
            print(f"[extract] ERROR: {zip_path} not a valid zip: {exc}")
            continue
        names = zf.namelist()
        for rx, subdir in patterns:
            hits = [n for n in names if rx.search(n)]
            if not hits:
                print(
                    f"[extract] WARN: no entry matching {rx.pattern} in {zip_path.name}"
                )
                continue
            entry = hits[0]
            out_path = base_dir / subdir / Path(entry).name
            if out_path.exists() and not args.force:
                print(f"[extract] skip (exists): {out_path.relative_to(PROJECT_ROOT)}")
                extracted.append(str(out_path.relative_to(PROJECT_ROOT)))
                continue
            print(
                f"[extract] {zip_path.name} :: {entry} -> {out_path.relative_to(PROJECT_ROOT)}"
            )
            size = extract_entry(zf, entry, out_path)
            print(f"[extract]   wrote {size / 1e6:.1f} MB")
            extracted.append(str(out_path.relative_to(PROJECT_ROOT)))

    print("[extract] DONE")
    for e in extracted:
        print("  ", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
