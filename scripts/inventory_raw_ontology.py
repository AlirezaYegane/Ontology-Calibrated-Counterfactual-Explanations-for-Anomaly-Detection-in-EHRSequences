"""
scripts/inventory_raw_ontology.py
=================================
Phase 2b -- Validate and inventory raw ontology assets.

Walks ``ontologies/raw`` and records, for every file: size, extension, whether it
is a structurally valid zip (reads the central directory; does NOT fully
decompress), whether it looks like a stray HTML/login error page, the expected
ontology type, and a status. Writes a JSON + Markdown inventory.

Usage
-----
    python scripts/inventory_raw_ontology.py
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "ontologies" / "raw"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase2b"

# Heuristic: map path fragments to an expected ontology type.
TYPE_HINTS = (
    ("umls", "UMLS Metathesaurus"),
    ("snomed", "SNOMED CT"),
    ("rxnorm", "RxNorm"),
    ("icd9", "ICD-9-CM"),
    ("icd10", "ICD-10-CM"),
    ("icd", "ICD"),
)

MIN_VALID_ZIP_BYTES = 100_000  # zips smaller than this are suspicious


def expected_type(path: Path) -> str:
    low = str(path).lower()
    for frag, label in TYPE_HINTS:
        if frag in low:
            return label
    return "unknown"


def looks_like_html(path: Path) -> bool:
    if path.suffix.lower() in {".html", ".htm"}:
        return True
    try:
        with path.open("rb") as fh:
            head = fh.read(512).lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html")
    except Exception:
        return False


def zip_status(path: Path) -> tuple[bool, int | None, str | None]:
    """Return (is_valid_zip, n_entries, error)."""
    try:
        with zipfile.ZipFile(path) as zf:
            return True, len(zf.namelist()), None
    except Exception as exc:
        return False, None, str(exc)


def inventory_file(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    ext = path.suffix.lower()
    rec: dict[str, Any] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": size,
        "size_mb": round(size / 1e6, 2),
        "extension": ext,
        "expected_type": expected_type(path),
        "is_html_or_login": looks_like_html(path),
        "is_valid_zip": False,
        "zip_entries": None,
        "extracted": None,
        "status": "unknown",
    }

    if rec["is_html_or_login"]:
        rec["status"] = "invalid_html_or_login_page"
        return rec

    if ext == ".zip":
        valid, n, err = zip_status(path)
        rec["is_valid_zip"] = valid
        rec["zip_entries"] = n
        if not valid:
            rec["status"] = f"invalid_zip: {err}"
        elif size < MIN_VALID_ZIP_BYTES:
            rec["status"] = "suspicious_small_zip"
        else:
            rec["status"] = "valid_zip"
    else:
        rec["status"] = "non_zip_file"
    return rec


def detect_extracted() -> dict[str, bool]:
    """Report whether key extracted RF2/RRF files exist."""
    checks = {
        "snomed_concept": RAW / "snomed" / "us_edition" / "snomed_extract",
        "umls_mrconso": RAW / "umls" / "2026AA" / "umls_extract" / "MRCONSO.RRF",
        "rxnorm_rxnconso": RAW / "rxnorm" / "rxnorm_extract" / "RXNCONSO.RRF",
    }
    out: dict[str, bool] = {}
    for key, p in checks.items():
        if p.is_dir():
            out[key] = any(p.glob("sct2_*")) if "snomed" in key else any(p.iterdir())
        else:
            out[key] = p.exists()
    return out


def render_md(records: list[dict[str, Any]], extracted: dict[str, bool]) -> str:
    lines = ["# Raw Ontology Asset Inventory (Phase 2b)\n"]
    lines.append(
        f"Scanned: `{RAW.relative_to(PROJECT_ROOT)}` — {len(records)} files.\n"
    )
    lines.append("| Path | Size (MB) | Type | Valid zip | Entries | Status |")
    lines.append("|---|---:|---|:---:|---:|---|")
    for r in sorted(records, key=lambda x: x["path"]):
        lines.append(
            f"| {r['path']} | {r['size_mb']} | {r['expected_type']} | "
            f"{'yes' if r['is_valid_zip'] else 'no'} | {r['zip_entries'] if r['zip_entries'] is not None else '-'} | {r['status']} |"
        )
    lines.append("\n## Extracted working files present\n")
    for k, v in extracted.items():
        lines.append(f"- `{k}`: {'yes' if v else 'no'}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2b raw ontology inventory.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise FileNotFoundError(f"Raw ontology directory not found: {RAW}")

    records = [inventory_file(p) for p in sorted(RAW.rglob("*")) if p.is_file()]
    extracted = detect_extracted()

    report = {
        "phase": "2b",
        "scanned_dir": str(RAW.relative_to(PROJECT_ROOT)),
        "n_files": len(records),
        "n_valid_zips": sum(1 for r in records if r["is_valid_zip"]),
        "n_invalid_or_html": sum(
            1 for r in records if r["status"].startswith("invalid")
        ),
        "extracted_working_files": extracted,
        "assets": records,
    }
    (out_dir / "raw_asset_inventory.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "raw_asset_inventory.md").write_text(
        render_md(records, extracted), encoding="utf-8"
    )
    print(
        f"[inventory] {len(records)} files; valid zips: {report['n_valid_zips']}; "
        f"invalid/html: {report['n_invalid_or_html']}"
    )
    print(f"[inventory] wrote {out_dir / 'raw_asset_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
