"""
scripts/run_phase7_external_validation_check.py
===============================================
Phase 7 -- External validation feasibility check (eICU).

Checks whether an external dataset (eICU) exists and whether the real ontology
scorer can actually score it. Does NOT fabricate results: if the schema is
incompatible, it writes an honest feasibility report.

Empirical test: the ontology scorer maps ICD/SNOMED/RxNorm tokens; eICU uses
APACHE / body-system tokens with no crosswalk to the ontology, so it will map
~0 tokens and fire ~0 rules -- confirming the mismatch quantitatively.

  python scripts/run_phase7_external_validation_check.py

Outputs: artifacts/phase7/external_validation_status.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_ONT = PROJECT_ROOT / "ontologies" / "processed"
OUT = PROJECT_ROOT / "artifacts" / "phase7"
EICU = PROJECT_ROOT / "data" / "processed" / "eicu_sequences.parquet"

# benchmark-v2 token prefixes the ontology scorer understands.
_MAPPABLE_PREFIXES = (
    "DX_10_",
    "DX_9_",
    "MED_",
    "DRUG_",
    "RAW_DRUG",
    "RX",
    "SNOMED:",
    "RXNORM:",
)


def run(sample: int = 500) -> dict[str, Any]:
    result: dict[str, Any] = {
        "phase": 7,
        "external_dataset": "eICU",
        "path": str(EICU.relative_to(PROJECT_ROOT)) if EICU.exists() else None,
    }

    if not EICU.exists():
        result.update(
            status="external_validation_blocked_missing_data",
            detail="No eICU (or other external) processed sequences found.",
        )
        _write(result)
        return result

    import pandas as pd

    from src.experiments.eval_common import ontology_scores
    from src.scoring.ontology_aware import (
        OntologyAwareScorer,
        ScoreWeights,
        map_tokens_to_ontology_codes,
    )

    df = pd.read_parquet(EICU).head(sample)
    seq_col = "sequence_tokens" if "sequence_tokens" in df.columns else None
    scorer = OntologyAwareScorer.from_processed_dir(
        PROCESSED_ONT, ontology_mode="real", weights=ScoreWeights()
    )

    # sample token vocabulary + mapping rate
    all_tokens: list[str] = []
    for s in df[seq_col]:
        if isinstance(s, (list, tuple)):
            all_tokens.extend(str(t) for t in s)
    uniq = sorted(set(all_tokens))
    n_prefix_mappable = sum(
        1
        for t in uniq
        if str(t).upper().startswith(tuple(p.upper() for p in _MAPPABLE_PREFIXES))
    )
    mapped_total = 0
    for s in df[seq_col]:
        toks = [str(t) for t in s] if isinstance(s, (list, tuple)) else []
        mapped_total += len(
            map_tokens_to_ontology_codes(toks, scorer.engine.index)["mapped_codes"]
        )

    # how often the ontology fires on eICU rows
    records = [
        {
            "seq": [str(t) for t in s] if isinstance(s, (list, tuple)) else [],
            "gender": g,
            "age_group": a,
        }
        for s, g, a in zip(df[seq_col], df.get("gender"), df.get("age_group"))
    ]
    ont = ontology_scores(scorer, records)
    fired = sum(1 for v in ont if v > 0)

    compatible = n_prefix_mappable > 0 and mapped_total > 0
    result.update(
        {
            "n_sampled_records": int(len(df)),
            "n_unique_tokens_sampled": len(uniq),
            "example_tokens": uniq[:8],
            "sequence_column": seq_col,
            "n_tokens_prefix_mappable": n_prefix_mappable,
            "total_ontology_codes_mapped": int(mapped_total),
            "n_records_ontology_fired": int(fired),
            "has_gender": "gender" in df.columns,
            "has_age_group": "age_group" in df.columns,
            "schema_compatible_with_ontology": bool(compatible),
            "status": "external_validation_completed"
            if compatible
            else "external_validation_blocked_schema_mismatch",
            "detail": (
                "eICU uses APACHE / body-system tokens (e.g. EICU_APACHE2_DX:*, "
                "EICU_BODYSYS:*), NOT ICD-10/ICD-9 or RxNorm. The ontology scorer + "
                "Phase 3b rule packs are keyed on ICD->SNOMED / drug->RxNorm mappings, "
                "which do not cover APACHE codes -> ~0 tokens map and ~0 rules fire. "
                "External validation would require (a) an APACHE->ICD/SNOMED crosswalk "
                "and (b) applying the benchmark-v2 anomaly injectors to eICU. Both are "
                "out of Phase 7 scope and are documented as future work."
            )
            if not compatible
            else "eICU tokens map to the ontology; aggregate external validation is feasible.",
            "recommended_paper_scope": "MIMIC-IV benchmark-v2 only; external validation = future work (pending APACHE crosswalk).",
        }
    )
    _write(result)
    return result


def _write(result: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "external_validation_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    md = [
        "# Phase 7 -- External Validation Status\n",
        f"**Status:** `{result['status']}`  |  dataset: {result['external_dataset']}\n",
        result.get("detail", ""),
    ]
    if "n_tokens_prefix_mappable" in result:
        md += [
            "\n## Empirical schema check",
            f"- sampled records: {result['n_sampled_records']}, unique tokens: {result['n_unique_tokens_sampled']}",
            f"- example tokens: {result['example_tokens']}",
            f"- ontology-mappable tokens: **{result['n_tokens_prefix_mappable']}** / total ontology codes mapped: **{result['total_ontology_codes_mapped']}**",
            f"- records where the ontology fired: **{result['n_records_ontology_fired']}**",
            f"- schema compatible with ontology: **{result['schema_compatible_with_ontology']}**",
            f"\n**Recommended paper scope:** {result['recommended_paper_scope']}",
        ]
    (OUT / "external_validation_status.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[phase7][external] status={result['status']}")


def main(argv: list[str] | None = None) -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
