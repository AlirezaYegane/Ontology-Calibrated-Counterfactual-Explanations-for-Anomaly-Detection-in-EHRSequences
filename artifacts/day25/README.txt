Day 25 — Ontology-aware scoring integration completed

Final status
- Day 25 technical integration: complete
- Day 25 semantic refinement: complete

What changed between run_ref and run_ref_v2
- ontology rules were aligned with the real token namespace used by the dataset
- demographic conflict rules were updated for DX_9_* and DX_10_* style codes
- initial medication-support heuristics were added
- Sont is now materially active instead of near-zero

Key outcome
- demographic_conflict now receives strong ontology contribution
- medication_mismatch now receives meaningful non-zero ontology contribution
- missing_diagnosis remains supported by ontology checks and combined scoring

Artifacts
- outputs/scoring/day25/run_ref/
- outputs/scoring/day25/run_ref_v2/
- artifacts/day25/day25_run1_summary.json
- artifacts/day25/day25_comparison_summary.json
