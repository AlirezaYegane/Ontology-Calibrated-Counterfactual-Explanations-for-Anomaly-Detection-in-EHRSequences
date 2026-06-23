# Phase 0 — Claim Re-scoping and Scientific Contract

This folder records the scientific contract for the project: exactly what may be
claimed, what needs new evidence, and what must be removed unless repaired. It exists
to enforce honesty before any further modeling.

## Contents
- `phase0_summary.json` — machine-readable claim matrix with status labels and buckets.

## Companion documents (in `docs/paper/`)
- `claims_matrix.md` — full claim-by-claim contract with evidence for/against.
- `contribution_matrix.md` — candidate contributions → paper plan and readiness.
- `revised_scientific_story.md` — the honest one-paragraph narrative we will defend.

## Bottom line
- **Safe to claim now:** real MIMIC-IV preprocessing (C1); supervised detector
  ROC-AUC ≈ 0.80 *on the current synthetic benchmark* (C3, number only).
- **Must be removed unless repaired:** non-circular benchmark (C2b), ontology ranking
  gain (C4), real ontology integration (C5), additive decomposition (C6), generative
  surprise (C7), realistic generation (C8), diffusion counterfactual (C9), ontology
  counterfactual repair (C10), and the A*-novelty framing (C12).
- **Diffusion** is provisionally demoted to a diagnostic component, pending the Phase 5
  decision gate.

See `../diagnostics/rf2_triviality.md` for the Phase 1a evidence that forces the C2b
"cut_unless_fixed" status.
