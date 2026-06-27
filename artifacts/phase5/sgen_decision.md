# Phase 5 -- Sgen Decision

## Decision: `remove_from_core`

- Sgen ROC-AUC 0.4868 is near-random or worse (<= 0.52).
- generation is mode-collapsed.
- keeping Sgen would weaken the paper's clarity/strength.

`w_gen` default stays **0.0**; Sgen in core score: **False**.
