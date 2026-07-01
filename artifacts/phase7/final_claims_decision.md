# Phase 7 — Final Claim Decisions (benchmark-v2)

Blunt and evidence-based. Unsupported claims are **not** upgraded.

| # | Claim | Decision | Key evidence |
|---|---|---|---|
| 1 | Benchmark-v2 is non-circular / leakage-controlled | **supported_now** | trivial signal 0.6127 < 0.80; subject-split; field separation |
| 2 | Real ontology integration | **supported_now** | UMLS/SNOMED/RxNorm; coverage 0.80/0.78 |
| 3 | Real ontology beats legacy rules | **supported_now** | 0.7881 vs 0.7358; paired +0.052, CI [0.033,0.072], p≈0 |
| 4 | Unsupervised detector improves detection | **unsupported** | full-scale detector 0.4525 (below chance) |
| 5 | Combined detector+ontology beats ontology-only | **unsupported** | 0.7036 < 0.7881; paired −0.085, p≈0 (hurts) |
| 6 | Sgen/diffusion improves detection | **removed_from_core** | 0.4868 below chance, harms combined (Phase 5) |
| 7 | Counterfactual repair is leakage-free | **supported_now** | leakage tests pass; answer keys change nothing |
| 8 | Counterfactual repair effective for ontology-flagged anomalies | **supported_now** | 89.99% valid among flagged; median 1 edit |
| 9 | Clinical validity externally validated | **future_work** | no clinician validation; validity = rule resolution |
| 10 | External dataset generalization | **future_work** | eICU schema mismatch (0/500 map); needs APACHE crosswalk |
| 11 | Method is reproducible | **supported_now** | Phase 6 config/resume/index; deterministic seeds |

## Headline
Real ontology rules provide the **main anomaly-ranking signal** on benchmark-v2
(0.79, significantly above legacy 0.74). The unsupervised detector is **below chance
and non-additive** for relational anomalies. **Sgen is removed** from the core.
Counterfactual repair is **leakage-free and effective (~90%)** for ontology-flagged
anomalies. External validation and clinical validation are **future work**.

**Recommended paper main score:** `S_main = S_ont` (ontology-centered), with the
detector reported as a diagnostic/negative result and Sgen excluded.
