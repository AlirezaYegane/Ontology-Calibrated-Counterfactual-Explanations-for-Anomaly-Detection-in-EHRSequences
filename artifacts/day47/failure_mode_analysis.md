# Day 47 — Risk and Failure Mode Analysis

## Status

Complete.

## Goal

Systematically identify failure modes in the ontology-calibrated anomaly explanation pipeline and define mitigation strategies suitable for a research paper.

## Why this matters for the paper

A publishable system should not only report positive metrics. It should also explain when the method may fail, how those failures are detected, and what safeguards or conservative interpretations are used.

## Evidence inventory summary

- Evidence files scanned: 75
- Measured failure modes: 4
- Not directly measured from available artifacts: 3
- Conceptual research risks: 2

## Failure mode matrix

| ID | Failure mode | Evidence status | Severity | Likelihood | Mitigation |
|---|---|---|---|---|---|
| FM-01 | Ambiguous or incomplete ontology mappings | measured | high | high | Report mapping coverage; compute Sont only for covered rule families; keep unmapped/ambiguous codes explicit rather than silently repairing them. |
| FM-02 | Uneven anomaly-family separation | measured | high | high | Report per-family breakdowns; tune thresholds per anomaly family if justified; add targeted rules/examples for weak families such as missing-diagnosis cases. |
| FM-03 | Weak diffusion-based generative surprise signal | measured | high | high | Keep Sgen as an auxiliary/diagnostic term; report detector and ontology-calibrated scores separately; use conservative weighting for w_gen. |
| FM-04 | Threshold sensitivity and precision-recall trade-off | not directly measured in available artifacts | medium | unknown | Export threshold_sweep.csv for every final evaluation run. |
| FM-05 | Unstable or non-minimal counterfactuals | measured | medium | medium | Prefer one- or two-edit candidates; flag no-improvement cases; report edit-count distribution and score reduction. |
| FM-06 | Records with multiple simultaneous issues | not directly measured in available artifacts | medium | unknown | Add violation_count / issue_count to exported explanation outputs. |
| FM-07 | Sensitivity to missing data and rare codes | conceptual risk | medium | high | Separate ontology violations from statistical rarity; report rare-code coverage; avoid treating high Sgen alone as an error. |
| FM-08 | Explanation overclaiming or unclear wording | not directly measured in available artifacts | high | medium | Keep template disclaimers; perform manual review on selected examples; separate documentation/coding suggestions from treatment claims. |
| FM-09 | Bias or brittleness in demographic rules | conceptual risk | high | medium | Make demographic rule sets explicit; log demographic attributes used; allow rule disabling and manual review. |

## Detailed evidence

### FM-01 — Ambiguous or incomplete ontology mappings

- Evidence status: measured
- Evidence: Found mapping audit: artifacts/day13/mapping_audit.json. edge_case_counts={'unknown_namespace_tokens': 124669, 'duplicate_tokens_within_record': 216498, 'malformed_tokens': 7821}
- Scientific impact: Ontology-based scores can be biased if mapping coverage is uneven across diagnoses, procedures, or medications.
- Mitigation: Report mapping coverage; compute Sont only for covered rule families; keep unmapped/ambiguous codes explicit rather than silently repairing them.

Paper-ready wording:

> Ontology-derived conclusions were restricted to code families with available mappings and rules; unmapped or ambiguous concepts were tracked as a coverage limitation.

### FM-02 — Uneven anomaly-family separation

- Evidence status: measured
- Evidence: From artifacts/day35/calibrated_scores.csv, hardest non-empty anomaly family by mean prob_anomaly is missing_diagnosis (mean=0.3476, median=0.2987, n=7552).
- Scientific impact: A single headline AUC may hide weaker performance on clinically important anomaly subtypes.
- Mitigation: Report per-family breakdowns; tune thresholds per anomaly family if justified; add targeted rules/examples for weak families such as missing-diagnosis cases.

Paper-ready wording:

> Performance was not uniform across anomaly categories; family-level breakdowns were therefore used to identify the main residual error modes.

### FM-03 — Weak diffusion-based generative surprise signal

- Evidence status: measured
- Evidence: Found artifacts/day34_final/day34_final_assessment.json; best timestep=12, Sgen ROC-AUC=0.5082, AP=0.5050.
- Scientific impact: Raw Sgen should not be overclaimed as a standalone anomaly detector.
- Mitigation: Keep Sgen as an auxiliary/diagnostic term; report detector and ontology-calibrated scores separately; use conservative weighting for w_gen.

Paper-ready wording:

> The diffusion-derived surprise proxy was retained as an auxiliary signal because its standalone discrimination was weak in the current benchmark.

### FM-04 — Threshold sensitivity and precision-recall trade-off

- Evidence status: not directly measured in available artifacts
- Evidence: No threshold_sweep CSV was found.
- Scientific impact: Cannot justify the selected operating threshold from available evidence.
- Mitigation: Export threshold_sweep.csv for every final evaluation run.

Paper-ready wording:

> Threshold-dependent metrics should be reported alongside threshold-free metrics such as ROC-AUC and average precision.

### FM-05 — Unstable or non-minimal counterfactuals

- Evidence status: measured
- Evidence: Found counterfactual/explanation evidence in artifacts/day36/counterfactuals.csv; rows=16442. mean_edits=1.008; pct_more_than_two_edits=0.000.
- Scientific impact: Explanations become less interpretable if they require many edits or fail to reduce the calibrated score.
- Mitigation: Prefer one- or two-edit candidates; flag no-improvement cases; report edit-count distribution and score reduction.

Paper-ready wording:

> Counterfactual quality was assessed using both score reduction and sparsity because clinically useful explanations should remain minimal.

### FM-06 — Records with multiple simultaneous issues

- Evidence status: not directly measured in available artifacts
- Evidence: No issue-count or violation-count column was found.
- Scientific impact: The current evidence may not distinguish simple anomalies from compound anomalies.
- Mitigation: Add violation_count / issue_count to exported explanation outputs.

Paper-ready wording:

> Future exports should track the number of simultaneous violations so that compound cases can be evaluated separately.

### FM-07 — Sensitivity to missing data and rare codes

- Evidence status: conceptual risk
- Evidence: This risk follows from EHR sparsity, incomplete medication/diagnosis coverage, and rare but valid clinical combinations.
- Scientific impact: The system may flag rare but valid patient trajectories or miss anomalies when required companion codes are absent.
- Mitigation: Separate ontology violations from statistical rarity; report rare-code coverage; avoid treating high Sgen alone as an error.

Paper-ready wording:

> Rare but ontology-consistent records were treated as review candidates rather than automatic errors.

### FM-08 — Explanation overclaiming or unclear wording

- Evidence status: not directly measured in available artifacts
- Evidence: No Day 46 plausibility review CSV with review labels was found.
- Scientific impact: Generated explanations may be misread as clinical advice if not carefully phrased.
- Mitigation: Keep template disclaimers; perform manual review on selected examples; separate documentation/coding suggestions from treatment claims.

Paper-ready wording:

> The interface and paper should explicitly state that counterfactual edits are explanatory, not prescriptive.

### FM-09 — Bias or brittleness in demographic rules

- Evidence status: conceptual risk
- Evidence: Demographic consistency rules can be useful but may become brittle if sex/age metadata or clinical context is incomplete.
- Scientific impact: Incorrect demographic assumptions can create false positives or inappropriate explanations.
- Mitigation: Make demographic rule sets explicit; log demographic attributes used; allow rule disabling and manual review.

Paper-ready wording:

> Demographic rules were implemented transparently and interpreted as data-consistency checks, not judgments about patient identity.

## Conservative interpretation policy

- Do not present counterfactual edits as treatment recommendations.
- Do not claim raw diffusion surprise is a strong standalone anomaly signal unless validated by separation metrics.
- Report ontology coverage and mapping limitations explicitly.
- Separate statistical rarity from ontology violation wherever possible.
- Report per-family anomaly behavior rather than relying only on aggregate scores.

## Day 47 conclusion

The main contribution of Day 47 is a paper-ready risk register and mitigation matrix. This strengthens the scientific framing of the project by making limitations explicit, measurable where possible, and connected to concrete engineering safeguards.
