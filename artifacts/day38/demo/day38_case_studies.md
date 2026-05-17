# Day 38 — Explanation Case Studies

These examples are generated from counterfactual evaluation outputs. They are intended for research inspection and paper-oriented qualitative analysis.

> Important: explanations are data-quality / model-explanation statements, not clinical recommendations.

## Case 1: `demo_demographic_001`

- Anomaly type: `demographic_conflict`
- Primary driver: `mixed detector-and-ontology signal`
- Edit count: `1`
- Score change: `0.9400` → `0.1800`
- ΔScal: `0.7600`

### Short explanation
Record demo_demographic_001 was flagged as a demographic-consistency anomaly. The main evidence comes from the mixed detector-and-ontology signal. The proposed counterfactual repair is: remove incompatible pregnancy-related code. This changes the calibrated score from 0.9400 to 0.1800 (ΔScal=0.7600, 80.9% reduction).

### Research explanation
For case demo_demographic_001, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores are Sdet=0.9100, Sgen=0.1200, Sont=1.0000, and Scal_before=0.9400. The selected counterfactual applies 1 edit(s): remove incompatible pregnancy-related code. The resulting score is Scal_after=0.1800, giving ΔScal=0.7600 (80.9% relative reduction). Ontology evidence: sex-incompatible code detected. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 2: `demo_medication_001`

- Anomaly type: `medication_mismatch`
- Primary driver: `mixed detector-and-ontology signal`
- Edit count: `1`
- Score change: `0.7300` → `0.3100`
- ΔScal: `0.4200`

### Short explanation
Record demo_medication_001 was flagged as a medication-indication mismatch. The main evidence comes from the mixed detector-and-ontology signal. The proposed counterfactual repair is: add compatible diagnosis / indication code. This changes the calibrated score from 0.7300 to 0.3100 (ΔScal=0.4200, 57.5% reduction).

### Research explanation
For case demo_medication_001, the explanation generator classifies the example as a medication-indication mismatch. The decomposed scores are Sdet=0.6800, Sgen=0.0800, Sont=0.7000, and Scal_before=0.7300. The selected counterfactual applies 1 edit(s): add compatible diagnosis / indication code. The resulting score is Scal_after=0.3100, giving ΔScal=0.4200 (57.5% relative reduction). Ontology evidence: medication appears without a compatible indication. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 3: `demo_missing_dx_001`

- Anomaly type: `missing_diagnosis`
- Primary driver: `mixed detector-and-ontology signal`
- Edit count: `1`
- Score change: `0.4200` → `0.3400`
- ΔScal: `0.0800`

### Short explanation
Record demo_missing_dx_001 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the mixed detector-and-ontology signal. The proposed counterfactual repair is: add expected diagnosis code. This changes the calibrated score from 0.4200 to 0.3400 (ΔScal=0.0800, 19.0% reduction).

### Research explanation
For case demo_missing_dx_001, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores are Sdet=0.3800, Sgen=0.1000, Sont=0.3500, and Scal_before=0.4200. The selected counterfactual applies 1 edit(s): add expected diagnosis code. The resulting score is Scal_after=0.3400, giving ΔScal=0.0800 (19.0% relative reduction). Ontology evidence: expected diagnosis not present. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.
