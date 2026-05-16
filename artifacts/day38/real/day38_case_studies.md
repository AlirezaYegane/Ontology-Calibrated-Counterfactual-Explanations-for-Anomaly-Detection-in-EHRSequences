# Day 38 — Explanation Case Studies

These examples are generated from counterfactual evaluation outputs. They are intended for research inspection and paper-oriented qualitative analysis.

> Important: explanations are data-quality / model-explanation statements, not clinical recommendations.

## Case 1: `0`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 0 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 64421. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 0, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 64421. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 2: `1`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 1 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 650. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 1, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 650. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 3: `2`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 2 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 Z3400. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 2, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 Z3400. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 4: `3`

- Anomaly type: `medication_mismatch`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.0000` → `0.0000`
- ΔScal: `1.0000`

### Short explanation
Record 3 was flagged as a medication-indication mismatch. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 25061. This changes the calibrated score from 1.0000 to 0.0000 (ΔScal=1.0000, 100.0% reduction).

### Research explanation
For case 3, the explanation generator classifies the example as a medication-indication mismatch. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.0000, and Scal_before=1.0000. The selected counterfactual applies 1 edit(s): add DX 9 25061. The resulting score is Scal_after=0.0000, giving ΔScal=1.0000 (100.0% relative reduction). Ontology/counterfactual evidence: a medication-related event appears without a sufficiently compatible diagnosis or indication context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 5: `4`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 4 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 N401. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 4, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 N401. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 6: `5`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 5 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 I420. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 5, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 10 I420. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 7: `6`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 6 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 6, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 8: `7`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 7 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 7, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 9: `8`

- Anomaly type: `medication_mismatch`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.0000` → `0.0000`
- ΔScal: `1.0000`

### Short explanation
Record 8 was flagged as a medication-indication mismatch. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 25000. This changes the calibrated score from 1.0000 to 0.0000 (ΔScal=1.0000, 100.0% reduction).

### Research explanation
For case 8, the explanation generator classifies the example as a medication-indication mismatch. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.0000, and Scal_before=1.0000. The selected counterfactual applies 1 edit(s): add DX 9 25000. The resulting score is Scal_after=0.0000, giving ΔScal=1.0000 (100.0% relative reduction). Ontology/counterfactual evidence: a medication-related event appears without a sufficiently compatible diagnosis or indication context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 10: `9`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 9 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 V220. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 9, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 V220. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 11: `10`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 10 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 30400. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 10, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 30400. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 12: `11`

- Anomaly type: `medication_mismatch`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.0000` → `0.0000`
- ΔScal: `1.0000`

### Short explanation
Record 11 was flagged as a medication-indication mismatch. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 E1051. This changes the calibrated score from 1.0000 to 0.0000 (ΔScal=1.0000, 100.0% reduction).

### Research explanation
For case 11, the explanation generator classifies the example as a medication-indication mismatch. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.0000, and Scal_before=1.0000. The selected counterfactual applies 1 edit(s): add DX 10 E1051. The resulting score is Scal_after=0.0000, giving ΔScal=1.0000 (100.0% relative reduction). Ontology/counterfactual evidence: a medication-related event appears without a sufficiently compatible diagnosis or indication context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 13: `12`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 12 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 Y848. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 12, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 10 Y848. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 14: `13`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 13 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 60001. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 13, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 60001. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 15: `14`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 14 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 14, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 16: `15`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 15 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 V220. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 15, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 V220. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 17: `16`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 16 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 16, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 18: `17`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 17 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 8248. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 17, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 8248. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 19: `18`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 18 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 Z3400. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 18, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 Z3400. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 20: `19`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 19 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 V220. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 19, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 V220. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 21: `20`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 20 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 Z3400. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 20, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 Z3400. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 22: `21`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 21 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 N401. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 21, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 N401. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 23: `22`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 22 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 42821. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 22, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 42821. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 24: `23`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 23 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 23, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 25: `24`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 24 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 Y92230. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 24, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 10 Y92230. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 26: `25`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 25 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 81500. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 25, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 81500. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 27: `26`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `4.0000` → `2.0000`
- ΔScal: `2.0000`

### Short explanation
Record 26 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 4.0000 to 2.0000 (ΔScal=2.0000, 50.0% reduction).

### Research explanation
For case 26, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=4.0000, and Scal_before=4.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=2.0000, giving ΔScal=2.0000 (50.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 28: `27`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 27 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 V220. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 27, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 V220. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 29: `28`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 28 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 650. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 28, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 650. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 30: `29`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 29 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 650. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 29, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 650. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 31: `30`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 30 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 C61. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 30, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 C61. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 32: `31`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 31 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 N401. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 31, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 N401. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 33: `32`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 32 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 99676. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 32, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 99676. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 34: `33`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 33 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 33, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 35: `34`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 34 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 O800. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 34, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 O800. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 36: `35`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 35 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 C61. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 35, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 C61. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 37: `36`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 36 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 E8889. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 36, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 E8889. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 38: `37`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 37 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 60001. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 37, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 60001. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 39: `38`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 38 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 1983. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 38, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 1983. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 40: `39`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 39 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 10 C61. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 39, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 10 C61. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 41: `40`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 40 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 9 30520. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 40, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 9 30520. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 42: `41`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 41 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 185. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 41, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 185. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 43: `42`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 42 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 650. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 42, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 650. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 44: `43`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 43 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 V220. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 43, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 V220. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 45: `44`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 44 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 L03114. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 44, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 10 L03114. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 46: `45`

- Anomaly type: `demographic_conflict`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `2.0000` → `0.0000`
- ΔScal: `2.0000`

### Short explanation
Record 45 was flagged as a demographic-consistency anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove DX 9 60001. This changes the calibrated score from 2.0000 to 0.0000 (ΔScal=2.0000, 100.0% reduction).

### Research explanation
For case 45, the explanation generator classifies the example as a demographic-consistency anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=2.0000, and Scal_before=2.0000. The selected counterfactual applies 1 edit(s): remove DX 9 60001. The resulting score is Scal_after=0.0000, giving ΔScal=2.0000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence contains a code pattern that is inconsistent with the available demographic context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 47: `46`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 46 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 Z6841. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 46, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 10 Z6841. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 48: `47`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 47 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 H04123. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 47, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 10 H04123. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 49: `48`

- Anomaly type: `missing_diagnosis`
- Primary driver: `ontology violation signal`
- Edit count: `1`
- Score change: `1.5000` → `0.0000`
- ΔScal: `1.5000`

### Short explanation
Record 48 was flagged as a possible missing-diagnosis / missing-indication anomaly. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: add DX 10 Z3A35. This changes the calibrated score from 1.5000 to 0.0000 (ΔScal=1.5000, 100.0% reduction).

### Research explanation
For case 48, the explanation generator classifies the example as a possible missing-diagnosis / missing-indication anomaly. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=1.5000, and Scal_before=1.5000. The selected counterfactual applies 1 edit(s): add DX 10 Z3A35. The resulting score is Scal_after=0.0000, giving ΔScal=1.5000 (100.0% relative reduction). Ontology/counterfactual evidence: the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.

## Case 50: `49`

- Anomaly type: `medication_mismatch`
- Primary driver: `ontology violation signal`
- Edit count: `2`
- Score change: `3.0000` → `0.0000`
- ΔScal: `3.0000`

### Short explanation
Record 49 was flagged as a medication-indication mismatch. The main evidence comes from the ontology violation signal. The proposed counterfactual repair is: remove MED PREGABALIN; add DX 10 E1165. This changes the calibrated score from 3.0000 to 0.0000 (ΔScal=3.0000, 100.0% reduction).

### Research explanation
For case 49, the explanation generator classifies the example as a medication-indication mismatch. The decomposed scores available to the text generator are Sdet=0.0000, Sgen=0.0000, Sont=3.0000, and Scal_before=3.0000. The selected counterfactual applies 2 edit(s): remove MED PREGABALIN; add DX 10 E1165. The resulting score is Scal_after=0.0000, giving ΔScal=3.0000 (100.0% relative reduction). Ontology/counterfactual evidence: a medication-related event appears without a sufficiently compatible diagnosis or indication context. The generative surprise score is reported as a diagnostic auxiliary signal only, because the current diffusion-based Sgen proxy was previously found to be weak for anomaly separation.
