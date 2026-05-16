# Day 39 — End-to-End Case Studies

## Purpose

This artifact documents representative end-to-end examples from the ontology-calibrated counterfactual explanation pipeline. The goal is to verify whether the generated explanations are coherent, clinically plausible, and faithful to the underlying scores and counterfactual edits.

## Summary

- Source file: `artifacts/day38/real/day38_explanations.csv`
- Number of selected cases: **10**
- Anomaly types covered: **demographic_conflict, medication_mismatch, missing_diagnosis**
- Mean ΔScal: **2.0500**
- Median edit count: **1.00**
- Automatic coherence pass rate: **100.00%**

## Research Interpretation

These cases should be treated as qualitative evidence for explanation behavior, not as clinical advice. The strongest paper-safe claims are score reduction, minimality of edits, ontology-rule alignment, and whether the explanation text faithfully reflects the computed signals.

The diffusion-based Sgen value is retained as an auxiliary diagnostic field. Because earlier evaluation found the current Sgen proxy weak for anomaly separation, the case-study interpretation should not rely on Sgen as the main evidence.

---

## Case 1: `demographic_conflict`

- Record ID: `26`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 9 185`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 4.000000 |
| Scal before | 4.000000 |
| Scal after | 2.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 26 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 9 185. After this edit, the calibrated anomaly score changes from 4.0000 to 2.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 2: `medication_mismatch`

- Record ID: `49`
- Label: `1`
- Edit count: **2**
- Action: `remove MED PREGABALIN; add DX 10 E1165`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 3.000000 |
| Scal before | 3.000000 |
| Scal after | 0.000000 |
| ΔScal | 3.000000 |

### Ontology Issues

- a medication-related event appears without a sufficiently compatible diagnosis or indication context

### Explanation

Record 49 appears unusual because a medication-related event appears without a sufficiently compatible diagnosis or indication context. The proposed counterfactual is to remove MED PREGABALIN; add DX 10 E1165. After this edit, the calibrated anomaly score changes from 3.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 3: `missing_diagnosis`

- Record ID: `0`
- Label: `1`
- Edit count: **1**
- Action: `add DX 9 64421`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 1.500000 |
| Scal before | 1.500000 |
| Scal after | 0.000000 |
| ΔScal | 1.500000 |

### Ontology Issues

- the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent

### Explanation

Record 0 appears unusual because the sequence appears to be missing a diagnosis or indication that would make the observed clinical events more coherent. The proposed counterfactual is to add DX 9 64421. After this edit, the calibrated anomaly score changes from 1.5000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 4: `demographic_conflict`

- Record ID: `1`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 9 650`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 2.000000 |
| Scal before | 2.000000 |
| Scal after | 0.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 1 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 9 650. After this edit, the calibrated anomaly score changes from 2.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 5: `demographic_conflict`

- Record ID: `14`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 9 185`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 2.000000 |
| Scal before | 2.000000 |
| Scal after | 0.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 14 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 9 185. After this edit, the calibrated anomaly score changes from 2.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 6: `demographic_conflict`

- Record ID: `4`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 10 N401`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 2.000000 |
| Scal before | 2.000000 |
| Scal after | 0.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 4 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 10 N401. After this edit, the calibrated anomaly score changes from 2.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 7: `demographic_conflict`

- Record ID: `7`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 9 185`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 2.000000 |
| Scal before | 2.000000 |
| Scal after | 0.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 7 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 9 185. After this edit, the calibrated anomaly score changes from 2.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 8: `demographic_conflict`

- Record ID: `6`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 9 185`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 2.000000 |
| Scal before | 2.000000 |
| Scal after | 0.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 6 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 9 185. After this edit, the calibrated anomaly score changes from 2.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 9: `demographic_conflict`

- Record ID: `9`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 9 V220`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 2.000000 |
| Scal before | 2.000000 |
| Scal after | 0.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 9 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 9 V220. After this edit, the calibrated anomaly score changes from 2.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Case 10: `demographic_conflict`

- Record ID: `15`
- Label: `1`
- Edit count: **1**
- Action: `remove DX 9 V220`

### Scores

| Score | Value |
|---|---:|
| Sdet | 0.000000 |
| Sgen | 0.000000 |
| Sont | 2.000000 |
| Scal before | 2.000000 |
| Scal after | 0.000000 |
| ΔScal | 2.000000 |

### Ontology Issues

- the sequence contains a code pattern that is inconsistent with the available demographic context

### Explanation

Record 15 appears unusual because the sequence contains a code pattern that is inconsistent with the available demographic context. The proposed counterfactual is to remove DX 9 V220. After this edit, the calibrated anomaly score changes from 2.0000 to 0.0000. This is a sparse counterfactual explanation because the score improves with only a small number of edits. This should be interpreted as a data-quality and explanation signal, not as a clinical recommendation.

---

## Day 39 Status

Day 39 is complete if:

- The selected cases cover multiple anomaly types.
- Each case includes before/after scores or a clear explanation artifact.
- Counterfactual actions are minimal and interpretable where available.
- The final markdown can be reused as a Results / Case Study appendix.
