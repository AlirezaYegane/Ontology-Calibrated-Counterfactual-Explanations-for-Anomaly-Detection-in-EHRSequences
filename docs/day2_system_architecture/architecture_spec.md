# Day 2 — System Architecture Design (Medium)
## Project: Ontology-Regularized Anomaly Detection + Counterfactual Explanations for EHR

### 1. Purpose
This document defines the end-to-end system architecture for detecting anomalies in electronic health record (EHR) event sequences and generating actionable, ontology-guided counterfactual explanations.

Day 2 outputs:
- High-level architecture diagram
- Module-by-module specification (inputs, outputs, responsibilities)
- Data flow + intermediate representations
- Feasibility notes aligned with project goals

---

### 2. Scope (MVP for Implementation)
**In scope**
- EHR record representation (sequence of coded events + demographics)
- Baseline anomaly detector module producing `S_det`
- Diffusion-based generative model producing `S_gen`
- Ontology engine (mappings, graph structures, rule checks) producing `S_ont`
- Calibrated anomaly score: `S_cal`
- Counterfactual generator (minimal edits) producing `X*`
- Explanation interface producing human-readable outputs

**Out of scope (Day 2)**
- Training pipelines and full model tuning
- Production deployment, scaling, full UI
- Exhaustive clinical validation

---

### 3. Core Data Representations

#### 3.1 Normalized Code Spaces
To enable reasoning and counterfactual edits, raw codes are normalized into:
- **SNOMED CT** for diagnoses / clinical concepts
- **RxNorm** for medications (or a consistent medication ontology)

Raw codes (e.g., ICD, NDC) must be preserved for traceability.

#### 3.2 Record (primary unit of inference)
A single EHR instance (e.g., admission-level or ICU-stay-level) is represented as:

```json
{
  "subject_id": "string|int",
  "encounter_id": "string|int",
  "demographics": { "sex": "M|F|Other|Unknown", "age": 0 },
  "events": ["SNOMED:...", "RXNORM:...", "..."],
  "raw_codes": ["ICD:...", "NDC:...", "..."],
  "meta": { "dataset": "MIMIC/eICU", "time_basis": "admission|icustay", "version": "v1" }
}
events is the canonical list used by all models.
If time is available, events may be time-bucketed; for MVP it can be a flattened sequence.

3.3 ScorePack (intermediate representation)
All scoring modules must output a consistent structure:

{
  "S_det": 0.0,
  "S_gen": 0.0,
  "S_ont": 0.0,
  "S_cal": 0.0,
  "violations": [
    { "rule_id": "R01", "severity": "low|med|high", "message": "..." }
  ],
  "attributions": [
    { "code": "SNOMED:...", "importance": 0.0 }
  ]
}
3.4 Counterfactual Output
Counterfactual generation returns:

{
  "original_record_id": "...",
  "counterfactual_record": { "...": "Record" },
  "edits": [
    { "op": "add|remove|replace", "from": "CODE_A", "to": "CODE_B", "reason": "..." }
  ],
  "scores_before": { "...": "ScorePack" },
  "scores_after": { "...": "ScorePack" }
}
4. System Architecture Overview (Modules)
4.1 Preprocessing & Mapping Module
Responsibilities

Extract per-encounter event sequences from raw EHR tables

Normalize codes into SNOMED/RxNorm via mapping dictionaries (e.g., UMLS-derived)

Create train/val/test splits and vocabularies

Input

Raw EHR tables (diagnoses, procedures, prescriptions, encounters, demographics)

Output

Record[] (normalized) + mapping logs + vocabulary artifacts

Notes

Preserve raw_codes for auditability

Maintain an unknown/unmapped token strategy

4.2 Baseline Anomaly Detector (Sequence Model)
Responsibilities

Compute detection score reflecting likelihood/surprise under normal patterns

Input

Record.events (sequence tokens)

Output

S_det(record) (float)

Optional: token-level surprise/attribution

Model options

GRU / LSTM / Transformer encoder

4.3 Ontology Engine (Mappings + Graph + Rules)
Responsibilities

Mappings (ICD→SNOMED, NDC/drug→RxNorm)

Graph traversal and neighborhood queries

Rule-based checks (contraindications, contradictions, demographic incompatibility, etc.)

Human-readable labels for codes

Input

Record (events + demographics)

Output

violations[] and S_ont(record) (float)

4.4 Diffusion-Based Generative Model (Ontology-Regularized)
Responsibilities

Model distribution of plausible EHR event sequences

Output generative anomaly score and “repair suggestions”

Input

Record.events

Output

S_gen(record) (float)

Optional: reconstructed/denoised candidates

4.5 Score Orchestrator & Calibration
Responsibilities

Combine anomaly evidence into a calibrated score for counterfactual search

Inputs

S_det, S_gen, S_ont, and violations

Output

ScorePack with S_cal(record)

Calibration (MVP)

S_cal(X) = w1 * S_gen(X) + w2 * S_ont(X) with default w2 > w1

4.6 Counterfactual Generator (Ontology-Constrained Minimal Edits)
Responsibilities

Find minimally modified record X* that reduces anomaly score and resolves key violations

Allowed edit operations

add / remove / replace (ontology-valid neighborhood)

Objective

Minimize J(X') = λ * (#edits) + S_cal(X') subject to ontology validity and optional constraints.

4.7 Explanation Interface
Responsibilities

Human-readable summary: why flagged + what minimal edit fixes it + score changes.

5. End-to-End Data Flow
Raw EHR → Preprocessing → Record
Record → Detector (S_det)
Record → Diffusion (S_gen)
Record → Ontology (S_ont, violations)
Orchestrator → ScorePack with S_cal
Counterfactual search → X*
Explanation → report + artifacts

6. Feasibility & Alignment Check
Feasible for MVP due to stable I/O contracts (Record, ScorePack, CounterfactualOutput) and bounded counterfactual search.

Open decisions:

Encounter granularity

Sequence representation (bucketed vs flat)

Initial ruleset for MVP

Weights/thresholds (w1, w2, cutoff, Δ)
