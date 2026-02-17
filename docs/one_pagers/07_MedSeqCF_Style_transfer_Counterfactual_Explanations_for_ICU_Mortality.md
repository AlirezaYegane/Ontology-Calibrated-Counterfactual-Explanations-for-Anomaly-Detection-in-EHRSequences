# [7] MedSeqCF (and augmented MedSeqCF): Style-transfer counterfactual explanations for ICU mortality

**Primary source read (open-access):** Zhendong Wang’s dissertation *Constrained Counterfactual Explanations for Temporal Data* (DiVA portal, includes Paper II: “Style-transfer counterfactual explanations: An application to mortality prevention of ICU patients”).

**Authors (Paper II):** Wang, Samsten, Kougia, Papapetrou  
**Venue/Year (Paper II):** *Artificial Intelligence in Medicine*, 2023 (DOI: 10.1016/j.artmed.2022.102457)

## What problem does it target?
Generate **counterfactual explanations** for clinical sequence models ("what minimal changes to an event sequence would flip a prediction") with a focus on **ICU mortality**.

## Data / representation
- **Representation:** each patient is represented as a **medical event sequence** (discrete events, ordered in time).
- **Dataset (in the dissertation summary):** **MIMIC-III**; Paper I uses a cardiovascular cohort; Paper II expands to **cardiovascular, sepsis, and ARDS** cohorts.

## Core idea / method
- Builds counterfactual generation for sequences by adapting a **text style-transfer** framework (Delete–Retrieve–Generate / DRG) to medical event sequences.
- **MedSeqCF (Paper I):** adaptation of DRG to produce a counterfactual event sequence.
- **Augmented MedSeqCF (Paper II):** adds domain-specific medical information to improve clinical relevance:
  - **MedSeqCF-D:** incorporates **historical diagnosis** information.
  - **MedSeqCF-C:** incorporates **event coexistence** signals (coexisting medical treatments).
  - **MedSeqCF-DC:** uses both diagnosis + coexistence mechanisms.
- **Black-box model used in the dissertation’s description:** a **2-layer bidirectional LSTM** is trained as the predictor.

## Evaluation (as described in the dissertation)
- Baselines include **1-NN** (nearest neighbor counterfactual in sequence space) and the **initial MedSeqCF** (without added medical inputs).
- Reported metrics include **validity** and similarity/plausibility style metrics (e.g., LOF, BLEU-4, edit distance).
- Paper II additionally includes **quantitative support analysis with domain experts** to validate clinical relevance scores.

## Key takeaways we can reuse
- **Feasibility:** Framing counterfactuals for discrete EHR sequences as a style-transfer problem works, and provides an actionable template for our “counterfactual generator” module.
- **Clinical constraints matter:** Adding diagnosis and coexistence information is a concrete example of injecting **medical knowledge** to reduce unrealistic edits.
- **Metric suite:** validity + minimality/edit distance + plausibility-style metrics is aligned with how we should evaluate our counterfactuals.

## Limitations / risks (project view)
- Style-transfer generation can still produce **clinically implausible** or **non-actionable** edits unless constraints are strong.
- Similarity metrics (BLEU/edit distance) can be misleading if they reward surface similarity over clinical validity.

## How it maps to our system
- **Counterfactual module starting point:** use MedSeqCF as a baseline counterfactual generator for sequences.
- **Where we extend:** replace / augment style-transfer priors with **diffusion-based generation** and enforce **ontology-calibrated constraints** (valid code hierarchy, permissible transitions, plausible co-treatments).

## Verification sources
- Open dissertation (includes Paper II summary + method details): DiVA portal PDF.
- Implementation reference: GitHub repo **zhendong3wang/counterfactuals-for-event-sequences**.
