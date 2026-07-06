# Abstract

Electronic health record (EHR) sequences contain documentation errors and clinically
incoherent combinations of codes that are hard to surface and harder to explain. Purely
statistical rarity is a poor detector for these cases, because the offending codes are
individually common; the incoherence lives in the *relationship* between fields (for
example, a sex-specific diagnosis paired with the wrong recorded sex, or a medication
without its required indication). We study whether a real medical ontology can both flag
such relational anomalies and explain them with minimal, coherent repairs.

We first build **benchmark-v2**, a leakage-controlled anomaly benchmark on MIMIC-IV in
which every anomaly is a relational violation while every individual model-visible token
stays common. Subject-level splits have zero overlap, model-visible fields are separated
from answer keys, and the strongest label-free trivial signal is 0.61 (below the 0.80
non-circularity gate), in contrast to an earlier circular benchmark where a single
token-presence feature recovered the label at 0.94.

On this benchmark, a **real UMLS / SNOMED CT / RxNorm ontology scorer** with three
auditable rule families reaches ROC-AUC 0.7881 (95% CI [0.774, 0.802]), significantly
above a legacy ICD-prefix baseline (0.7358; paired bootstrap +0.052, p ≈ 0). We report two
transparent negative results: a full-scale unsupervised sequence detector is *below chance*
(0.4525) and, when combined with the ontology score, significantly *lowers* ROC-AUC
(−0.085, p ≈ 0); and a diffusion-based generative-surprise term is near-random and
mode-collapsed and is removed from the core method. The recommended method is therefore
**ontology-only**.

Finally, we present a **leakage-free, ontology-guided counterfactual repair** that never
reads the corruption answer key: it proposes minimal ontology-valid edits validated by an
independent scorer, achieving 89.99% valid repair among ontology-flagged anomalies with a
median of one edit. All evidence is on MIMIC-IV benchmark-v2; external validation on eICU
is blocked by a token-schema mismatch (APACHE/body-system codes do not map to the
ICD/SNOMED/RxNorm ontology) and is left as future work. The contribution is a
leakage-controlled benchmark, a real ontology scorer that beats the legacy baseline, and a
leakage-free counterfactual repair method — reported honestly alongside the components that
did not work.
