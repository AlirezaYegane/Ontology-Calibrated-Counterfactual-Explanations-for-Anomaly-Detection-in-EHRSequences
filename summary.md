# Day 1 Summary (Literature Survey) — Updated

This document consolidates the initial literature survey into a module-aligned map and a set of defensible design decisions for the proposal pipeline.

## What we collected today

- **Total core papers:** 10 (across representation learning, anomaly detection, generative modeling, ontologies/KG, counterfactuals, and interpretability)

## Module-aligned takeaways (actionable)

- **Backbone encoder candidates:** Med-BERT + CEHR-BERT show that BERT-style pretraining can transfer to structured EHR and that explicit time encoding improves predictions.

- **Ontology-aware semantics:** GRAM and G-BERT demonstrate how hierarchical medical code structure can regularize embeddings and improve downstream tasks.

- **Anomaly detection tracks:** DeepAnT gives a strong baseline for continuous signals; EHR-BERT tailors masked-token prediction for discrete EHR anomaly detection.

- **Generative backbone:** EHRDiff indicates diffusion models can synthesize realistic EHR while considering privacy risk.

- **Counterfactuals:** MedSeqCF (and augmented MedSeqCF; details available in the author's open dissertation + official repo) applies sequence-style transfer to generate counterfactual event sequences for ICU cohorts.

- **Interpretability:** SHy proposes self-explaining hypergraph phenotypes for concise/flexible explanations.

- **Rare events/imbalance:** CONAN motivates uncertainty-aware augmentation when positives are extremely rare.


## Design decisions we can justify now (v1)

1) **Use a two-stream detector**: (a) continuous signals baseline (e.g., CNN-forecasting style) + (b) discrete event anomaly scoring using masked-token prediction.

2) **Introduce ontology calibration as a constraint layer**: represent codes with hierarchy-aware embeddings and use ontology violations as an additional anomaly signal.

3) **Use diffusion as the generative prior** for realistic EHR edits/synthesis, but **constrain sampling** with clinical + ontology plausibility checks.

4) **Counterfactual generation must be constrained**: minimize edits, respect temporal order, and enforce clinical/ontology validity; MedSeqCF informs feasibility but needs stricter constraints.

5) **Evaluation must be multi-axis**: detection (AUC/PR-AUC), explanation validity, plausibility, distance/edit size, and privacy/leakage checks.


## Sources (starting points)

- Paper [1]: https://www.nature.com/articles/s41746-021-00455-y

- Paper [2]: https://proceedings.mlr.press/v158/pang21a.html

- Paper [3]: https://dl.acm.org/doi/10.1145/3097983.3098126

- Paper [4]: https://www.ijcai.org/proceedings/2019/825

- Paper [5]: https://www.dfki.de/fileadmin/user_upload/import/10175_DeepAnt.pdf

- Paper [6]: https://www.sciencedirect.com/science/article/pii/S1532046424000236

- Paper [7]: https://pubmed.ncbi.nlm.nih.gov/36628793/
- Paper [7] (official repo): https://github.com/zhendong3wang/counterfactuals-for-event-sequences
- Paper [7] (open dissertation write-up): https://www.diva-portal.org/smash/get/diva2:1906268/FULLTEXT03.pdf


- Paper [8]: https://arxiv.org/abs/2303.05656

- Paper [9]: https://proceedings.mlr.press/v287/yu25a.html

- Paper [10]: https://ojs.aaai.org/index.php/AAAI/article/view/5401
