# Day 1 Literature Map (v1)

> **Note:** These summaries are *abstract/official-page level* and should be confirmed/expanded after full PDF reading.


## Anomaly Detection (Rare Events)

- **[10] CONAN: Complementary Pattern Augmentation for Rare Event Detection** (2020, AAAI) — *Augmentation-driven rare event detection*  
  Link:     https://arxiv.org/abs/1911.13232


## Anomaly Detection in EHR/ICU

- **[5] DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series** (2019, IEEE Access) — *Unsupervised CNN-based forecasting/anomaly scoring*  
  Link:    https://doi.org/10.1109/ACCESS.2018.2886457

- **[6] EHR-BERT: A BERT-based Model for Effective Anomaly Detection in EHRs** (2024, Journal of Biomedical Informatics) — *Transformer anomaly detection via masked token prediction objective*  
  Link:    https://www.osti.gov/biblio/2351748


## EHR Representation Learning

- **[1] Med-BERT: Pre-trained Contextualized Embeddings on Large-scale Structured EHRs** (2021, NPJ Digital Medicine) — *Transformer/BERT; self-supervised pretraining; time encoding*  
  Link:    https://www.nature.com/articles/s41746-021-00455-y

- **[2] CEHR-BERT: Incorporating Temporal Information from Structured EHR Data** (2021, PMLR (ML4H)) — *Transformer with explicit temporal encoding for irregular intervals*  
  Link:    https://proceedings.mlr.press/v158/pang21a.html


## Generative Counterfactuals

- **[7] MedSeqCF / augmented MedSeqCF: Style-transfer counterfactual explanations for ICU mortality** (Artificial Intelligence in Medicine, 2023) — *Counterfactual generation for medical event sequences using style-transfer (DRG) + medical-knowledge augmentations*  
  Link(s):
  - PubMed (metadata/abstract): https://pubmed.ncbi.nlm.nih.gov/36628793/
  - GitHub (implementation): https://github.com/zhendong3wang/counterfactuals-for-event-sequences
  - Open dissertation (includes Paper II details): https://www.diva-portal.org/smash/get/diva2:1906268/FULLTEXT03.pdf


## Generative Modeling for Clinical Data

- **[8] EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models** (2023, TMLR) — *Diffusion models for EHR synthesis*  
  Link:    https://github.com/zzh16mz/EHRDiff


## Hypergraph / Interpretable Modeling

- **[9] SHy: Self-Explaining Hypergraph Neural Networks for Disease Diagnosis Prediction** (2025, CHIL) — *Hypergraph model with self-explaining components*  
  Link:    https://www.researchgate.net/scientific-contributions/Leisheng-Yu-2284841331


## Medical Ontologies / Knowledge Graph

- **[3] GRAM: Graph-based Attention Model for Healthcare Representation Learning** (2017, KDD) — *Ontology-aware attention over code ancestors*  
  Link:    https://doi.org/10.1145/3097983.3098126

- **[4] G-BERT: Pre-training of Graph Augmented Transformers for Medication Recommendation** (2019, IJCAI) — *Graph + Transformer fusion (GNN for ontology, BERT for sequence)*  
  Link:    https://www.ijcai.org/proceedings/2019/825    https://arxiv.org/abs/1906.00346


## Mapping to our pipeline modules

- **S_det (anomaly detector):** DeepAnT (continuous signals) + EHR-BERT (discrete EHR events) + CONAN (rare disease detection).

- **S_gen (generative model):** EHRDiff (diffusion for EHR synthesis) + (CONAN uses a GAN-style augmentation for rare disease detection).

- **S_ont (ontology/knowledge):** GRAM + G-BERT (explicit code hierarchy / graph augmentation).

- **Counterfactual explanations:** MedSeqCF provides a sequence counterfactual approach aligned with clinical actionability.

- **Interpretability / higher-order structure:** SHy uses patient hypergraphs and self-explaining phenotypes.


## Gaps / Opportunities (what we can claim as novelty)

- **Anomaly ≠ explanation:** Many methods detect/predict anomalies, but do not produce *clinically plausible*, constrained counterfactuals.

- **Ontology constraints are often separate:** Ontology-aware embeddings exist, but are not consistently integrated into *anomaly scoring + counterfactual generation* as a unified constraint system.

- **Generative EHR models focus on synthesis:** Diffusion-based synthesis (e.g., EHRDiff) is rarely coupled to counterfactual generation for *targeted* “minimal-change” edits under constraints.

- **Rare events:** Rare disease/event detection (e.g., CONAN) highlights imbalance; our pipeline should explicitly address imbalance and uncertainty-aware augmentation.

- **Evaluation gap:** Need a combined evaluation: (i) detection quality, (ii) plausibility/ontology-validity of edits, (iii) privacy & leakage, (iv) clinical usefulness.
