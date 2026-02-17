# Day 1 Reading Notes (starter pack)

> **Note:** For papers **[1–6, 8–10]** the PDFs are downloaded and these notes are based on **title+abstract + quick skim of methods**. Paper **[7] MedSeqCF** is referenced via the author's **open dissertation (DiVA) + official GitHub repo**, because the journal PDF is not openly downloadable.


## [1] Med-BERT: Pre-trained Contextualized Embeddings on Large-scale Structured EHRs (NPJ Digital Medicine, 2021)

- **Bucket:** EHR Representation Learning

- **Data:** Structured EHR tokens

- **Method:** Transformer/BERT; self-supervised pretraining; time encoding

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Candidate backbone encoder for our pipeline; aligns with representation learning foundation

- **Quick source links:**    https://www.nature.com/articles/s41746-021-00455-y


## [2] CEHR-BERT: Incorporating Temporal Information from Structured EHR Data (PMLR (ML4H), 2021)

- **Bucket:** EHR Representation Learning

- **Data:** Structured EHR + time

- **Method:** Transformer with explicit temporal encoding for irregular intervals

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Supports temporal anomaly detection component in ICU setting

- **Quick source links:**    https://proceedings.mlr.press/v158/pang21a.html


## [3] GRAM: Graph-based Attention Model for Healthcare Representation Learning (KDD, 2017)

- **Bucket:** Medical Ontologies / Knowledge Graph

- **Data:** ICD/clinical code hierarchies

- **Method:** Ontology-aware attention over code ancestors

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Core for ontology-calibrated embeddings and anomaly/counterfactual plausibility constraints

- **Quick source links:**    https://doi.org/10.1145/3097983.3098126


## [4] G-BERT: Pre-training of Graph Augmented Transformers for Medication Recommendation (IJCAI, 2019)

- **Bucket:** Medical Ontologies / Knowledge Graph

- **Data:** Ontology graph + EHR sequences

- **Method:** Graph + Transformer fusion (GNN for ontology, BERT for sequence)

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Blueprint for fusing KG/ontology into the encoder stage before anomaly/counterfactual modules

- **Quick source links:**    https://www.ijcai.org/proceedings/2019/825    https://arxiv.org/abs/1906.00346


## [5] DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series (IEEE Access, 2019)

- **Bucket:** Anomaly Detection in EHR/ICU

- **Data:** Continuous time-series (vitals/waveforms)

- **Method:** Unsupervised CNN-based forecasting/anomaly scoring

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Baseline comparator for vitals/waveform anomaly detection stream

- **Quick source links:**    https://doi.org/10.1109/ACCESS.2018.2886457


## [6] EHR-BERT: A BERT-based Model for Effective Anomaly Detection in EHRs (Journal of Biomedical Informatics, 2024)

- **Bucket:** Anomaly Detection in EHR/ICU

- **Data:** Discrete EHR event logs

- **Method:** Transformer anomaly detection via masked token prediction objective

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Anchor method for discrete-event anomaly stream before ontology calibration and counterfactuals

- **Quick source links:**    https://www.osti.gov/biblio/2351748


## [7] MedSeqCF: Style-transfer Counterfactual Explanations for ICU Mortality (Artificial Intelligence in Medicine, 2023)

- **Bucket:** Generative Counterfactuals

- **Data:** ICU sequences

- **Method:** Counterfactual generation via style-transfer framing

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Direct anchor for counterfactual module design and evaluation

- **Quick source links:**    https://doi.org/10.1016/j.artmed.2022.102457               https://github.com/zzachw/MedSeqCF


## [8] EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models (TMLR, 2023)

- **Bucket:** Generative Modeling for Clinical Data

- **Data:** EHR records

- **Method:** Diffusion models for EHR synthesis

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Primary candidate generative backbone for counterfactual generation

- **Quick source links:**    https://github.com/zzh16mz/EHRDiff


## [9] SHy: Self-Explaining Hypergraph Neural Networks for Disease Diagnosis Prediction (CHIL, 2025)

- **Bucket:** Hypergraph / Interpretable Modeling

- **Data:** Higher-order clinical interactions

- **Method:** Hypergraph model with self-explaining components

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Supports interpretable, ontology-aware anomaly and counterfactual explanations

- **Quick source links:**    https://www.researchgate.net/scientific-contributions/Leisheng-Yu-2284841331


## [10] CONAN: Complementary Pattern Augmentation for Rare Event Detection (AAAI, 2020)

- **Bucket:** Anomaly Detection (Rare Events)

- **Data:** Event sequences / rare patterns

- **Method:** Augmentation-driven rare event detection

- **What to extract when you read the full PDF:**
  - Exact objective(s) + loss formulation
  - Data preprocessing/tokenization + time handling
  - Metrics and headline numbers
  - Any ablation that supports your design decision

- **Why it matters for us (summary):** Motivates coupling generative module with anomaly detection for rare anomaly sensitivity

- **Quick source links:**     https://arxiv.org/abs/1911.13232
