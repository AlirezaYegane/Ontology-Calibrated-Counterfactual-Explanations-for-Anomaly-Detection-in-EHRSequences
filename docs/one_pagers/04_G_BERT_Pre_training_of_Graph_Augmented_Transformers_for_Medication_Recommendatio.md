# [4] G-BERT: Pre-training of Graph Augmented Transformers for Medication Recommendation

**Authors:** Shang et al.  
**Venue/Year:** IJCAI, 2019  
**Bucket:** Medical Ontologies / Knowledge Graph

## What problem does it target?
- (From the paper description) Improve modeling of EHR data for prediction / anomaly detection / generation / explanation.

## Data / representation

- **Data/Modality:** Ontology graph + EHR sequences

## Core idea / method

- **Model:** Graph + Transformer fusion (GNN for ontology, BERT for sequence)

## Evaluation (as reported)

- Medication recommendation

## Key takeaways we can reuse

- **Key contribution (project view):** Graph-augmented transformer framework for clinically aware sequence modeling

- **Strengths (project view):** Good semantic fusion; aligns with ontology-calibrated representations

## Limitations / risks (project view)

- Graph construction choices matter; sensitive to missing edges

## How it maps to our system

- Blueprint for fusing KG/ontology into the encoder stage before anomaly/counterfactual modules

## Sources (for verification)

- IJCAI: G-BERT (2019): https://www.ijcai.org/proceedings/2019/825
