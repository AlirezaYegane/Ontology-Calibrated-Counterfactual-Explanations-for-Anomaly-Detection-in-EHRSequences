Day 33 — Ontology-Regularized Diffusion

Status: complete

What was added
- Lightweight ontology-consistency loss L_ont
- Proxy rules for medication/procedure context and unknown-token mass
- Generated/reconstructed violation monitoring
- Best checkpoint selection by validation loss

Main outputs
- outputs/diffusion/day33_ontology_regularized_fixed/best.pt
- outputs/diffusion/day33_ontology_regularized_fixed/last.pt
- outputs/diffusion/day33_ontology_regularized_fixed/metrics.jsonl
- outputs/diffusion/day33_ontology_regularized_fixed/summary.json
- artifacts/day33_fixed/day33_ontology_regularization_report.json

Important limitation
This is a conservative ontology-regularization pass. Full SNOMED graph-distance loss,
demographic metadata constraints, and counterfactual repair search remain deferred.
