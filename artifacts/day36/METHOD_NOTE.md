# Day 36 Method Note — Repair-Ready Counterfactual Generation

The Day 36 counterfactual module performs constrained generate-and-test repair over anomalous EHR sequences.

Given an anomalous sequence X, the system proposes a small set of clinically interpretable edits:
- adding a missing expected diagnosis code,
- removing an injected or demographically incompatible code,
- replacing a code when explicit replacement evidence exists.

Each candidate X* is evaluated by an ontology-violation score and an edit penalty:

Cost(X* | X) = ontology_violation_score(X*) + λ_edit × number_of_edits

The selected counterfactual is the candidate with the lowest cost, with preference for sparse edits.

A key design choice is that repair targets are not inferred from labels alone. Instead, they are reconstructed from the synthetic anomaly construction process by comparing `codes_original` and `codes_corrupted`. This gives each counterfactual edit a traceable basis.

For the current synthetic anomaly benchmark:
- missing-diagnosis anomalies are repaired by adding back the removed diagnosis code,
- medication-mismatch anomalies are repaired by adding back the supporting diagnosis code,
- demographic-conflict anomalies are repaired by removing the injected incompatible code.

This positions the counterfactual generator as a conservative, traceable explanation module rather than an unconstrained sequence editor.
