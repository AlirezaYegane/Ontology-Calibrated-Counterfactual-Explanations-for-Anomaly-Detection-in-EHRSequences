# Discussion

## The ontology carries the signal

The clearest finding is that a real, auditable ontology rule engine is the discriminative
component. It reaches ROC-AUC 0.7881 and significantly beats the legacy ICD-prefix baseline
(+0.052, p ≈ 0). The rule-family ablation shows *why* it works: no single family exceeds
about 0.63 alone, but together they cover three distinct kinds of relational incoherence,
and the combination is what reaches 0.79. The improvement over the legacy baseline is
concentrated in the forbidden-co-occurrence family (the legacy prefix rules essentially miss
it: per-family ROC-AUC 0.42 vs 0.93), which is exactly the kind of relationship that a real
concept hierarchy captures and a flat prefix match does not.

## Why the detector is below chance

A full-scale unsupervised next-token detector scores *below* chance (0.4525) on this
benchmark. This is not a training failure — the run converged with sensible validation
behavior — but a consequence of the benchmark's construction. The anomalies are
**relational**: a gender flip injects no token, an indication removal deletes a common
diagnosis, and a forbidden co-occurrence adds a common code. None of these makes the
surrounding token stream more surprising, so a language-model detector has nothing to grip.
The fact that the detector lands slightly below 0.5 rather than exactly at it reflects mild
correlations (e.g., anomalous records tend to be marginally longer or shorter), but the
central message is that next-token surprise is the wrong signal for relational anomalies.

## Why adding the detector hurts

Because `S_det` is essentially noise (and slightly anti-correlated with the label), mixing
it into the calibrated score dilutes the strong ontology term: the combination drops to
0.7036, a significant −0.085 versus ontology-only. This is the honest reason the final
method is ontology-only and not the calibrated combination the project originally planned to
headline. The calibrated-score machinery remains in the repository — it is how we *measured*
the non-additivity — but `w_det` is best set to 0 in practice on this benchmark.

## Why Sgen was removed

The diffusion-based generative-surprise term (`Sgen`) is near-random (0.4868), points the
wrong way (anomalies score lower than normals), does not correlate with the ontology signal,
and *harms* the combined score. The underlying generator is also mode-collapsed (127 of
4,587 tokens used; generated length ≈ 254 vs real ≈ 47). We remove it from the core
(`w_gen = 0`) and report it as a negative diagnostic rather than a component. A retrained,
anti-collapse generative variant could be revisited in future work, but nothing in the
current evidence supports keeping it.

## Counterfactual repair is strong where the ontology flags the anomaly

Repair success is 89.99% among ontology-flagged anomalies but 61.4% overall. The gap is
entirely explained by *detection* coverage, not by the repair search: the ontology does not
flag ~33% of the injected anomalies (mostly the deliberately noisy medication-indication
family), and an explainer cannot repair a violation the scorer never raises. Where the
ontology does flag — mutual exclusion and missing-required-context — repair is 100%, with a
median of one edit. The weaker demographic case (65.7%) is an edit-budget artifact: records
with many obstetric codes would need more than three token edits, whereas the clinically
minimal fix (correct the recorded sex) is outside the token-edit vocabulary this phase used.

## Negative results strengthen the paper

Taken together, the two negative results are not gaps — they are the explanation for the
positive one. They establish that the discriminative structure of these anomalies is
relational and ontological, not statistical, which is precisely what motivates an
ontology-centered method. Reporting them with confidence intervals and paired significance
tests makes the ontology claim credible rather than selectively favorable.
