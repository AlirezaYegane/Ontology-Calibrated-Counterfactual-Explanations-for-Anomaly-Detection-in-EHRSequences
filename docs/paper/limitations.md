# Limitations

We state the boundary of the evidence explicitly.

- **MIMIC-IV only.** All final evidence comes from a single source database. We make no
  cross-database generalization claim.

- **Synthetic, constructed anomalies.** Benchmark-v2 injects relational anomalies into real
  MIMIC-IV records. It is a controlled, ontology-backed *synthetic* benchmark, not a corpus
  of clinician-confirmed real-world documentation errors. It is designed to be
  non-circular, not to be a census of naturally occurring errors.

- **No clinician validation.** Counterfactual "validity" means an ontology violation is
  resolved with a minimal, coherent edit that an independent scorer accepts. It does **not**
  mean a clinician reviewed and endorsed the edit. Clinical usefulness of the repairs is
  future work.

- **No external validation.** The attempt on eICU is blocked by a token-schema mismatch:
  eICU uses APACHE/body-system tokens that do not map to the ICD/SNOMED/RxNorm ontology
  (0/500 sampled records fire any rule). External validation would require an
  APACHE→ICD/SNOMED crosswalk plus re-injecting the benchmark anomalies on eICU.

- **Curated rule packs.** The three rule families are hand-authored and auditable, not
  learned from data. Their coverage is deliberately scoped (three anomaly families), and the
  medication-indication family is intentionally noisy (recall ≈ 0.50) because its clinical
  semantics are genuinely ambiguous. Broader rule coverage is future work.

- **Detector and generative components are negative results.** The full-scale detector is
  below chance and non-additive on this benchmark, and the diffusion generative term is
  removed from the core. We report them for transparency; they are not part of the method.

- **Counterfactual edit vocabulary.** Repairs operate on model-visible tokens with a small
  edit budget. Some clinically minimal fixes (e.g., correcting a recorded sex rather than
  removing obstetric codes) fall outside this vocabulary, which limits demographic-case
  repair.

- **Scope of the reproducibility claim.** Reproducibility covers the aggregate pipeline and
  tests. Exactly reproducing the numbers requires the restricted MIMIC-IV data and licensed
  ontologies, which cannot be redistributed here.
