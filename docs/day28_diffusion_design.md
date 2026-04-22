# Day 28 — Final Diffusion Model Design

## Goal
Lock the diffusion-model design after Day 27 data preparation so Day 29 can begin implementation with no major architectural ambiguity.

## Day 27 dependency
The diffusion-ready artifact already exists and includes padded / truncated sequence data, summary metadata, and loader views prepared for generative modeling.

## Locked decisions

### 1) Primary training view
- Main training view: sequence-based representation
- Keep multi-hot / both-view support only for ablations or diagnostics
- Reason: sequence structure is more aligned with anomaly explanation and token-level repair than a very wide flat multi-hot input

### 2) Input shape
- max_len: 256
- variable-length support via attention mask / pad mask
- pad_idx: 0
- unknown_idx: 1

### 3) Vocabulary strategy
- Reuse the existing sequence vocabulary already used in detector/generative data preparation
- No new tokenization layer in Day 29

### 4) Diffusion parameterization
- model family: DDPM-style discrete-sequence diffusion in embedding space
- prediction target: epsilon (noise prediction)
- diffusion steps T: 64
- beta schedule: cosine
- evaluation sampling steps: 16 (DDIM-style fast sampling later if needed)

### 5) Denoiser architecture
- backbone: small Transformer encoder
- d_model: 128
- n_heads: 4
- n_layers: 4
- ff_dim: 512
- dropout: 0.10
- time embedding: sinusoidal -> projected to d_model
- output head: project hidden states back to embedding/noise space

### 6) Training objective
- primary loss: mask-aware MSE on predicted noise over non-pad positions
- optional auxiliary reconstruction monitoring on non-pad positions
- gradient clipping: 1.0
- mixed precision on GPU when stable

### 7) Generative surprise score Sgen
- initial operational definition: mean non-pad denoising reconstruction error
- record-level score aggregates token-level error only over valid positions
- keep midpoint-step reconstruction diagnostics as optional analysis, not the core metric

### 8) Ontology regularization strategy
Initial Day 29/30 implementation will use lightweight rule penalties instead of expensive graph shortest-path losses.

Use:
- demographic consistency penalty
- medication-indication mismatch penalty
- forbidden co-occurrence penalty

Combined objective:
L = L_diff + lambda_total * L_ont

Initial weights:
- lambda_total = 0.10
- demographic_penalty = 0.04
- medication_indication_penalty = 0.04
- forbidden_pair_penalty = 0.02

### 9) Explicit deferrals
Deferred beyond baseline implementation:
- flat multi-hot MLP as the mainline architecture
- temporal U-Net backbone
- graph shortest-path regularization inside every batch step
- full counterfactual repair sampling loop

These remain valid future ablations/extensions, but not the Day 29 baseline.

## Engineering rationale
- Sequence view preserves local token context and supports later counterfactual editing better than a huge flattened multi-hot vector.
- T=64 stays inside the proposal range (50-100) while remaining practical on a single GPU.
- A small Transformer is expressive enough for token interactions yet still lightweight.
- Lightweight ontology penalties are much easier to stabilize first than full graph-distance regularization.

## Ready-for-Day-29 checklist
- [x] Input representation locked
- [x] Diffusion steps locked
- [x] Denoiser architecture locked
- [x] Ontology regularization strategy locked
- [x] Design written to a durable document

## Day 29 implementation targets
- src/models/diffusion.py
- src/training/train_diffusion.py
- dataloader that reads the Day 27 artifact
- smoke forward pass on GPU
