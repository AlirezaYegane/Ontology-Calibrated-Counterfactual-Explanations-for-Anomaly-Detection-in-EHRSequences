Day 29 — Diffusion Model Implementation

Status
- DiffusionModel implemented in src/models/diffusion.py
- Sinusoidal time embedding implemented
- Cosine beta schedule implemented
- Embedding-space q_sample implemented
- Time-conditioned Transformer denoiser implemented
- Mask-aware training_loss implemented
- Initial Sgen proxy via surprise_score implemented
- Basic reverse sampling and nearest-token decoding implemented
- Unit tests passed
- Ruff format/check passed
- Day 27 artifact smoke test completed

Main design carried from Day 28
- Primary view: sequence embeddings
- Diffusion steps: 64
- Beta schedule: cosine
- Denoiser: small Transformer encoder
- d_model: 128
- n_heads: 4
- n_layers: 4
- ff_dim: 512
- prediction target: epsilon/noise

Important note
Day 29 is not the full diffusion training day. The goal is model implementation and smoke validation.
Full train_diffusion.py belongs to Day 30.

Artifacts
- src/models/diffusion.py
- tests/test_diffusion_model.py
- scripts/smoke_day29_diffusion.py
- artifacts/day29/day29_smoke_summary.json
- artifacts/day29/README.txt
