Day 32 — Diffusion Training Refinement

Goal
- Refine the Day 31 full diffusion training run into a final baseline diffusion checkpoint.
- Keep the Day 28 architecture unchanged.
- Use a more conservative learning rate for final baseline training.
- Save a report that records checkpoint paths, training metrics, and real-data sequence statistics.

Final baseline run
- Output directory: outputs/diffusion/day32_final_baseline/
- Final checkpoint alias: outputs/diffusion/day32_final_baseline/diffusion.pt
- Report: artifacts/day32/day32_final_diffusion_report.json

Design choices
- Sequence-based diffusion baseline
- Small Transformer denoiser
- 64 diffusion steps
- Cosine beta schedule
- lr = 2e-4
- batch_size = 16 unless GPU memory requires fallback to 8

Interpretation
- Day 32 closes the non-ontology diffusion baseline.
- Day 33 should start ontology regularization and compare against this baseline.
