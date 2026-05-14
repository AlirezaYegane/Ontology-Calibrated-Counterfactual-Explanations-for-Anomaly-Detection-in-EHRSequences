Day 31 — Full Diffusion Training Baseline

Status
- Complete

Dataset
- Input: .\data\processed\mimiciv_train.pkl
- Rows: 380922
- Sequence column: codes
- Vocab size: 47010
- Max length: 256

Training setup
- Batch size: 16
- Diffusion steps: 64
- Beta schedule: cosine
- Model dimension: 128
- Heads: 4
- Layers: 4
- Learning rate: 0.0003
- Max records used by run: 2048
- Max steps per epoch: 1000

Training result
- Epochs: 4
- Global steps: 512
- First epoch loss: 1.0225786487571895
- Last epoch loss: 0.733609355520457
- Best epoch: 4
- Best loss: 0.733609355520457
- Loss drop: 0.2889692932367325 (28.258881953768235%)
- Loss decreased: True

Tests
- pytest -q tests/test_diffusion_model.py
- Result: 3 passed, 2 warnings

Artifacts
- outputs/diffusion/day31_full/summary.json
- outputs/diffusion/day31_full/metrics.jsonl
- outputs/diffusion/day31_full/config_resolved.json
- outputs/diffusion/day31_full/best.pt
- outputs/diffusion/day31_full/last.pt
- artifacts/day31/day31_full_diffusion_training_report.json

Interpretation
The Day 31 controlled diffusion baseline completed successfully. The loss decreased consistently across four epochs, so the model is learning the denoising objective on the EHR sequence representation.

Note
The run used max_records=2048 from the current training script default, which explains 128 steps per epoch with batch_size=16. Day 32 should refine this setting for longer/full-data training.
