# Environment

## Python

Python 3.10+ (developed and tested on Windows 11 and Linux). A CUDA GPU is only required to
*retrain* the unsupervised detector; the aggregate evaluations, counterfactual evaluation,
and the full test suite run on CPU.

## Packages

Direct dependencies are pinned by lower bound in [`requirements.txt`](../../requirements.txt):

```
torch>=2.0        pandas>=2.0     numpy>=1.24     scipy>=1.10
networkx>=3.0     scikit-learn>=1.3   pyyaml>=6.0
matplotlib>=3.7   seaborn>=0.13   tqdm>=4.65      pyarrow>=14.0
```

Install:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
pip install pytest
```

For GPU detector retraining, additionally install a CUDA torch build; a known-good pin set
is in [`docs/setup/requirements-torch-cu128.txt`](../setup/requirements-torch-cu128.txt).

## Determinism and seeds

- benchmark-v2 build seed: **42**.
- The counterfactual generator is deterministic (a seed is accepted only for reproducibility;
  no randomness affects the result).
- Detector training uses fixed seeds; the full run is `phase6_detector_full_gpu`
  (25 epochs, best epoch 19).
- All headline metrics are reported with bootstrap confidence intervals, so small numerical
  differences across platforms remain within the reported CIs.

## Dataset roots

Point [`dataset_roots.yaml`](../../dataset_roots.yaml) at your local copies of MIMIC-IV /
eICU if you intend to rebuild data. You do **not** need these to run the tests.

## Sanity check

```bash
python -m pytest -q         # expect all tests to pass (data-dependent tests skip if splits absent)
python scripts/run_phase8_final_checks.py
```
