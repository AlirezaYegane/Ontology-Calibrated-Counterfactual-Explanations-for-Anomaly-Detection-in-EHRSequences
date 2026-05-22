# Day 48 — Reproducibility Audit

## Overview

This directory contains machine-readable outputs of the Day 48 reproducibility audit.  The audit captures repository state, environment metadata, source-file inventory, artifact manifest, and private-data boundary checks.

## Generated Files

| File | Description |
| ---- | ----------- |
| `day48_reproducibility_audit.json` | Full audit payload (git, env, files, warnings) |
| `day48_reproducibility_matrix.csv` | High-level reproducibility status matrix |
| `README.md` | This file |

## Snapshot

- **Branch:** `main`
- **Commit:** `439e868`
- **Working tree clean:** No ✗
- **Risky tracked files:** 6
- **Warnings:** 3

## Warnings

- ⚠️  Working tree is NOT clean — uncommitted changes present.
- ⚠️  6 risky file(s) tracked by git (private data, checkpoints, or large binary artifacts).
- ⚠️  4 key source file(s) missing: src/training/train_detector_supervised.py, src/training/build_detector_supervised_data.py, src/training/detector_supervised_utils.py, src/evaluation/evaluate_detector_supervised.py

## Regeneration

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/build_day48_reproducibility_audit.py --project_root . --out_dir artifacts/day48 --docs_dir docs
```
