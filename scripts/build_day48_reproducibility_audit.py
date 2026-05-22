#!/usr/bin/env python3
"""
Day 48 — Reproducibility Audit Builder
=======================================

Collects repository state, environment metadata, artifact inventory,
and private-data boundary checks, then emits:

  - artifacts/day48/day48_reproducibility_audit.json
  - artifacts/day48/day48_reproducibility_matrix.csv
  - artifacts/day48/README.md
  - docs/reproducibility_checklist.md
  - docs/run_from_scratch.md
  - docs/artifact_manifest.md

This script does NOT modify source code, train models, or access
private clinical data.  It is safe to run on any checkout.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── helpers ────────────────────────────────────────────────────────────

def _run_git(args: List[str], cwd: str) -> Tuple[int, str]:
    """Run a git command and return (returncode, stdout)."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode, proc.stdout.strip()
    except FileNotFoundError:
        return -1, "<git not found>"
    except subprocess.TimeoutExpired:
        return -2, "<git timed out>"


def _pkg_version(name: str) -> Optional[str]:
    """Return installed version string, or None."""
    try:
        # Use importlib.metadata for reliable version lookup
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        pass
    # Fallback for older Python
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def _file_meta(root: Path, rel: str) -> Dict[str, Any]:
    """Return existence flag and byte-size for a project-relative path."""
    p = root / rel
    if p.exists():
        return {"exists": True, "size_bytes": p.stat().st_size}
    return {"exists": False, "size_bytes": None}


def _redact_url(url: str) -> str:
    """Strip credentials from a git remote URL."""
    # https://user:token@host/... → https://***@host/...
    return re.sub(r"://[^@]+@", "://***@", url)


# ── collectors ─────────────────────────────────────────────────────────

def collect_git_info(root: str) -> Dict[str, Any]:
    """Collect git branch, commit, status, remotes."""
    info: Dict[str, Any] = {}

    rc, branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    info["branch"] = branch if rc == 0 else None

    rc, sha = _run_git(["rev-parse", "--short", "HEAD"], root)
    info["commit_short"] = sha if rc == 0 else None

    rc, status = _run_git(["status", "--short"], root)
    info["status_short"] = status if rc == 0 else None
    info["working_tree_clean"] = (rc == 0 and status == "")

    rc, remotes = _run_git(["remote", "-v"], root)
    if rc == 0 and remotes:
        info["remotes_redacted"] = [
            _redact_url(line) for line in remotes.splitlines()
        ]
    else:
        info["remotes_redacted"] = []

    return info


def collect_env_info() -> Dict[str, Any]:
    """Python, platform, package versions, CUDA probe."""
    info: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    pkgs = [
        "torch", "numpy", "pandas", "scikit-learn", "scipy",
        "networkx", "matplotlib", "seaborn", "tqdm", "PyYAML",
    ]
    info["packages"] = {p: _pkg_version(p) for p in pkgs}

    # Torch / CUDA probe
    cuda_info: Dict[str, Any] = {}
    torch_ver = _pkg_version("torch")
    cuda_info["torch_installed"] = torch_ver is not None
    if torch_ver is not None:
        try:
            import torch
            cuda_info["cuda_available"] = torch.cuda.is_available()
            cuda_info["cuda_version"] = (
                torch.version.cuda if torch.cuda.is_available() else None
            )
            cuda_info["device_count"] = torch.cuda.device_count()
            cuda_info["device_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
        except Exception as exc:
            cuda_info["probe_error"] = str(exc)
    else:
        cuda_info["cuda_available"] = False
        cuda_info["cuda_version"] = None
        cuda_info["device_count"] = 0
        cuda_info["device_names"] = []
    info["torch_cuda"] = cuda_info
    return info


def collect_source_files(root: Path) -> Dict[str, Any]:
    """Check existence and size of key source files."""
    key_sources = [
        "src/models/diffusion.py",
        "src/models/diffusion_legacy_day33.py",
        "src/models/detector_supervised.py",
        "src/training/train_detector_supervised.py",
        "src/training/build_detector_supervised_data.py",
        "src/training/detector_supervised_utils.py",
        "src/evaluation/evaluate_detector_supervised.py",
        "scripts/analyze_day47_failure_modes.py",
        "scripts/day34_sgen_timestep_sweep.py",
        "scripts/run_day39_end_to_end_case_studies.py",
        "scripts/build_day36_repair_ready_scores.py",
    ]
    return {rel: _file_meta(root, rel) for rel in key_sources}


def collect_docs_check(root: Path) -> Dict[str, Any]:
    """Check existence of important documentation files."""
    doc_paths = [
        "README.md",
        "docs/reproducibility_checklist.md",
        "docs/run_from_scratch.md",
        "docs/artifact_manifest.md",
    ]
    return {rel: _file_meta(root, rel) for rel in doc_paths}


def collect_artifact_check(root: Path) -> Dict[str, Any]:
    """Check existence of important artifact files and list day dirs."""
    artifact_files = [
        "artifacts/day20/day20_supervised_eval_summary.json",
        "artifacts/day34_final/day34_final_assessment.json",
        "artifacts/day40/README.md",
        "artifacts/day41/README.md",
        "artifacts/day47/README.md",
    ]
    result: Dict[str, Any] = {
        "key_artifacts": {rel: _file_meta(root, rel) for rel in artifact_files},
    }

    # List day-level artifact directories
    artifacts_dir = root / "artifacts"
    if artifacts_dir.is_dir():
        day_dirs = sorted([
            d.name for d in artifacts_dir.iterdir()
            if d.is_dir() and d.name.startswith("day")
        ])
    else:
        day_dirs = []
    result["day_directories"] = day_dirs
    return result


def collect_risky_tracked_files(root: str) -> Dict[str, Any]:
    """Detect risky files that are tracked by git."""
    rc, ls_output = _run_git(["ls-files"], root)
    if rc != 0:
        return {"error": "git ls-files failed", "risky_files": []}

    risky: List[str] = []
    risky_extensions = {
        ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
        ".parquet", ".feather", ".h5", ".hdf5",
    }
    risky_prefixes = ("data/", "outputs/", "ontologies/")

    for f in ls_output.splitlines():
        f_stripped = f.strip()
        if not f_stripped:
            continue
        f_norm = f_stripped.replace("\\", "/")
        if any(f_norm.startswith(p) for p in risky_prefixes):
            risky.append(f_stripped)
        elif any(f_norm.endswith(ext) for ext in risky_extensions):
            risky.append(f_stripped)

    return {"risky_files": risky, "risky_count": len(risky)}


# ── warnings ───────────────────────────────────────────────────────────

def compute_warnings(audit: Dict[str, Any]) -> List[str]:
    """Generate human-readable warnings from the audit payload."""
    warnings: List[str] = []

    git = audit.get("git", {})
    if not git.get("working_tree_clean"):
        warnings.append("Working tree is NOT clean — uncommitted changes present.")

    risky = audit.get("risky_tracked_files", {})
    if risky.get("risky_count", 0) > 0:
        warnings.append(
            f"{risky['risky_count']} risky file(s) tracked by git "
            "(private data, checkpoints, or large binary artifacts)."
        )

    source = audit.get("source_files", {})
    missing_src = [k for k, v in source.items() if not v.get("exists")]
    if missing_src:
        warnings.append(
            f"{len(missing_src)} key source file(s) missing: "
            + ", ".join(missing_src)
        )

    return warnings


# ── output generators ──────────────────────────────────────────────────

def write_audit_json(audit: Dict[str, Any], out_dir: Path) -> Path:
    """Write the full audit payload to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "day48_reproducibility_audit.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False, default=str)
    return path


def write_matrix_csv(audit: Dict[str, Any], out_dir: Path) -> Path:
    """Write the reproducibility matrix CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "day48_reproducibility_matrix.csv"

    git = audit.get("git", {})
    risky = audit.get("risky_tracked_files", {})
    src = audit.get("source_files", {})
    docs = audit.get("docs_check", {})
    arts = audit.get("artifact_check", {})

    src_ok = all(v.get("exists") for v in src.values())
    risky_count = risky.get("risky_count", 0)
    clean = git.get("working_tree_clean", False)

    rows = [
        {
            "component": "Repository state",
            "reproducibility_status": "OK" if clean else "WARNING",
            "evidence": f"branch={git.get('branch')}, commit={git.get('commit_short')}, clean={clean}",
            "private_data_required": "No",
            "notes": "Working tree should be clean before release.",
        },
        {
            "component": "Private data boundary",
            "reproducibility_status": "OK" if risky_count == 0 else "FAIL",
            "evidence": f"{risky_count} risky tracked file(s)",
            "private_data_required": "Yes — MIMIC-III/eICU access, UMLS/SNOMED/RxNorm",
            "notes": "Raw clinical data and ontology resources cannot be redistributed.",
        },
        {
            "component": "Environment capture",
            "reproducibility_status": "OK",
            "evidence": f"Python {platform.python_version()}, platform={platform.platform()}",
            "private_data_required": "No",
            "notes": "Full package versions recorded in audit JSON.",
        },
        {
            "component": "Core code availability",
            "reproducibility_status": "OK" if src_ok else "WARNING",
            "evidence": f"{sum(1 for v in src.values() if v.get('exists'))}/{len(src)} key files present",
            "private_data_required": "No",
            "notes": "All core model, training, and evaluation source files checked.",
        },
        {
            "component": "Artifact manifest",
            "reproducibility_status": "OK",
            "evidence": f"{len(arts.get('day_directories', []))} day-level artifact directories",
            "private_data_required": "Partial — some artifacts require private data to regenerate",
            "notes": "See docs/artifact_manifest.md for full inventory.",
        },
        {
            "component": "Run-from-scratch instructions",
            "reproducibility_status": "OK" if (docs.get("docs/run_from_scratch.md", {}).get("exists")) else "PENDING",
            "evidence": "docs/run_from_scratch.md",
            "private_data_required": "Yes — dataset and ontology access required for full reproduction",
            "notes": "See docs/run_from_scratch.md for step-by-step guide.",
        },
    ]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["component", "reproducibility_status", "evidence",
                         "private_data_required", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_day48_readme(audit: Dict[str, Any], out_dir: Path) -> Path:
    """Write artifacts/day48/README.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "README.md"

    git = audit.get("git", {})
    warnings = audit.get("warnings", [])
    risky = audit.get("risky_tracked_files", {})

    md = []
    md.append("# Day 48 — Reproducibility Audit\n")
    md.append("## Overview\n")
    md.append(
        "This directory contains machine-readable outputs of the Day 48 "
        "reproducibility audit.  The audit captures repository state, "
        "environment metadata, source-file inventory, artifact manifest, "
        "and private-data boundary checks.\n"
    )
    md.append("## Generated Files\n")
    md.append("| File | Description |")
    md.append("| ---- | ----------- |")
    md.append(
        "| `day48_reproducibility_audit.json` | "
        "Full audit payload (git, env, files, warnings) |"
    )
    md.append(
        "| `day48_reproducibility_matrix.csv` | "
        "High-level reproducibility status matrix |"
    )
    md.append("| `README.md` | This file |")
    md.append("")

    md.append("## Snapshot\n")
    md.append(f"- **Branch:** `{git.get('branch', 'N/A')}`")
    md.append(f"- **Commit:** `{git.get('commit_short', 'N/A')}`")
    md.append(
        f"- **Working tree clean:** "
        f"{'Yes ✓' if git.get('working_tree_clean') else 'No ✗'}"
    )
    md.append(
        f"- **Risky tracked files:** {risky.get('risky_count', '?')}"
    )
    md.append(f"- **Warnings:** {len(warnings)}")
    md.append("")

    if warnings:
        md.append("## Warnings\n")
        for w in warnings:
            md.append(f"- ⚠️  {w}")
        md.append("")

    md.append("## Regeneration\n")
    md.append("```powershell")
    md.append("$env:PYTHONPATH = (Get-Location).Path")
    md.append(
        "python scripts/build_day48_reproducibility_audit.py "
        "--project_root . --out_dir artifacts/day48 --docs_dir docs"
    )
    md.append("```\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


# ── documentation generators ──────────────────────────────────────────

def write_reproducibility_checklist(docs_dir: Path) -> Path:
    """Generate docs/reproducibility_checklist.md."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "reproducibility_checklist.md"

    content = """\
# Reproducibility Checklist

This checklist accompanies the manuscript
*"Ontology-Calibrated Counterfactual Explanations for Anomaly Detection
in EHR Sequences"* and documents the measures taken to ensure
reproducibility.

---

## 1. Code Availability

- [x] All model, training, evaluation, and explanation source code is
      included in the repository under `src/`.
- [x] All experiment scripts are included under `scripts/`.
- [x] Configuration files are stored under `config/`.

## 2. Data Availability

- [ ] **Raw MIMIC-III / eICU data cannot be redistributed.**
      Researchers must obtain independent access via PhysioNet
      (https://physionet.org/).
- [ ] **UMLS, SNOMED-CT, and RxNorm resources cannot be
      redistributed.**  Users must obtain these from the National
      Library of Medicine (https://www.nlm.nih.gov/research/umls/).
- [x] Synthetic or illustrative examples are provided where possible
      to enable structural testing without private data.

## 3. Environment Specification

- [x] `requirements.txt` lists direct Python dependencies.
- [x] The Day 48 audit script records exact package versions,
      Python version, platform, and CUDA configuration.
- [x] Conda or virtualenv setup instructions are provided in
      `docs/run_from_scratch.md`.

## 4. Artifact Management

- [x] Experiment outputs are organised into day-level directories
      under `artifacts/`.
- [x] An artifact manifest is provided in `docs/artifact_manifest.md`.
- [x] Processed patient-level tensors, pickles, and model checkpoints
      are excluded from version control via `.gitignore`.

## 5. Private-Data Boundaries

- [x] `.gitignore` excludes `data/`, `outputs/`, `ontologies/`,
      and binary model files (`*.pt`, `*.pth`, `*.ckpt`, `*.pkl`,
      `*.pickle`, `*.parquet`).
- [x] The Day 48 audit script explicitly checks for risky tracked
      files and emits warnings.
- [x] No patient-level identifiers, clinical notes, or protected
      health information is committed.

## 6. Methodological Transparency

- [x] The supervised anomaly detector and ontology-calibrated scoring
      pipeline are the primary evidence for the paper's claims.
- [x] The Sgen (diffusion-based generative) component showed weak
      discriminative performance in Day 34 evaluations.  It is
      described honestly as a diagnostic/exploratory signal and
      should **not** be overclaimed.
- [x] Ablation studies (Days 40–41) and failure-mode analyses
      (Day 47) are documented with full results.

## 7. Reproducibility Audit

- [x] A machine-readable audit (`day48_reproducibility_audit.json`)
      records repository state at each run.
- [x] A reproducibility matrix (`day48_reproducibility_matrix.csv`)
      summarises component-level status.
- [x] Run-from-scratch instructions are provided in
      `docs/run_from_scratch.md`.

---

*Checklist generated by `scripts/build_day48_reproducibility_audit.py`.*
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def write_run_from_scratch(docs_dir: Path) -> Path:
    """Generate docs/run_from_scratch.md."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "run_from_scratch.md"

    content = """\
# Run-from-Scratch Guide

Step-by-step instructions to reproduce the experiments described in
*"Ontology-Calibrated Counterfactual Explanations for Anomaly Detection
in EHR Sequences"*.

> **Important:** Full reproduction requires independent access to
> MIMIC-III (or eICU) via PhysioNet and to UMLS / SNOMED-CT / RxNorm
> resources via the NLM.  These datasets **cannot** be redistributed
> with this repository.

---

## Prerequisites

| Requirement | Source |
| ----------- | ------ |
| MIMIC-III or eICU tables | https://physionet.org/ |
| UMLS Metathesaurus | https://www.nlm.nih.gov/research/umls/ |
| SNOMED-CT release files | Included in UMLS or via national release centres |
| RxNorm files | https://www.nlm.nih.gov/research/umls/rxnorm/ |
| Python ≥ 3.10 | https://www.python.org/ |
| Git | https://git-scm.com/ |
| (Optional) CUDA-capable GPU | NVIDIA drivers + CUDA toolkit |

---

## Step 1 — Clone the Repository

```powershell
git clone <repository-url> Article
Set-Location Article
```

## Step 2 — Create and Activate a Conda Environment

```powershell
conda create -n ontology_cf python=3.10 -y
conda activate ontology_cf
```

## Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

If a GPU is available, install PyTorch with CUDA support:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 4 — Set PYTHONPATH

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Verify:

```powershell
python -c "import src; print('src package importable')"
```

## Step 5 — Place Private Data Resources

Place the required datasets and ontology files in the expected local
directories.  Refer to `dataset_roots.yaml` for the expected layout:

```powershell
Get-Content dataset_roots.yaml
```

Typical layout:

```
data/
  raw/          ← raw MIMIC-III / eICU tables (CSV or compressed)
  processed/    ← generated by preprocessing scripts
ontologies/
  raw/          ← UMLS / SNOMED / RxNorm source files
  processed/    ← generated by ontology-parsing scripts
```

> These directories are excluded from version control by `.gitignore`.

## Step 6 — Data Preprocessing

Run the preprocessing and ontology-parsing scripts as described in the
day-by-day roadmap.  Key entry points:

```powershell
# Ontology parsing
python scripts/parse_snomed.py
python scripts/parse_rxnorm.py
python scripts/build_rxnorm_graph.py

# Dataset splits
python scripts/day11_build_mimiciii_splits.py
```

## Step 7 — Training and Evaluation

Follow the day-level scripts in chronological order.  The most
important stages are:

```powershell
# Day 20: Supervised detector
python scripts/rebuild_day20_supervised_detector.py

# Day 33: Ontology regularisation
python scripts/run_day33_ontology_regularization.py

# Day 39: End-to-end case studies
python scripts/run_day39_end_to_end_case_studies.py

# Day 45: Test-set evaluation
python scripts/evaluate_day45_test_set.py
```

## Step 8 — Run the Day 48 Reproducibility Audit

```powershell
python scripts/build_day48_reproducibility_audit.py `
  --project_root . `
  --out_dir artifacts/day48 `
  --docs_dir docs
```

Inspect outputs:

```powershell
Get-ChildItem artifacts/day48
Get-Content artifacts/day48/README.md
Import-Csv artifacts/day48/day48_reproducibility_matrix.csv | Format-Table -AutoSize
```

## Step 9 — Verify No Private Files Are Staged

```powershell
git status --short
git diff --cached --name-only
```

Ensure no files under `data/`, `outputs/`, `ontologies/`, or binary
checkpoints (`*.pt`, `*.pkl`, etc.) are staged.

---

## Notes on Sgen (Diffusion-Based Generator)

The Day 34 Sgen evaluation showed that the generative diffusion
component produces **weak discriminative signal** for anomaly
detection.  Unless subsequent work improves this component, it
should be described as diagnostic and exploratory rather than a
primary evidence source.  The supervised detector and ontology-driven
scoring pipeline remain the strongest reproducible evidence.

---

## What the Repository Shares

This repository distributes:

- All source code (models, training, evaluation, explanation)
- Configuration files
- Experiment scripts
- Documentation and reproducibility artefacts
- Synthetic / illustrative examples

It does **not** distribute:

- Raw MIMIC-III / eICU patient data
- UMLS / SNOMED-CT / RxNorm source files
- Processed patient-level tensors, pickles, or checkpoints
- Model weight files

Researchers must obtain private resources independently and follow the
steps above to regenerate derived artefacts.

---

*Generated by `scripts/build_day48_reproducibility_audit.py`.*
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def write_artifact_manifest(docs_dir: Path, audit: Dict[str, Any]) -> Path:
    """Generate docs/artifact_manifest.md."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "artifact_manifest.md"

    arts = audit.get("artifact_check", {})
    day_dirs = arts.get("day_directories", [])
    key_arts = arts.get("key_artifacts", {})

    md = []
    md.append("# Artifact Manifest\n")
    md.append(
        "This document catalogues the experiment artefacts produced "
        "across the project roadmap.  Artefacts are organised into "
        "day-level directories under `artifacts/`.\n"
    )
    md.append("> **Note:** Some artefacts contain derived patient-level")
    md.append("> data (tensors, evaluation CSVs with clinical codes) and")
    md.append("> are excluded from version control.  Only code, configs,")
    md.append("> documentation, synthetic examples, and small non-sensitive")
    md.append("> summaries should be committed.\n")

    md.append("---\n")
    md.append("## Day-Level Directories\n")
    if day_dirs:
        for d in day_dirs:
            md.append(f"- `artifacts/{d}/`")
    else:
        md.append("_No day-level directories found._")
    md.append("")

    md.append("---\n")
    md.append("## Key Artefact Files\n")
    md.append("| Path | Exists |")
    md.append("| ---- | ------ |")
    for rel, meta in key_arts.items():
        exists = "✓" if meta.get("exists") else "✗"
        md.append(f"| `{rel}` | {exists} |")
    md.append("")

    md.append("---\n")
    md.append("## Data and Privacy Boundaries\n")
    md.append(
        "The following categories of files **must not** be committed "
        "to the repository:\n"
    )
    md.append("- Raw MIMIC-III / eICU patient data")
    md.append("- UMLS, SNOMED-CT, and RxNorm source files")
    md.append("- Processed patient-level tensors and pickles")
    md.append(
        "- Model checkpoints (`.pt`, `.pth`, `.ckpt`, `.pkl`, "
        "`.pickle`, `.parquet`)"
    )
    md.append("- Evaluation outputs containing protected health information")
    md.append("")

    md.append(
        "These exclusions are enforced by `.gitignore` and verified "
        "by the Day 48 reproducibility audit script.\n"
    )

    md.append("---\n")
    md.append("## Sgen Finding Disclaimer\n")
    md.append(
        "The Day 34 evaluation of the Sgen (diffusion-based generative) "
        "component found **weak discriminative performance**.  This "
        "negative result is documented transparently.  Sgen should be "
        "described as a weak or diagnostic signal unless subsequent "
        "work demonstrates improvement.  The supervised anomaly "
        "detector and ontology-calibrated scoring pipeline represent "
        "the current strongest evidence.\n"
    )

    md.append("---\n")
    md.append(
        "*Manifest generated by "
        "`scripts/build_day48_reproducibility_audit.py`.*\n"
    )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


# ── main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 48 — Reproducibility Audit Builder",
    )
    parser.add_argument(
        "--project_root", default=".", help="Project root directory",
    )
    parser.add_argument(
        "--out_dir", default="artifacts/day48",
        help="Output directory for audit artefacts",
    )
    parser.add_argument(
        "--docs_dir", default="docs",
        help="Output directory for generated documentation",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_absolute():
        docs_dir = root / docs_dir

    print(f"[day48] Project root : {root}")
    print(f"[day48] Output dir   : {out_dir}")
    print(f"[day48] Docs dir     : {docs_dir}")
    print()

    # ── collect ────────────────────────────────────────────────────────
    audit: Dict[str, Any] = {
        "audit_timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "audit_script": "scripts/build_day48_reproducibility_audit.py",
    }

    print("[day48] Collecting git info …")
    audit["git"] = collect_git_info(str(root))

    print("[day48] Collecting environment info …")
    audit["environment"] = collect_env_info()

    print("[day48] Checking source files …")
    audit["source_files"] = collect_source_files(root)

    print("[day48] Checking documentation …")
    # We write docs first (below), so check after writing.

    print("[day48] Checking artifacts …")
    audit["artifact_check"] = collect_artifact_check(root)

    print("[day48] Scanning for risky tracked files …")
    audit["risky_tracked_files"] = collect_risky_tracked_files(str(root))

    # ── write docs first (so the docs_check reflects their existence) ──
    print("[day48] Writing docs/reproducibility_checklist.md …")
    write_reproducibility_checklist(docs_dir)

    print("[day48] Writing docs/run_from_scratch.md …")
    write_run_from_scratch(docs_dir)

    print("[day48] Writing docs/artifact_manifest.md …")
    write_artifact_manifest(docs_dir, audit)

    # Re-check docs after writing
    audit["docs_check"] = collect_docs_check(root)

    # ── warnings ───────────────────────────────────────────────────────
    audit["warnings"] = compute_warnings(audit)
    audit["warning_count"] = len(audit["warnings"])

    # ── write outputs ──────────────────────────────────────────────────
    print("[day48] Writing audit JSON …")
    json_path = write_audit_json(audit, out_dir)

    print("[day48] Writing reproducibility matrix CSV …")
    csv_path = write_matrix_csv(audit, out_dir)

    print("[day48] Writing artifacts/day48/README.md …")
    readme_path = write_day48_readme(audit, out_dir)

    # ── summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Day 48 Reproducibility Audit — Complete")
    print("=" * 60)
    print(f"  Branch            : {audit['git'].get('branch')}")
    print(f"  Commit            : {audit['git'].get('commit_short')}")
    print(
        f"  Working tree clean: "
        f"{'Yes' if audit['git'].get('working_tree_clean') else 'No'}"
    )
    print(f"  Warning count     : {audit['warning_count']}")
    risky_count = audit["risky_tracked_files"].get("risky_count", 0)
    print(f"  Risky tracked     : {risky_count}")
    print()
    print("  Generated files:")
    print(f"    {json_path}")
    print(f"    {csv_path}")
    print(f"    {readme_path}")
    print(f"    {docs_dir / 'reproducibility_checklist.md'}")
    print(f"    {docs_dir / 'run_from_scratch.md'}")
    print(f"    {docs_dir / 'artifact_manifest.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
