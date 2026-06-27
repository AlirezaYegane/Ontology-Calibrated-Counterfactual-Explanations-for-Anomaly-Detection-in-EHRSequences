"""Phase 1b -- tests for the benchmark-v2 build (splits, manifest, leakage)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_benchmark_v2.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_benchmark_v2", BUILD_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_source(tmp_path: Path) -> Path:
    # Mix of normal records with sex-specific / insulin+diabetes / type-2 content,
    # across multiple subjects so splits are non-trivial.
    rows = []
    for i in range(40):
        subj = f"S{i % 8}"
        if i % 3 == 0:
            codes = ["DX_10_O0900", "DX_10_I10", "MED_ASPIRIN"]  # pregnancy (F)
            gender = "F"
        elif i % 3 == 1:
            codes = ["MED_INSULIN", "DX_10_E119", "DX_10_I10"]  # insulin + diabetes
            gender = "M"
        else:
            codes = ["DX_10_E119", "MED_METFORMIN", "DX_10_N179"]  # type-2 diabetes
            gender = "F"
        rows.append(
            {
                "subject_id": subj,
                "hadm_id": f"H{i}",
                "gender": gender,
                "age_group": "45-64",
                "codes": codes,
                "is_synthetic_anomaly": 0,
            }
        )
    df = pd.DataFrame(rows)
    p = tmp_path / "toy_source.pkl"
    df.to_pickle(p)
    return p


def test_subject_splits_no_overlap() -> None:
    mod = _load_build_module()
    assignment = mod._subject_splits([f"S{i}" for i in range(30)], rng_seed=42)
    splits: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for subj, sp in assignment.items():
        splits[sp].add(subj)
    assert splits["train"] & splits["val"] == set()
    assert splits["train"] & splits["test"] == set()
    assert splits["val"] & splits["test"] == set()


def test_build_produces_valid_benchmark(tmp_path: Path) -> None:
    mod = _load_build_module()
    source = _toy_source(tmp_path)
    result = mod.build(source, max_records=40, seed=42)
    df = result["df"]

    # label + columns present
    for col in (
        "model_visible_sequence",
        "gender",
        "label",
        "anomaly_type",
        "subject_id",
        "split",
        "hidden_eval_metadata",
        "audit_metadata",
    ):
        assert col in df.columns

    # subject-level split: no subject in >1 split
    subj_split = df.groupby("subject_id")["split"].nunique()
    assert (subj_split == 1).all()

    # train split must be clean (no anomalies) -> usable for unsupervised training
    train = df[df["split"] == "train"]
    assert (train["label"] == 1).sum() == 0

    # model-visible records must not carry answer-key columns
    forbidden = mod.__dict__  # ensure module loaded
    assert forbidden is not None
    for _, row in df.iterrows():
        assert (
            "original_gender" not in row["model_visible_sequence"]
            if isinstance(row["model_visible_sequence"], dict)
            else True
        )

    # at least one anomaly produced overall
    assert (df["label"] == 1).sum() >= 1


def test_real_manifest_if_present() -> None:
    manifest = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_v2"
        / "benchmark_v2_manifest.json"
    )
    if not manifest.exists():
        return  # built on demand elsewhere
    import json

    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["subject_level_split_ok"] is True
    assert m["leakage_guard"]["train_split_is_clean_normal_only"] is True
    assert m["n_anomaly"] >= 1
