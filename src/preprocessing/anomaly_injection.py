"""
src/preprocessing/anomaly_injection.py
=======================================
Day 17 -- Synthetic anomaly injection for detector evaluation.

Provides routines to inject controlled anomalies into clinical code
sequences and build a labeled test set of normal vs. anomalous records.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectedAnomaly:
    """Record of a single injected anomaly."""

    original_codes: tuple[str, ...]
    anomalous_codes: tuple[str, ...]
    anomaly_type: str
    description: str


# ---------------------------------------------------------------------------
# Default pregnancy codes (subset of descendants of SNOMED 77386006)
# ---------------------------------------------------------------------------

_DEFAULT_PREGNANCY_CODES: tuple[str, ...] = (
    "SNOMED:77386006",   # Pregnant
    "SNOMED:72892002",   # Normal pregnancy
    "SNOMED:69532007",   # Complications of pregnancy
    "SNOMED:237240001",  # Pregnancy with threatened abortion
    "SNOMED:16356006",   # Multiple pregnancy
)


# ---------------------------------------------------------------------------
# Injection functions
# ---------------------------------------------------------------------------


def inject_missing_indication(
    codes: list[str],
    rng: random.Random,
) -> InjectedAnomaly:
    """Remove all SNOMED diagnosis tokens, simulating a missing indication.

    Parameters
    ----------
    codes:
        Original token list.
    rng:
        Random number generator (unused here but kept for consistent API).

    Returns
    -------
    InjectedAnomaly with diagnosis tokens stripped.
    """
    anomalous = [c for c in codes if not c.startswith("SNOMED:")]
    removed = [c for c in codes if c.startswith("SNOMED:")]

    return InjectedAnomaly(
        original_codes=tuple(codes),
        anomalous_codes=tuple(anomalous) if anomalous else ("UNK_EMPTY",),
        anomaly_type="missing_indication",
        description=f"Removed {len(removed)} SNOMED diagnosis tokens",
    )


def inject_random_code_swap(
    codes: list[str],
    all_codes_pool: list[str],
    rng: random.Random,
) -> InjectedAnomaly | None:
    """Replace one random SNOMED token with a random code from the pool.

    Parameters
    ----------
    codes:
        Original token list.
    all_codes_pool:
        Pool of all possible codes to draw a replacement from.
    rng:
        Random number generator.

    Returns
    -------
    InjectedAnomaly, or ``None`` if no SNOMED tokens exist in *codes*.
    """
    snomed_indices = [i for i, c in enumerate(codes) if c.startswith("SNOMED:")]
    if not snomed_indices:
        return None

    idx = rng.choice(snomed_indices)
    original_code = codes[idx]

    # Pick a replacement that differs from the original
    replacement = rng.choice(all_codes_pool)
    attempts = 0
    while replacement == original_code and attempts < 20:
        replacement = rng.choice(all_codes_pool)
        attempts += 1

    anomalous = list(codes)
    anomalous[idx] = replacement

    return InjectedAnomaly(
        original_codes=tuple(codes),
        anomalous_codes=tuple(anomalous),
        anomaly_type="random_code_swap",
        description=f"Swapped {original_code} at position {idx} with {replacement}",
    )


def inject_demographic_conflict(
    codes: list[str],
    sex: str,
    pregnancy_codes: list[str] | tuple[str, ...] | None,
    rng: random.Random,
) -> InjectedAnomaly | None:
    """Add a pregnancy code to a male patient's record.

    Parameters
    ----------
    codes:
        Original token list.
    sex:
        Patient sex (``"M"`` or ``"F"``).
    pregnancy_codes:
        Pool of pregnancy-related codes.  Falls back to a built-in list
        if ``None``.
    rng:
        Random number generator.

    Returns
    -------
    InjectedAnomaly, or ``None`` if the patient is not male.
    """
    if sex != "M":
        return None

    if pregnancy_codes is None:
        pregnancy_codes = list(_DEFAULT_PREGNANCY_CODES)

    added = rng.choice(list(pregnancy_codes))
    anomalous = list(codes) + [added]

    return InjectedAnomaly(
        original_codes=tuple(codes),
        anomalous_codes=tuple(anomalous),
        anomaly_type="demographic_conflict",
        description=f"Added pregnancy code {added} to male patient",
    )


# ---------------------------------------------------------------------------
# Test-set builder
# ---------------------------------------------------------------------------


def build_anomaly_test_set(
    df: pd.DataFrame,
    code_col: str = "codes_ont",
    n_per_type: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a labeled dataset of normal and anomalous records.

    Samples normal records, then generates anomalous variants using
    each injection function.

    Parameters
    ----------
    df:
        Source DataFrame with at least *code_col* and ``gender`` columns.
    code_col:
        Column name holding token lists (list[str] or JSON-encoded).
    n_per_type:
        Number of anomalous records to generate per anomaly type.
    seed:
        Random seed.

    Returns
    -------
    DataFrame with columns: ``codes``, ``label`` (0=normal, 1=anomalous),
    ``anomaly_type``, ``gender``.
    """
    rng = random.Random(seed)

    # Parse list columns if JSON-encoded
    code_series = df[code_col].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Collect all unique codes for the swap pool
    all_codes: set[str] = set()
    for codes in code_series:
        if isinstance(codes, list):
            all_codes.update(codes)
    all_codes_pool = sorted(all_codes)

    if not all_codes_pool:
        log.warning("No tokens found in column '%s'", code_col)
        return pd.DataFrame(columns=["codes", "label", "anomaly_type", "gender"])

    # Identify male records for demographic conflict injection
    has_gender = "gender" in df.columns
    male_indices = (
        df.index[df["gender"].str.upper() == "M"].tolist()
        if has_gender else []
    )

    # Collect all valid indices (non-empty sequences)
    valid_indices = [
        i for i, codes in enumerate(code_series)
        if isinstance(codes, list) and len(codes) > 0
    ]

    records: list[dict[str, Any]] = []

    # --- Normal records ---
    n_normal = min(n_per_type, len(valid_indices))
    normal_sample = rng.sample(valid_indices, n_normal)
    for idx in normal_sample:
        codes = code_series.iloc[idx]
        gender = df["gender"].iloc[idx] if has_gender else None
        records.append({
            "codes": codes,
            "label": 0,
            "anomaly_type": "normal",
            "gender": gender,
        })

    # --- Missing indication ---
    sample_mi = rng.sample(valid_indices, min(n_per_type, len(valid_indices)))
    count_mi = 0
    for idx in sample_mi:
        codes = list(code_series.iloc[idx])
        gender = df["gender"].iloc[idx] if has_gender else None
        result = inject_missing_indication(codes, rng)
        records.append({
            "codes": list(result.anomalous_codes),
            "label": 1,
            "anomaly_type": result.anomaly_type,
            "gender": gender,
        })
        count_mi += 1

    # --- Random code swap ---
    sample_rcs = rng.sample(valid_indices, min(n_per_type, len(valid_indices)))
    count_rcs = 0
    for idx in sample_rcs:
        codes = list(code_series.iloc[idx])
        gender = df["gender"].iloc[idx] if has_gender else None
        result = inject_random_code_swap(codes, all_codes_pool, rng)
        if result is not None:
            records.append({
                "codes": list(result.anomalous_codes),
                "label": 1,
                "anomaly_type": result.anomaly_type,
                "gender": gender,
            })
            count_rcs += 1

    # --- Demographic conflict ---
    count_dc = 0
    if male_indices:
        sample_dc = rng.sample(male_indices, min(n_per_type, len(male_indices)))
        for idx in sample_dc:
            codes = list(code_series.iloc[idx])
            sex = df["gender"].iloc[idx]
            result = inject_demographic_conflict(codes, sex, None, rng)
            if result is not None:
                records.append({
                    "codes": list(result.anomalous_codes),
                    "label": 1,
                    "anomaly_type": result.anomaly_type,
                    "gender": sex,
                })
                count_dc += 1

    log.info(
        "Anomaly test set: %d normal, %d missing_indication, "
        "%d random_code_swap, %d demographic_conflict",
        n_normal, count_mi, count_rcs, count_dc,
    )

    result_df = pd.DataFrame(records)
    return result_df.sample(frac=1, random_state=seed).reset_index(drop=True)
