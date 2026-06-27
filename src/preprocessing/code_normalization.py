"""
src/preprocessing/code_normalization.py
========================================
Phase 2b -- Token -> source-vocabulary code normalization.

Processed EHR tokens look like ``DX_9_4019``, ``DX_10_E785``, ``PROC_9_3961``,
``MED_ACETAMINOPHEN_IV``. UMLS/RxNorm maps key on the *source* vocabulary code
form (e.g. ICD-9 ``401.9``, ICD-10 ``E78.5``, drug name ``ACETAMINOPHEN``).

This module produces a small set of *candidate* normalized forms per token so the
coverage/mapping code can try each against the crosswalk maps. It is deliberately
conservative: it returns the raw and dotted forms (which are unambiguous) rather
than aggressive transformations that could create false matches.
"""

from __future__ import annotations

_DX9_PREFIX = "DX_9_"
_DX10_PREFIX = "DX_10_"
_MED_PREFIXES = ("MED_", "DRUG_", "RAW_DRUG_")


def icd9_candidates(body: str) -> list[str]:
    """Candidate ICD-9-CM code forms for an undotted body like ``4019``/``V270``.

    ICD-9-CM dotted convention:
      * numeric: 3-digit category + '.' + remainder  -> ``401.9``
      * V codes: ``V`` + 2 digits + '.' + remainder   -> ``V27.0``
      * E codes: ``E`` + 3 digits + '.' + remainder   -> ``E849.7``
    """
    body = body.strip().upper()
    cands: list[str] = [body]
    if not body:
        return cands
    if body.startswith("E"):
        if len(body) > 4:
            cands.append(f"{body[:4]}.{body[4:]}")
    elif body.startswith("V"):
        if len(body) > 3:
            cands.append(f"{body[:3]}.{body[3:]}")
    elif body[:3].isdigit() and len(body) > 3:
        cands.append(f"{body[:3]}.{body[3:]}")
    return _dedup(cands)


def icd10_candidates(body: str) -> list[str]:
    """Candidate ICD-10-CM code forms for an undotted body like ``E785``.

    ICD-10-CM dotted convention: first 3 characters + '.' + remainder
    (``E785`` -> ``E78.5``, ``Z9861`` -> ``Z98.61``).
    """
    body = body.strip().upper()
    cands: list[str] = [body]
    if len(body) > 3:
        cands.append(f"{body[:3]}.{body[3:]}")
    return _dedup(cands)


def normalize_diagnosis_token(token: str) -> list[str]:
    """Return candidate ICD code forms for a ``DX_9_*`` / ``DX_10_*`` token."""
    up = token.strip()
    if up.startswith(_DX9_PREFIX):
        return icd9_candidates(up[len(_DX9_PREFIX) :])
    if up.startswith(_DX10_PREFIX):
        return icd10_candidates(up[len(_DX10_PREFIX) :])
    # Unknown DX layout: strip a leading DX_ and return the remainder.
    if up.upper().startswith("DX_"):
        return [up.split("_", 2)[-1].upper()]
    return [up.upper()]


def normalize_drug_token(token: str) -> list[str]:
    """Return candidate normalized drug-name keys for a ``MED_*`` token.

    The RxNorm map keys are ``"_".join(name.upper().split())`` (e.g.
    ``ACETAMINOPHEN``). We return the full uppercased suffix plus a few
    conservative trailing-descriptor trims (e.g. dropping a trailing route/form
    like ``_IV``) WITHOUT aggressive token surgery that could mis-map.
    """
    up = token.strip().upper()
    body = up
    for pref in (p.upper() for p in _MED_PREFIXES):
        if up.startswith(pref):
            body = up[len(pref) :]
            break
    cands = [body]
    # drop a single trailing common route/form descriptor as a fallback
    parts = body.split("_")
    _TRAILING = {
        "IV",
        "PO",
        "ORAL",
        "INJ",
        "INJECTION",
        "TAB",
        "TABLET",
        "NEB",
        "INHALER",
    }
    if len(parts) > 1 and parts[-1] in _TRAILING:
        cands.append("_".join(parts[:-1]))
    return _dedup([c for c in cands if c])


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# RxNorm ingredient-level matching (Phase 2b-fix)
# ---------------------------------------------------------------------------

# Descriptor words removed before ingredient matching (dose/form/route/release).
_MED_DESCRIPTOR_WORDS: frozenset[str] = frozenset(
    {
        # units / concentration
        "MG",
        "ML",
        "MCG",
        "G",
        "GM",
        "UNIT",
        "UNITS",
        "PERCENT",
        "MEQ",
        "MMOL",
        "NS",
        "D5",
        "D5W",
        "D10",
        "D50",
        # release
        "IMMEDIATE",
        "EXTENDED",
        "RELEASE",
        "ER",
        "IR",
        "XR",
        "SR",
        "CR",
        # form
        "TABLET",
        "TAB",
        "TABS",
        "CAPSULE",
        "CAP",
        "CAPS",
        "SOLUTION",
        "SOLN",
        "INJECTION",
        "INJ",
        "NEB",
        "NEBULIZER",
        "INHALER",
        "INHALATION",
        "FLUSH",
        "BAG",
        "VIAL",
        "SYRINGE",
        "DROPS",
        "CREAM",
        "OINTMENT",
        "PATCH",
        "SUPPOSITORY",
        "SUSPENSION",
        "SUSP",
        "ELIXIR",
        "SYRUP",
        # route
        "ORAL",
        "PO",
        "IV",
        "IM",
        "SUBCUT",
        "SUBCUTANEOUS",
        "TOPICAL",
        "INHALED",
        "INTRAVENOUS",
        "OPHTHALMIC",
        "OTIC",
        "NASAL",
        "RECTAL",
    }
)

# Single-word ingredients too generic to map alone (electrolyte/filler fragments).
# They may only match as part of a longer multi-word ingredient phrase.
_AMBIGUOUS_SINGLE_INGREDIENTS: frozenset[str] = frozenset(
    {
        "SODIUM",
        "CHLORIDE",
        "CALCIUM",
        "POTASSIUM",
        "MAGNESIUM",
        "PHOSPHATE",
        "ACID",
        "WATER",
        "BAG",
        "BICARBONATE",
        "GLUCONATE",
        "SULFATE",
    }
)

_MED_PREFIX_LIST = ("RAW_DRUG_", "MED_", "DRUG_", "RX_", "RXNORM_")


def _strip_med_prefix(token: str) -> str:
    up = token.strip().upper()
    for pref in _MED_PREFIX_LIST:
        if up.startswith(pref):
            return up[len(pref) :]
    return up


def strip_medication_descriptors(words: list[str]) -> list[str]:
    """Drop dose/form/route/release descriptors and pure-number tokens.

    Conservative: only removes recognised descriptor words and numeric fragments,
    never the (potential) ingredient words themselves.
    """
    out: list[str] = []
    for w in words:
        if not w:
            continue
        if w.isdigit():
            continue
        if w in _MED_DESCRIPTOR_WORDS:
            continue
        out.append(w)
    return out


def match_medication_ingredient(
    token: str,
    ingredient_set: frozenset[str] | set[str],
    min_single_word_len: int = 5,
) -> str | None:
    """Return the matched RxNorm ingredient phrase for a ``MED_*`` token, or None.

    Strategy (conservative, longest-match first):
      1. strip the MED_/DRUG_ prefix and dose/form/route descriptors,
      2. try the longest contiguous word-span that is a known ingredient,
      3. reject ambiguous single-word electrolyte fragments (e.g. ``SODIUM``) and
         very short single words, so nothing is mapped on a weak partial match.

    Returns the matched ingredient phrase (underscore-joined) or ``None`` when no
    safe match exists (e.g. ``MED_INSULIN`` -> None, because RxNorm has no bare
    "insulin" ingredient, only specific insulins).
    """
    body = _strip_med_prefix(token)
    words = strip_medication_descriptors(body.split("_"))
    n = len(words)
    if n == 0:
        return None

    for length in range(n, 0, -1):
        for start in range(0, n - length + 1):
            phrase = "_".join(words[start : start + length])
            if phrase not in ingredient_set:
                continue
            if length == 1:
                if phrase in _AMBIGUOUS_SINGLE_INGREDIENTS:
                    continue
                if len(phrase) < min_single_word_len:
                    continue
            return phrase
    return None


__all__ = [
    "icd9_candidates",
    "icd10_candidates",
    "normalize_diagnosis_token",
    "normalize_drug_token",
    "strip_medication_descriptors",
    "match_medication_ingredient",
]
