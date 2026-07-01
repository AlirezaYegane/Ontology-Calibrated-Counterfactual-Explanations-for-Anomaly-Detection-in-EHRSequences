"""
src/explanations/counterfactual.py
==================================
Phase 4 -- Leakage-free ontology counterfactual repair.

The pre-Phase-4 module (audited in ``artifacts/phase4/counterfactual_audit_before.md``)
was circular: it read injection answer keys (``bad_code`` / ``expected_code`` /
``replacement_code`` / ``anomaly_type``) to undo the synthetic anomaly. This rewrite
removes ALL of that. A counterfactual is generated using ONLY:

  * the model-visible token sequence + demographics (gender / age_group),
  * the real :class:`OntologyAwareScorer` (mode="real") violations + S_ont,
  * real ontology neighborhoods / distance (``src.ontology.distance``),
  * the unsupervised detector as a DIAGNOSTIC-only signal (never drives repair).

It never reads ``label``, ``anomaly_type``, ``hidden_eval_metadata``,
``audit_metadata``, ``bad_code``, ``expected_code``, ``replacement_code`` or any
benchmark answer key. Repair is a deterministic beam search over edits derived from
the scorer's violations; candidates are validated by re-scoring with the same
independent scorer.

Repair families (driven by the violation ``kind`` reported by the scorer, NOT by
``anomaly_type``):
  * ``demographic_mismatch``   -> remove (or ontology-neighbor replace) the
    sex-incompatible diagnosis token.
  * ``missing_required_code``  -> remove the unsupported medication (preferred), or
    add a curated required-context diagnosis (higher clinical-risk, so the search
    prefers removal).
  * ``mutual_exclusion``       -> remove (or generalize) one side of the conflicting
    pair, chosen by score reduction -- NOT by knowing which side was injected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from src.ontology.distance import neighborhood, shortest_path_distance

# ---------------------------------------------------------------------------
# Model-visible field allowlist (structural leakage guard)
# ---------------------------------------------------------------------------
# generate_counterfactual reads ONLY these keys off the input record. Everything
# else (labels, answer keys, hidden/audit metadata) is structurally ignored.
_SEQUENCE_KEYS = ("model_visible_sequence", "codes", "sequence_tokens", "tokens")
_GENDER_KEYS = ("gender", "sex")
_AGE_KEYS = ("age_group", "age")

_DIAG_PREFIXES = ("DX_", "DIAG_", "ICD")
_MED_PREFIXES = ("MED_", "DRUG_", "RAW_DRUG", "RX")

# Default repair-objective weights (documented in the README): minimize residual
# S_ont first; then prefer fewer edits, smaller ontology distance, lower risk.
_LAMBDA_EDIT = 0.05
_LAMBDA_DIST = 0.02
_LAMBDA_RISK = 0.10

# Per-operation clinical-risk priors (a removed *impossible* code is safest; a
# fabricated context diagnosis is riskiest).
_RISK = {
    "remove_diagnosis": 0.0,
    "remove_medication": 0.3,
    "replace_diagnosis": 0.2,
    "add_context": 1.0,
}


@dataclass(frozen=True)
class CounterfactualEdit:
    """A single, structured, ontology-grounded edit."""

    operation: str  # "remove" | "replace" | "add"
    token_before: str | None = None
    token_after: str | None = None
    ontology_source: str = ""
    distance: int | None = None
    clinical_risk: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualResult:
    """Result of a leakage-free counterfactual repair."""

    status: str  # "valid" | "improved_not_valid" | "not_improved" | "no_candidate" | "no_violation"
    validity: bool
    original_codes: list[str]
    repaired_codes: list[str]
    edits: list[CounterfactualEdit]
    # ontology score (primary, trustworthy signal)
    s_ont_before: float
    s_ont_after: float
    # calibrated / detector scores (s_det diagnostic-only; smoke-scale)
    s_cal_before: float | None = None
    s_cal_after: float | None = None
    s_det_before: float | None = None
    s_det_after: float | None = None
    rule_violations_before: list[dict[str, Any]] = field(default_factory=list)
    rule_violations_after: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_edits(self) -> int:
        return len(self.edits)

    @property
    def delta_s_ont(self) -> float:
        return self.s_ont_before - self.s_ont_after

    @property
    def delta_s_cal(self) -> float | None:
        if self.s_cal_before is None or self.s_cal_after is None:
            return None
        return self.s_cal_before - self.s_cal_after

    @property
    def delta_s_det(self) -> float | None:
        if self.s_det_before is None or self.s_det_after is None:
            return None
        return self.s_det_before - self.s_det_after

    @property
    def sparsity(self) -> float:
        """Fraction of the original sequence left unchanged (1.0 = no edits)."""
        n = max(1, len(self.original_codes))
        return 1.0 - (self.num_edits / n)

    @property
    def total_distance(self) -> int:
        return sum(e.distance for e in self.edits if e.distance is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "validity": self.validity,
            "num_edits": self.num_edits,
            "sparsity": round(self.sparsity, 4),
            "total_ontology_distance": self.total_distance,
            "s_ont_before": round(self.s_ont_before, 6),
            "s_ont_after": round(self.s_ont_after, 6),
            "delta_s_ont": round(self.delta_s_ont, 6),
            "s_cal_before": _round_opt(self.s_cal_before),
            "s_cal_after": _round_opt(self.s_cal_after),
            "delta_s_cal": _round_opt(self.delta_s_cal),
            "s_det_before": _round_opt(self.s_det_before),
            "s_det_after": _round_opt(self.s_det_after),
            "delta_s_det": _round_opt(self.delta_s_det),
            "edits": [e.to_dict() for e in self.edits],
            "rule_violations_before": self.rule_violations_before,
            "rule_violations_after": self.rule_violations_after,
            "failure_reason": self.failure_reason,
            "metadata": self.metadata,
        }


def _round_opt(v: float | None) -> float | None:
    return None if v is None else round(float(v), 6)


# ---------------------------------------------------------------------------
# Model-visible extraction (the only place the record is read)
# ---------------------------------------------------------------------------


def _first(record: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return None


def extract_model_visible(record: Mapping[str, Any]) -> tuple[list[str], Any, Any]:
    """Read ONLY model-visible fields: (tokens, gender, age_group)."""
    seq = _first(record, _SEQUENCE_KEYS)
    tokens = [str(t) for t in seq] if isinstance(seq, (list, tuple)) else []
    return tokens, _first(record, _GENDER_KEYS), _first(record, _AGE_KEYS)


# ---------------------------------------------------------------------------
# Token / ontology helpers
# ---------------------------------------------------------------------------


def _icd_body(token: str) -> str | None:
    up = str(token).strip().upper()
    if up.startswith("DX_10_"):
        return up[len("DX_10_") :]
    if up.startswith("DX_9_"):
        return up[len("DX_9_") :]
    return None


def _is_diag(token: str) -> bool:
    up = token.upper()
    return up.startswith(_DIAG_PREFIXES) or up.startswith("SNOMED:")


def _is_med(token: str) -> bool:
    up = token.upper()
    return up.startswith(_MED_PREFIXES)


def _token_concepts(tokens: list[str], index: Any) -> dict[str, list[str]]:
    """Map each token to its canonical ontology codes (per-token, cached)."""
    from src.scoring.ontology_aware import map_tokens_to_ontology_codes

    out: dict[str, list[str]] = {}
    for t in tokens:
        if t not in out:
            out[t] = map_tokens_to_ontology_codes([t], index)["mapped_codes"]
    return out


def _evidence_tokens(
    evidence: list[str], tokens: list[str], tok2concepts: dict[str, list[str]]
) -> list[str]:
    """Resolve a violation's evidence codes to the model-visible tokens that
    produced them (so they can be removed/replaced)."""
    token_set = set(tokens)
    out: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        if t not in seen:
            seen.add(t)
            out.append(t)

    for e in evidence:
        e = str(e)
        if e in token_set:  # source token
            _add(e)
        elif e.startswith(("RXNORM:", "SNOMED:")):
            for t in tokens:
                if e in tok2concepts.get(t, []):
                    _add(t)
        elif e.startswith("DRUGNAME:"):
            name = e.split(":", 1)[1].upper()
            for t in tokens:
                if _is_med(t) and name in t.upper():
                    _add(t)
        else:  # ICD body e.g. "E10_9" / "E119" / "O80"
            for t in tokens:
                body = _icd_body(t)
                if body is not None and body.startswith(e):
                    _add(t)
    return out


def _neighbor_replacements(
    token: str, tok2concepts: dict[str, list[str]], index: Any, max_candidates: int = 2
) -> list[tuple[str, str, str, int]]:
    """Ontology-neighbor replacement candidates for a diagnosis token.

    Returns ``(replacement_token, from_concept, to_concept, distance)`` tuples,
    nearest first. The replacement is a canonical ``SNOMED:<id>`` token (the scorer
    consumes these directly). Only generalizing/sibling moves within 1 hop.
    """
    if not _is_diag(token):
        return []
    cands: list[tuple[str, str, str, int]] = []
    seen: set[str] = set()
    for concept in tok2concepts.get(token, []):
        for nb in neighborhood(index, concept, radius=1):
            if nb == concept or nb in seen:
                continue
            seen.add(nb)
            dist = shortest_path_distance(index, concept, nb) or 1
            repl = nb if nb.upper().startswith("SNOMED:") else f"SNOMED:{nb}"
            cands.append((repl, concept, nb, int(dist)))
    cands.sort(key=lambda x: (x[3], x[0]))
    return cands[:max_candidates]


# ---------------------------------------------------------------------------
# Candidate edit generation (per violation kind; from the scorer, not labels)
# ---------------------------------------------------------------------------


def _candidate_edits(
    tokens: list[str], violations: list[dict[str, Any]], index: Any
) -> list[CounterfactualEdit]:
    tok2concepts = _token_concepts(tokens, index)
    edits: list[CounterfactualEdit] = []
    seen: set[tuple[str, str | None, str | None]] = set()

    def _push(e: CounterfactualEdit) -> None:
        key = (e.operation, e.token_before, e.token_after)
        if key not in seen:
            seen.add(key)
            edits.append(e)

    for v in violations:
        kind = v.get("kind", "")
        evidence = [str(c) for c in v.get("codes", [])]
        implicated = _evidence_tokens(evidence, tokens, tok2concepts)

        if kind == "demographic_mismatch":
            for tok in implicated:
                _push(
                    CounterfactualEdit(
                        operation="remove",
                        token_before=tok,
                        ontology_source=f"violation:{v.get('rule_id')}",
                        distance=0,
                        clinical_risk=_RISK["remove_diagnosis"],
                        rationale="remove sex-incompatible diagnosis (resolves demographic conflict)",
                    )
                )
                for repl, frm, to, dist in _neighbor_replacements(
                    tok, tok2concepts, index
                ):
                    _push(
                        CounterfactualEdit(
                            operation="replace",
                            token_before=tok,
                            token_after=repl,
                            ontology_source=f"neighbor_of:{frm}",
                            distance=dist,
                            clinical_risk=_RISK["replace_diagnosis"],
                            rationale="replace with a nearby non-restricted ontology neighbor",
                        )
                    )

        elif kind == "missing_required_code":
            for tok in implicated:  # the unsupported medication token(s)
                _push(
                    CounterfactualEdit(
                        operation="remove",
                        token_before=tok,
                        ontology_source=f"violation:{v.get('rule_id')}",
                        distance=0,
                        clinical_risk=_RISK["remove_medication"],
                        rationale="remove medication lacking a supporting indication",
                    )
                )
            for ctx in _context_add_candidates(evidence):
                _push(
                    CounterfactualEdit(
                        operation="add",
                        token_after=ctx,
                        ontology_source="curated_required_context_group",
                        distance=None,
                        clinical_risk=_RISK["add_context"],
                        rationale="add a curated required-context diagnosis (conservative; fabricates a dx)",
                    )
                )

        elif kind == "mutual_exclusion":
            for tok in implicated:
                _push(
                    CounterfactualEdit(
                        operation="remove",
                        token_before=tok,
                        ontology_source=f"violation:{v.get('rule_id')}",
                        distance=0,
                        clinical_risk=_RISK["remove_diagnosis"],
                        rationale="remove one side of the mutually-exclusive pair",
                    )
                )
                for repl, frm, to, dist in _neighbor_replacements(
                    tok, tok2concepts, index
                ):
                    _push(
                        CounterfactualEdit(
                            operation="replace",
                            token_before=tok,
                            token_after=repl,
                            ontology_source=f"neighbor_of:{frm}",
                            distance=dist,
                            clinical_risk=_RISK["replace_diagnosis"],
                            rationale="generalize one side to a non-conflicting ontology neighbor",
                        )
                    )
    return edits


# Curated drug-class -> representative required-context SNOMED concept (from the
# Phase 3b rule packs). Used ONLY for the optional add-context edit.
def _context_add_candidates(evidence: list[str]) -> list[str]:
    from src.ontology import rule_packs as rp

    anticoag = set(rp.ANTICOAGULANT_RXCUIS)
    levo = set(rp.LEVOTHYROXINE_RXCUIS)
    out: list[str] = []
    for e in evidence:
        if e == "DRUGNAME:INSULIN" or e in set(rp.INSULIN_RXCUIS):
            out.append("SNOMED:73211009")  # diabetes mellitus
        elif e in anticoag:
            out.append("SNOMED:49436004")  # atrial fibrillation
        elif e in levo:
            out.append("SNOMED:40930008")  # hypothyroidism
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _apply_edit(tokens: list[str], edit: CounterfactualEdit) -> list[str]:
    if edit.operation == "remove" and edit.token_before is not None:
        out, removed = [], False
        for t in tokens:
            if t == edit.token_before and not removed:
                removed = True
                continue
            out.append(t)
        return out
    if edit.operation == "replace" and edit.token_before is not None:
        out, done = [], False
        for t in tokens:
            if t == edit.token_before and not done:
                out.append(edit.token_after if edit.token_after is not None else t)
                done = True
            else:
                out.append(t)
        return out
    if edit.operation == "add" and edit.token_after is not None:
        return tokens if edit.token_after in tokens else [*tokens, edit.token_after]
    return list(tokens)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_ont(scorer: Any, tokens: list[str], gender: Any, age: Any) -> dict[str, Any]:
    row = {"codes": tokens, "gender": gender, "age_group": age}
    out = scorer.score(row, s_det=0.0)
    return {
        "s_ont": float(out["s_ont"]),
        "s_cal": float(out["s_cal"]),
        "violations": list(out.get("violations", [])),
    }


def _max_severity(violations: list[dict[str, Any]]) -> float:
    return max((float(v.get("severity", 0.0)) for v in violations), default=0.0)


def _cost(state: dict[str, Any]) -> float:
    edits: list[CounterfactualEdit] = state["edits"]
    dist = sum(e.distance for e in edits if e.distance is not None)
    risk = sum(e.clinical_risk for e in edits)
    return (
        state["info"]["s_ont"]
        + _LAMBDA_EDIT * len(edits)
        + _LAMBDA_DIST * dist
        + _LAMBDA_RISK * risk
    )


def _better(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """a strictly better than b: lower S_ont, then fewer edits, then lower cost."""
    ai, bi = a["info"], b["info"]
    if ai["s_ont"] != bi["s_ont"]:
        return ai["s_ont"] < bi["s_ont"]
    if len(a["edits"]) != len(b["edits"]):
        return len(a["edits"]) < len(b["edits"])
    return _cost(a) < _cost(b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_counterfactual(
    record: Mapping[str, Any],
    scorer: Any,
    ontology_index: Any,
    *,
    detector: Any = None,
    max_edits: int = 3,
    beam_size: int = 20,
    mode: str = "ontology",
    seed: int = 42,
    min_delta: float = 0.05,
    allowed_operations: tuple[str, ...] | None = None,
) -> CounterfactualResult:
    """Generate a leakage-free counterfactual repair for one record.

    Deterministic (no randomness; ``seed`` is accepted for reproducibility/API
    stability). Reads only model-visible fields. Drives the beam search with the
    real ontology S_ont; the optional ``detector`` is scored before/after the final
    repair as a DIAGNOSTIC-only signal and never influences which edits are chosen.

    ``allowed_operations`` (e.g. ``("remove",)``) restricts the candidate edit
    operations -- used by the Phase 7 edit-strategy ablation. ``None`` allows all.
    """
    tokens, gender, age = extract_model_visible(record)
    before = _score_ont(scorer, tokens, gender, age)

    s_det_before = _detector_score(detector, tokens)
    s_det_after = s_det_before  # updated if a repair is found
    s_cal_before = _s_cal(before["s_ont"], s_det_before)

    if not before["violations"]:
        return CounterfactualResult(
            status="no_violation",
            validity=False,
            original_codes=list(tokens),
            repaired_codes=list(tokens),
            edits=[],
            s_ont_before=before["s_ont"],
            s_ont_after=before["s_ont"],
            s_cal_before=s_cal_before,
            s_cal_after=s_cal_before,
            s_det_before=s_det_before,
            s_det_after=s_det_before,
            rule_violations_before=[],
            rule_violations_after=[],
            failure_reason="no_ontology_violation_to_repair",
            metadata={"mode": mode, "max_edits": max_edits, "beam_size": beam_size},
        )

    start = {"tokens": list(tokens), "edits": [], "info": before}
    beam = [start]
    best = start
    any_candidate = False

    for _ in range(max_edits):
        candidates: list[dict[str, Any]] = []
        for state in beam:
            if not state["info"]["violations"]:
                continue
            for edit in _candidate_edits(
                state["tokens"], state["info"]["violations"], ontology_index
            ):
                if (
                    allowed_operations is not None
                    and edit.operation not in allowed_operations
                ):
                    continue
                new_tokens = _apply_edit(state["tokens"], edit)
                if new_tokens == state["tokens"]:
                    continue
                any_candidate = True
                info = _score_ont(scorer, new_tokens, gender, age)
                candidates.append(
                    {
                        "tokens": new_tokens,
                        "edits": [*state["edits"], edit],
                        "info": info,
                    }
                )
        if not candidates:
            break
        candidates.sort(key=_cost)
        beam = candidates[:beam_size]
        for s in beam:
            if _better(s, best):
                best = s
        if best["info"]["s_ont"] == 0.0:
            break

    after = best["info"]
    edits: list[CounterfactualEdit] = best["edits"]

    if not edits:
        status = "no_candidate" if not any_candidate else "not_improved"
        return CounterfactualResult(
            status=status,
            validity=False,
            original_codes=list(tokens),
            repaired_codes=list(tokens),
            edits=[],
            s_ont_before=before["s_ont"],
            s_ont_after=before["s_ont"],
            s_cal_before=s_cal_before,
            s_cal_after=s_cal_before,
            s_det_before=s_det_before,
            s_det_after=s_det_before,
            rule_violations_before=before["violations"],
            rule_violations_after=before["violations"],
            failure_reason="no_score_reducing_edit_found",
            metadata={"mode": mode, "max_edits": max_edits, "beam_size": beam_size},
        )

    s_det_after = _detector_score(detector, best["tokens"])
    s_cal_after = _s_cal(after["s_ont"], s_det_after)

    valid, reason = _validate(
        before, after, edits, max_edits, min_delta, best["tokens"]
    )
    if valid:
        status = "valid"
    elif after["s_ont"] < before["s_ont"]:
        status = "improved_not_valid"
    else:
        status = "not_improved"

    return CounterfactualResult(
        status=status,
        validity=valid,
        original_codes=list(tokens),
        repaired_codes=list(best["tokens"]),
        edits=edits,
        s_ont_before=before["s_ont"],
        s_ont_after=after["s_ont"],
        s_cal_before=s_cal_before,
        s_cal_after=s_cal_after,
        s_det_before=s_det_before,
        s_det_after=s_det_after,
        rule_violations_before=before["violations"],
        rule_violations_after=after["violations"],
        failure_reason=None if valid else reason,
        metadata={
            "mode": mode,
            "max_edits": max_edits,
            "beam_size": beam_size,
            "seed": seed,
            "detector_used": detector is not None,
            "detector_note": "s_det is smoke-scale, diagnostic-only; does not drive repair",
        },
    )


def _validate(
    before: dict[str, Any],
    after: dict[str, Any],
    edits: list[CounterfactualEdit],
    max_edits: int,
    min_delta: float,
    after_tokens: list[str],
) -> tuple[bool, str]:
    """Conservative validity: meaningful S_ont reduction, no new higher-severity
    violation, within budget, non-empty record."""
    if len(edits) < 1 or len(edits) > max_edits:
        return False, "edit_budget_violation"
    if not after_tokens:
        return False, "empty_record"
    delta = before["s_ont"] - after["s_ont"]
    resolved = len(after["violations"]) == 0
    if not (resolved or delta >= min_delta):
        return False, "insufficient_score_reduction"
    # no NEW violation of higher severity than the worst original
    max_before = _max_severity(before["violations"])
    before_keys = {
        (v.get("rule_id"), v.get("kind"), tuple(v.get("codes", [])))
        for v in before["violations"]
    }
    for v in after["violations"]:
        key = (v.get("rule_id"), v.get("kind"), tuple(v.get("codes", [])))
        if key not in before_keys and float(v.get("severity", 0.0)) > max_before:
            return False, "introduced_higher_severity_violation"
    return True, "ok"


def _detector_score(detector: Any, tokens: list[str]) -> float | None:
    if detector is None:
        return None
    try:
        return float(detector.anomaly_scores([tokens])[0])
    except Exception:
        return None


def _s_cal(s_ont: float, s_det: float | None) -> float:
    from src.scoring.ontology_aware import ScoreWeights, compute_calibrated_score

    return compute_calibrated_score(
        s_det=0.0 if s_det is None else float(s_det),
        s_ont=s_ont,
        weights=ScoreWeights(),
        include_ont=True,
    )


__all__ = [
    "CounterfactualEdit",
    "CounterfactualResult",
    "generate_counterfactual",
    "extract_model_visible",
]
