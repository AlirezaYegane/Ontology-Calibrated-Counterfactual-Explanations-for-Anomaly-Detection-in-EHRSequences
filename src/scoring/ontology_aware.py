from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # torch is only needed by the legacy supervised Day-25 scorer
    import torch
except ImportError:  # pragma: no cover - allows Phase 3 scoring without torch
    torch = None  # type: ignore[assignment]

from src.ontology.rules import compute_s_ont, extract_tokens_from_row, normalize_tokens


def combine_scores(
    sdet: float,
    sont: float,
    sgen: float = 0.0,
    w_det: float = 0.70,
    w_ont: float = 0.30,
    w_gen: float = 0.0,
) -> float:
    sont_norm = 1.0 - math.exp(-max(float(sont), 0.0))
    sgen_norm = max(0.0, min(1.0, float(sgen)))

    raw = (w_det * float(sdet)) + (w_ont * sont_norm) + (w_gen * sgen_norm)
    denom = max(w_det + w_ont + w_gen, 1e-8)
    return float(raw / denom)


class Day25OntologyAwareScorer:
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "cuda",
        max_len: int | None = None,
        truncate_strategy: str | None = None,
    ) -> None:
        from src.models.detector_supervised import GRUSequenceBinaryClassifier

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.config = ckpt["config"]

        import json
        from pathlib import Path

        self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))

        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.max_len = int(max_len or self.config.get("max_len", 256))
        self.truncate_strategy = str(
            truncate_strategy or self.config.get("truncate_strategy", "tail")
        )

        self.model = GRUSequenceBinaryClassifier(
            vocab_size=len(self.vocab),
            embed_dim=int(self.config["embed_dim"]),
            hidden_dim=int(self.config["hidden_dim"]),
            num_layers=int(self.config["num_layers"]),
            dropout=float(self.config["dropout"]),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.unk_idx = int(self.vocab.get("<unk>", 1))

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        tokens = list(tokens)
        if not tokens:
            return [self.unk_idx]

        if len(tokens) > self.max_len:
            if self.truncate_strategy == "head":
                tokens = tokens[: self.max_len]
            else:
                tokens = tokens[-self.max_len :]

        return [int(self.vocab.get(token, self.unk_idx)) for token in tokens]

    def detector_score(self, tokens: list[str]) -> float:
        with torch.no_grad():
            ids = self.encode_tokens(tokens)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

            logits = self.model(input_ids, lengths)
            prob = torch.sigmoid(logits).item()
            return float(prob)

    def detector_token_contributions(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}

        base = self.detector_score(tokens)
        contrib: dict[str, float] = defaultdict(float)

        if len(tokens) == 1:
            contrib[tokens[0]] += base
            return {
                k: round(v, 6)
                for k, v in sorted(contrib.items(), key=lambda x: x[1], reverse=True)
            }

        for idx, token in enumerate(tokens):
            reduced = tokens[:idx] + tokens[idx + 1 :]
            new_score = self.detector_score(reduced)
            delta = max(0.0, base - new_score)
            contrib[token] += float(delta)

        return {
            k: round(v, 6)
            for k, v in sorted(contrib.items(), key=lambda x: x[1], reverse=True)
        }

    def score_record(self, record: Mapping[str, Any] | list[str]) -> dict[str, Any]:
        row = record.to_dict() if hasattr(record, "to_dict") else record
        tokens = (
            extract_tokens_from_row(row)
            if isinstance(row, Mapping)
            else normalize_tokens(row)
        )

        sdet = self.detector_score(tokens)
        ont = compute_s_ont(row if isinstance(row, Mapping) else tokens)
        sgen = 0.0
        scal = combine_scores(sdet=sdet, sont=ont["sont"], sgen=sgen)

        combined: dict[str, float] = defaultdict(float)
        det_contrib = self.detector_token_contributions(tokens)

        if det_contrib:
            max_det = max(det_contrib.values()) or 1.0
            for token, value in det_contrib.items():
                combined[token] += 0.50 * (float(value) / max_det)

        if ont["token_weights"]:
            max_ont = max(ont["token_weights"].values()) or 1.0
            for token, value in ont["token_weights"].items():
                combined[token] += 0.50 * (float(value) / max_ont)

        ranked = [
            {"token": token, "score": round(score, 6)}
            for token, score in sorted(
                combined.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        return {
            "tokens": tokens,
            "sdet": round(float(sdet), 6),
            "sont": round(float(ont["sont"]), 6),
            "sgen": round(float(sgen), 6),
            "scal": round(float(scal), 6),
            "violations": ont["violations"],
            "detector_token_contributions": det_contrib,
            "ontology_token_weights": ont["token_weights"],
            "top_implicated_tokens": ranked,
        }


# ===========================================================================
# Phase 3 -- Explicit, configurable, real-ontology-aware calibrated scoring.
# ===========================================================================
#
# The legacy `combine_scores` / `Day25OntologyAwareScorer` above are kept for
# backward compatibility (ICD-prefix `compute_s_ont` fallback). The API below is
# the canonical Phase 3 scoring path: explicit weights (no hidden coefficients),
# explicit ontology mode (real / legacy / disabled); real mode uses the canonical
# OntologyEngine (Phase 2) and never silently falls back to the legacy rules.


@dataclass(frozen=True)
class ScoreWeights:
    """Explicit calibrated-score weights. `w_gen` defaults to 0 (diagnostic-only)."""

    w_det: float = 0.70
    w_ont: float = 0.30
    w_gen: float = 0.0


def normalize_sont(sont: float) -> float:
    """Squash an unbounded ontology severity sum into [0, 1)."""
    return 1.0 - math.exp(-max(float(sont), 0.0))


def compute_calibrated_score(
    s_det: float,
    s_ont: float,
    s_gen: float | None = None,
    weights: ScoreWeights | None = None,
    normalize_ont: bool = True,
    include_ont: bool = True,
) -> float:
    """Explicit normalized weighted score over the ACTIVE components.

    ``S_cal = (w_det*S_det + w_ont*S_ont' + w_gen*S_gen') / (sum of active weights)``

    Only active terms enter both numerator and denominator, so:
      * ontology disabled (``include_ont=False``) -> ``S_cal == S_det``;
      * ``s_gen`` is active only when supplied AND ``weights.w_gen > 0``
        (diagnostic-only until Phase 5).
    This avoids a zero/absent component spuriously diluting the detector score.
    """
    weights = weights or ScoreWeights()

    use_ont = include_ont
    s_ont_term = (
        (normalize_sont(s_ont) if normalize_ont else max(float(s_ont), 0.0))
        if use_ont
        else 0.0
    )
    w_ont = weights.w_ont if use_ont else 0.0

    use_gen = s_gen is not None and weights.w_gen > 0.0
    s_gen_term = max(0.0, min(1.0, float(s_gen))) if use_gen else 0.0
    w_gen = weights.w_gen if use_gen else 0.0

    raw = weights.w_det * float(s_det) + w_ont * s_ont_term + w_gen * s_gen_term
    denom = max(weights.w_det + w_ont + w_gen, 1e-8)
    return float(raw / denom)


_DIAG_PREFIXES = ("DX_", "DIAG_", "ICD")
_MED_PREFIXES = ("MED_", "DRUG_", "RAW_DRUG", "RX")


def map_tokens_to_ontology_codes(tokens: list[str], index: Any) -> dict[str, list[str]]:
    """Map raw EHR tokens to canonical ontology codes using the Phase 2 index.

    Diagnoses (DX_*) -> ``SNOMED:<id>`` via ICD->SNOMED crosswalk.
    Medications (MED_*) -> ``RXNORM:<cui>`` via drug-name map.
    """
    from src.preprocessing.code_normalization import (
        normalize_diagnosis_token,
        normalize_drug_token,
    )

    mapped_codes: list[str] = []
    unmapped: list[str] = []
    for tok in tokens:
        up = tok.upper()
        if up.startswith(("SNOMED:", "RXNORM:")):
            # already-canonical ontology code (e.g. a counterfactual replace edit
            # that inserts an ontology-neighbor concept directly); pass through.
            mapped_codes.append(tok)
        elif any(up.startswith(p) for p in _DIAG_PREFIXES):
            hit = None
            for cand in normalize_diagnosis_token(tok):
                snomed = index.map_icd_to_snomed(cand)
                if snomed:
                    hit = snomed
                    break
            if hit:
                mapped_codes.extend(hit)
            else:
                unmapped.append(tok)
        elif any(up.startswith(p) for p in _MED_PREFIXES):
            cui = None
            for cand in normalize_drug_token(tok):
                cui = index.map_drug_to_rxcui(cand)
                if cui:
                    break
            if cui:
                mapped_codes.append(f"RXNORM:{cui}")
            else:
                unmapped.append(tok)
        else:
            unmapped.append(tok)

    seen: set[str] = set()
    ordered = [c for c in mapped_codes if not (c in seen or seen.add(c))]
    return {"mapped_codes": ordered, "unmapped_tokens": unmapped}


class OntologyAwareScorer:
    """Phase 3 calibrated scorer with an explicit ontology mode.

    ontology_mode: "real" (canonical engine over mapped SNOMED/RxNorm codes),
    "legacy" (ICD-prefix compute_s_ont fallback), or "disabled" (S_ont forced 0).
    Real mode never silently falls back: with require_assets=True a missing engine
    raises; with require_assets=False it returns status="ontology_unavailable".
    """

    VALID_MODES = ("real", "legacy", "disabled")

    def __init__(
        self,
        ontology_mode: str = "real",
        engine: Any | None = None,
        weights: ScoreWeights | None = None,
        require_assets: bool = True,
    ) -> None:
        if ontology_mode not in self.VALID_MODES:
            raise ValueError(
                f"ontology_mode must be one of {self.VALID_MODES}, got {ontology_mode!r}"
            )
        self.ontology_mode = ontology_mode
        self.engine = engine
        self.weights = weights or ScoreWeights()
        self.require_assets = require_assets
        if ontology_mode == "real" and engine is None and require_assets:
            raise ValueError(
                "ontology_mode='real' requires a loaded OntologyEngine (use "
                "from_processed_dir) or set require_assets=False."
            )

    @classmethod
    def from_processed_dir(
        cls,
        processed_dir: str | Path,
        ontology_mode: str = "real",
        weights: ScoreWeights | None = None,
        require_assets: bool = True,
    ) -> "OntologyAwareScorer":
        engine = None
        processed_dir = Path(processed_dir)
        hierarchy = processed_dir / "snomed_hierarchy.json"
        if hierarchy.exists():
            from src.ontology.loader import load_ontology_engine, load_ontology_index

            engine = load_ontology_engine(processed_dir)
            engine.index = load_ontology_index(processed_dir)
        elif ontology_mode == "real" and require_assets:
            raise FileNotFoundError(
                f"Real ontology assets not found under {processed_dir} "
                f"(missing {hierarchy.name}). Run Phase 2/2b parsers first."
            )
        return cls(
            ontology_mode=ontology_mode,
            engine=engine,
            weights=weights,
            require_assets=require_assets,
        )

    def _s_ont_real(self, row: Mapping[str, Any], tokens: list[str]) -> dict[str, Any]:
        from src.ontology.records import ClinicalRecord

        mapping = map_tokens_to_ontology_codes(tokens, self.engine.index)
        sex = (
            (row.get("gender") or row.get("sex")) if isinstance(row, Mapping) else None
        )
        age = row.get("age_group") if isinstance(row, Mapping) else None
        record = ClinicalRecord(
            record_id=str(row.get("record_id"))
            if isinstance(row, Mapping) and row.get("record_id")
            else None,
            codes=tuple(mapping["mapped_codes"]),
            sex=str(sex) if sex is not None else None,
            age_group=str(age) if age is not None else None,
            source_tokens=tuple(str(t) for t in tokens),
        )
        score, violations = self.engine.score_violations(record)
        return {
            "s_ont": float(score),
            "violations": [v.to_dict() for v in violations],
            "status": "ok",
            "mapping": {
                "n_mapped": len(mapping["mapped_codes"]),
                "n_unmapped": len(mapping["unmapped_tokens"]),
            },
        }

    def _s_ont_legacy(self, row_or_tokens: Any) -> dict[str, Any]:
        ont = compute_s_ont(row_or_tokens)
        return {
            "s_ont": float(ont["sont"]),
            "violations": ont["violations"],
            "status": "ok_legacy",
            "mapping": {},
        }

    def score(
        self,
        record: Mapping[str, Any] | list[str],
        s_det: float,
        s_gen: float | None = None,
    ) -> dict[str, Any]:
        row = record.to_dict() if hasattr(record, "to_dict") else record
        tokens = (
            extract_tokens_from_row(row)
            if isinstance(row, Mapping)
            else normalize_tokens(row)
        )

        if self.ontology_mode == "disabled":
            ont = {"s_ont": 0.0, "violations": [], "status": "disabled", "mapping": {}}
        elif self.ontology_mode == "legacy":
            ont = self._s_ont_legacy(row if isinstance(row, Mapping) else tokens)
        else:  # real
            if self.engine is None:
                if self.require_assets:
                    raise RuntimeError(
                        "Real ontology scoring requested but no engine is loaded."
                    )
                ont = {
                    "s_ont": 0.0,
                    "violations": [],
                    "status": "ontology_unavailable",
                    "mapping": {},
                }
            else:
                ont = self._s_ont_real(row if isinstance(row, Mapping) else {}, tokens)

        include_ont = self.ontology_mode != "disabled" and ont["status"] not in {
            "ontology_unavailable",
            "disabled",
        }
        s_cal = compute_calibrated_score(
            s_det=s_det,
            s_ont=ont["s_ont"],
            s_gen=s_gen,
            weights=self.weights,
            include_ont=include_ont,
        )
        return {
            "tokens": tokens,
            "s_det": round(float(s_det), 6),
            "s_ont": round(float(ont["s_ont"]), 6),
            "s_gen": (None if s_gen is None else round(float(s_gen), 6)),
            "s_cal": round(float(s_cal), 6),
            "violations": ont["violations"],
            "ontology_mode": self.ontology_mode,
            "ontology_status": ont["status"],
            "mapping": ont.get("mapping", {}),
            "weights": {
                "w_det": self.weights.w_det,
                "w_ont": self.weights.w_ont,
                "w_gen": self.weights.w_gen,
            },
        }


__all__ = [
    "combine_scores",
    "Day25OntologyAwareScorer",
    "ScoreWeights",
    "normalize_sont",
    "compute_calibrated_score",
    "map_tokens_to_ontology_codes",
    "OntologyAwareScorer",
]
