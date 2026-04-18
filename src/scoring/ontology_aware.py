from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import torch

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

        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.max_len = int(max_len or self.config.get("max_len", 256))
        self.truncate_strategy = str(truncate_strategy or self.config.get("truncate_strategy", "tail"))

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

    @torch.no_grad()
    def detector_score(self, tokens: list[str]) -> float:
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
            return {k: round(v, 6) for k, v in sorted(contrib.items(), key=lambda x: x[1], reverse=True)}

        for idx, token in enumerate(tokens):
            reduced = tokens[:idx] + tokens[idx + 1 :]
            new_score = self.detector_score(reduced)
            delta = max(0.0, base - new_score)
            contrib[token] += float(delta)

        return {k: round(v, 6) for k, v in sorted(contrib.items(), key=lambda x: x[1], reverse=True)}

    def score_record(self, record: Mapping[str, Any] | list[str]) -> dict[str, Any]:
        row = record.to_dict() if hasattr(record, "to_dict") else record
        tokens = extract_tokens_from_row(row) if isinstance(row, Mapping) else normalize_tokens(row)

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
            for token, score in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]
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
