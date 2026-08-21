from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable


class TokenRarityBaseline:
    """B0 token-rarity baseline fitted only on clean training data."""

    def __init__(
        self,
        alpha: float = 1.0,
        rare_quantile: float = 0.05,
    ) -> None:
        self.alpha = float(alpha)
        self.rare_quantile = float(rare_quantile)

        self.counts: Counter[str] = Counter()
        self.total_tokens = 0
        self.vocab_size = 0
        self.rare_count_threshold = 0.0
        self._fitted = False

    def fit(
        self,
        sequences: Iterable[list[str]],
    ) -> "TokenRarityBaseline":
        self.counts.clear()

        for sequence in sequences:
            self.counts.update(sequence)

        self.total_tokens = int(sum(self.counts.values()))
        self.vocab_size = int(len(self.counts))

        if self.total_tokens <= 0:
            raise ValueError("Cannot fit B0 on empty training data.")

        sorted_counts = sorted(self.counts.values())

        index = int(
            self.rare_quantile * max(len(sorted_counts) - 1, 0)
        )

        self.rare_count_threshold = float(sorted_counts[index])
        self._fitted = True

        return self

    def token_probability(self, token: str) -> float:
        self._require_fitted()

        denominator = (
            self.total_tokens
            + self.alpha * (self.vocab_size + 1)
        )

        return (
            self.counts.get(token, 0) + self.alpha
        ) / denominator

    def score(self, sequence: list[str]) -> dict[str, float]:
        self._require_fitted()

        if not sequence:
            return {
                "max_token_surprisal": 0.0,
                "mean_negative_log_frequency": 0.0,
                "rare_code_fraction": 0.0,
            }

        surprisals = [
            -math.log(
                max(self.token_probability(token), 1e-12)
            )
            for token in sequence
        ]

        rare_count = sum(
            self.counts.get(token, 0)
            <= self.rare_count_threshold
            for token in sequence
        )

        return {
            "max_token_surprisal": float(max(surprisals)),
            "mean_negative_log_frequency": float(
                sum(surprisals) / len(surprisals)
            ),
            "rare_code_fraction": float(
                rare_count / len(sequence)
            ),
        }

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")
