from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from itertools import combinations

import numpy as np


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class StatisticalRelationalBaseline:
    """
    B1 statistical-relational baseline.

    Uses clean-training empirical statistics only.
    No ontology information is used.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_support: int = 5,
        quantile: float = 0.90,
        bottom_k: int = 5,
    ) -> None:
        self.alpha = float(alpha)
        self.min_support = int(min_support)
        self.quantile = float(quantile)
        self.bottom_k = int(bottom_k)

        self.n_records = 0
        self.item_support: Counter[str] = Counter()
        self.pair_support: Counter[tuple[str, str]] = Counter()

        self._fitted = False

    @staticmethod
    def relation_pairs(
        codes: list[str],
        demographics: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        demographics = demographics or []

        unique_codes = sorted(set(codes))
        unique_demographics = sorted(set(demographics))

        pairs = [
            _pair_key(a, b)
            for a, b in combinations(unique_codes, 2)
        ]

        for code in unique_codes:
            for demographic in unique_demographics:
                pairs.append(
                    _pair_key(code, demographic)
                )

        return pairs

    def fit(
        self,
        records: Iterable[
            tuple[list[str], list[str]]
        ],
    ) -> "StatisticalRelationalBaseline":
        self.n_records = 0
        self.item_support.clear()
        self.pair_support.clear()

        for codes, demographics in records:
            unique_codes = sorted(set(codes))
            unique_demographics = sorted(set(demographics))

            if not unique_codes:
                continue

            self.n_records += 1

            self.item_support.update(
                set(unique_codes + unique_demographics)
            )

            self.pair_support.update(
                set(
                    self.relation_pairs(
                        unique_codes,
                        unique_demographics,
                    )
                )
            )

        if self.n_records <= 0:
            raise ValueError("Cannot fit B1 on empty training data.")

        self._fitted = True
        return self

    def marginal_probability(self, item: str) -> float:
        self._require_fitted()

        return (
            self.item_support.get(item, 0) + self.alpha
        ) / (
            self.n_records + 2.0 * self.alpha
        )

    def relation_statistics(
        self,
        a: str,
        b: str,
    ) -> dict[str, float]:
        self._require_fitted()

        pair = _pair_key(a, b)

        count_a = self.item_support.get(a, 0)
        count_b = self.item_support.get(b, 0)
        count_ab = self.pair_support.get(pair, 0)

        p_a = self.marginal_probability(a)
        p_b = self.marginal_probability(b)

        if count_ab >= self.min_support:
            p_ab = (
                count_ab + self.alpha
            ) / (
                self.n_records + 2.0 * self.alpha
            )

            p_a_given_b = (
                count_ab + self.alpha
            ) / (
                count_b + 2.0 * self.alpha
            )

            p_b_given_a = (
                count_ab + self.alpha
            ) / (
                count_a + 2.0 * self.alpha
            )

            conditional = min(
                p_a_given_b,
                p_b_given_a,
            )

            pmi = math.log(
                max(
                    p_ab / max(p_a * p_b, 1e-12),
                    1e-12,
                )
            )

            denominator = max(
                -math.log(max(p_ab, 1e-12)),
                1e-12,
            )

            npmi = pmi / denominator

            confidence = conditional
            lift = p_ab / max(p_a * p_b, 1e-12)

            used_backoff = False

        else:
            # Conservative low-support / unseen-pair backoff.
            conditional = math.sqrt(
                max(p_a * p_b, 1e-12)
            )

            npmi = 0.0
            confidence = min(p_a, p_b)
            lift = 1.0

            used_backoff = True

        return {
            "conditional": float(conditional),
            "npmi": float(npmi),
            "confidence": float(confidence),
            "lift": float(lift),
            "support": float(count_ab),
            "used_backoff": float(used_backoff),
        }

    def score(
        self,
        codes: list[str],
        demographics: list[str] | None = None,
    ) -> dict[str, float]:
        self._require_fitted()

        pairs = self.relation_pairs(
            codes,
            demographics or [],
        )

        if not pairs:
            return {
                "worst_relation": 0.0,
                "mean_relation": 0.0,
                "q90_relation": 0.0,
                "topk_relation": 0.0,
                "npmi_anomaly": 0.0,
                "confidence_anomaly": 0.0,
                "lift_anomaly": 0.0,
                "candidate_pair_count": 0.0,
                "supported_pair_fraction": 0.0,
            }

        relation_scores = []
        npmi_scores = []
        confidence_scores = []
        lift_scores = []

        supported = 0

        for a, b in pairs:
            stats = self.relation_statistics(a, b)

            if stats["support"] >= self.min_support:
                supported += 1

            relation_scores.append(
                -math.log(
                    max(stats["conditional"], 1e-12)
                )
            )

            npmi_scores.append(
                (1.0 - stats["npmi"]) / 2.0
            )

            confidence_scores.append(
                -math.log(
                    max(stats["confidence"], 1e-12)
                )
            )

            lift_scores.append(
                -math.log(
                    max(stats["lift"], 1e-12)
                )
            )

        arr = np.asarray(
            relation_scores,
            dtype=float,
        )

        k = max(
            1,
            min(self.bottom_k, len(arr)),
        )

        worst_k = np.sort(arr)[-k:]

        return {
            "worst_relation": float(np.max(arr)),
            "mean_relation": float(np.mean(arr)),
            "q90_relation": float(
                np.quantile(arr, self.quantile)
            ),
            "topk_relation": float(
                np.mean(worst_k)
            ),
            "npmi_anomaly": float(
                np.mean(npmi_scores)
            ),
            "confidence_anomaly": float(
                np.mean(confidence_scores)
            ),
            "lift_anomaly": float(
                np.mean(lift_scores)
            ),
            "candidate_pair_count": float(len(pairs)),
            "supported_pair_fraction": float(
                supported / len(pairs)
            ),
        }

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")
