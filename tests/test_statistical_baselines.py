from __future__ import annotations

import math

import pandas as pd

from src.baselines.common import (
    demographic_tokens,
    load_dataframe,
)
from src.baselines.cooccurrence_pmi import (
    StatisticalRelationalBaseline,
)
from src.baselines.rarity import (
    TokenRarityBaseline,
)


def test_b0_unseen_token_more_surprising() -> None:
    model = TokenRarityBaseline().fit(
        [
            ["A", "B"],
            ["A", "B"],
            ["A", "C"],
        ]
    )

    known = model.score(["A"])
    unseen = model.score(["NEVER_SEEN"])

    assert (
        unseen["max_token_surprisal"]
        > known["max_token_surprisal"]
    )


def test_b0_scores_are_finite() -> None:
    model = TokenRarityBaseline().fit(
        [["A", "B"], ["A", "C"]]
    )

    scores = model.score(["A", "B"])

    assert all(
        math.isfinite(value)
        for value in scores.values()
    )


def test_b1_common_pair_less_anomalous() -> None:
    model = StatisticalRelationalBaseline(
        min_support=2
    ).fit(
        [
            (["A", "B"], []),
            (["A", "B"], []),
            (["A", "B"], []),
            (["A", "B"], []),
            (["A", "C"], []),
        ]
    )

    common = model.score(["A", "B"])
    rare = model.score(["B", "C"])

    assert (
        common["worst_relation"]
        < rare["worst_relation"]
    )


def test_b1_demographics_are_relations() -> None:
    model = StatisticalRelationalBaseline(
        min_support=1
    ).fit(
        [
            (["DX_A"], ["__DEMOG__SEX=F"]),
            (["DX_A"], ["__DEMOG__SEX=F"]),
            (["DX_B"], ["__DEMOG__SEX=M"]),
        ]
    )

    score = model.score(
        ["DX_A"],
        ["__DEMOG__SEX=F"],
    )

    assert score["candidate_pair_count"] == 1.0


def test_b1_length_robust_scores_exist() -> None:
    model = StatisticalRelationalBaseline(
        min_support=1
    ).fit(
        [
            (["A", "B", "C"], []),
            (["A", "B", "C"], []),
            (["A", "B", "D"], []),
        ]
    )

    score = model.score(["A", "B", "C"])

    assert "q90_relation" in score
    assert "topk_relation" in score
    assert score["candidate_pair_count"] == 3.0


def test_loader_uses_only_model_visible_fields(
    tmp_path,
) -> None:
    path = tmp_path / "canonical.pkl"

    pd.DataFrame(
        [
            {
                "model_visible_sequence": [
                    "DX_A",
                    "MED_B",
                ],
                "gender": "F",
                "age_group": "30-44",
                "label": 1,
                "anomaly_type": "example",
                "hidden_eval_metadata": {
                    "answer": "SECRET_SIGNAL"
                },
                "audit_metadata": {
                    "source": "synthetic"
                },
            }
        ]
    ).to_pickle(path)

    loaded = load_dataframe(path)
    row = loaded.iloc[0]

    assert (
        loaded.attrs["source_sequence_column"]
        == "model_visible_sequence"
    )

    assert row["_sequence_tokens"] == [
        "DX_A",
        "MED_B",
    ]

    assert demographic_tokens(row) == [
        "__DEMOG__SEX=F",
        "__DEMOG__AGE=30-44",
    ]

    assert "SECRET_SIGNAL" not in " ".join(
        row["_sequence_tokens"]
    )
