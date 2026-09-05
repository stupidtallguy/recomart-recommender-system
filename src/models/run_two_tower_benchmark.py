from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import (
    evaluate_model,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidates"
    / "two_tower_v1_top50.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_interactions.parquet"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "two_tower_v1.json"
)


class PrecomputedTwoTower:

    def __init__(
        self,
        recommendations,
    ):

        self.recommendations = (
            recommendations
        )


    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:

        return (
            self.recommendations
            .get(
                int(user_id),
                [],
            )
            [:k]
        )


def main():

    print(
        "Loading Two-Tower candidates..."
    )

    candidate_df = pd.read_parquet(
        CANDIDATE_PATH,
        columns=[
            "user_id",
            "recommendations",
        ],
    )

    recommendations = {

        int(row.user_id):
            [
                int(x)
                for x
                in row.recommendations
            ]

        for row
        in candidate_df.itertuples(
            index=False
        )
    }

    model = PrecomputedTwoTower(
        recommendations
    )

    print(
        "Loading validation data..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    print(
        "Evaluating Two-Tower..."
    )

    metrics = evaluate_model(
        model=model,
        evaluation_data=validation,
        k=10,
    )

    result = {

        "model":
            "two_tower_v1",

        "users_evaluated":
            metrics[
                "users_evaluated"
            ],

        "recall_at_10":
            metrics[
                "recall_at_k"
            ],

        "precision_at_10":
            metrics[
                "precision_at_k"
            ],

        "ndcg_at_10":
            metrics[
                "ndcg_at_k"
            ],
    }

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TWO-TOWER V1 RESULTS"
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()