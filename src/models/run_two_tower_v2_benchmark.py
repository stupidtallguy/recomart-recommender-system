from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import (
    evaluate_model,
)


# ============================================================
# Paths
# ============================================================

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
    / "two_tower_v2_top50.parquet"
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
    / "two_tower_v2.json"
)


# ============================================================
# Precomputed recommender
# ============================================================

class PrecomputedTwoTowerV2:

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
                int(
                    user_id
                ),
                [],
            )
            [:k]
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading Two-Tower v2 candidates..."
    )

    candidate_df = pd.read_parquet(
        CANDIDATE_PATH,
        columns=[
            "user_id",
            "recommendations",
        ],
    )

    recommendations = {

        int(
            row.user_id
        ):
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

    print(
        f"Users with recommendations: "
        f"{len(recommendations):,}"
    )

    model = (
        PrecomputedTwoTowerV2(
            recommendations
        )
    )

    print(
        "Loading official validation data..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    # --------------------------------------------------------
    # Top-10 ranking benchmark
    # --------------------------------------------------------

    print(
        "\nEvaluating Two-Tower v2 Top-10..."
    )

    metrics_10 = evaluate_model(
        model=model,
        evaluation_data=
            validation,
        k=10,
    )

    # --------------------------------------------------------
    # Top-50 candidate recall
    # --------------------------------------------------------

    print(
        "\nEvaluating Two-Tower v2 "
        "candidate Recall@50..."
    )

    metrics_50 = evaluate_model(
        model=model,
        evaluation_data=
            validation,
        k=50,
    )

    result = {

        "model":
            "two_tower_v2_history_aware",

        "users_evaluated":
            int(
                metrics_10[
                    "users_evaluated"
                ]
            ),

        "top10": {

            "recall_at_10":
                float(
                    metrics_10[
                        "recall_at_k"
                    ]
                ),

            "precision_at_10":
                float(
                    metrics_10[
                        "precision_at_k"
                    ]
                ),

            "ndcg_at_10":
                float(
                    metrics_10[
                        "ndcg_at_k"
                    ]
                ),
        },

        "candidate_retrieval": {

            "recall_at_50":
                float(
                    metrics_50[
                        "recall_at_k"
                    ]
                ),

            "precision_at_50":
                float(
                    metrics_50[
                        "precision_at_k"
                    ]
                ),

            "ndcg_at_50":
                float(
                    metrics_50[
                        "ndcg_at_k"
                    ]
                ),
        },
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "TWO-TOWER V2 OFFICIAL "
        "VALIDATION RESULTS"
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()