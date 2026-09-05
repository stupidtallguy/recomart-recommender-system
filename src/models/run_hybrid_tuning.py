from __future__ import annotations

import gc
import json
from functools import reduce
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.evaluation.evaluate import (
    evaluate_model,
)

from src.models.hybrid_rrf import (
    HybridRRFRecommender,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CANDIDATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidates"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_interactions.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "hybrid_rrf_tuning.json"
)


SOURCE_FILES = {
    "repeat":
        "repeat_purchase_top50.parquet",

    "als":
        "als_v2_top50.parquet",

    "content":
        "content_v1_top50.parquet",
}


EXPERIMENTS = [

    # Important sanity check:
    # this should reproduce the Repeat Purchase
    # baseline almost exactly.
    {
        "name":
            "repeat_control",

        "weights": {
            "repeat": 1.0,
            "als": 0.0,
            "content": 0.0,
        },
    },

    {
        "name":
            "equal_weights",

        "weights": {
            "repeat": 1.0,
            "als": 1.0,
            "content": 1.0,
        },
    },

    {
        "name":
            "repeat3_als1_content1",

        "weights": {
            "repeat": 3.0,
            "als": 1.0,
            "content": 1.0,
        },
    },

    {
        "name":
            "repeat5_als2_content1",

        "weights": {
            "repeat": 5.0,
            "als": 2.0,
            "content": 1.0,
        },
    },

    {
        "name":
            "repeat5_als1_content2",

        "weights": {
            "repeat": 5.0,
            "als": 1.0,
            "content": 2.0,
        },
    },

    {
        "name":
            "repeat5_als2",

        "weights": {
            "repeat": 5.0,
            "als": 2.0,
            "content": 0.0,
        },
    },

    {
        "name":
            "repeat8_als2_content1",

        "weights": {
            "repeat": 8.0,
            "als": 2.0,
            "content": 1.0,
        },
    },
]


def load_candidates() -> pd.DataFrame:

    frames = []

    for (
        source,
        filename,
    ) in SOURCE_FILES.items():

        path = (
            CANDIDATE_DIR
            / filename
        )

        print(
            f"Loading {source}: "
            f"{path.name}"
        )

        frame = pd.read_parquet(
            path,
            columns=[
                "user_id",
                "recommendations",
            ],
        )

        frame = frame.rename(
            columns={
                "recommendations":
                    source
            }
        )

        frames.append(
            frame
        )

    candidates = reduce(
        lambda left, right:
            left.merge(
                right,
                on="user_id",
                how="outer",
                validate="one_to_one",
            ),
        frames,
    )

    candidates = (
        candidates
        .sort_values("user_id")
        .reset_index(drop=True)
    )

    print(
        "Users in candidate store: "
        f"{len(candidates):,}"
    )

    return candidates


def main():

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading candidate store..."
    )

    candidates = (
        load_candidates()
    )

    print(
        "\nLoading validation data..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    all_results = []

    for index, config in enumerate(
        EXPERIMENTS,
        start=1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Hybrid experiment "
            f"{index}/"
            f"{len(EXPERIMENTS)}"
        )

        print(
            config["name"]
        )

        print(
            "Weights:"
        )

        print(
            json.dumps(
                config["weights"],
                indent=4,
            )
        )

        model = (
            HybridRRFRecommender(
                source_weights=
                    config[
                        "weights"
                    ],
                rrf_k=60,
                precompute_k=20,
            )
        )

        start = perf_counter()

        model.fit(
            candidates
        )

        fusion_seconds = (
            perf_counter()
            - start
        )

        metrics = evaluate_model(
            model=model,
            evaluation_data=validation,
            k=10,
        )

        result = {
            "name":
                config["name"],

            "weights":
                config[
                    "weights"
                ],

            "rrf_k":
                60,

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

            "fusion_seconds":
                fusion_seconds,
        }

        all_results.append(
            result
        )

        print(
            json.dumps(
                result,
                indent=4,
            )
        )

        RESULTS_PATH.write_text(
            json.dumps(
                {
                    "users_evaluated":
                        206209,

                    "fusion":
                        "weighted_rrf",

                    "experiments":
                        all_results,
                },
                indent=4,
            ),
            encoding="utf-8",
        )

        del model

        gc.collect()

    best_ndcg = max(
        all_results,
        key=lambda x:
            x["ndcg_at_10"],
    )

    best_recall = max(
        all_results,
        key=lambda x:
            x["recall_at_10"],
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST BY NDCG@10"
    )

    print(
        json.dumps(
            best_ndcg,
            indent=4,
        )
    )

    print(
        "\nBEST BY RECALL@10"
    )

    print(
        json.dumps(
            best_recall,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()