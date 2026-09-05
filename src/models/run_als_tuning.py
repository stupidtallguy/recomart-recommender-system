from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from pyspark import StorageLevel
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.evaluation.evaluate import evaluate_model


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_interactions.parquet"
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
    / "als_tuning.json"
)


EXPERIMENTS = [
    {
        "name": "rank64_alpha10",
        "rank": 64,
        "reg_param": 0.05,
        "alpha": 10.0,
    },
    {
        "name": "rank64_alpha10_reg10",
        "rank": 64,
        "reg_param": 0.10,
        "alpha": 10.0,
    },
    {
        "name": "rank48_alpha10",
        "rank": 48,
        "reg_param": 0.05,
        "alpha": 10.0,
    },
    {
        "name": "rank32_alpha10_reg10",
        "rank": 32,
        "reg_param": 0.10,
        "alpha": 10.0,
    },
    {
        "name": "rank64_alpha5",
        "rank": 64,
        "reg_param": 0.05,
        "alpha": 5.0,
    },
]

class PrecomputedRecommender:

    def __init__(
        self,
        recommendations: dict[int, list[int]],
    ):
        self.recommendations = recommendations


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


def create_spark_session():

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(
            "RecoMart-ALS-Tuning"
        )
        .config(
            "spark.driver.memory",
            "6g",
        )
        .config(
            "spark.sql.shuffle.partitions",
            "64",
        )
        .config(
            "spark.default.parallelism",
            "64",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    return spark


def collect_recommendations(
    als_model,
    k: int = 10,
):

    rows = (
        als_model
        .recommendForAllUsers(k)
        .collect()
    )

    recommendations = {}

    for row in rows:

        recommendations[
            int(row["user_id"])
        ] = [
            int(rec["product_id"])
            for rec
            in row["recommendations"]
        ]

    return recommendations


def main():

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = create_spark_session()

    try:

        print(
            "Reading training interactions..."
        )

        interactions = (
            spark.read
            .parquet(
                str(TRAIN_PATH)
            )
            .select(
                F.col("user_id")
                .cast("int"),

                F.col("product_id")
                .cast("int"),
            )
        )

        print(
            "Building purchase-count "
            "interaction strengths ONCE..."
        )

        strengths = (
            interactions
            .groupBy(
                "user_id",
                "product_id",
            )
            .agg(
                F.count("*")
                .cast("float")
                .alias("strength")
            )
            .persist(
                StorageLevel.MEMORY_AND_DISK
            )
        )

        pair_count = strengths.count()

        print(
            "Unique user-product pairs: "
            f"{pair_count:,}"
        )

        print(
            "Loading validation data once..."
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
                f"Experiment "
                f"{index}/{len(EXPERIMENTS)}:"
                f" {config['name']}"
            )

            print(
                "rank="
                f"{config['rank']}, "
                "reg="
                f"{config['reg_param']}, "
                "alpha="
                f"{config['alpha']}"
            )

            als = ALS(
                userCol="user_id",
                itemCol="product_id",
                ratingCol="strength",

                implicitPrefs=True,

                rank=config["rank"],
                maxIter=10,
                regParam=config[
                    "reg_param"
                ],
                alpha=config[
                    "alpha"
                ],

                coldStartStrategy="drop",
                seed=42,
            )

            start = perf_counter()

            als_model = als.fit(
                strengths
            )

            training_seconds = (
                perf_counter()
                - start
            )

            print(
                "Training complete."
            )

            start = perf_counter()

            recommendations = (
                collect_recommendations(
                    als_model,
                    k=10,
                )
            )

            recommendation_seconds = (
                perf_counter()
                - start
            )

            recommender = (
                PrecomputedRecommender(
                    recommendations
                )
            )

            metrics = evaluate_model(
                model=recommender,
                evaluation_data=validation,
                k=10,
            )

            result = {
                "name":
                    config["name"],

                "rank":
                    config["rank"],

                "reg_param":
                    config[
                        "reg_param"
                    ],

                "alpha":
                    config["alpha"],

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

                "training_seconds":
                    training_seconds,

                "recommendation_seconds":
                    recommendation_seconds,
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

            # Save after every experiment so
            # completed results survive an
            # interrupted later run.
            RESULTS_PATH.write_text(
                json.dumps(
                    {
                        "interaction_signal":
                            "purchase_count",

                        "users_evaluated":
                            206209,

                        "unique_user_item_pairs":
                            pair_count,

                        "experiments":
                            all_results,
                    },
                    indent=4,
                ),
                encoding="utf-8",
            )

            del recommendations
            del recommender
            del als_model

            gc.collect()

        best = max(
            all_results,
            key=lambda x:
                x["ndcg_at_10"],
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "BEST CONFIGURATION"
        )

        print(
            json.dumps(
                best,
                indent=4,
            )
        )

        strengths.unpersist()

    finally:

        spark.stop()


if __name__ == "__main__":
    main()