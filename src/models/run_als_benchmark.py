from pathlib import Path
import json

import pandas as pd

from pyspark.sql import SparkSession

from src.evaluation.evaluate import (
    evaluate_model,
)

from src.models.als import (
    ALSRecommender,
)


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
    / "als_v1.json"
)


def create_spark_session():

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(
            "RecoMart-ALS"
        )

        # Reasonable starting point for local development.
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


def main():

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = create_spark_session()

    try:

        model = ALSRecommender(
            rank=32,
            max_iter=10,
            reg_param=0.05,
            alpha=20.0,
            precompute_k=20,
            seed=42,
        )

        model.fit(
            spark=spark,
            train_path=TRAIN_PATH,
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

        print(
            "\nRunning FULL validation benchmark..."
        )

        results = evaluate_model(
            model=model,
            evaluation_data=validation,
            k=10,
        )

        results.update(
            {
                "model":
                    "spark_als_implicit_v1",

                "interaction_signal":
                    "purchase_count",

                "rank":
                    model.rank,

                "max_iter":
                    model.max_iter,

                "reg_param":
                    model.reg_param,

                "alpha":
                    model.alpha,

                "unique_user_item_pairs":
                    model.unique_user_item_pairs,

                "training_seconds":
                    model.training_seconds,

                "recommendation_generation_seconds":
                    model.recommendation_seconds,
            }
        )

        print(
            "\nALS Results"
        )

        print(
            json.dumps(
                results,
                indent=4,
            )
        )

        RESULTS_PATH.write_text(
            json.dumps(
                results,
                indent=4,
            ),
            encoding="utf-8",
        )

        print(
            "\nSaved results to:"
        )

        print(
            RESULTS_PATH
        )

    finally:

        print(
            "\nStopping Spark..."
        )

        spark.stop()


if __name__ == "__main__":
    main()