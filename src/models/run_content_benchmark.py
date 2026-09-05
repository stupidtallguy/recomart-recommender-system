from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.evaluation.evaluate import (
    evaluate_model,
)

from src.models.content_based import (
    ContentBasedRecommender,
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

ITEM_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "item_features.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "content_v1.json"
)


def main():

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading training interactions..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    print(
        f"Training interactions: "
        f"{len(train):,}"
    )

    print(
        "Loading item features..."
    )

    item_features = pd.read_parquet(
        ITEM_FEATURE_PATH
    )

    model = ContentBasedRecommender(
        top_aisles=5,
        top_departments=3,
        products_per_aisle=50,
        products_per_department=50,
        precompute_k=20,
        department_weight=0.5,
    )

    print(
        "Training content recommender..."
    )

    start = perf_counter()

    model.fit(
        interactions=train,
        item_features=item_features,
    )

    training_seconds = (
        perf_counter()
        -
        start
    )

    print(
        "Content model fit/precompute "
        f"seconds: {training_seconds:.2f}"
    )

    del train
    del item_features

    gc.collect()

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
        "Running FULL validation benchmark..."
    )

    results = evaluate_model(
        model=model,
        evaluation_data=validation,
        k=10,
    )

    results.update(
        {
            "model":
                "content_v1",

            "top_aisles":
                model.top_aisles,

            "top_departments":
                model.top_departments,

            "products_per_aisle":
                model.products_per_aisle,

            "products_per_department":
                model.products_per_department,

            "department_weight":
                model.department_weight,

            "training_precompute_seconds":
                training_seconds,
        }
    )

    print(
        "\nContent-Based Results"
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


if __name__ == "__main__":
    main()