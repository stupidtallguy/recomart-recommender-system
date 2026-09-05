from pathlib import Path
import gc
import json

import pandas as pd

from src.models.item_similarity import (
    ItemSimilarityRecommender,
)

from src.evaluation.evaluate import (
    evaluate_model,
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
    / "item_similarity.json"
)


def main():

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading required training columns..."
    )

    # Item similarity does not require:
    # order_number,
    # reordered,
    # timestamps,
    # eval_set, etc.
    #
    # Avoid loading unnecessary data.
    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    print(
        f"Training rows: {len(train):,}"
    )

    model = ItemSimilarityRecommender(
        n_neighbors=50,
        candidate_pool_size=100,
        precompute_k=20,
    )

    print(
        "Training item-similarity model..."
    )

    model.fit(
        train
    )

    del train
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

    results[
        "model"
    ] = "item_similarity"

    results[
        "n_neighbors"
    ] = model.n_neighbors

    results[
        "candidate_pool_size"
    ] = (
        model.candidate_pool_size
    )

    print(
        "\nItem Similarity Results"
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