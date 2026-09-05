from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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

PRODUCT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "item_features.parquet"
)


def build_item_features() -> pd.DataFrame:

    print("Loading training interactions...")

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "product_id",
            "reordered",
        ],
    )

    print("Aggregating product behavior...")

    behavior = (
        train
        .groupby(
            "product_id",
            as_index=False,
        )
        .agg(
            purchase_count=(
                "product_id",
                "size",
            ),
            reorder_rate=(
                "reordered",
                "mean",
            ),
        )
    )

    behavior[
        "log_purchase_count"
    ] = np.log1p(
        behavior["purchase_count"]
    )

    print("Loading product metadata...")

    products = pd.read_parquet(
        PRODUCT_PATH
    )

    features = (
        products
        .merge(
            behavior,
            on="product_id",
            how="left",
            validate="one_to_one",
        )
    )

    features[
        "purchase_count"
    ] = (
        features[
            "purchase_count"
        ]
        .fillna(0)
        .astype("int64")
    )

    features[
        "reorder_rate"
    ] = (
        features[
            "reorder_rate"
        ]
        .fillna(0.0)
        .astype("float32")
    )

    features[
        "log_purchase_count"
    ] = (
        features[
            "log_purchase_count"
        ]
        .fillna(0.0)
        .astype("float32")
    )

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    print(
        "\nItem feature table created."
    )

    print(
        f"Products: {len(features):,}"
    )

    print(
        f"Aisles: "
        f"{features['aisle_id'].nunique():,}"
    )

    print(
        f"Departments: "
        f"{features['department_id'].nunique():,}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    return features


if __name__ == "__main__":
    build_item_features()