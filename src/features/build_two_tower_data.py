from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_PATH = (
    PROCESSED_DIR
    / "train_interactions.parquet"
)

ITEM_FEATURE_PATH = (
    PROCESSED_DIR
    / "item_features.parquet"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "two_tower"
)

PAIR_PATH = (
    OUTPUT_DIR
    / "positive_pairs.parquet"
)

FEATURE_PATH = (
    OUTPUT_DIR
    / "feature_arrays.npz"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "metadata.json"
)


def standardize(
    values: np.ndarray,
    valid_start: int = 1,
) -> np.ndarray:

    values = values.astype(
        np.float32,
        copy=True,
    )

    valid = values[
        valid_start:
    ]

    mean = float(
        valid.mean()
    )

    std = float(
        valid.std()
    )

    if std < 1e-8:
        std = 1.0

    values[
        valid_start:
    ] = (
        valid - mean
    ) / std

    values[:valid_start] = 0.0

    return values


def main():

    OUTPUT_DIR.mkdir(
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
            "reordered",
        ],
    )

    train["user_id"] = (
        train["user_id"]
        .astype(np.int32)
    )

    train["product_id"] = (
        train["product_id"]
        .astype(np.int32)
    )

    train["reordered"] = (
        train["reordered"]
        .astype(np.float32)
    )

    print(
        f"Interactions: {len(train):,}"
    )

    # --------------------------------------------------
    # Unique positive user-product pairs
    # --------------------------------------------------

    print(
        "Building unique positive pairs..."
    )

    pairs = (
        train
        .groupby(
            [
                "user_id",
                "product_id",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            purchase_count=(
                "product_id",
                "size",
            ),
        )
    )

    pairs[
        "purchase_count"
    ] = (
        pairs[
            "purchase_count"
        ]
        .astype(np.int16)
    )

    # Repeat purchases receive somewhat more weight,
    # without allowing very frequent products to
    # dominate the objective linearly.
    pairs[
        "sample_weight"
    ] = (
        1.0
        +
        np.log1p(
            pairs[
                "purchase_count"
            ]
            .astype(np.float32)
        )
    ).astype(
        np.float32
    )

    pairs = (
        pairs
        .sort_values(
            [
                "user_id",
                "product_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pairs.to_parquet(
        PAIR_PATH,
        index=False,
        compression="snappy",
    )

    print(
        f"Positive pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------
    # User numerical features
    # --------------------------------------------------

    print(
        "Building user features..."
    )

    user_stats = (
        train
        .groupby(
            "user_id",
            as_index=False,
        )
        .agg(
            total_interactions=(
                "product_id",
                "size",
            ),

            unique_products=(
                "product_id",
                "nunique",
            ),

            reorder_rate=(
                "reordered",
                "mean",
            ),
        )
    )

    max_user_id = int(
        max(
            train[
                "user_id"
            ].max(),
            user_stats[
                "user_id"
            ].max(),
        )
    )

    max_product_id = int(
        train[
            "product_id"
        ].max()
    )

    user_log_interactions = (
        np.zeros(
            max_user_id + 1,
            dtype=np.float32,
        )
    )

    user_log_unique = (
        np.zeros(
            max_user_id + 1,
            dtype=np.float32,
        )
    )

    user_reorder_rate = (
        np.zeros(
            max_user_id + 1,
            dtype=np.float32,
        )
    )

    user_ids = (
        user_stats[
            "user_id"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    user_log_interactions[
        user_ids
    ] = np.log1p(
        user_stats[
            "total_interactions"
        ].to_numpy(
            dtype=np.float32
        )
    )

    user_log_unique[
        user_ids
    ] = np.log1p(
        user_stats[
            "unique_products"
        ].to_numpy(
            dtype=np.float32
        )
    )

    user_reorder_rate[
        user_ids
    ] = (
        user_stats[
            "reorder_rate"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    user_features = np.column_stack(
        [
            standardize(
                user_log_interactions
            ),

            standardize(
                user_log_unique
            ),

            standardize(
                user_reorder_rate
            ),
        ]
    ).astype(
        np.float32
    )

    # --------------------------------------------------
    # Item features
    # --------------------------------------------------

    print(
        "Loading item feature store..."
    )

    items = pd.read_parquet(
        ITEM_FEATURE_PATH,
        columns=[
            "product_id",
            "aisle_id",
            "department_id",
            "log_purchase_count",
            "reorder_rate",
        ],
    )

    max_product_id = max(
        max_product_id,
        int(
            items[
                "product_id"
            ].max()
        ),
    )

    max_aisle_id = int(
        items[
            "aisle_id"
        ].max()
    )

    max_department_id = int(
        items[
            "department_id"
        ].max()
    )

    item_aisle = np.zeros(
        max_product_id + 1,
        dtype=np.int64,
    )

    item_department = np.zeros(
        max_product_id + 1,
        dtype=np.int64,
    )

    item_log_popularity = np.zeros(
        max_product_id + 1,
        dtype=np.float32,
    )

    item_reorder_rate = np.zeros(
        max_product_id + 1,
        dtype=np.float32,
    )

    item_ids = (
        items[
            "product_id"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    item_aisle[
        item_ids
    ] = (
        items[
            "aisle_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    item_department[
        item_ids
    ] = (
        items[
            "department_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    item_log_popularity[
        item_ids
    ] = (
        items[
            "log_purchase_count"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    item_reorder_rate[
        item_ids
    ] = (
        items[
            "reorder_rate"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    item_features = np.column_stack(
        [
            standardize(
                item_log_popularity
            ),

            standardize(
                item_reorder_rate
            ),
        ]
    ).astype(
        np.float32
    )

    # --------------------------------------------------
    # Observed products
    # --------------------------------------------------

    observed_product_ids = np.sort(
        pairs[
            "product_id"
        ]
        .unique()
        .astype(
            np.int32
        )
    )

    # --------------------------------------------------
    # Save compact arrays
    # --------------------------------------------------

    np.savez_compressed(
        FEATURE_PATH,

        user_features=
            user_features,

        item_features=
            item_features,

        item_aisle=
            item_aisle,

        item_department=
            item_department,

        observed_product_ids=
            observed_product_ids,
    )

    metadata = {

        "interactions":
            int(
                len(train)
            ),

        "positive_pairs":
            int(
                len(pairs)
            ),

        "users":
            int(
                user_stats[
                    "user_id"
                ].nunique()
            ),

        "max_user_id":
            max_user_id,

        "products_observed":
            int(
                len(
                    observed_product_ids
                )
            ),

        "max_product_id":
            max_product_id,

        "max_aisle_id":
            max_aisle_id,

        "max_department_id":
            max_department_id,

        "user_numeric_features": [
            "log_total_interactions",
            "log_unique_products",
            "reorder_rate",
        ],

        "item_numeric_features": [
            "log_purchase_count",
            "reorder_rate",
        ],
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TWO-TOWER DATA COMPLETE"
    )

    print(
        json.dumps(
            metadata,
            indent=4,
        )
    )

    print(
        "\nSaved:"
    )

    print(
        PAIR_PATH
    )

    print(
        FEATURE_PATH
    )

    print(
        METADATA_PATH
    )


if __name__ == "__main__":
    main()