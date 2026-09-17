from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

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
    / "two_tower_v2"
)

CONTEXT_PATH = (
    OUTPUT_DIR
    / "contexts.npz"
)

SERVING_CONTEXT_PATH = (
    OUTPUT_DIR
    / "serving_contexts.npz"
)

PAIR_PATH = (
    OUTPUT_DIR
    / "positive_pairs.parquet"
)

ITEM_ARRAY_PATH = (
    OUTPUT_DIR
    / "item_features.npz"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "metadata.json"
)


# ============================================================
# Configuration
# ============================================================

# A pseudo-target order must have at least this many
# earlier orders available as history.
MIN_HISTORY_ORDERS = 2

# Maximum number of historical target points retained
# per user.
#
# Example:
# orders 1-3 -> target 4
# orders 1-4 -> target 5
# orders 1-5 -> target 6
# orders 1-6 -> target 7
MAX_CONTEXTS_PER_USER = 4

# Only recent orders enter the user-history encoder.
HISTORY_ORDERS = 5

# Maximum number of product occurrences in one
# history sequence.
MAX_HISTORY_PRODUCTS = 80

# Recent orders receive larger aggregation weights.
RECENCY_DECAY = 0.35

PAIR_WRITE_CHUNK_SIZE = 500_000


# ============================================================
# Helpers
# ============================================================

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

    if len(valid) == 0:
        return values

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

    values[
        :valid_start
    ] = 0.0

    return values


def history_numeric_features(
    history_products: list[int],
    history_order_count: int,
) -> np.ndarray:

    """
    Compact history statistics in approximately [0, 1].

    Features:
        history length ratio
        unique product ratio
        history order ratio
        repeated-product fraction
    """

    length = len(
        history_products
    )

    if length == 0:

        return np.zeros(
            4,
            dtype=np.float32,
        )

    unique_count = len(
        set(
            history_products
        )
    )

    repeat_fraction = (
        1.0
        -
        (
            unique_count
            /
            length
        )
    )

    return np.asarray(
        [
            min(
                length
                /
                MAX_HISTORY_PRODUCTS,
                1.0,
            ),

            min(
                unique_count
                /
                MAX_HISTORY_PRODUCTS,
                1.0,
            ),

            min(
                history_order_count
                /
                HISTORY_ORDERS,
                1.0,
            ),

            repeat_fraction,
        ],
        dtype=np.float32,
    )


def build_history(
    order_products: list[np.ndarray],
    target_position: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    """
    Build context for predicting order[target_position].

    Only orders BEFORE target_position are visible.

    Most recent orders are processed first.
    Repeated product occurrences are intentionally retained.
    """

    start_position = max(
        0,
        target_position
        -
        HISTORY_ORDERS,
    )

    selected_positions = list(
        range(
            start_position,
            target_position,
        )
    )

    history_products: list[int] = []
    history_weights: list[float] = []

    # Most recent order first.
    for order_position in reversed(
        selected_positions
    ):

        # Previous order:
        # gap = 1
        #
        # Two orders ago:
        # gap = 2
        gap = (
            target_position
            -
            order_position
        )

        weight = math.exp(
            -RECENCY_DECAY
            *
            (
                gap - 1
            )
        )

        products = (
            order_products[
                order_position
            ]
        )

        for product_id in products:

            history_products.append(
                int(product_id)
            )

            history_weights.append(
                float(weight)
            )

            if (
                len(
                    history_products
                )
                >=
                MAX_HISTORY_PRODUCTS
            ):
                break

        if (
            len(
                history_products
            )
            >=
            MAX_HISTORY_PRODUCTS
        ):
            break

    product_array = np.zeros(
        MAX_HISTORY_PRODUCTS,
        dtype=np.int32,
    )

    weight_array = np.zeros(
        MAX_HISTORY_PRODUCTS,
        dtype=np.float16,
    )

    if history_products:

        count = len(
            history_products
        )

        product_array[
            :count
        ] = np.asarray(
            history_products,
            dtype=np.int32,
        )

        weight_array[
            :count
        ] = np.asarray(
            history_weights,
            dtype=np.float16,
        )

    numeric = history_numeric_features(
        history_products=
            history_products,

        history_order_count=
            len(
                selected_positions
            ),
    )

    return (
        product_array,
        weight_array,
        numeric,
    )


def build_serving_history(
    order_products: list[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    """
    Build the context used when recommending after the
    final available TRAIN order.

    Unlike a training pseudo-context, all training orders
    are allowed here.
    """

    n_orders = len(
        order_products
    )

    start_position = max(
        0,
        n_orders
        -
        HISTORY_ORDERS,
    )

    selected_positions = list(
        range(
            start_position,
            n_orders,
        )
    )

    history_products: list[int] = []
    history_weights: list[float] = []

    # Last known train order has recency gap 0 and weight 1.
    for reverse_offset, order_position in enumerate(
        reversed(
            selected_positions
        )
    ):

        weight = math.exp(
            -RECENCY_DECAY
            *
            reverse_offset
        )

        products = (
            order_products[
                order_position
            ]
        )

        for product_id in products:

            history_products.append(
                int(product_id)
            )

            history_weights.append(
                float(weight)
            )

            if (
                len(
                    history_products
                )
                >=
                MAX_HISTORY_PRODUCTS
            ):
                break

        if (
            len(
                history_products
            )
            >=
            MAX_HISTORY_PRODUCTS
        ):
            break

    product_array = np.zeros(
        MAX_HISTORY_PRODUCTS,
        dtype=np.int32,
    )

    weight_array = np.zeros(
        MAX_HISTORY_PRODUCTS,
        dtype=np.float16,
    )

    if history_products:

        count = len(
            history_products
        )

        product_array[
            :count
        ] = np.asarray(
            history_products,
            dtype=np.int32,
        )

        weight_array[
            :count
        ] = np.asarray(
            history_weights,
            dtype=np.float16,
        )

    numeric = history_numeric_features(
        history_products=
            history_products,

        history_order_count=
            len(
                selected_positions
            ),
    )

    return (
        product_array,
        weight_array,
        numeric,
    )


# ============================================================
# Positive-pair Parquet writer
# ============================================================

class PositivePairWriter:

    def __init__(
        self,
        path: Path,
    ):

        self.path = path

        self.tmp_path = (
            path.with_suffix(
                path.suffix
                + ".tmp"
            )
        )

        if self.tmp_path.exists():
            self.tmp_path.unlink()

        self.writer = None

        self.context_ids: list[int] = []
        self.user_ids: list[int] = []
        self.product_ids: list[int] = []
        self.splits: list[int] = []

        self.total_rows = 0


    def append(
        self,
        context_id: int,
        user_id: int,
        product_ids: np.ndarray,
        split: int,
    ):

        # Basket is a set for ranking relevance.
        unique_products = np.unique(
            product_ids
        )

        for product_id in unique_products:

            self.context_ids.append(
                int(
                    context_id
                )
            )

            self.user_ids.append(
                int(
                    user_id
                )
            )

            self.product_ids.append(
                int(
                    product_id
                )
            )

            self.splits.append(
                int(
                    split
                )
            )

        if (
            len(
                self.context_ids
            )
            >=
            PAIR_WRITE_CHUNK_SIZE
        ):
            self.flush()


    def flush(
        self,
    ):

        if not self.context_ids:
            return

        table = pa.Table.from_pydict(
            {
                "context_id":
                    np.asarray(
                        self.context_ids,
                        dtype=np.int32,
                    ),

                "user_id":
                    np.asarray(
                        self.user_ids,
                        dtype=np.int32,
                    ),

                "product_id":
                    np.asarray(
                        self.product_ids,
                        dtype=np.int32,
                    ),

                # 0 = train pseudo-target
                # 1 = temporal internal validation
                "split":
                    np.asarray(
                        self.splits,
                        dtype=np.int8,
                    ),
            }
        )

        if self.writer is None:

            self.writer = (
                pq.ParquetWriter(
                    self.tmp_path,
                    table.schema,
                    compression="snappy",
                )
            )

        self.writer.write_table(
            table
        )

        self.total_rows += len(
            self.context_ids
        )

        self.context_ids.clear()
        self.user_ids.clear()
        self.product_ids.clear()
        self.splits.clear()


    def close(
        self,
    ):

        self.flush()

        if self.writer is None:

            raise RuntimeError(
                "No positive pairs were generated."
            )

        self.writer.close()

        self.writer = None

        if self.path.exists():
            self.path.unlink()

        self.tmp_path.replace(
            self.path
        )


# ============================================================
# Item feature arrays
# ============================================================

def build_item_arrays(
    max_product_id: int,
):

    print(
        "Loading item feature store..."
    )

    items = pd.read_parquet(
        ITEM_FEATURE_PATH,
        columns=[
            "product_id",
            "aisle_id",
            "department_id",
            "purchase_count",
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

    item_purchase_count = np.zeros(
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

    item_purchase_count[
        item_ids
    ] = (
        items[
            "purchase_count"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    item_numeric = np.column_stack(
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

    observed_product_ids = (
        items.loc[
            items[
                "purchase_count"
            ]
            > 0,
            "product_id",
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    observed_product_ids = np.sort(
        observed_product_ids
    )

    # Frequency-smoothed negative sampling.
    #
    # p(item) proportional to count^0.75
    sampling_weights = (
        item_purchase_count[
            observed_product_ids
        ]
        ** 0.75
    )

    sampling_weights = (
        sampling_weights
        /
        sampling_weights.sum()
    ).astype(
        np.float64
    )

    np.savez_compressed(
        ITEM_ARRAY_PATH,

        item_numeric=
            item_numeric,

        item_aisle=
            item_aisle,

        item_department=
            item_department,

        observed_product_ids=
            observed_product_ids,

        negative_sampling_probabilities=
            sampling_weights,
    )

    return {
        "max_product_id":
            max_product_id,

        "max_aisle_id":
            int(
                items[
                    "aisle_id"
                ].max()
            ),

        "max_department_id":
            int(
                items[
                    "department_id"
                ].max()
            ),

        "observed_products":
            int(
                len(
                    observed_product_ids
                )
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading temporal training interactions..."
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "user_id",
            "order_number",
            "product_id",
        ],
    )

    train[
        "user_id"
    ] = (
        train[
            "user_id"
        ]
        .astype(
            np.int32
        )
    )

    train[
        "order_number"
    ] = (
        train[
            "order_number"
        ]
        .astype(
            np.int16
        )
    )

    train[
        "product_id"
    ] = (
        train[
            "product_id"
        ]
        .astype(
            np.int32
        )
    )

    train = (
        train
        .sort_values(
            [
                "user_id",
                "order_number",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Training interactions: "
        f"{len(train):,}"
    )

    print(
        f"Users: "
        f"{train['user_id'].nunique():,}"
    )

    print(
        f"Products: "
        f"{train['product_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Determine required context array size first.
    # --------------------------------------------------------

    print(
        "\nCounting eligible temporal contexts..."
    )

    order_counts = (
        train
        .groupby(
            "user_id",
            sort=False,
        )[
            "order_number"
        ]
        .nunique()
    )

    eligible_counts = (
        order_counts
        -
        MIN_HISTORY_ORDERS
    ).clip(
        lower=0
    )

    selected_counts = (
        eligible_counts
        .clip(
            upper=
                MAX_CONTEXTS_PER_USER
        )
    )

    total_contexts = int(
        selected_counts.sum()
    )

    total_users = int(
        len(
            order_counts
        )
    )

    print(
        f"Temporal contexts to build: "
        f"{total_contexts:,}"
    )

    # --------------------------------------------------------
    # Allocate compact context arrays.
    # --------------------------------------------------------

    context_user_ids = np.zeros(
        total_contexts,
        dtype=np.int32,
    )

    context_history_products = (
        np.zeros(
            (
                total_contexts,
                MAX_HISTORY_PRODUCTS,
            ),
            dtype=np.int32,
        )
    )

    context_history_weights = (
        np.zeros(
            (
                total_contexts,
                MAX_HISTORY_PRODUCTS,
            ),
            dtype=np.float16,
        )
    )

    context_numeric = (
        np.zeros(
            (
                total_contexts,
                4,
            ),
            dtype=np.float32,
        )
    )

    # 0 = train pseudo-target
    # 1 = temporal internal validation target
    context_split = np.zeros(
        total_contexts,
        dtype=np.int8,
    )

    # --------------------------------------------------------
    # Serving contexts: one per user.
    # --------------------------------------------------------

    serving_user_ids = np.zeros(
        total_users,
        dtype=np.int32,
    )

    serving_history_products = (
        np.zeros(
            (
                total_users,
                MAX_HISTORY_PRODUCTS,
            ),
            dtype=np.int32,
        )
    )

    serving_history_weights = (
        np.zeros(
            (
                total_users,
                MAX_HISTORY_PRODUCTS,
            ),
            dtype=np.float16,
        )
    )

    serving_numeric = (
        np.zeros(
            (
                total_users,
                4,
            ),
            dtype=np.float32,
        )
    )

    pair_writer = (
        PositivePairWriter(
            PAIR_PATH
        )
    )

    context_id = 0
    serving_index = 0

    train_context_count = 0
    validation_context_count = 0

    train_pair_count = 0
    validation_pair_count = 0

    # --------------------------------------------------------
    # Build contexts user by user.
    # --------------------------------------------------------

    print(
        "\nBuilding temporal histories..."
    )

    grouped = train.groupby(
        "user_id",
        sort=False,
    )

    for user_id, user_frame in tqdm(
        grouped,
        total=total_users,
        desc="Building Two-Tower v2 contexts",
    ):

        # ----------------------------------------------
        # Convert this user's interactions into
        # ordered baskets.
        # ----------------------------------------------

        order_products: list[np.ndarray] = []

        for _, order_frame in (
            user_frame.groupby(
                "order_number",
                sort=True,
            )
        ):

            products = (
                order_frame[
                    "product_id"
                ]
                .to_numpy(
                    dtype=np.int32
                )
            )

            order_products.append(
                products
            )

        n_orders = len(
            order_products
        )

        # ----------------------------------------------
        # Serving context uses ALL training history.
        # ----------------------------------------------

        (
            serving_products,
            serving_weights,
            serving_stats,
        ) = build_serving_history(
            order_products
        )

        serving_user_ids[
            serving_index
        ] = int(
            user_id
        )

        serving_history_products[
            serving_index
        ] = serving_products

        serving_history_weights[
            serving_index
        ] = serving_weights

        serving_numeric[
            serving_index
        ] = serving_stats

        serving_index += 1

        # ----------------------------------------------
        # Temporal pseudo-target positions.
        # ----------------------------------------------

        first_target_position = (
            MIN_HISTORY_ORDERS
        )

        if (
            n_orders
            <=
            first_target_position
        ):
            continue

        eligible_positions = list(
            range(
                first_target_position,
                n_orders,
            )
        )

        # Keep only latest contexts.
        selected_positions = (
            eligible_positions[
                -MAX_CONTEXTS_PER_USER:
            ]
        )

        # If a user has >=2 temporal contexts:
        #
        # latest target = internal validation
        #
        # earlier target(s) = training
        #
        # If only one exists, keep it as training.
        has_internal_validation = (
            len(
                selected_positions
            )
            >= 2
        )

        latest_position = (
            selected_positions[
                -1
            ]
            if has_internal_validation
            else None
        )

        for target_position in (
            selected_positions
        ):

            split = (
                1
                if (
                    has_internal_validation
                    and
                    target_position
                    ==
                    latest_position
                )
                else
                0
            )

            (
                history_products,
                history_weights,
                history_stats,
            ) = build_history(
                order_products=
                    order_products,

                target_position=
                    target_position,
            )

            context_user_ids[
                context_id
            ] = int(
                user_id
            )

            context_history_products[
                context_id
            ] = history_products

            context_history_weights[
                context_id
            ] = history_weights

            context_numeric[
                context_id
            ] = history_stats

            context_split[
                context_id
            ] = split

            target_products = (
                order_products[
                    target_position
                ]
            )

            unique_target_count = len(
                np.unique(
                    target_products
                )
            )

            pair_writer.append(
                context_id=
                    context_id,

                user_id=
                    int(
                        user_id
                    ),

                product_ids=
                    target_products,

                split=
                    split,
            )

            if split == 0:

                train_context_count += 1

                train_pair_count += (
                    unique_target_count
                )

            else:

                validation_context_count += 1

                validation_pair_count += (
                    unique_target_count
                )

            context_id += 1

    pair_writer.close()

    if context_id != total_contexts:

        raise RuntimeError(
            "Context count mismatch: "
            f"expected {total_contexts:,}, "
            f"built {context_id:,}"
        )

    # --------------------------------------------------------
    # Save context arrays.
    # --------------------------------------------------------

    print(
        "\nSaving temporal context arrays..."
    )

    np.savez_compressed(
        CONTEXT_PATH,

        user_ids=
            context_user_ids,

        history_product_ids=
            context_history_products,

        history_weights=
            context_history_weights,

        numeric_features=
            context_numeric,

        split=
            context_split,
    )

    print(
        "Saving serving contexts..."
    )

    np.savez_compressed(
        SERVING_CONTEXT_PATH,

        user_ids=
            serving_user_ids,

        history_product_ids=
            serving_history_products,

        history_weights=
            serving_history_weights,

        numeric_features=
            serving_numeric,
    )

    # --------------------------------------------------------
    # Item arrays.
    # --------------------------------------------------------

    item_metadata = (
        build_item_arrays(
            max_product_id=
                int(
                    train[
                        "product_id"
                    ].max()
                )
        )
    )

    # --------------------------------------------------------
    # Metadata / quality checks.
    # --------------------------------------------------------

    users_with_internal_validation = (
        len(
            np.unique(
                context_user_ids[
                    context_split
                    == 1
                ]
            )
        )
    )

    metadata = {

        "version":
            "two_tower_v2_temporal",

        "source":
            "train_interactions_only",

        "leakage_policy": {

            "official_validation_used":
                False,

            "official_test_used":
                False,

            "internal_validation":
                (
                    "latest retained pseudo-target "
                    "inside temporal training history"
                ),
        },

        "training_interactions":
            int(
                len(train)
            ),

        "users":
            total_users,

        "temporal_contexts":
            total_contexts,

        "train_contexts":
            int(
                train_context_count
            ),

        "validation_contexts":
            int(
                validation_context_count
            ),

        "users_with_internal_validation":
            int(
                users_with_internal_validation
            ),

        "positive_pairs":
            int(
                pair_writer.total_rows
            ),

        "train_positive_pairs":
            int(
                train_pair_count
            ),

        "validation_positive_pairs":
            int(
                validation_pair_count
            ),

        "history_orders":
            HISTORY_ORDERS,

        "max_history_products":
            MAX_HISTORY_PRODUCTS,

        "min_history_orders":
            MIN_HISTORY_ORDERS,

        "max_contexts_per_user":
            MAX_CONTEXTS_PER_USER,

        "recency_decay":
            RECENCY_DECAY,

        "history_numeric_features": [
            "history_length_ratio",
            "unique_product_ratio",
            "history_order_ratio",
            "repeat_fraction",
        ],

        **item_metadata,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TWO-TOWER V2 TEMPORAL DATA COMPLETE"
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
        CONTEXT_PATH
    )

    print(
        SERVING_CONTEXT_PATH
    )

    print(
        PAIR_PATH
    )

    print(
        ITEM_ARRAY_PATH
    )

    print(
        METADATA_PATH
    )


if __name__ == "__main__":
    main()