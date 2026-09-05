from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from src.utils.spark_io import (
    write_spark_df_local_parquet,
)


# ============================================================
# Python / PySpark configuration
# ============================================================

PYTHON_EXECUTABLE = str(
    Path(sys.executable).resolve()
)

# Force Spark workers to use the active RecoMart .venv
# instead of Windows' generic "python" alias.
os.environ[
    "PYSPARK_PYTHON"
] = PYTHON_EXECUTABLE

os.environ[
    "PYSPARK_DRIVER_PYTHON"
] = PYTHON_EXECUTABLE


# ============================================================
# Project paths
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

CANDIDATE_DIR = (
    PROCESSED_DIR
    / "candidates"
)

TRAIN_PATH = (
    PROCESSED_DIR
    / "train_interactions.parquet"
)

VALIDATION_PATH = (
    PROCESSED_DIR
    / "validation_interactions.parquet"
)

ITEM_FEATURE_PATH = (
    PROCESSED_DIR
    / "item_features.parquet"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "ranker_v1_corrected"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "ranker_train"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "ranker_validation"
)


# ============================================================
# Configuration
# ============================================================

NEGATIVES_PER_USER = 15


RANK_COLUMNS = {
    "repeat_rank": 0,
    "als_rank": 0,
    "content_rank": 0,
}


# ============================================================
# Spark session
# ============================================================

def create_spark_session():

    print(
        "PySpark Python executable:"
    )

    print(
        PYTHON_EXECUTABLE
    )

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(
            "RecoMart-Ranker-V1-Corrected"
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
        .config(
            "spark.pyspark.python",
            PYTHON_EXECUTABLE,
        )
        .config(
            "spark.pyspark.driver.python",
            PYTHON_EXECUTABLE,
        )
        .config(
            "spark.executorEnv.PYSPARK_PYTHON",
            PYTHON_EXECUTABLE,
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    return spark


# ============================================================
# Candidate sources
# ============================================================

def load_candidate_source(
    spark,
    filename: str,
    source: str,
):
    """
    Load a precomputed candidate file.

    Input schema:
        user_id
        recommendations: list[product_id]

    Output schema:
        user_id
        product_id
        <source>_rank
    """

    path = (
        CANDIDATE_DIR
        / filename
    )

    print(
        f"Loading {source} candidates..."
    )

    df = (
        spark.read
        .parquet(
            str(path)
        )
        .select(
            F.col(
                "user_id"
            )
            .cast("int")
            .alias(
                "user_id"
            ),

            F.posexplode(
                "recommendations"
            ).alias(
                "position",
                "product_id",
            ),
        )
        .withColumn(
            "product_id",
            F.col(
                "product_id"
            ).cast("int"),
        )
        .withColumn(
            f"{source}_rank",
            (
                F.col(
                    "position"
                )
                + 1
            )
            .cast("int"),
        )
        .drop(
            "position"
        )
    )

    return df


def build_candidate_union(
    spark,
):
    """
    Build the original three-source candidate union:

        Repeat Purchase
        ALS-v2
        Content-v1

    This intentionally excludes Two-Tower so that
    this dataset is a controlled baseline for Ranker v2.
    """

    print(
        "Loading ranked candidate sources..."
    )

    repeat = load_candidate_source(
        spark,
        "repeat_purchase_top50.parquet",
        "repeat",
    )

    als = load_candidate_source(
        spark,
        "als_v2_top50.parquet",
        "als",
    )

    content = load_candidate_source(
        spark,
        "content_v1_top50.parquet",
        "content",
    )

    print(
        "Building three-source candidate union..."
    )

    candidates = (
        repeat
        .join(
            als,
            on=[
                "user_id",
                "product_id",
            ],
            how="full",
        )
        .join(
            content,
            on=[
                "user_id",
                "product_id",
            ],
            how="full",
        )
    )

    # ----------------------------------------------------
    # Candidate-source membership
    # ----------------------------------------------------

    candidates = (
        candidates

        .withColumn(
            "is_repeat_candidate",

            F.col(
                "repeat_rank"
            )
            .isNotNull()
            .cast("int"),
        )

        .withColumn(
            "is_als_candidate",

            F.col(
                "als_rank"
            )
            .isNotNull()
            .cast("int"),
        )

        .withColumn(
            "is_content_candidate",

            F.col(
                "content_rank"
            )
            .isNotNull()
            .cast("int"),
        )
    )

    # ----------------------------------------------------
    # Number of retrieval sources supporting item
    # ----------------------------------------------------

    candidates = (
        candidates
        .withColumn(
            "source_count",

            (
                F.col(
                    "is_repeat_candidate"
                )
                +
                F.col(
                    "is_als_candidate"
                )
                +
                F.col(
                    "is_content_candidate"
                )
            )
            .cast("int"),
        )
    )

    # ----------------------------------------------------
    # Reciprocal-rank signals
    # ----------------------------------------------------

    candidates = (
        candidates

        .withColumn(
            "repeat_rr",

            F.when(
                F.col(
                    "repeat_rank"
                )
                .isNotNull(),

                1.0
                /
                F.col(
                    "repeat_rank"
                ),
            )
            .otherwise(
                0.0
            )
            .cast("float"),
        )

        .withColumn(
            "als_rr",

            F.when(
                F.col(
                    "als_rank"
                )
                .isNotNull(),

                1.0
                /
                F.col(
                    "als_rank"
                ),
            )
            .otherwise(
                0.0
            )
            .cast("float"),
        )

        .withColumn(
            "content_rr",

            F.when(
                F.col(
                    "content_rank"
                )
                .isNotNull(),

                1.0
                /
                F.col(
                    "content_rank"
                ),
            )
            .otherwise(
                0.0
            )
            .cast("float"),
        )
    )

    return candidates


# ============================================================
# Historical behavior features
# ============================================================

def build_history_features(
    spark,
):
    """
    Build historical user-product and user-level
    behavior features from the temporal training split.
    """

    print(
        "Loading historical interactions..."
    )

    train = (
        spark.read
        .parquet(
            str(TRAIN_PATH)
        )
        .select(
            F.col(
                "user_id"
            )
            .cast("int"),

            F.col(
                "product_id"
            )
            .cast("int"),

            F.col(
                "order_number"
            )
            .cast("int"),

            F.col(
                "reordered"
            )
            .cast("float"),
        )
    )

    print(
        "Building user-product features..."
    )

    user_product = (
        train
        .groupBy(
            "user_id",
            "product_id",
        )
        .agg(
            F.count("*")
            .cast("float")
            .alias(
                "user_product_purchase_count"
            ),

            F.avg(
                "reordered"
            )
            .cast("float")
            .alias(
                "user_product_reorder_rate"
            ),

            F.max(
                "order_number"
            )
            .cast("int")
            .alias(
                "last_purchase_order"
            ),
        )
    )

    print(
        "Building user summary features..."
    )

    user_summary = (
        train
        .groupBy(
            "user_id"
        )
        .agg(
            F.max(
                "order_number"
            )
            .cast("int")
            .alias(
                "user_latest_order"
            ),

            F.count("*")
            .cast("float")
            .alias(
                "user_total_interactions"
            ),

            F.countDistinct(
                "product_id"
            )
            .cast("float")
            .alias(
                "user_unique_products"
            ),
        )
    )

    return (
        train,
        user_product,
        user_summary,
    )


# ============================================================
# Category affinity features
# ============================================================

def build_category_affinities(
    train,
    item_features,
):
    """
    Calculate user preference for aisles and
    departments from historical purchases.
    """

    print(
        "Building user category affinities..."
    )

    product_categories = (
        item_features
        .select(
            "product_id",
            "aisle_id",
            "department_id",
        )
    )

    enriched = (
        train
        .join(
            product_categories,
            on="product_id",
            how="left",
        )
    )

    user_totals = (
        enriched
        .groupBy(
            "user_id"
        )
        .agg(
            F.count("*")
            .alias(
                "category_user_total"
            )
        )
    )

    # ----------------------------------------------------
    # User → aisle affinity
    # ----------------------------------------------------

    user_aisle = (
        enriched
        .groupBy(
            "user_id",
            "aisle_id",
        )
        .agg(
            F.count("*")
            .alias(
                "user_aisle_count"
            )
        )
        .join(
            user_totals,
            on="user_id",
            how="inner",
        )
        .withColumn(
            "aisle_affinity",

            (
                F.col(
                    "user_aisle_count"
                )
                /
                F.col(
                    "category_user_total"
                )
            )
            .cast("float"),
        )
        .select(
            "user_id",
            "aisle_id",
            "aisle_affinity",
        )
    )

    # ----------------------------------------------------
    # User → department affinity
    # ----------------------------------------------------

    user_department = (
        enriched
        .groupBy(
            "user_id",
            "department_id",
        )
        .agg(
            F.count("*")
            .alias(
                "user_department_count"
            )
        )
        .join(
            user_totals,
            on="user_id",
            how="inner",
        )
        .withColumn(
            "department_affinity",

            (
                F.col(
                    "user_department_count"
                )
                /
                F.col(
                    "category_user_total"
                )
            )
            .cast("float"),
        )
        .select(
            "user_id",
            "department_id",
            "department_affinity",
        )
    )

    return (
        user_aisle,
        user_department,
    )


# ============================================================
# Validation labels
# ============================================================

def add_labels(
    spark,
    candidates,
):
    """
    Mark candidate as positive when it appears in the
    user's next validation basket.
    """

    print(
        "Adding validation labels..."
    )

    positives = (
        spark.read
        .parquet(
            str(
                VALIDATION_PATH
            )
        )
        .select(
            F.col(
                "user_id"
            )
            .cast("int"),

            F.col(
                "product_id"
            )
            .cast("int"),
        )
        .dropDuplicates(
            [
                "user_id",
                "product_id",
            ]
        )
        .withColumn(
            "label",
            F.lit(1),
        )
    )

    labeled = (
        candidates
        .join(
            positives,
            on=[
                "user_id",
                "product_id",
            ],
            how="left",
        )
        .fillna(
            {
                "label": 0,
            }
        )
        .withColumn(
            "label",
            F.col(
                "label"
            )
            .cast("int"),
        )
    )

    return labeled


# ============================================================
# Main pipeline
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = (
        create_spark_session()
    )

    try:

        # ====================================================
        # 1. Build original three-source candidate union
        # ====================================================

        candidates = (
            build_candidate_union(
                spark
            )
        )

        # ====================================================
        # 2. Historical features
        # ====================================================

        (
            train,
            user_product,
            user_summary,
        ) = (
            build_history_features(
                spark
            )
        )

        # ====================================================
        # 3. Item feature store
        # ====================================================

        print(
            "Loading item feature store..."
        )

        item_features = (
            spark.read
            .parquet(
                str(
                    ITEM_FEATURE_PATH
                )
            )
            .select(
                F.col(
                    "product_id"
                )
                .cast("int"),

                F.col(
                    "aisle_id"
                )
                .cast("int"),

                F.col(
                    "department_id"
                )
                .cast("int"),

                F.col(
                    "purchase_count"
                )
                .cast("float")
                .alias(
                    "product_purchase_count"
                ),

                F.col(
                    "reorder_rate"
                )
                .cast("float")
                .alias(
                    "product_reorder_rate"
                ),

                F.col(
                    "log_purchase_count"
                )
                .cast("float")
                .alias(
                    "product_log_purchase_count"
                ),
            )
        )

        # ====================================================
        # 4. User/category affinity
        # ====================================================

        (
            user_aisle,
            user_department,
        ) = (
            build_category_affinities(
                train,
                item_features,
            )
        )

        # ====================================================
        # 5. Join all ranker features
        # ====================================================

        print(
            "Joining ranking features..."
        )

        features = (
            candidates

            .join(
                user_product,
                on=[
                    "user_id",
                    "product_id",
                ],
                how="left",
            )

            .join(
                user_summary,
                on="user_id",
                how="left",
            )

            .join(
                item_features,
                on="product_id",
                how="left",
            )

            .join(
                user_aisle,
                on=[
                    "user_id",
                    "aisle_id",
                ],
                how="left",
            )

            .join(
                user_department,
                on=[
                    "user_id",
                    "department_id",
                ],
                how="left",
            )
        )

        # ====================================================
        # 6. Fill missing historical values
        # ====================================================

        features = (
            features
            .fillna(
                {
                    "user_product_purchase_count":
                        0.0,

                    "user_product_reorder_rate":
                        0.0,

                    "aisle_affinity":
                        0.0,

                    "department_affinity":
                        0.0,
                }
            )
        )

        # ====================================================
        # 7. Seen-before indicator
        # ====================================================

        features = (
            features
            .withColumn(
                "seen_before",

                (
                    F.col(
                        "user_product_purchase_count"
                    )
                    > 0
                )
                .cast("int"),
            )
        )

        # ====================================================
        # 8. User-relative recency
        #
        # Seen product:
        # user_latest_order - last_purchase_order
        #
        # Unseen product:
        # user_latest_order + 1
        # ====================================================

        features = (
            features
            .withColumn(
                "orders_since_last_purchase",

                F.when(
                    F.col(
                        "last_purchase_order"
                    )
                    .isNull(),

                    (
                        F.col(
                            "user_latest_order"
                        )
                        + 1
                    ),
                )
                .otherwise(
                    F.col(
                        "user_latest_order"
                    )
                    -
                    F.col(
                        "last_purchase_order"
                    )
                )
                .cast("float"),
            )
        )

        # ====================================================
        # 9. Validation labels
        # ====================================================

        features = (
            add_labels(
                spark,
                features,
            )
        )

        # ====================================================
        # 10. Missing retrieval ranks
        #
        # rank=0 means the source did not retrieve it.
        # ====================================================

        features = (
            features
            .fillna(
                RANK_COLUMNS
            )
        )

        # ====================================================
        # 11. Deterministic USER-level ranker split
        #
        # Same split logic used by v1 and v2:
        #
        # hash % 5 == 0 → validation
        # otherwise     → train
        #
        # Approximately 80/20.
        # ====================================================

        print(
            "Creating deterministic "
            "user-level ranker split..."
        )

        features = (
            features
            .withColumn(
                "ranker_split",

                F.when(
                    F.pmod(
                        F.xxhash64(
                            "user_id"
                        ),
                        F.lit(5),
                    )
                    == 0,

                    F.lit(
                        "validation"
                    ),
                )
                .otherwise(
                    F.lit(
                        "train"
                    )
                ),
            )
        )

        # ====================================================
        # 12. Full ranker validation dataset
        #
        # Do NOT negative-sample validation.
        # Every candidate must remain available.
        # ====================================================

        validation_features = (
            features
            .filter(
                F.col(
                    "ranker_split"
                )
                == "validation"
            )
            .drop(
                "ranker_split"
            )
        )

        # ====================================================
        # 13. Ranker training split
        # ====================================================

        train_features = (
            features
            .filter(
                F.col(
                    "ranker_split"
                )
                == "train"
            )
        )

        # ====================================================
        # 14. CORRECTED negative sampling
        #
        # IMPORTANT:
        #
        # Positives and negatives are separated BEFORE
        # row_number().
        #
        # This ensures each user gets up to exactly
        # NEGATIVES_PER_USER negatives.
        #
        # This is the correction relative to original v1.
        # ====================================================

        print(
            "Sampling ranker training negatives..."
        )

        train_positives = (
            train_features
            .filter(
                F.col(
                    "label"
                )
                == 1
            )
        )

        train_negatives = (
            train_features
            .filter(
                F.col(
                    "label"
                )
                == 0
            )
        )

        negative_window = (
            Window
            .partitionBy(
                "user_id"
            )
            .orderBy(
                F.xxhash64(
                    "user_id",
                    "product_id",
                )
            )
        )

        sampled_negatives = (
            train_negatives
            .withColumn(
                "negative_rank",

                F.row_number()
                .over(
                    negative_window
                ),
            )
            .filter(
                F.col(
                    "negative_rank"
                )
                <=
                NEGATIVES_PER_USER
            )
            .drop(
                "negative_rank"
            )
        )

        train_features = (
            train_positives
            .unionByName(
                sampled_negatives
            )
            .drop(
                "ranker_split"
            )
        )

        # ====================================================
        # 15. Write ranker training data
        #
        # Use our PyArrow local writer because native
        # Spark/Hadoop local Parquet writing causes
        # winutils.exe issues on this Windows setup.
        # ====================================================

        print(
            "Writing corrected ranker v1 "
            "training data..."
        )

        train_count = (
            write_spark_df_local_parquet(
                df=train_features,

                output_dir=
                    TRAIN_OUTPUT,

                num_partitions=
                    32,

                partition_column=
                    "user_id",
            )
        )

        # ====================================================
        # 16. Write validation data
        # ====================================================

        print(
            "Writing corrected ranker v1 "
            "validation data..."
        )

        val_count = (
            write_spark_df_local_parquet(
                df=validation_features,

                output_dir=
                    VALIDATION_OUTPUT,

                num_partitions=
                    16,

                partition_column=
                    "user_id",
            )
        )

        # ====================================================
        # 17. Dataset statistics
        # ====================================================

        print(
            "Computing dataset statistics..."
        )

        train_users = (
            train_features
            .select(
                "user_id"
            )
            .distinct()
            .count()
        )

        validation_users = (
            validation_features
            .select(
                "user_id"
            )
            .distinct()
            .count()
        )

        train_positive_count = (
            train_features
            .filter(
                F.col(
                    "label"
                )
                == 1
            )
            .count()
        )

        validation_positive_count = (
            validation_features
            .filter(
                F.col(
                    "label"
                )
                == 1
            )
            .count()
        )

        train_negative_count = (
            train_features
            .filter(
                F.col(
                    "label"
                )
                == 0
            )
            .count()
        )

        validation_negative_count = (
            validation_features
            .filter(
                F.col(
                    "label"
                )
                == 0
            )
            .count()
        )

        # ====================================================
        # 18. Sanity calculations
        # ====================================================

        expected_max_negatives = (
            train_users
            *
            NEGATIVES_PER_USER
        )

        average_negatives_per_user = (
            train_negative_count
            /
            train_users
            if train_users > 0
            else 0.0
        )

        average_candidates_per_validation_user = (
            val_count
            /
            validation_users
            if validation_users > 0
            else 0.0
        )

        # ====================================================
        # 19. Final output
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CORRECTED RANKER V1 DATASET COMPLETE"
        )

        print()

        print(
            "TRAIN"
        )

        print(
            f"Rows: "
            f"{train_count:,}"
        )

        print(
            f"Users: "
            f"{train_users:,}"
        )

        print(
            f"Positives: "
            f"{train_positive_count:,}"
        )

        print(
            f"Negatives: "
            f"{train_negative_count:,}"
        )

        print(
            f"Maximum expected negatives "
            f"(users × {NEGATIVES_PER_USER}): "
            f"{expected_max_negatives:,}"
        )

        print(
            f"Average negatives/user: "
            f"{average_negatives_per_user:.4f}"
        )

        print()

        print(
            "VALIDATION"
        )

        print(
            f"Rows: "
            f"{val_count:,}"
        )

        print(
            f"Users: "
            f"{validation_users:,}"
        )

        print(
            f"Positives: "
            f"{validation_positive_count:,}"
        )

        print(
            f"Negatives: "
            f"{validation_negative_count:,}"
        )

        print(
            "Average candidates/user: "
            f"{average_candidates_per_validation_user:.2f}"
        )

        print(
            "\nSaved under:"
        )

        print(
            OUTPUT_DIR
        )

    finally:

        spark.stop()


if __name__ == "__main__":
    main()