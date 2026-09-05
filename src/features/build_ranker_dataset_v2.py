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

# Force Spark Python workers to use the current
# RecoMart virtual environment instead of Windows'
# generic "python" command / Microsoft Store alias.
os.environ[
    "PYSPARK_PYTHON"
] = PYTHON_EXECUTABLE

os.environ[
    "PYSPARK_DRIVER_PYTHON"
] = PYTHON_EXECUTABLE


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
    / "ranker_v2"
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
    "two_tower_rank": 0,
}


# ============================================================
# Spark
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
            "RecoMart-Ranker-V2-Features"
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
# Candidate Sources
# ============================================================

def load_candidate_source(
    spark,
    filename: str,
    source: str,
):
    """
    Load a candidate file containing:

        user_id
        recommendations: list[product_id]

    and expand it to:

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
            ).cast("int"),
        )
        .drop(
            "position"
        )
    )

    return df


def load_two_tower_source(
    spark,
):
    """
    Load Two-Tower candidates.

    The Two-Tower candidate file contains:

        user_id
        recommendations
        scores

    recommendations[i] and scores[i]
    must stay aligned.

    Output:

        user_id
        product_id
        two_tower_rank
        two_tower_score
    """

    path = (
        CANDIDATE_DIR
        / "two_tower_v1_top50.parquet"
    )

    print(
        "Loading two_tower candidates..."
    )

    raw = (
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

            "recommendations",
            "scores",
        )
    )

    zipped = (
        raw
        .withColumn(
            "candidate_pairs",
            F.arrays_zip(
                "recommendations",
                "scores",
            ),
        )
        .select(
            "user_id",

            F.posexplode(
                "candidate_pairs"
            ).alias(
                "position",
                "candidate",
            ),
        )
    )

    df = (
        zipped
        .select(
            "user_id",

            F.col(
                "candidate.recommendations"
            )
            .cast("int")
            .alias(
                "product_id"
            ),

            (
                F.col(
                    "position"
                )
                + 1
            )
            .cast("int")
            .alias(
                "two_tower_rank"
            ),

            F.col(
                "candidate.scores"
            )
            .cast("float")
            .alias(
                "two_tower_score"
            ),
        )
    )

    return df


def build_candidate_union(
    spark,
):
    """
    Build the four-source candidate union:

        Repeat Purchase
        ALS
        Content
        Two-Tower

    Each candidate retains source-specific rank and
    membership information.
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

    two_tower = (
        load_two_tower_source(
            spark
        )
    )

    print(
        "Building four-source candidate union..."
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
        .join(
            two_tower,
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

        .withColumn(
            "is_two_tower_candidate",
            F.col(
                "two_tower_rank"
            )
            .isNotNull()
            .cast("int"),
        )
    )

    # ----------------------------------------------------
    # Number of retrieval sources supporting candidate
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
                +
                F.col(
                    "is_two_tower_candidate"
                )
            ).cast("int"),
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
                ).isNotNull(),

                1.0
                /
                F.col(
                    "repeat_rank"
                ),
            )
            .otherwise(
                0.0
            )
            .cast(
                "float"
            ),
        )

        .withColumn(
            "als_rr",

            F.when(
                F.col(
                    "als_rank"
                ).isNotNull(),

                1.0
                /
                F.col(
                    "als_rank"
                ),
            )
            .otherwise(
                0.0
            )
            .cast(
                "float"
            ),
        )

        .withColumn(
            "content_rr",

            F.when(
                F.col(
                    "content_rank"
                ).isNotNull(),

                1.0
                /
                F.col(
                    "content_rank"
                ),
            )
            .otherwise(
                0.0
            )
            .cast(
                "float"
            ),
        )

        .withColumn(
            "two_tower_rr",

            F.when(
                F.col(
                    "two_tower_rank"
                ).isNotNull(),

                1.0
                /
                F.col(
                    "two_tower_rank"
                ),
            )
            .otherwise(
                0.0
            )
            .cast(
                "float"
            ),
        )
    )

    return candidates


# ============================================================
# Historical Features
# ============================================================

def build_history_features(
    spark,
):
    """
    Build historical user-product and user-level
    behavioral features from the training period.
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
            ).cast("int"),

            F.col(
                "product_id"
            ).cast("int"),

            F.col(
                "order_number"
            ).cast("int"),

            F.col(
                "reordered"
            ).cast("float"),
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
# Category Affinity Features
# ============================================================

def build_category_affinities(
    train,
    item_features,
):
    """
    Calculate the fraction of a user's historical
    purchases belonging to each aisle and department.
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
# Validation Labels
# ============================================================

def add_labels(
    spark,
    candidates,
):
    """
    Candidate is positive if the user purchased that
    product in the next validation basket.
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
            ).cast("int"),

            F.col(
                "product_id"
            ).cast("int"),
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
            ).cast("int"),
        )
    )

    return labeled


# ============================================================
# Main Pipeline
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
        # 1. Four-source candidate retrieval
        # ====================================================

        candidates = (
            build_candidate_union(
                spark
            )
        )

        # ====================================================
        # 2. Historical user/product features
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
        # 4. Category affinity
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
        # 5. Join all ranking features
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
        # 6. Missing historical features
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
        # 7. Previously seen indicator
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
        # 8. Recency
        #
        # For unseen candidates:
        # user_latest_order + 1
        #
        # For seen candidates:
        # latest user order - last product order
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
        # 9. Add future-basket labels
        # ====================================================

        features = add_labels(
            spark,
            features,
        )

        # ====================================================
        # 10. Fill missing retrieval ranks
        #
        # rank=0 means that source did NOT retrieve
        # the candidate.
        # ====================================================

        features = (
            features
            .fillna(
                RANK_COLUMNS
            )
        )

        # ====================================================
        # 11. Two-Tower missing score
        #
        # Normal cosine similarity should be in [-1, 1].
        # -2 means "not returned by Two-Tower".
        # ====================================================

        features = (
            features
            .fillna(
                {
                    "two_tower_score":
                        -2.0,
                }
            )
        )

        # Make Two-Tower score explicitly float.
        features = (
            features
            .withColumn(
                "two_tower_score",
                F.col(
                    "two_tower_score"
                )
                .cast("float"),
            )
        )

        # ====================================================
        # 12. Deterministic user-level ranker split
        #
        # ~80% train
        # ~20% validation
        #
        # Every candidate for the same user stays
        # entirely in one split.
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
        # 13. Ranker validation
        #
        # KEEP ALL candidates.
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
        # 14. Ranker training
        #
        # Keep ALL positives.
        # Sample deterministic negatives.
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

        print(
            "Sampling ranker training negatives..."
        )

        # Separate positives and negatives before
        # row_number so positive rows never affect
        # negative numbering.
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
        # 15. Write training dataset
        #
        # PyArrow compatibility writer is used instead
        # of Spark's Hadoop local filesystem writer
        # because this project is running on Windows.
        # ====================================================

        print(
            "Writing ranker v2 training data..."
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
        # 16. Write full ranker validation dataset
        # ====================================================

        print(
            "Writing ranker v2 validation data..."
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

        val_users = (
            validation_features
            .select(
                "user_id"
            )
            .distinct()
            .count()
        )

        train_positives_count = (
            train_features
            .filter(
                F.col(
                    "label"
                )
                == 1
            )
            .count()
        )

        val_positives_count = (
            validation_features
            .filter(
                F.col(
                    "label"
                )
                == 1
            )
            .count()
        )

        train_two_tower_candidates = (
            train_features
            .filter(
                F.col(
                    "is_two_tower_candidate"
                )
                == 1
            )
            .count()
        )

        val_two_tower_candidates = (
            validation_features
            .filter(
                F.col(
                    "is_two_tower_candidate"
                )
                == 1
            )
            .count()
        )

        train_two_tower_positive = (
            train_features
            .filter(
                (
                    F.col(
                        "is_two_tower_candidate"
                    )
                    == 1
                )
                &
                (
                    F.col(
                        "label"
                    )
                    == 1
                )
            )
            .count()
        )

        val_two_tower_positive = (
            validation_features
            .filter(
                (
                    F.col(
                        "is_two_tower_candidate"
                    )
                    == 1
                )
                &
                (
                    F.col(
                        "label"
                    )
                    == 1
                )
            )
            .count()
        )

        # ====================================================
        # 18. Final output
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RANKER V2 DATASET COMPLETE"
        )

        print(
            f"Train rows: "
            f"{train_count:,}"
        )

        print(
            f"Train users: "
            f"{train_users:,}"
        )

        print(
            f"Train positives: "
            f"{train_positives_count:,}"
        )

        print(
            f"Train Two-Tower candidate rows: "
            f"{train_two_tower_candidates:,}"
        )

        print(
            f"Train Two-Tower positive rows: "
            f"{train_two_tower_positive:,}"
        )

        print()

        print(
            f"Validation rows: "
            f"{val_count:,}"
        )

        print(
            f"Validation users: "
            f"{val_users:,}"
        )

        print(
            f"Validation positives: "
            f"{val_positives_count:,}"
        )

        print(
            f"Validation Two-Tower candidate rows: "
            f"{val_two_tower_candidates:,}"
        )

        print(
            f"Validation Two-Tower positive rows: "
            f"{val_two_tower_positive:,}"
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