from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import (
    SparkSession,
    Window,
)

from pyspark.sql import (
    functions as F,
)

from src.utils.spark_io import (
    write_spark_df_local_parquet,
)

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
            "RecoMart-Ranker-Features"
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
PYTHON_EXECUTABLE = str(
    Path(sys.executable).resolve()
)

os.environ[
    "PYSPARK_PYTHON"
] = PYTHON_EXECUTABLE

os.environ[
    "PYSPARK_DRIVER_PYTHON"
] = PYTHON_EXECUTABLE

RANK_COLUMNS = {
    "repeat_rank": 0,
    "als_rank": 0,
    "content_rank": 0,
}
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
    / "ranker"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "ranker_train"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "ranker_validation"
)

NEGATIVES_PER_USER = 15


def create_spark_session():

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(
            "RecoMart-Ranker-Features"
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


def load_candidate_source(
    spark,
    filename: str,
    source: str,
):

    path = (
        CANDIDATE_DIR
        / filename
    )

    df = (
        spark.read
        .parquet(str(path))
        .select(
            F.col("user_id")
            .cast("int")
            .alias("user_id"),

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
                F.col("position")
                + 1
            ).cast("int"),
        )
        .drop(
            "position"
        )
    )

    return df


def build_candidate_union(
    spark,
):

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
        "Building candidate union..."
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

    candidates = (
        candidates
        .withColumn(
            "is_repeat_candidate",
            F.col(
                "repeat_rank"
            ).isNotNull().cast("int"),
        )
        .withColumn(
            "is_als_candidate",
            F.col(
                "als_rank"
            ).isNotNull().cast("int"),
        )
        .withColumn(
            "is_content_candidate",
            F.col(
                "content_rank"
            ).isNotNull().cast("int"),
        )
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
            ),
        )
    )

    # Reciprocal-rank signals give the learner
    # smooth versions of source position.
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
            ).otherwise(0.0),
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
            ).otherwise(0.0),
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
            ).otherwise(0.0),
        )
    )

    return candidates


def build_history_features(
    spark,
):

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
            .alias(
                "last_purchase_order"
            ),
        )
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
            .alias(
                "user_latest_order"
            ),

            F.count("*")
            .alias(
                "user_total_interactions"
            ),

            F.countDistinct(
                "product_id"
            )
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


def build_category_affinities(
    train,
    item_features,
):

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
            ).cast("float"),
        )
        .select(
            "user_id",
            "aisle_id",
            "aisle_affinity",
        )
    )

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
            ).cast("float"),
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


def add_labels(
    spark,
    candidates,
):

    print(
        "Adding validation labels..."
    )

    positives = (
        spark.read
        .parquet(
            str(VALIDATION_PATH)
        )
        .select(
            F.col(
                "user_id"
            ).cast("int"),

            F.col(
                "product_id"
            ).cast("int"),
        )
        .dropDuplicates()
        .withColumn(
            "label",
            F.lit(1),
        )
    )

    return (
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


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = (
        create_spark_session()
    )

    try:

        candidates = (
            build_candidate_union(
                spark
            )
        )

        (
            train,
            user_product,
            user_summary,
        ) = build_history_features(
            spark
        )

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
                ).cast("int"),

                F.col(
                    "aisle_id"
                ).cast("int"),

                F.col(
                    "department_id"
                ).cast("int"),

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

        (
            user_aisle,
            user_department,
        ) = build_category_affinities(
            train,
            item_features,
        )

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
            .withColumn(
                "seen_before",
                (
                    F.col(
                        "user_product_purchase_count"
                    )
                    > 0
                ).cast("int"),
            )
            .withColumn(
                "orders_since_last_purchase",

                F.when(
                    F.col(
                        "last_purchase_order"
                    ).isNull(),

                    F.col(
                        "user_latest_order"
                    )
                    + 1,
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

        features = add_labels(
            spark,
            features,
        )
        features = features.fillna(
            RANK_COLUMNS
        )

        # -------------------------------------------------
        # User-level ranker train/validation split.
        #
        # All candidates from one user remain in the
        # same split.
        # -------------------------------------------------

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

        train_features = (
            features
            .filter(
                F.col(
                    "ranker_split"
                )
                == "train"
            )
        )

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

        # -------------------------------------------------
        # Keep every positive example but only a
        # deterministic subset of negatives for
        # ranker training.
        # -------------------------------------------------

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

        train_features = (
            train_features
            .withColumn(
                "negative_rank",

                F.when(
                    F.col(
                        "label"
                    )
                    == 0,

                    F.row_number()
                    .over(
                        negative_window
                    ),
                ),
            )
            .filter(
                (
                    F.col(
                        "label"
                    )
                    == 1
                )
                |
                (
                    F.col(
                        "negative_rank"
                    )
                    <=
                    NEGATIVES_PER_USER
                )
            )
            .drop(
                "negative_rank",
                "ranker_split",
            )
        )

        print(
            "Writing ranker training data..."
        )

        train_count = (
            write_spark_df_local_parquet(
                df=train_features,
                output_dir=TRAIN_OUTPUT,
                num_partitions=32,
                partition_column="user_id",
            )
        )

        print(
            "Writing ranker validation data..."
        )

        val_count = (
            write_spark_df_local_parquet(
                df=validation_features,
                output_dir=
                VALIDATION_OUTPUT,
                num_partitions=16,
                partition_column="user_id",
            )
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

        train_positives = (
            train_features
            .filter(
                F.col("label")
                == 1
            )
            .count()
        )

        val_positives = (
            validation_features
            .filter(
                F.col("label")
                == 1
            )
            .count()
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RANKER DATASET COMPLETE"
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
            f"{train_positives:,}"
        )

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
            f"{val_positives:,}"
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