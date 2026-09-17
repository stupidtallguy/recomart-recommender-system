from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from src.utils.spark_io import write_spark_df_local_parquet


# ============================================================
# PySpark / Python configuration
# ============================================================

PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())

os.environ["PYSPARK_PYTHON"] = PYTHON_EXECUTABLE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXECUTABLE


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CANDIDATE_DIR = PROCESSED_DIR / "candidates"

TRAIN_PATH = PROCESSED_DIR / "train_interactions.parquet"
VALIDATION_PATH = PROCESSED_DIR / "validation_interactions.parquet"
ITEM_FEATURE_PATH = PROCESSED_DIR / "item_features.parquet"

OUTPUT_DIR = PROCESSED_DIR / "ranker_v3"

TRAIN_OUTPUT = OUTPUT_DIR / "ranker_train"
VALIDATION_OUTPUT = OUTPUT_DIR / "ranker_validation"


# ============================================================
# Configuration
# ============================================================

NEGATIVES_PER_USER = 15

RANK_COLUMNS = {
    "repeat_rank": 0,
    "als_rank": 0,
    "content_rank": 0,
    "two_tower_v1_rank": 0,
    "two_tower_v2_rank": 0,
}

SCORE_COLUMNS = {
    "two_tower_v1_score": -2.0,
    "two_tower_v2_score": -2.0,
}


# ============================================================
# Spark
# ============================================================

def create_spark_session():

    print("PySpark Python executable:")
    print(PYTHON_EXECUTABLE)

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("RecoMart-Ranker-V3-Five-Source")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
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

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# Standard candidate loader
# ============================================================

def load_candidate_source(
    spark,
    filename: str,
    source_name: str,
):
    """
    Loads candidate files containing:

        user_id
        recommendations

    and converts to:

        user_id
        product_id
        <source>_rank
    """

    path = CANDIDATE_DIR / filename

    print(f"Loading {source_name}:")
    print(path)

    return (
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
        .select(
            "user_id",

            F.col("product_id")
            .cast("int")
            .alias("product_id"),

            (
                F.col("position") + 1
            )
            .cast("int")
            .alias(
                f"{source_name}_rank"
            ),
        )
    )


# ============================================================
# Two-Tower candidate loader
# ============================================================

def load_two_tower_source(
    spark,
    filename: str,
    version: str,
):
    """
    Two-Tower candidate files contain aligned arrays:

        recommendations
        scores

    Output:

        user_id
        product_id
        two_tower_vX_rank
        two_tower_vX_score
    """

    path = CANDIDATE_DIR / filename

    print(f"Loading {version}:")
    print(path)

    raw = (
        spark.read
        .parquet(str(path))
        .select(
            F.col("user_id")
            .cast("int")
            .alias("user_id"),
            "recommendations",
            "scores",
        )
    )

    exploded = (
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

    return (
        exploded
        .select(
            "user_id",

            F.col(
                "candidate.recommendations"
            )
            .cast("int")
            .alias("product_id"),

            (
                F.col("position") + 1
            )
            .cast("int")
            .alias(
                f"{version}_rank"
            ),

            F.col(
                "candidate.scores"
            )
            .cast("float")
            .alias(
                f"{version}_score"
            ),
        )
    )


# ============================================================
# Candidate union
# ============================================================

def build_candidate_union(spark):

    print("\nLoading five candidate sources...")

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

    tt_v1 = load_two_tower_source(
        spark,
        "two_tower_v1_top50.parquet",
        "two_tower_v1",
    )

    tt_v2 = load_two_tower_source(
        spark,
        "two_tower_v2_top50.parquet",
        "two_tower_v2",
    )

    print("\nBuilding five-source union...")

    candidates = (
        repeat
        .join(
            als,
            ["user_id", "product_id"],
            "full",
        )
        .join(
            content,
            ["user_id", "product_id"],
            "full",
        )
        .join(
            tt_v1,
            ["user_id", "product_id"],
            "full",
        )
        .join(
            tt_v2,
            ["user_id", "product_id"],
            "full",
        )
    )

    # --------------------------------------------------------
    # Membership indicators
    # --------------------------------------------------------

    candidates = (
        candidates
        .withColumn(
            "is_repeat_candidate",
            F.col("repeat_rank")
            .isNotNull()
            .cast("int"),
        )
        .withColumn(
            "is_als_candidate",
            F.col("als_rank")
            .isNotNull()
            .cast("int"),
        )
        .withColumn(
            "is_content_candidate",
            F.col("content_rank")
            .isNotNull()
            .cast("int"),
        )
        .withColumn(
            "is_two_tower_v1_candidate",
            F.col("two_tower_v1_rank")
            .isNotNull()
            .cast("int"),
        )
        .withColumn(
            "is_two_tower_v2_candidate",
            F.col("two_tower_v2_rank")
            .isNotNull()
            .cast("int"),
        )
    )

    # --------------------------------------------------------
    # Number of retrieval sources supporting candidate
    # --------------------------------------------------------

    candidates = candidates.withColumn(
        "source_count",
        (
            F.col("is_repeat_candidate")
            + F.col("is_als_candidate")
            + F.col("is_content_candidate")
            + F.col("is_two_tower_v1_candidate")
            + F.col("is_two_tower_v2_candidate")
        ).cast("int"),
    )

    # --------------------------------------------------------
    # Reciprocal-rank features
    # --------------------------------------------------------

    candidates = (
        candidates
        .withColumn(
            "repeat_rr",
            F.when(
                F.col("repeat_rank").isNotNull(),
                1.0 / F.col("repeat_rank"),
            )
            .otherwise(0.0)
            .cast("float"),
        )
        .withColumn(
            "als_rr",
            F.when(
                F.col("als_rank").isNotNull(),
                1.0 / F.col("als_rank"),
            )
            .otherwise(0.0)
            .cast("float"),
        )
        .withColumn(
            "content_rr",
            F.when(
                F.col("content_rank").isNotNull(),
                1.0 / F.col("content_rank"),
            )
            .otherwise(0.0)
            .cast("float"),
        )
        .withColumn(
            "two_tower_v1_rr",
            F.when(
                F.col(
                    "two_tower_v1_rank"
                ).isNotNull(),

                1.0
                / F.col(
                    "two_tower_v1_rank"
                ),
            )
            .otherwise(0.0)
            .cast("float"),
        )
        .withColumn(
            "two_tower_v2_rr",
            F.when(
                F.col(
                    "two_tower_v2_rank"
                ).isNotNull(),

                1.0
                / F.col(
                    "two_tower_v2_rank"
                ),
            )
            .otherwise(0.0)
            .cast("float"),
        )
    )

    return candidates


# ============================================================
# Historical behavior features
# ============================================================

def build_history_features(spark):

    print("\nLoading historical training interactions...")

    train = (
        spark.read
        .parquet(str(TRAIN_PATH))
        .select(
            F.col("user_id").cast("int"),
            F.col("product_id").cast("int"),
            F.col("order_number").cast("int"),
            F.col("reordered").cast("float"),
        )
    )

    print("Building user-product features...")

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

            F.avg("reordered")
            .cast("float")
            .alias(
                "user_product_reorder_rate"
            ),

            F.max("order_number")
            .cast("int")
            .alias(
                "last_purchase_order"
            ),
        )
    )

    print("Building user summary features...")

    user_summary = (
        train
        .groupBy("user_id")
        .agg(
            F.max("order_number")
            .cast("int")
            .alias(
                "user_latest_order"
            ),

            F.count("*")
            .cast("float")
            .alias(
                "user_total_interactions"
            ),

            F.countDistinct("product_id")
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
# Category affinity
# ============================================================

def build_category_affinities(
    train,
    item_features,
):

    print("Building user category affinities...")

    categories = (
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
            categories,
            on="product_id",
            how="left",
        )
    )

    user_totals = (
        enriched
        .groupBy("user_id")
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
                F.col("user_aisle_count")
                /
                F.col("category_user_total")
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


# ============================================================
# Labels
# ============================================================

def add_labels(
    spark,
    candidates,
):

    print("Adding official validation labels...")

    positives = (
        spark.read
        .parquet(str(VALIDATION_PATH))
        .select(
            F.col("user_id").cast("int"),
            F.col("product_id").cast("int"),
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

    return (
        candidates
        .join(
            positives,
            [
                "user_id",
                "product_id",
            ],
            "left",
        )
        .fillna(
            {
                "label": 0,
            }
        )
        .withColumn(
            "label",
            F.col("label").cast("int"),
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = create_spark_session()

    try:

        # ====================================================
        # 1. Candidate union
        # ====================================================

        candidates = build_candidate_union(
            spark
        )

        # ====================================================
        # 2. Historical features
        # ====================================================

        (
            train,
            user_product,
            user_summary,
        ) = build_history_features(
            spark
        )

        # ====================================================
        # 3. Item features
        # ====================================================

        print("\nLoading item feature store...")

        item_features = (
            spark.read
            .parquet(
                str(ITEM_FEATURE_PATH)
            )
            .select(
                F.col("product_id")
                .cast("int"),

                F.col("aisle_id")
                .cast("int"),

                F.col("department_id")
                .cast("int"),

                F.col("purchase_count")
                .cast("float")
                .alias(
                    "product_purchase_count"
                ),

                F.col("reorder_rate")
                .cast("float")
                .alias(
                    "product_reorder_rate"
                ),

                F.col("log_purchase_count")
                .cast("float")
                .alias(
                    "product_log_purchase_count"
                ),
            )
        )

        # ====================================================
        # 4. Category affinities
        # ====================================================

        (
            user_aisle,
            user_department,
        ) = build_category_affinities(
            train,
            item_features,
        )

        # ====================================================
        # 5. Join features
        # ====================================================

        print("Joining all LambdaMART v3 features...")

        features = (
            candidates
            .join(
                user_product,
                [
                    "user_id",
                    "product_id",
                ],
                "left",
            )
            .join(
                user_summary,
                "user_id",
                "left",
            )
            .join(
                item_features,
                "product_id",
                "left",
            )
            .join(
                user_aisle,
                [
                    "user_id",
                    "aisle_id",
                ],
                "left",
            )
            .join(
                user_department,
                [
                    "user_id",
                    "department_id",
                ],
                "left",
            )
        )

        # ====================================================
        # 6. Missing historical features
        # ====================================================

        features = features.fillna(
            {
                "user_product_purchase_count": 0.0,
                "user_product_reorder_rate": 0.0,
                "aisle_affinity": 0.0,
                "department_affinity": 0.0,
            }
        )

        # ====================================================
        # 7. Seen-before
        # ====================================================

        features = features.withColumn(
            "seen_before",
            (
                F.col(
                    "user_product_purchase_count"
                )
                > 0
            ).cast("int"),
        )

        # ====================================================
        # 8. User-relative recency
        # ====================================================

        features = features.withColumn(
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

        # ====================================================
        # 9. Labels
        # ====================================================

        features = add_labels(
            spark,
            features,
        )

        # ====================================================
        # 10. Missing retrieval ranks
        # ====================================================

        features = features.fillna(
            RANK_COLUMNS
        )

        # ====================================================
        # 11. Missing TT scores
        #
        # Both are cosine-like normalized scores.
        # -2 means candidate absent from that source.
        # ====================================================

        features = features.fillna(
            SCORE_COLUMNS
        )

        features = (
            features
            .withColumn(
                "two_tower_v1_score",
                F.col(
                    "two_tower_v1_score"
                ).cast("float"),
            )
            .withColumn(
                "two_tower_v2_score",
                F.col(
                    "two_tower_v2_score"
                ).cast("float"),
            )
        )

        # ====================================================
        # 12. Same deterministic user-level split
        # ====================================================

        print(
            "Creating deterministic user-level split..."
        )

        features = features.withColumn(
            "ranker_split",

            F.when(
                F.pmod(
                    F.xxhash64("user_id"),
                    F.lit(5),
                )
                == 0,

                F.lit("validation"),
            )
            .otherwise(
                F.lit("train")
            ),
        )

        # ====================================================
        # 13. Full ranker validation
        # ====================================================

        validation_features = (
            features
            .filter(
                F.col("ranker_split")
                == "validation"
            )
            .drop("ranker_split")
        )

        # ====================================================
        # 14. Training split
        # ====================================================

        train_features_raw = (
            features
            .filter(
                F.col("ranker_split")
                == "train"
            )
        )

        # ====================================================
        # 15. Correct negative sampling
        #
        # ALL positives
        # + up to exactly 15 negatives/user
        # ====================================================

        print(
            "Sampling corrected ranker negatives..."
        )

        train_positives = (
            train_features_raw
            .filter(
                F.col("label") == 1
            )
        )

        train_negatives = (
            train_features_raw
            .filter(
                F.col("label") == 0
            )
        )

        negative_window = (
            Window
            .partitionBy("user_id")
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

                F.row_number().over(
                    negative_window
                ),
            )
            .filter(
                F.col("negative_rank")
                <= NEGATIVES_PER_USER
            )
            .drop("negative_rank")
        )

        train_features = (
            train_positives
            .unionByName(
                sampled_negatives
            )
            .drop("ranker_split")
        )

        # ====================================================
        # 16. Write train
        # ====================================================

        print(
            "Writing LambdaMART v3 training data..."
        )

        train_count = (
            write_spark_df_local_parquet(
                df=train_features,
                output_dir=TRAIN_OUTPUT,
                num_partitions=32,
                partition_column="user_id",
            )
        )

        # ====================================================
        # 17. Write validation
        # ====================================================

        print(
            "Writing LambdaMART v3 validation data..."
        )

        val_count = (
            write_spark_df_local_parquet(
                df=validation_features,
                output_dir=VALIDATION_OUTPUT,
                num_partitions=16,
                partition_column="user_id",
            )
        )

        # ====================================================
        # 18. Statistics
        # ====================================================

        print("Computing dataset statistics...")

        train_users = (
            train_features
            .select("user_id")
            .distinct()
            .count()
        )

        val_users = (
            validation_features
            .select("user_id")
            .distinct()
            .count()
        )

        train_positive_count = (
            train_features
            .filter(
                F.col("label") == 1
            )
            .count()
        )

        validation_positive_count = (
            validation_features
            .filter(
                F.col("label") == 1
            )
            .count()
        )

        train_negative_count = (
            train_features
            .filter(
                F.col("label") == 0
            )
            .count()
        )

        validation_negative_count = (
            validation_features
            .filter(
                F.col("label") == 0
            )
            .count()
        )

        val_tt_v1_rows = (
            validation_features
            .filter(
                F.col(
                    "is_two_tower_v1_candidate"
                )
                == 1
            )
            .count()
        )

        val_tt_v2_rows = (
            validation_features
            .filter(
                F.col(
                    "is_two_tower_v2_candidate"
                )
                == 1
            )
            .count()
        )

        val_tt_v1_positive = (
            validation_features
            .filter(
                (
                    F.col(
                        "is_two_tower_v1_candidate"
                    )
                    == 1
                )
                &
                (
                    F.col("label") == 1
                )
            )
            .count()
        )

        val_tt_v2_positive = (
            validation_features
            .filter(
                (
                    F.col(
                        "is_two_tower_v2_candidate"
                    )
                    == 1
                )
                &
                (
                    F.col("label") == 1
                )
            )
            .count()
        )

        avg_validation_pool = (
            val_count
            /
            val_users
        )

        # ====================================================
        # Final report
        # ====================================================

        print(
            "\n"
            + "=" * 75
        )

        print(
            "LAMBDAMART V3 FIVE-SOURCE DATASET COMPLETE"
        )

        print(
            "=" * 75
        )

        print("\nTRAIN")

        print(
            f"Rows: {train_count:,}"
        )

        print(
            f"Users: {train_users:,}"
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
            "Average negatives/user: "
            f"{train_negative_count / train_users:.4f}"
        )

        print("\nVALIDATION")

        print(
            f"Rows: {val_count:,}"
        )

        print(
            f"Users: {val_users:,}"
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
            "Average candidate pool/user: "
            f"{avg_validation_pool:.2f}"
        )

        print("\nNEURAL SOURCE VALIDATION STATS")

        print(
            f"TT-v1 candidate rows: "
            f"{val_tt_v1_rows:,}"
        )

        print(
            f"TT-v1 positive rows: "
            f"{val_tt_v1_positive:,}"
        )

        print(
            f"TT-v2 candidate rows: "
            f"{val_tt_v2_rows:,}"
        )

        print(
            f"TT-v2 positive rows: "
            f"{val_tt_v2_positive:,}"
        )

        print("\nSaved under:")
        print(OUTPUT_DIR)

    finally:

        spark.stop()


if __name__ == "__main__":
    main()