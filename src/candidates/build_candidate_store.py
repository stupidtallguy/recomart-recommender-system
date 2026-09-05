from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession

from src.models.als import ALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.repeat_purchase import RepeatPurchaseRecommender


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

CANDIDATE_DIR = (
    PROCESSED_DIR
    / "candidates"
)

METADATA_PATH = (
    CANDIDATE_DIR
    / "candidate_store_metadata.json"
)

TOP_N = 50


def write_candidate_dict(
    recommendations: dict,
    path: Path,
    top_n: int,
    chunk_size: int = 20_000,
) -> None:
    """
    Persist user recommendation lists in batches.

    Schema:
        user_id
        recommendations: list[product_id]
    """

    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    if tmp_path.exists():
        tmp_path.unlink()

    writer = None

    try:

        items = list(
            recommendations.items()
        )

        for start in range(
            0,
            len(items),
            chunk_size,
        ):

            batch = items[
                start:
                start + chunk_size
            ]

            user_ids = []

            recommendation_lists = []

            for user_id, recs in batch:

                user_ids.append(
                    int(user_id)
                )

                recommendation_lists.append(
                    [
                        int(product_id)
                        for product_id
                        in recs[:top_n]
                    ]
                )

            table = pa.Table.from_pydict(
                {
                    "user_id":
                        user_ids,

                    "recommendations":
                        recommendation_lists,
                }
            )

            if writer is None:

                writer = (
                    pq.ParquetWriter(
                        tmp_path,
                        table.schema,
                        compression="snappy",
                    )
                )

            writer.write_table(
                table
            )

        if writer is None:

            raise RuntimeError(
                "No candidate rows written."
            )

        writer.close()

        tmp_path.replace(
            path
        )

    except Exception:

        if writer is not None:
            writer.close()

        if tmp_path.exists():
            tmp_path.unlink()

        raise


def create_spark_session():

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(
            "RecoMart-Candidate-Store"
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


def build_repeat_candidates():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING REPEAT-PURCHASE CANDIDATES"
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "user_id",
            "product_id",
            "order_number",
            "reordered",
        ],
    )

    model = RepeatPurchaseRecommender()

    model.fit(
        train
    )

    output = (
        CANDIDATE_DIR
        / "repeat_purchase_top50.parquet"
    )

    write_candidate_dict(
        recommendations=
            model.user_products,
        path=output,
        top_n=TOP_N,
    )

    users = len(
        model.user_products
    )

    print(
        f"Saved {users:,} user "
        "candidate lists."
    )

    del train
    del model

    gc.collect()

    return users


def build_content_candidates():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING CONTENT CANDIDATES"
    )

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    features = pd.read_parquet(
        ITEM_FEATURE_PATH
    )

    model = ContentBasedRecommender(
        top_aisles=5,
        top_departments=3,
        products_per_aisle=100,
        products_per_department=100,
        precompute_k=TOP_N,
        department_weight=0.5,
    )

    model.fit(
        interactions=train,
        item_features=features,
    )

    output = (
        CANDIDATE_DIR
        / "content_v1_top50.parquet"
    )

    write_candidate_dict(
        recommendations=
            model.recommendations,
        path=output,
        top_n=TOP_N,
    )

    users = len(
        model.recommendations
    )

    print(
        f"Saved {users:,} user "
        "candidate lists."
    )

    del train
    del features
    del model

    gc.collect()

    return users


def build_als_candidates():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING ALS-V2 CANDIDATES"
    )

    spark = create_spark_session()

    try:

        model = ALSRecommender(
            rank=64,
            max_iter=10,
            reg_param=0.05,
            alpha=5.0,
            precompute_k=TOP_N,
            seed=42,
        )

        model.fit(
            spark=spark,
            train_path=TRAIN_PATH,
        )

        output = (
            CANDIDATE_DIR
            / "als_v2_top50.parquet"
        )

        write_candidate_dict(
            recommendations=
                model.recommendations,
            path=output,
            top_n=TOP_N,
        )

        users = len(
            model.recommendations
        )

        print(
            f"Saved {users:,} user "
            "candidate lists."
        )

        return users

    finally:

        spark.stop()

        gc.collect()


def main():

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    repeat_users = (
        build_repeat_candidates()
    )

    content_users = (
        build_content_candidates()
    )

    als_users = (
        build_als_candidates()
    )

    metadata = {
        "top_n_per_source":
            TOP_N,

        "sources": {

            "repeat_purchase": {
                "users":
                    repeat_users,

                "file":
                    "repeat_purchase_top50.parquet",
            },

            "content_v1": {
                "users":
                    content_users,

                "file":
                    "content_v1_top50.parquet",
            },

            "als_v2": {
                "users":
                    als_users,

                "file":
                    "als_v2_top50.parquet",

                "rank":
                    64,

                "alpha":
                    5.0,

                "reg_param":
                    0.05,
            },
        },
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
        "CANDIDATE STORE COMPLETE"
    )

    print(
        f"Saved under:\n"
        f"{CANDIDATE_DIR}"
    )


if __name__ == "__main__":
    main()