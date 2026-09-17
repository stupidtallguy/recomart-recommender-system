from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


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

VALIDATION_TRUTH_PATH = (
    PROCESSED_DIR
    / "validation_interactions.parquet"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "ranking_segment_diagnostics.json"
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 10

CATALOG_SIZE = 49_641


V1_FEATURES = [
    "repeat_rank",
    "als_rank",
    "content_rank",

    "repeat_rr",
    "als_rr",
    "content_rr",

    "is_repeat_candidate",
    "is_als_candidate",
    "is_content_candidate",
    "source_count",

    "user_product_purchase_count",
    "user_product_reorder_rate",
    "orders_since_last_purchase",
    "seen_before",

    "user_total_interactions",
    "user_unique_products",
    "user_latest_order",

    "product_purchase_count",
    "product_reorder_rate",
    "product_log_purchase_count",

    "aisle_affinity",
    "department_affinity",
]


V2_FEATURES = [
    "repeat_rank",
    "als_rank",
    "content_rank",
    "two_tower_rank",

    "repeat_rr",
    "als_rr",
    "content_rr",
    "two_tower_rr",

    "is_repeat_candidate",
    "is_als_candidate",
    "is_content_candidate",
    "is_two_tower_candidate",
    "source_count",

    "two_tower_score",

    "user_product_purchase_count",
    "user_product_reorder_rate",
    "orders_since_last_purchase",
    "seen_before",

    "user_total_interactions",
    "user_unique_products",
    "user_latest_order",

    "product_purchase_count",
    "product_reorder_rate",
    "product_log_purchase_count",

    "aisle_affinity",
    "department_affinity",
]


V3_FEATURES = [
    "repeat_rank",
    "als_rank",
    "content_rank",
    "two_tower_v1_rank",
    "two_tower_v2_rank",

    "repeat_rr",
    "als_rr",
    "content_rr",
    "two_tower_v1_rr",
    "two_tower_v2_rr",

    "is_repeat_candidate",
    "is_als_candidate",
    "is_content_candidate",
    "is_two_tower_v1_candidate",
    "is_two_tower_v2_candidate",
    "source_count",

    "two_tower_v1_score",
    "two_tower_v2_score",

    "user_product_purchase_count",
    "user_product_reorder_rate",
    "orders_since_last_purchase",
    "seen_before",

    "user_total_interactions",
    "user_unique_products",
    "user_latest_order",

    "product_purchase_count",
    "product_reorder_rate",
    "product_log_purchase_count",

    "aisle_affinity",
    "department_affinity",
]


MODEL_CONFIGS = {

    "lambda_v1_corrected": {

        "ranker_dir":
            PROCESSED_DIR
            / "ranker_v1_corrected",

        "model_path":
            PROJECT_ROOT
            / "models"
            / "xgb_ranker_v1_corrected.json",

        "benchmark_path":
            PROJECT_ROOT
            / "results"
            / "benchmarks"
            / "xgb_ranker_v1_corrected.json",

        "features":
            V1_FEATURES,
    },

    "lambda_v2_tt_v1": {

        "ranker_dir":
            PROCESSED_DIR
            / "ranker_v2",

        "model_path":
            PROJECT_ROOT
            / "models"
            / "xgb_ranker_v2.json",

        "benchmark_path":
            PROJECT_ROOT
            / "results"
            / "benchmarks"
            / "xgb_ranker_v2.json",

        "features":
            V2_FEATURES,
    },

    "lambda_v3_tt_v1_tt_v2": {

        "ranker_dir":
            PROCESSED_DIR
            / "ranker_v3",

        "model_path":
            PROJECT_ROOT
            / "models"
            / "xgb_ranker_v3.json",

        "benchmark_path":
            PROJECT_ROOT
            / "results"
            / "benchmarks"
            / "xgb_ranker_v3.json",

        "features":
            V3_FEATURES,
    },
}


# ============================================================
# History bucket
# ============================================================

def history_bucket(
    latest_order: int,
) -> str:

    if latest_order <= 5:
        return "01_<=5_orders"

    if latest_order <= 10:
        return "02_6_10_orders"

    if latest_order <= 20:
        return "03_11_20_orders"

    return "04_>20_orders"


# ============================================================
# Load official truth
# ============================================================

def load_truth():

    print(
        "Loading official validation truth..."
    )

    df = pd.read_parquet(
        VALIDATION_TRUTH_PATH,
        columns=[
            "user_id",
            "product_id",
            "reordered",
        ],
    )

    df["user_id"] = (
        df["user_id"]
        .astype(np.int32)
    )

    df["product_id"] = (
        df["product_id"]
        .astype(np.int32)
    )

    df["reordered"] = (
        df["reordered"]
        .fillna(0)
        .astype(np.int8)
    )

    all_truth = {}

    repeat_truth = {}

    novel_truth = {}

    for (
        user_id,
        user_frame,
    ) in df.groupby(
        "user_id",
        sort=False,
    ):

        user_id = int(
            user_id
        )

        all_products = set(
            int(x)
            for x
            in user_frame[
                "product_id"
            ]
        )

        repeat_products = set(
            int(x)
            for x
            in user_frame.loc[
                user_frame[
                    "reordered"
                ]
                == 1,
                "product_id",
            ]
        )

        novel_products = (
            all_products
            -
            repeat_products
        )

        all_truth[
            user_id
        ] = all_products

        repeat_truth[
            user_id
        ] = repeat_products

        novel_truth[
            user_id
        ] = novel_products

    print(
        f"Users: {len(all_truth):,}"
    )

    return (
        all_truth,
        repeat_truth,
        novel_truth,
    )


# ============================================================
# Prediction generation
# ============================================================

def generate_predictions(
    name: str,
    config: dict,
):

    validation_path = (
        config[
            "ranker_dir"
        ]
        / "ranker_validation"
    )

    features = config[
        "features"
    ]

    columns = (
        [
            "user_id",
            "product_id",
            "seen_before",
            "user_latest_order",
        ]
        +
        features
    )

    # Remove duplicate column names.
    columns = list(
        dict.fromkeys(
            columns
        )
    )

    print(
        f"\nLoading validation candidates "
        f"for {name}..."
    )

    df = pd.read_parquet(
        validation_path,
        columns=columns,
    )

    print(
        f"Candidate rows: "
        f"{len(df):,}"
    )

    df["user_id"] = (
        df["user_id"]
        .astype(np.int32)
    )

    df["product_id"] = (
        df["product_id"]
        .astype(np.int32)
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    booster = xgb.Booster()

    booster.load_model(
        config[
            "model_path"
        ]
    )

    benchmark = json.loads(
        config[
            "benchmark_path"
        ]
        .read_text(
            encoding="utf-8"
        )
    )

    best_iteration = int(
        benchmark[
            "best_iteration"
        ]
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print(
        f"Predicting {name}..."
    )

    matrix = xgb.DMatrix(
        df[
            features
        ].astype(
            np.float32
        ),
        feature_names=
            features,
    )

    scores = booster.predict(
        matrix,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )

    df["score"] = (
        scores.astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Top-10
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "user_id",
                "score",
                "product_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="mergesort",
        )
    )

    topk = (
        df
        .groupby(
            "user_id",
            sort=False,
        )
        .head(
            TOP_K
        )
        [
            [
                "user_id",
                "product_id",
                "seen_before",
                "user_latest_order",
                "score",
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Candidate pool sets for ceiling diagnostics
    # --------------------------------------------------------

    candidate_sets = (
        df
        .groupby(
            "user_id",
            sort=False,
        )[
            "product_id"
        ]
        .apply(
            lambda values:
                set(
                    int(x)
                    for x
                    in values
                )
        )
        .to_dict()
    )

    user_latest_order = (
        df
        .groupby(
            "user_id",
            sort=False,
        )[
            "user_latest_order"
        ]
        .first()
        .astype(int)
        .to_dict()
    )

    del matrix
    del scores
    del df

    gc.collect()

    return (
        topk,
        candidate_sets,
        user_latest_order,
    )


# ============================================================
# Metric calculation
# ============================================================

def evaluate_model(
    topk: pd.DataFrame,
    candidate_sets: dict,
    latest_order: dict,
    all_truth: dict,
    repeat_truth: dict,
    novel_truth: dict,
):

    predictions = (
        topk
        .groupby(
            "user_id",
            sort=False,
        )[
            "product_id"
        ]
        .apply(
            lambda values:
                [
                    int(x)
                    for x
                    in values
                ]
        )
        .to_dict()
    )

    # ============================================================
    # IMPORTANT:
    # Evaluate ONLY users belonging to the deterministic
    # LambdaMART validation split.
    #
    # Ranker validation files contain ~41,114 users, not all
    # 206,209 users in the official temporal validation set.
    # ============================================================

    candidate_users = set(
        int(user_id)
        for user_id
        in candidate_sets.keys()
    )

    truth_users = set(
        int(user_id)
        for user_id
        in all_truth.keys()
    )

    users = sorted(
        candidate_users
        &
        truth_users
    )

    if not users:
        raise RuntimeError(
            "No overlapping users between ranker "
            "validation candidates and official truth."
        )

    print(
        f"Evaluation users: {len(users):,}"
    )

    # Every evaluated ranker user should also have its
    # historical-order information available.
    missing_latest_order = [
        user_id
        for user_id
        in users
        if user_id not in latest_order
    ]

    if missing_latest_order:
        raise RuntimeError(
            f"{len(missing_latest_order):,} evaluated users "
            "are missing user_latest_order."
        )

    recall_sum = 0.0

    repeat_recall_sum = 0.0
    repeat_user_count = 0

    novel_recall_sum = 0.0
    novel_user_count = 0

    novel_hit_users = 0

    candidate_recall_sum = 0.0

    candidate_repeat_recall_sum = 0.0
    candidate_repeat_users = 0

    candidate_novel_recall_sum = 0.0
    candidate_novel_users = 0

    bucket_stats = {}

    recommended_products = set()

    total_recommendations = 0
    total_seen_recommendations = 0

    for user_id in users:

        truth = (
            all_truth[
                user_id
            ]
        )

        truth_repeat = (
            repeat_truth[
                user_id
            ]
        )

        truth_novel = (
            novel_truth[
                user_id
            ]
        )

        predicted = predictions.get(
            user_id,
            [],
        )

        predicted_set = set(
            predicted
        )

        candidate_set = (
            candidate_sets.get(
                user_id,
                set(),
            )
        )

        # ----------------------------------------------------
        # Overall ranking recall
        # ----------------------------------------------------

        hits = len(
            predicted_set
            &
            truth
        )

        recall = (
            hits
            /
            len(truth)
            if truth
            else 0.0
        )

        recall_sum += recall

        # ----------------------------------------------------
        # Repeat-item recall
        # ----------------------------------------------------

        if truth_repeat:

            repeat_hits = len(
                predicted_set
                &
                truth_repeat
            )

            repeat_recall_sum += (
                repeat_hits
                /
                len(
                    truth_repeat
                )
            )

            repeat_user_count += 1

        # ----------------------------------------------------
        # Novel-item recall
        # ----------------------------------------------------

        if truth_novel:

            novel_hits = len(
                predicted_set
                &
                truth_novel
            )

            novel_recall_sum += (
                novel_hits
                /
                len(
                    truth_novel
                )
            )

            novel_user_count += 1

            if novel_hits > 0:

                novel_hit_users += 1

        # ----------------------------------------------------
        # Candidate ceiling — overall
        # ----------------------------------------------------

        if truth:

            candidate_recall_sum += (
                len(
                    candidate_set
                    &
                    truth
                )
                /
                len(
                    truth
                )
            )

        # ----------------------------------------------------
        # Candidate ceiling — repeat
        # ----------------------------------------------------

        if truth_repeat:

            candidate_repeat_recall_sum += (
                len(
                    candidate_set
                    &
                    truth_repeat
                )
                /
                len(
                    truth_repeat
                )
            )

            candidate_repeat_users += 1

        # ----------------------------------------------------
        # Candidate ceiling — novel
        # ----------------------------------------------------

        if truth_novel:

            candidate_novel_recall_sum += (
                len(
                    candidate_set
                    &
                    truth_novel
                )
                /
                len(
                    truth_novel
                )
            )

            candidate_novel_users += 1

        # ----------------------------------------------------
        # History bucket
        # ----------------------------------------------------

        bucket = history_bucket(
            int(
                latest_order.get(
                    user_id,
                    0,
                )
            )
        )

        if bucket not in bucket_stats:

            bucket_stats[
                bucket
            ] = {
                "users": 0,
                "recall_sum": 0.0,
                "novel_users": 0,
                "novel_recall_sum": 0.0,
            }

        bucket_stats[
            bucket
        ][
            "users"
        ] += 1

        bucket_stats[
            bucket
        ][
            "recall_sum"
        ] += recall

        if truth_novel:

            bucket_stats[
                bucket
            ][
                "novel_users"
            ] += 1

            bucket_stats[
                bucket
            ][
                "novel_recall_sum"
            ] += (
                len(
                    predicted_set
                    &
                    truth_novel
                )
                /
                len(
                    truth_novel
                )
            )

    # --------------------------------------------------------
    # Recommendation composition / coverage
    # --------------------------------------------------------

    recommended_products = set(
        int(x)
        for x
        in topk[
            "product_id"
        ]
    )

    total_recommendations = len(
        topk
    )

    total_seen_recommendations = int(
        (
            topk[
                "seen_before"
            ]
            > 0
        )
        .sum()
    )

    finalized_buckets = {}

    for bucket, values in sorted(
        bucket_stats.items()
    ):

        finalized_buckets[
            bucket
        ] = {

            "users":
                values[
                    "users"
                ],

            "recall_at_10":
                (
                    values[
                        "recall_sum"
                    ]
                    /
                    values[
                        "users"
                    ]
                ),

            "users_with_novel_truth":
                values[
                    "novel_users"
                ],

            "novel_recall_at_10":
                (
                    values[
                        "novel_recall_sum"
                    ]
                    /
                    values[
                        "novel_users"
                    ]

                    if values[
                        "novel_users"
                    ]
                    else 0.0
                ),
        }

    return {

        "users":
            len(
                users
            ),

        "ranking": {

            "overall_recall_at_10":
                recall_sum
                /
                len(
                    users
                ),

            "repeat_recall_at_10":
                repeat_recall_sum
                /
                repeat_user_count,

            "novel_recall_at_10":
                novel_recall_sum
                /
                novel_user_count,

            "novel_hit_rate":
                novel_hit_users
                /
                novel_user_count,
        },

        "candidate_ceiling": {

            "overall_candidate_recall":
                candidate_recall_sum
                /
                len(
                    users
                ),

            "repeat_candidate_recall":
                candidate_repeat_recall_sum
                /
                candidate_repeat_users,

            "novel_candidate_recall":
                candidate_novel_recall_sum
                /
                candidate_novel_users,
        },

        "recommendation_composition": {

            "unique_products_recommended":
                len(
                    recommended_products
                ),

            "catalog_coverage_at_10":
                len(
                    recommended_products
                )
                /
                CATALOG_SIZE,

            "fraction_recommendations_seen_before":
                total_seen_recommendations
                /
                total_recommendations,

            "fraction_recommendations_novel":
                1.0
                -
                (
                    total_seen_recommendations
                    /
                    total_recommendations
                ),
        },

        "history_buckets":
            finalized_buckets,
    }


# ============================================================
# Main
# ============================================================

def main():

    (
        all_truth,
        repeat_truth,
        novel_truth,
    ) = load_truth()

    result = {

        "top_k":
            TOP_K,

        "catalog_size":
            CATALOG_SIZE,

        "models":
            {},
    }

    for (
        name,
        config,
    ) in MODEL_CONFIGS.items():

        print(
            "\n"
            + "=" * 75
        )

        print(
            f"DIAGNOSING {name}"
        )

        print(
            "=" * 75
        )

        (
            topk,
            candidate_sets,
            latest_order,
        ) = generate_predictions(
            name,
            config,
        )

        metrics = evaluate_model(
            topk=
                topk,

            candidate_sets=
                candidate_sets,

            latest_order=
                latest_order,

            all_truth=
                all_truth,

            repeat_truth=
                repeat_truth,

            novel_truth=
                novel_truth,
        )

        result[
            "models"
        ][
            name
        ] = metrics

        print(
            json.dumps(
                metrics,
                indent=4,
            )
        )

        del topk
        del candidate_sets
        del latest_order

        gc.collect()

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RANKING SEGMENT DIAGNOSTICS COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"\nSaved:\n{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()