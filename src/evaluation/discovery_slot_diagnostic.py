from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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

RANKER_DIR = (
    PROCESSED_DIR
    / "ranker_v3"
)

VALIDATION_PATH = (
    RANKER_DIR
    / "ranker_validation"
)

OFFICIAL_VALIDATION_PATH = (
    PROCESSED_DIR
    / "validation_interactions.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgb_ranker_v3.json"
)

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "xgb_ranker_v3.json"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "discovery_slot_diagnostic.json"
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 10

CATALOG_SIZE = 49_641

# Minimum number of unseen / novel recommendations that
# must appear in final Top-10.
DISCOVERY_SLOT_POLICIES = [
    0,
    1,
    2,
    3,
]


# ============================================================
# LambdaMART v3 feature columns
# ============================================================

FEATURE_COLUMNS = [

    # --------------------------------------------------------
    # Retrieval ranks
    # --------------------------------------------------------

    "repeat_rank",
    "als_rank",
    "content_rank",
    "two_tower_v1_rank",
    "two_tower_v2_rank",

    # --------------------------------------------------------
    # Reciprocal ranks
    # --------------------------------------------------------

    "repeat_rr",
    "als_rr",
    "content_rr",
    "two_tower_v1_rr",
    "two_tower_v2_rr",

    # --------------------------------------------------------
    # Candidate source membership
    # --------------------------------------------------------

    "is_repeat_candidate",
    "is_als_candidate",
    "is_content_candidate",
    "is_two_tower_v1_candidate",
    "is_two_tower_v2_candidate",

    "source_count",

    # --------------------------------------------------------
    # Neural retrieval scores
    # --------------------------------------------------------

    "two_tower_v1_score",
    "two_tower_v2_score",

    # --------------------------------------------------------
    # User-product behavior
    # --------------------------------------------------------

    "user_product_purchase_count",
    "user_product_reorder_rate",
    "orders_since_last_purchase",
    "seen_before",

    # --------------------------------------------------------
    # User behavior
    # --------------------------------------------------------

    "user_total_interactions",
    "user_unique_products",
    "user_latest_order",

    # --------------------------------------------------------
    # Product features
    # --------------------------------------------------------

    "product_purchase_count",
    "product_reorder_rate",
    "product_log_purchase_count",

    # --------------------------------------------------------
    # Category affinities
    # --------------------------------------------------------

    "aisle_affinity",
    "department_affinity",
]


# ============================================================
# Utility: history bucket
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
# Ranking metrics
# ============================================================

def dcg_at_k(
    recommendations: list[int],
    truth: set[int],
    k: int,
) -> float:

    score = 0.0

    for rank, product_id in enumerate(
        recommendations[:k]
    ):

        if product_id in truth:

            score += (
                1.0
                /
                np.log2(
                    rank + 2
                )
            )

    return float(score)


def ndcg_at_k(
    recommendations: list[int],
    truth: set[int],
    k: int,
) -> float:

    if not truth:
        return 0.0

    actual_dcg = dcg_at_k(
        recommendations,
        truth,
        k,
    )

    ideal_hits = min(
        len(truth),
        k,
    )

    ideal_dcg = sum(

        1.0
        /
        np.log2(
            rank + 2
        )

        for rank
        in range(
            ideal_hits
        )
    )

    if ideal_dcg <= 0:
        return 0.0

    return float(
        actual_dcg
        /
        ideal_dcg
    )


# ============================================================
# Load official validation truth
# ============================================================

def load_truth():

    print(
        "Loading official validation truth..."
    )

    df = pd.read_parquet(
        OFFICIAL_VALIDATION_PATH,
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
        f"Official validation users: "
        f"{len(all_truth):,}"
    )

    return (
        all_truth,
        repeat_truth,
        novel_truth,
    )


# ============================================================
# Load + score LambdaMART v3 candidates
# ============================================================

def load_scored_candidates():

    load_columns = list(
        dict.fromkeys(
            [
                "user_id",
                "product_id",
                "seen_before",
                "user_latest_order",
            ]
            +
            FEATURE_COLUMNS
        )
    )

    print(
        "\nLoading LambdaMART v3 "
        "validation candidate pool..."
    )

    df = pd.read_parquet(
        VALIDATION_PATH,
        columns=load_columns,
    )

    print(
        f"Candidate rows: "
        f"{len(df):,}"
    )

    print(
        f"Candidate users: "
        f"{df['user_id'].nunique():,}"
    )

    df[
        "user_id"
    ] = (
        df[
            "user_id"
        ]
        .astype(
            np.int32
        )
    )

    df[
        "product_id"
    ] = (
        df[
            "product_id"
        ]
        .astype(
            np.int32
        )
    )

    df[
        "seen_before"
    ] = (
        df[
            "seen_before"
        ]
        .astype(
            np.int8
        )
    )

    # --------------------------------------------------------
    # Load trained LambdaMART
    # --------------------------------------------------------

    booster = xgb.Booster()

    booster.load_model(
        MODEL_PATH
    )

    benchmark = json.loads(
        BENCHMARK_PATH.read_text(
            encoding="utf-8"
        )
    )

    best_iteration = int(
        benchmark[
            "best_iteration"
        ]
    )

    print(
        f"Using checkpoint iteration: "
        f"{best_iteration}"
    )

    # --------------------------------------------------------
    # Predict scores
    # --------------------------------------------------------

    print(
        "Predicting LambdaMART scores..."
    )

    matrix = xgb.DMatrix(
        df[
            FEATURE_COLUMNS
        ].astype(
            np.float32
        ),
        feature_names=
            FEATURE_COLUMNS,
    )

    scores = booster.predict(
        matrix,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )

    df[
        "score"
    ] = scores.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Sort once globally
    # --------------------------------------------------------

    print(
        "Sorting candidate rankings..."
    )

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
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# Discovery quota reranker
# ============================================================

def rerank_with_minimum_novel_slots(
    candidate_frame: pd.DataFrame,
    minimum_novel_slots: int,
    k: int = 10,
) -> list[dict]:
    """
    Produce final Top-K while guaranteeing AT LEAST
    minimum_novel_slots unseen items when enough unseen
    candidates exist.

    Strategy
    --------
    1. Start with ordinary LambdaMART Top-K.
    2. Count how many novel items already appear.
    3. If quota is already satisfied, change nothing.
    4. Otherwise:
       - take highest-scoring novel candidate outside Top-K
       - replace lowest-ranked SEEN candidate in Top-K
       - repeat until quota is met.
    5. Re-sort selected items by original LambdaMART score.

    This makes the smallest possible intervention in the
    learned ranking.
    """

    records = (
        candidate_frame[
            [
                "product_id",
                "seen_before",
                "score",
            ]
        ]
        .to_dict(
            "records"
        )
    )

    if len(records) <= k:

        return records[
            :k
        ]

    selected = list(
        records[:k]
    )

    current_novel = sum(
        1
        for item
        in selected
        if int(
            item[
                "seen_before"
            ]
        )
        == 0
    )

    # No intervention needed.
    if (
        current_novel
        >= minimum_novel_slots
    ):

        return selected

    # --------------------------------------------------------
    # Novel candidates outside current Top-K
    # --------------------------------------------------------

    outside_novel = [

        item

        for item
        in records[k:]

        if int(
            item[
                "seen_before"
            ]
        )
        == 0
    ]

    if not outside_novel:

        return selected

    # Already score-sorted because input frame is sorted.
    outside_index = 0

    while (
        current_novel
        <
        minimum_novel_slots
    ):

        if (
            outside_index
            >=
            len(
                outside_novel
            )
        ):
            break

        replacement_candidate = (
            outside_novel[
                outside_index
            ]
        )

        outside_index += 1

        # ----------------------------------------------------
        # Find lowest-ranked SEEN item in current Top-K.
        # Search backwards to minimize ranking disruption.
        # ----------------------------------------------------

        replace_index = None

        for index in range(
            len(selected) - 1,
            -1,
            -1,
        ):

            if int(
                selected[
                    index
                ][
                    "seen_before"
                ]
            ) == 1:

                replace_index = index
                break

        # Nothing left that can be replaced without removing
        # an existing novel recommendation.
        if replace_index is None:
            break

        selected[
            replace_index
        ] = (
            replacement_candidate
        )

        current_novel += 1

    # --------------------------------------------------------
    # Keep original LambdaMART score ordering among selected
    # products.
    # --------------------------------------------------------

    selected = sorted(
        selected,
        key=lambda item:
            (
                -float(
                    item[
                        "score"
                    ]
                ),
                int(
                    item[
                        "product_id"
                    ]
                ),
            ),
    )

    return selected[:k]


# ============================================================
# Build policies for all users
# ============================================================

def build_policy_predictions(
    scored_candidates: pd.DataFrame,
):

    policies = {

        slots: {}

        for slots
        in DISCOVERY_SLOT_POLICIES
    }

    latest_orders = {}

    groups = (
        scored_candidates
        .groupby(
            "user_id",
            sort=False,
        )
    )

    print(
        "\nBuilding discovery-slot policies..."
    )

    for (
        user_id,
        user_frame,
    ) in tqdm(
        groups,
        total=
            scored_candidates[
                "user_id"
            ]
            .nunique(),
        desc="Discovery reranking",
    ):

        user_id = int(
            user_id
        )

        latest_orders[
            user_id
        ] = int(
            user_frame[
                "user_latest_order"
            ]
            .iloc[0]
        )

        for slots in (
            DISCOVERY_SLOT_POLICIES
        ):

            selected = (
                rerank_with_minimum_novel_slots(
                    candidate_frame=
                        user_frame,

                    minimum_novel_slots=
                        slots,

                    k=
                        TOP_K,
                )
            )

            policies[
                slots
            ][
                user_id
            ] = selected

    return (
        policies,
        latest_orders,
    )


# ============================================================
# Policy evaluation
# ============================================================

def evaluate_policy(
    predictions: dict[int, list[dict]],
    latest_orders: dict[int, int],
    all_truth: dict[int, set[int]],
    repeat_truth: dict[int, set[int]],
    novel_truth: dict[int, set[int]],
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Evaluate only the deterministic ranker validation users.
    # --------------------------------------------------------

    users = sorted(
        set(
            predictions.keys()
        )
        &
        set(
            all_truth.keys()
        )
    )

    if not users:

        raise RuntimeError(
            "No overlapping evaluation users."
        )

    print(
        f"Evaluation users: "
        f"{len(users):,}"
    )

    overall_recall_sum = 0.0
    precision_sum = 0.0
    ndcg_sum = 0.0

    repeat_recall_sum = 0.0
    repeat_user_count = 0

    novel_recall_sum = 0.0
    novel_user_count = 0

    novel_hit_users = 0

    # --------------------------------------------------------
    # Recommendation composition
    # --------------------------------------------------------

    unique_recommended_products = set()

    total_recommendations = 0
    total_novel_recommendations = 0
    total_seen_recommendations = 0

    # --------------------------------------------------------
    # How many users actually satisfy each number of novel
    # recommendations?
    # --------------------------------------------------------

    novel_count_histogram = {
        str(i): 0
        for i
        in range(
            0,
            TOP_K + 1,
        )
    }

    # --------------------------------------------------------
    # History-segment stats
    # --------------------------------------------------------

    bucket_stats = {}

    for user_id in users:

        selected = predictions[
            user_id
        ]

        recommended = [
            int(
                item[
                    "product_id"
                ]
            )
            for item
            in selected
        ]

        predicted_set = set(
            recommended
        )

        truth = all_truth[
            user_id
        ]

        truth_repeat = repeat_truth[
            user_id
        ]

        truth_novel = novel_truth[
            user_id
        ]

        # ====================================================
        # Overall metrics
        # ====================================================

        hits = len(
            predicted_set
            &
            truth
        )

        if truth:

            overall_recall = (
                hits
                /
                len(
                    truth
                )
            )

        else:

            overall_recall = 0.0

        overall_recall_sum += (
            overall_recall
        )

        # Standard Precision@K denominator = K.
        precision_sum += (
            hits
            /
            TOP_K
        )

        ndcg_sum += ndcg_at_k(
            recommendations=
                recommended,

            truth=
                truth,

            k=
                TOP_K,
        )

        # ====================================================
        # Repeat Recall
        # ====================================================

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

        # ====================================================
        # Novel Recall
        # ====================================================

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

        # ====================================================
        # Recommendation composition
        # ====================================================

        user_novel_count = 0

        for item in selected:

            product_id = int(
                item[
                    "product_id"
                ]
            )

            seen_before = int(
                item[
                    "seen_before"
                ]
            )

            unique_recommended_products.add(
                product_id
            )

            total_recommendations += 1

            if seen_before == 0:

                total_novel_recommendations += (
                    1
                )

                user_novel_count += 1

            else:

                total_seen_recommendations += (
                    1
                )

        novel_count_histogram[
            str(
                user_novel_count
            )
        ] += 1

        # ====================================================
        # History bucket
        # ====================================================

        bucket = history_bucket(
            latest_orders[
                user_id
            ]
        )

        if bucket not in bucket_stats:

            bucket_stats[
                bucket
            ] = {

                "users":
                    0,

                "overall_recall_sum":
                    0.0,

                "precision_sum":
                    0.0,

                "ndcg_sum":
                    0.0,

                "repeat_users":
                    0,

                "repeat_recall_sum":
                    0.0,

                "novel_users":
                    0,

                "novel_recall_sum":
                    0.0,

                "novel_hit_users":
                    0,

                "novel_recommendations":
                    0,

                "recommendations":
                    0,
            }

        stats = bucket_stats[
            bucket
        ]

        stats[
            "users"
        ] += 1

        stats[
            "overall_recall_sum"
        ] += overall_recall

        stats[
            "precision_sum"
        ] += (
            hits
            /
            TOP_K
        )

        stats[
            "ndcg_sum"
        ] += ndcg_at_k(
            recommended,
            truth,
            TOP_K,
        )

        if truth_repeat:

            stats[
                "repeat_users"
            ] += 1

            stats[
                "repeat_recall_sum"
            ] += (
                len(
                    predicted_set
                    &
                    truth_repeat
                )
                /
                len(
                    truth_repeat
                )
            )

        if truth_novel:

            stats[
                "novel_users"
            ] += 1

            novel_hits = len(
                predicted_set
                &
                truth_novel
            )

            stats[
                "novel_recall_sum"
            ] += (
                novel_hits
                /
                len(
                    truth_novel
                )
            )

            if novel_hits > 0:

                stats[
                    "novel_hit_users"
                ] += 1

        stats[
            "novel_recommendations"
        ] += user_novel_count

        stats[
            "recommendations"
        ] += len(
            recommended
        )

    # ========================================================
    # Finalize bucket metrics
    # ========================================================

    finalized_buckets = {}

    for (
        bucket,
        stats,
    ) in sorted(
        bucket_stats.items()
    ):

        finalized_buckets[
            bucket
        ] = {

            "users":
                stats[
                    "users"
                ],

            "overall_recall_at_10":
                stats[
                    "overall_recall_sum"
                ]
                /
                stats[
                    "users"
                ],

            "precision_at_10":
                stats[
                    "precision_sum"
                ]
                /
                stats[
                    "users"
                ],

            "ndcg_at_10":
                stats[
                    "ndcg_sum"
                ]
                /
                stats[
                    "users"
                ],

            "repeat_recall_at_10":
                (
                    stats[
                        "repeat_recall_sum"
                    ]
                    /
                    stats[
                        "repeat_users"
                    ]

                    if stats[
                        "repeat_users"
                    ]
                    else 0.0
                ),

            "novel_recall_at_10":
                (
                    stats[
                        "novel_recall_sum"
                    ]
                    /
                    stats[
                        "novel_users"
                    ]

                    if stats[
                        "novel_users"
                    ]
                    else 0.0
                ),

            "novel_hit_rate":
                (
                    stats[
                        "novel_hit_users"
                    ]
                    /
                    stats[
                        "novel_users"
                    ]

                    if stats[
                        "novel_users"
                    ]
                    else 0.0
                ),

            "fraction_recommendations_novel":
                (
                    stats[
                        "novel_recommendations"
                    ]
                    /
                    stats[
                        "recommendations"
                    ]

                    if stats[
                        "recommendations"
                    ]
                    else 0.0
                ),
        }

    # ========================================================
    # Final result
    # ========================================================

    return {

        "users":
            len(
                users
            ),

        "overall": {

            "recall_at_10":
                overall_recall_sum
                /
                len(
                    users
                ),

            "precision_at_10":
                precision_sum
                /
                len(
                    users
                ),

            "ndcg_at_10":
                ndcg_sum
                /
                len(
                    users
                ),
        },

        "repeat": {

            "users_with_repeat_truth":
                repeat_user_count,

            "repeat_recall_at_10":
                repeat_recall_sum
                /
                repeat_user_count,
        },

        "novel": {

            "users_with_novel_truth":
                novel_user_count,

            "novel_recall_at_10":
                novel_recall_sum
                /
                novel_user_count,

            "novel_hit_rate":
                novel_hit_users
                /
                novel_user_count,
        },

        "composition": {

            "unique_products_recommended":
                len(
                    unique_recommended_products
                ),

            "catalog_coverage_at_10":
                len(
                    unique_recommended_products
                )
                /
                CATALOG_SIZE,

            "fraction_recommendations_seen_before":
                total_seen_recommendations
                /
                total_recommendations,

            "fraction_recommendations_novel":
                total_novel_recommendations
                /
                total_recommendations,

            "mean_novel_recommendations_per_user":
                total_novel_recommendations
                /
                len(
                    users
                ),

            "novel_count_histogram":
                novel_count_histogram,
        },

        "history_buckets":
            finalized_buckets,
    }


# ============================================================
# Comparison helper
# ============================================================

def build_comparison(
    policy_results: dict,
):

    baseline = policy_results[
        "minimum_0_novel_slots"
    ]

    baseline_overall = (
        baseline[
            "overall"
        ]
    )

    baseline_repeat = (
        baseline[
            "repeat"
        ][
            "repeat_recall_at_10"
        ]
    )

    baseline_novel = (
        baseline[
            "novel"
        ][
            "novel_recall_at_10"
        ]
    )

    baseline_hit_rate = (
        baseline[
            "novel"
        ][
            "novel_hit_rate"
        ]
    )

    comparison = {}

    for (
        policy_name,
        metrics,
    ) in policy_results.items():

        comparison[
            policy_name
        ] = {

            "recall_at_10":
                metrics[
                    "overall"
                ][
                    "recall_at_10"
                ],

            "delta_recall_at_10":
                (
                    metrics[
                        "overall"
                    ][
                        "recall_at_10"
                    ]
                    -
                    baseline_overall[
                        "recall_at_10"
                    ]
                ),

            "precision_at_10":
                metrics[
                    "overall"
                ][
                    "precision_at_10"
                ],

            "delta_precision_at_10":
                (
                    metrics[
                        "overall"
                    ][
                        "precision_at_10"
                    ]
                    -
                    baseline_overall[
                        "precision_at_10"
                    ]
                ),

            "ndcg_at_10":
                metrics[
                    "overall"
                ][
                    "ndcg_at_10"
                ],

            "delta_ndcg_at_10":
                (
                    metrics[
                        "overall"
                    ][
                        "ndcg_at_10"
                    ]
                    -
                    baseline_overall[
                        "ndcg_at_10"
                    ]
                ),

            "repeat_recall_at_10":
                metrics[
                    "repeat"
                ][
                    "repeat_recall_at_10"
                ],

            "delta_repeat_recall":
                (
                    metrics[
                        "repeat"
                    ][
                        "repeat_recall_at_10"
                    ]
                    -
                    baseline_repeat
                ),

            "novel_recall_at_10":
                metrics[
                    "novel"
                ][
                    "novel_recall_at_10"
                ],

            "delta_novel_recall":
                (
                    metrics[
                        "novel"
                    ][
                        "novel_recall_at_10"
                    ]
                    -
                    baseline_novel
                ),

            "relative_novel_recall_gain":
                (
                    (
                        metrics[
                            "novel"
                        ][
                            "novel_recall_at_10"
                        ]
                        /
                        baseline_novel
                    )
                    -
                    1.0

                    if baseline_novel > 0

                    else 0.0
                ),

            "novel_hit_rate":
                metrics[
                    "novel"
                ][
                    "novel_hit_rate"
                ],

            "delta_novel_hit_rate":
                (
                    metrics[
                        "novel"
                    ][
                        "novel_hit_rate"
                    ]
                    -
                    baseline_hit_rate
                ),

            "fraction_recommendations_novel":
                metrics[
                    "composition"
                ][
                    "fraction_recommendations_novel"
                ],

            "catalog_coverage_at_10":
                metrics[
                    "composition"
                ][
                    "catalog_coverage_at_10"
                ],
        }

    return comparison


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Official truth
    # ========================================================

    (
        all_truth,
        repeat_truth,
        novel_truth,
    ) = load_truth()

    # ========================================================
    # Existing five-source LambdaMART scores
    # ========================================================

    scored_candidates = (
        load_scored_candidates()
    )

    # ========================================================
    # Build all slot policies
    # ========================================================

    (
        policies,
        latest_orders,
    ) = build_policy_predictions(
        scored_candidates
    )

    # Free large candidate dataframe before metric stage.
    del scored_candidates

    # ========================================================
    # Evaluate
    # ========================================================

    policy_results = {}

    for slots in (
        DISCOVERY_SLOT_POLICIES
    ):

        print(
            "\n"
            + "=" * 75
        )

        print(
            f"EVALUATING MINIMUM "
            f"{slots} NOVEL SLOT(S)"
        )

        print(
            "=" * 75
        )

        metrics = evaluate_policy(

            predictions=
                policies[
                    slots
                ],

            latest_orders=
                latest_orders,

            all_truth=
                all_truth,

            repeat_truth=
                repeat_truth,

            novel_truth=
                novel_truth,
        )

        policy_name = (
            f"minimum_{slots}_novel_slots"
        )

        policy_results[
            policy_name
        ] = metrics

        print(
            json.dumps(
                metrics,
                indent=4,
            )
        )

    # ========================================================
    # Compact comparison
    # ========================================================

    comparison = (
        build_comparison(
            policy_results
        )
    )

    # ========================================================
    # Final artifact
    # ========================================================

    result = {

        "experiment":
            "discovery_slot_diagnostic",

        "model":
            "xgboost_lambdamart_v3_five_source",

        "top_k":
            TOP_K,

        "policy_definition":
            (
                "Minimum number of unseen candidates "
                "required in final Top-10. "
                "If quota is unmet, lowest-ranked seen "
                "items are replaced by highest-scoring "
                "novel candidates outside the original "
                "Top-10."
            ),

        "policies":
            policy_results,

        "comparison_vs_zero_slot_baseline":
            comparison,
    }

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

    # ========================================================
    # Human-readable comparison
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DISCOVERY SLOT COMPARISON"
    )

    print(
        "=" * 90
    )

    header = (
        f"{'Policy':<28}"
        f"{'Recall':>10}"
        f"{'NDCG':>10}"
        f"{'RepeatR':>10}"
        f"{'NovelR':>10}"
        f"{'NovelHit':>10}"
        f"{'Novel%':>10}"
    )

    print(
        header
    )

    print(
        "-" * 90
    )

    for (
        policy_name,
        metrics,
    ) in comparison.items():

        print(
            f"{policy_name:<28}"

            f"{metrics['recall_at_10']:>10.6f}"

            f"{metrics['ndcg_at_10']:>10.6f}"

            f"{metrics['repeat_recall_at_10']:>10.6f}"

            f"{metrics['novel_recall_at_10']:>10.6f}"

            f"{metrics['novel_hit_rate']:>10.6f}"

            f"{metrics['fraction_recommendations_novel']:>10.4%}"
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DELTAS VS NORMAL LAMBDAMART"
    )

    print(
        "=" * 90
    )

    for (
        policy_name,
        metrics,
    ) in comparison.items():

        print(
            f"\n{policy_name}"
        )

        print(
            f"  Delta Recall@10:       "
            f"{metrics['delta_recall_at_10']:+.6f}"
        )

        print(
            f"  Delta Precision@10:    "
            f"{metrics['delta_precision_at_10']:+.6f}"
        )

        print(
            f"  Delta NDCG@10:         "
            f"{metrics['delta_ndcg_at_10']:+.6f}"
        )

        print(
            f"  Delta Repeat Recall:   "
            f"{metrics['delta_repeat_recall']:+.6f}"
        )

        print(
            f"  Delta Novel Recall:    "
            f"{metrics['delta_novel_recall']:+.6f}"
        )

        print(
            f"  Relative Novel Delta:  "
            f"{metrics['relative_novel_recall_gain']:+.2%}"
        )

        print(
            f"  Novel Hit Rate:    "
            f"{metrics['novel_hit_rate']:.4%}"
        )

        print(
            f"  Novel Rec Share:   "
            f"{metrics['fraction_recommendations_novel']:.4%}"
        )

        print(
            f"  Catalog Coverage:  "
            f"{metrics['catalog_coverage_at_10']:.4%}"
        )

    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()
