from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

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

RANKER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ranker_v3"
)

TRAIN_PATH = (
    RANKER_DIR
    / "ranker_train"
)

VALIDATION_PATH = (
    RANKER_DIR
    / "ranker_validation"
)

OFFICIAL_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_interactions.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgb_ranker_v3.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "xgb_ranker_v3.json"
)

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "xgb_ranker_v3_feature_importance.csv"
)


# ============================================================
# Feature columns
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
    # Candidate-source membership
    # --------------------------------------------------------

    "is_repeat_candidate",
    "is_als_candidate",
    "is_content_candidate",
    "is_two_tower_v1_candidate",
    "is_two_tower_v2_candidate",

    "source_count",

    # --------------------------------------------------------
    # Neural similarity scores
    # --------------------------------------------------------

    "two_tower_v1_score",
    "two_tower_v2_score",

    # --------------------------------------------------------
    # User-product history
    # --------------------------------------------------------

    "user_product_purchase_count",
    "user_product_reorder_rate",
    "orders_since_last_purchase",
    "seen_before",

    # --------------------------------------------------------
    # User features
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
    # User-category affinity
    # --------------------------------------------------------

    "aisle_affinity",
    "department_affinity",
]


LOAD_COLUMNS = (
    [
        "user_id",
        "product_id",
        "label",
    ]
    +
    FEATURE_COLUMNS
)


# ============================================================
# XGBoost configuration
#
# IMPORTANT:
# Keep these identical to previous LambdaMART experiments.
# ============================================================

XGB_PARAMS = {

    "objective":
        "rank:ndcg",

    "eval_metric":
        "ndcg@10",

    "tree_method":
        "hist",

    "max_bin":
        256,

    "eta":
        0.05,

    "max_depth":
        8,

    "min_child_weight":
        20,

    "subsample":
        0.8,

    "colsample_bytree":
        0.9,

    "reg_lambda":
        1.0,

    "reg_alpha":
        0.0,

    "lambdarank_pair_method":
        "topk",

    "lambdarank_num_pair_per_sample":
        12,

    "seed":
        42,
}


NUM_BOOST_ROUND = 800
EARLY_STOPPING_ROUNDS = 50
TOP_K = 10


# ============================================================
# Data loading
# ============================================================

def prepare_dataframe(
    path: Path,
) -> pd.DataFrame:

    print(f"Loading: {path}")

    df = pd.read_parquet(
        path,
        columns=LOAD_COLUMNS,
    )

    df["user_id"] = (
        df["user_id"]
        .astype(np.int32)
    )

    df["product_id"] = (
        df["product_id"]
        .astype(np.int32)
    )

    df["label"] = (
        df["label"]
        .astype(np.float32)
    )

    # Ranking queries MUST be contiguous.
    df = (
        df
        .sort_values(
            [
                "user_id",
                "product_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Keep only queries containing >=1 positive
# ============================================================

def remove_zero_positive_queries(
    df: pd.DataFrame,
) -> pd.DataFrame:

    positive_users = (
        df.loc[
            df["label"] > 0,
            "user_id",
        ]
        .unique()
    )

    filtered = (
        df[
            df["user_id"].isin(
                positive_users
            )
        ]
        .copy()
    )

    filtered = (
        filtered
        .sort_values(
            [
                "user_id",
                "product_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return filtered


# ============================================================
# Official validation truth
# ============================================================

def load_official_truth():

    print(
        "\nLoading official validation truth..."
    )

    validation = pd.read_parquet(
        OFFICIAL_VALIDATION_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    grouped = (
        validation
        .groupby(
            "user_id"
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
    )

    return {
        int(user_id):
            products

        for user_id, products
        in grouped.items()
    }


# ============================================================
# Ranking metrics
# ============================================================

def dcg_at_k(
    predictions: list[int],
    truth: set[int],
    k: int,
) -> float:

    score = 0.0

    for rank, product_id in enumerate(
        predictions[:k]
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
    predictions: list[int],
    truth: set[int],
    k: int,
) -> float:

    if not truth:
        return 0.0

    actual_dcg = dcg_at_k(
        predictions,
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

    if ideal_dcg == 0:
        return 0.0

    return float(
        actual_dcg
        /
        ideal_dcg
    )


def evaluate_predictions(
    predictions: dict[int, list[int]],
    truth: dict[int, set[int]],
    users: np.ndarray,
    k: int = 10,
):

    recall_sum = 0.0
    precision_sum = 0.0
    ndcg_sum = 0.0

    n_users = len(users)

    for user_id in users:

        user_id = int(user_id)

        relevant = truth.get(
            user_id,
            set(),
        )

        recommended = (
            predictions.get(
                user_id,
                [],
            )[:k]
        )

        hits = len(
            set(recommended)
            &
            relevant
        )

        if relevant:

            recall_sum += (
                hits
                /
                len(relevant)
            )

        # Precision@K uses K even when a user receives a short list.
        precision_sum += hits / k if k else 0.0

        ndcg_sum += ndcg_at_k(
            recommended,
            relevant,
            k,
        )

    return {

        "recall_at_10":
            recall_sum
            /
            n_users,

        "precision_at_10":
            precision_sum
            /
            n_users,

        "ndcg_at_10":
            ndcg_sum
            /
            n_users,
    }


# ============================================================
# Build predictions from LambdaMART scores
# ============================================================

def build_ranker_predictions(
    validation: pd.DataFrame,
    scores: np.ndarray,
    k: int,
):

    ranking_frame = pd.DataFrame(
        {
            "user_id":
                validation[
                    "user_id"
                ].to_numpy(),

            "product_id":
                validation[
                    "product_id"
                ].to_numpy(),

            "score":
                scores,
        }
    )

    ranking_frame = (
        ranking_frame
        .sort_values(
            [
                "user_id",
                "score",
            ],
            ascending=[
                True,
                False,
            ],
            kind="mergesort",
        )
    )

    topk = (
        ranking_frame
        .groupby(
            "user_id",
            sort=False,
        )
        .head(k)
    )

    grouped = (
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
    )

    return {
        int(user_id):
            products

        for user_id, products
        in grouped.items()
    }


# ============================================================
# Repeat-purchase control
# ============================================================

def build_repeat_predictions(
    validation: pd.DataFrame,
    k: int,
):

    repeat = (
        validation[
            validation[
                "repeat_rank"
            ]
            > 0
        ][
            [
                "user_id",
                "product_id",
                "repeat_rank",
            ]
        ]
        .copy()
    )

    repeat = (
        repeat
        .sort_values(
            [
                "user_id",
                "repeat_rank",
            ],
            ascending=[
                True,
                True,
            ],
            kind="mergesort",
        )
    )

    repeat = (
        repeat
        .groupby(
            "user_id",
            sort=False,
        )
        .head(k)
    )

    grouped = (
        repeat
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
    )

    return {
        int(user_id):
            products

        for user_id, products
        in grouped.items()
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load datasets
    # ========================================================

    train = prepare_dataframe(
        TRAIN_PATH
    )

    validation = prepare_dataframe(
        VALIDATION_PATH
    )

    print(
        f"\nRaw ranker train rows: "
        f"{len(train):,}"
    )

    print(
        "Raw ranker validation rows: "
        f"{len(validation):,}"
    )

    # ========================================================
    # Remove zero-positive train queries
    # ========================================================

    train = (
        remove_zero_positive_queries(
            train
        )
    )

    train_users = (
        train[
            "user_id"
        ]
        .nunique()
    )

    print(
        "Train rows after removing "
        "zero-positive queries: "
        f"{len(train):,}"
    )

    print(
        "Train users used by LambdaMART: "
        f"{train_users:,}"
    )

    # ========================================================
    # Early-stop validation
    #
    # Only queries containing at least one positive candidate
    # can provide useful LambdaMART ranking feedback.
    # ========================================================

    early_validation = (
        remove_zero_positive_queries(
            validation
        )
    )

    early_validation_users = (
        early_validation[
            "user_id"
        ]
        .nunique()
    )

    print(
        "Early-stop validation users: "
        f"{early_validation_users:,}"
    )

    # ========================================================
    # Matrices
    # ========================================================

    X_train = train[
        FEATURE_COLUMNS
    ].astype(
        np.float32
    )

    y_train = train[
        "label"
    ].to_numpy(
        dtype=np.float32
    )

    qid_train = train[
        "user_id"
    ].to_numpy(
        dtype=np.int32
    )

    X_early = early_validation[
        FEATURE_COLUMNS
    ].astype(
        np.float32
    )

    y_early = early_validation[
        "label"
    ].to_numpy(
        dtype=np.float32
    )

    qid_early = early_validation[
        "user_id"
    ].to_numpy(
        dtype=np.int32
    )

    print("\nBuilding QuantileDMatrix...")

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        qid=qid_train,
        max_bin=256,
    )

    deval = xgb.QuantileDMatrix(
        X_early,
        label=y_early,
        qid=qid_early,
        max_bin=256,
        ref=dtrain,
    )

    # ========================================================
    # Train LambdaMART
    # ========================================================

    print(
        "\nTraining XGBoost LambdaMART v3..."
    )

    training_start = perf_counter()

    booster = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,

        num_boost_round=
            NUM_BOOST_ROUND,

        evals=[
            (
                dtrain,
                "train",
            ),
            (
                deval,
                "validation",
            ),
        ],

        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,

        verbose_eval=25,
    )

    training_seconds = (
        perf_counter()
        -
        training_start
    )

    best_iteration = int(
        booster.best_iteration
    )

    best_internal_ndcg = float(
        booster.best_score
    )

    print("\nTraining complete.")

    print(
        f"Training seconds: "
        f"{training_seconds:.2f}"
    )

    print(
        f"Best iteration: "
        f"{best_iteration}"
    )

    print(
        "Best validation NDCG: "
        f"{best_internal_ndcg}"
    )

    # ========================================================
    # Full candidate validation prediction
    # ========================================================

    print(
        "\nBuilding full validation matrix..."
    )

    X_full = (
        validation[
            FEATURE_COLUMNS
        ]
        .astype(
            np.float32
        )
    )

    dfull = xgb.QuantileDMatrix(
        X_full,
        max_bin=256,
        ref=dtrain,
    )

    print(
        "Predicting ranking scores..."
    )

    scores = booster.predict(
        dfull,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )

    # ========================================================
    # Official validation truth
    # ========================================================

    truth = load_official_truth()

    evaluation_users = (
        validation[
            "user_id"
        ]
        .drop_duplicates()
        .to_numpy(
            dtype=np.int32
        )
    )

    # ========================================================
    # LambdaMART predictions
    # ========================================================

    ranker_predictions = (
        build_ranker_predictions(
            validation=validation,
            scores=scores,
            k=TOP_K,
        )
    )

    ranker_metrics = (
        evaluate_predictions(
            predictions=
                ranker_predictions,

            truth=truth,

            users=
                evaluation_users,

            k=TOP_K,
        )
    )

    # ========================================================
    # Repeat control
    # ========================================================

    repeat_predictions = (
        build_repeat_predictions(
            validation=validation,
            k=TOP_K,
        )
    )

    repeat_metrics = (
        evaluate_predictions(
            predictions=
                repeat_predictions,

            truth=truth,

            users=
                evaluation_users,

            k=TOP_K,
        )
    )

    # ========================================================
    # Deltas
    # ========================================================

    delta = {

        metric:
            ranker_metrics[
                metric
            ]
            -
            repeat_metrics[
                metric
            ]

        for metric
        in ranker_metrics
    }

    # ========================================================
    # Feature importance
    # ========================================================

    raw_importance = (
        booster.get_score(
            importance_type="gain"
        )
    )

    importance_rows = []

    for feature in FEATURE_COLUMNS:

        importance_rows.append(
            {
                "feature":
                    feature,

                "gain":
                    float(
                        raw_importance.get(
                            feature,
                            0.0,
                        )
                    ),
            }
        )

    importance_df = pd.DataFrame(
        importance_rows
    )

    importance_df = (
        importance_df
        .sort_values(
            "gain",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # Results
    # ========================================================

    result = {

        "model":
            "xgboost_lambdamart_v3_five_source",

        "objective":
            "rank:ndcg",

        "users_evaluated":
            int(
                len(
                    evaluation_users
                )
            ),

        "features":
            FEATURE_COLUMNS,

        "best_iteration":
            best_iteration,

        "best_internal_ndcg":
            best_internal_ndcg,

        "training_seconds":
            float(
                training_seconds
            ),

        "ranker":
            ranker_metrics,

        "repeat_control_same_users":
            repeat_metrics,

        "delta_vs_repeat":
            delta,
    }

    # ========================================================
    # Save artifacts
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    booster.save_model(
        MODEL_PATH
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            result,
            indent=4,
        ),
        encoding="utf-8",
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    # ========================================================
    # Report
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "XGBOOST LAMBDAMART V3 RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    print(
        "\nTop feature importance:"
    )

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

    print("\nSaved model:")
    print(MODEL_PATH)

    print("\nSaved benchmark:")
    print(RESULTS_PATH)

    print("\nSaved feature importance:")
    print(IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
