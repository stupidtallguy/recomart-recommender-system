from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RANKER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ranker_v1_corrected"
)

TRAIN_PATH = (
    RANKER_DIR
    / "ranker_train"
)

VALIDATION_PATH = (
    RANKER_DIR
    / "ranker_validation"
)

ORIGINAL_VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_interactions.parquet"
)

REPEAT_CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidates"
    / "repeat_purchase_top50.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgb_ranker_v1_corrected.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "xgb_ranker_v1_corrected.json"
)

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "xgb_ranker_v1_corrected_feature_importance.csv"
)

FEATURE_COLUMNS = [
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


LOAD_COLUMNS = [
    "user_id",
    "product_id",
    "label",
    *FEATURE_COLUMNS,
]


def prepare_dataframe(
    path: Path,
) -> pd.DataFrame:

    print(
        f"Loading: {path}"
    )

    df = pd.read_parquet(
        path,
        columns=LOAD_COLUMNS,
    )

    df = df.sort_values(
        [
            "user_id",
            "product_id",
        ]
    ).reset_index(
        drop=True
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
        .astype(np.int8)
    )

    for column in FEATURE_COLUMNS:

        df[column] = (
            df[column]
            .fillna(0)
            .astype(np.float32)
        )

    return df


def keep_users_with_positive(
    df: pd.DataFrame,
) -> pd.DataFrame:

    positive_by_user = (
        df
        .groupby(
            "user_id",
            sort=False,
        )["label"]
        .transform("max")
    )

    filtered = (
        df[
            positive_by_user > 0
        ]
        .copy()
    )

    return filtered


def build_quantile_matrix(
    df: pd.DataFrame,
    ref=None,
):

    X = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )

    y = (
        df["label"]
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )

    qid = (
        df["user_id"]
        .to_numpy(
            dtype=np.int32,
            copy=False,
        )
    )

    matrix = xgb.QuantileDMatrix(
        X,
        label=y,
        qid=qid,
        ref=ref,
        feature_names=
            FEATURE_COLUMNS,
        max_bin=256,
    )

    return matrix


def top_k_predictions(
    validation_df: pd.DataFrame,
    scores: np.ndarray,
    k: int = 10,
) -> dict[int, list[int]]:

    scored = validation_df[
        [
            "user_id",
            "product_id",
        ]
    ].copy()

    scored[
        "prediction"
    ] = scores

    scored = (
        scored
        .sort_values(
            [
                "user_id",
                "prediction",
                "product_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
    )

    topk = (
        scored
        .groupby(
            "user_id",
            sort=False,
        )
        .head(k)
    )

    predictions = (
        topk
        .groupby(
            "user_id",
            sort=False,
        )["product_id"]
        .apply(list)
        .to_dict()
    )

    return predictions


def evaluate_predictions(
    predictions: dict[int, list[int]],
    truth: dict[int, set[int]],
    users,
    k: int = 10,
):

    recall_scores = []
    precision_scores = []
    ndcg_scores = []

    for user_id in users:

        recs = predictions.get(
            int(user_id),
            [],
        )

        relevant = truth.get(
            int(user_id),
            set(),
        )

        recall_scores.append(
            recall_at_k(
                recs,
                relevant,
                k,
            )
        )

        precision_scores.append(
            precision_at_k(
                recs,
                relevant,
                k,
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                recs,
                relevant,
                k,
            )
        )

    return {
        "recall_at_10":
            float(
                np.mean(
                    recall_scores
                )
            ),

        "precision_at_10":
            float(
                np.mean(
                    precision_scores
                )
            ),

        "ndcg_at_10":
            float(
                np.mean(
                    ndcg_scores
                )
            ),
    }


def evaluate_repeat_control(
    users,
    truth,
):

    repeat = pd.read_parquet(
        REPEAT_CANDIDATE_PATH,
        columns=[
            "user_id",
            "recommendations",
        ],
    )

    user_set = set(
        int(x)
        for x in users
    )

    repeat = repeat[
        repeat["user_id"]
        .isin(
            user_set
        )
    ]

    predictions = {
        int(row.user_id):
            [
                int(x)
                for x
                in row.recommendations[:10]
            ]

        for row
        in repeat.itertuples(
            index=False
        )
    }

    return evaluate_predictions(
        predictions=
            predictions,

        truth=
            truth,

        users=
            users,

        k=10,
    )


def main():

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load ranking datasets
    # --------------------------------------------------

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
        f"Raw ranker validation rows: "
        f"{len(validation):,}"
    )

    # --------------------------------------------------
    # Remove training groups with no positive candidate.
    #
    # Such groups provide no useful pairwise ranking
    # signal to LambdaMART.
    # --------------------------------------------------

    train = keep_users_with_positive(
        train
    )

    print(
        "Train rows after removing "
        "zero-positive queries: "
        f"{len(train):,}"
    )

    print(
        "Train users used by LambdaMART: "
        f"{train['user_id'].nunique():,}"
    )

    # For early stopping only, use validation
    # queries that actually contain a relevant
    # candidate.
    validation_fit = (
        keep_users_with_positive(
            validation
        )
    )

    print(
        "Early-stop validation users: "
        f"{validation_fit['user_id'].nunique():,}"
    )

    # --------------------------------------------------
    # XGBoost matrices
    # --------------------------------------------------

    print(
        "\nBuilding QuantileDMatrix..."
    )

    dtrain = build_quantile_matrix(
        train
    )

    dvalid = build_quantile_matrix(
        validation_fit,
        ref=dtrain,
    )

    # QuantileDMatrix is designed for hist-based
    # training with reduced memory overhead.
    # Validation reuses the training quantile bins.
    # XGBoost recommends using ref for this.
    # --------------------------------------------------

    params = {

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

        "nthread":
            max(
                1,
                (
                    os.cpu_count()
                    or 2
                )
                - 1,
            ),
    }


    print(
        "\nTraining XGBoost LambdaMART..."
    )

    start = perf_counter()

    early_stop = (
        xgb.callback.EarlyStopping(
            rounds=50,
            save_best=True,
            maximize=True,
        )
    )

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=800,

        evals=[
            (
                dtrain,
                "train",
            ),
            (
                dvalid,
                "validation",
            ),
        ],

        callbacks=[
            early_stop
        ],

        verbose_eval=25,
    )

    training_seconds = (
        perf_counter()
        - start
    )

    print(
        "\nTraining complete."
    )

    print(
        "Training seconds: "
        f"{training_seconds:.2f}"
    )

    print(
        "Best iteration: "
        f"{booster.best_iteration}"
    )

    print(
        "Best validation NDCG: "
        f"{booster.best_score}"
    )

    booster.save_model(
        MODEL_PATH
    )

    # Release training memory before
    # full validation prediction.
    del dtrain
    del dvalid
    del train
    del validation_fit

    gc.collect()

    # --------------------------------------------------
    # Predict ALL 41,114 ranker-validation users
    # --------------------------------------------------

    print(
        "\nBuilding full validation matrix..."
    )

    X_validation = (
        validation[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )

    dvalidation_full = (
        xgb.DMatrix(
            X_validation,
            feature_names=
                FEATURE_COLUMNS,
        )
    )

    print(
        "Predicting ranking scores..."
    )

    scores = booster.predict(
        dvalidation_full
    )

    predictions = top_k_predictions(
        validation_df=
            validation,

        scores=
            scores,

        k=10,
    )

    validation_users = (
        validation[
            "user_id"
        ]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------

    original_validation = (
        pd.read_parquet(
            ORIGINAL_VALIDATION_PATH,
            columns=[
                "user_id",
                "product_id",
            ],
        )
    )

    validation_user_set = set(
        validation_users
    )

    original_validation = (
        original_validation[
            original_validation[
                "user_id"
            ]
            .isin(
                validation_user_set
            )
        ]
    )

    truth = (
        original_validation
        .groupby(
            "user_id"
        )["product_id"]
        .apply(set)
        .to_dict()
    )

    # --------------------------------------------------
    # Ranker metrics
    # --------------------------------------------------

    ranker_metrics = (
        evaluate_predictions(
            predictions=
                predictions,

            truth=
                truth,

            users=
                validation_users,

            k=10,
        )
    )

    # --------------------------------------------------
    # Repeat Purchase control on SAME users
    # --------------------------------------------------

    repeat_metrics = (
        evaluate_repeat_control(
            users=
                validation_users,

            truth=
                truth,
        )
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = (
        booster
        .get_score(
            importance_type="gain"
        )
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature":
                    FEATURE_COLUMNS,

                "gain":
                    [
                        float(
                            importance
                            .get(
                                feature,
                                0.0,
                            )
                        )

                        for feature
                        in FEATURE_COLUMNS
                    ],
            }
        )
        .sort_values(
            "gain",
            ascending=False,
        )
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results = {

        "model":
            "xgboost_lambdamart_v1_corrected",

        "objective":
            "rank:ndcg",

        "users_evaluated":
            len(
                validation_users
            ),

        "features":
            FEATURE_COLUMNS,

        "best_iteration":
            int(
                booster.best_iteration
            ),

        "best_internal_ndcg":
            float(
                booster.best_score
            ),

        "training_seconds":
            training_seconds,

        "ranker": {
            **ranker_metrics,
        },

        "repeat_control_same_users": {
            **repeat_metrics,
        },

        "delta_vs_repeat": {

            "recall_at_10":
                (
                    ranker_metrics[
                        "recall_at_10"
                    ]
                    -
                    repeat_metrics[
                        "recall_at_10"
                    ]
                ),

            "precision_at_10":
                (
                    ranker_metrics[
                        "precision_at_10"
                    ]
                    -
                    repeat_metrics[
                        "precision_at_10"
                    ]
                ),

            "ndcg_at_10":
                (
                    ranker_metrics[
                        "ndcg_at_10"
                    ]
                    -
                    repeat_metrics[
                        "ndcg_at_10"
                    ]
                ),
        },
    }


    RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=4,
        ),
        encoding="utf-8",
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "XGBOOST LAMBDAMART RESULTS"
    )

    print(
        json.dumps(
            results,
            indent=4,
        )
    )

    print(
        "\nTop feature importance:"
    )

    print(
        importance_df.head(
            12
        ).to_string(
            index=False
        )
    )

    print(
        "\nSaved model:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nSaved benchmark:"
    )

    print(
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()