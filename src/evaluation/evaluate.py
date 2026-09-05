from pathlib import Path
import json
from datetime import datetime, timezone
from tqdm.auto import tqdm
import pandas as pd

from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "benchmarks"
)


def evaluate_model(
    model,
    evaluation_data: pd.DataFrame,
    k: int = 10,
    sample_users: int | None = None,
):

    """
    Evaluate a recommender model.

    Each user's future basket is compared
    against generated recommendations.
    """

    user_truth = (
        evaluation_data
        .groupby("user_id")
        ["product_id"]
        .apply(set)
    )


    if sample_users:

        user_truth = (
            user_truth
            .head(sample_users)
        )


    recall_scores = []
    precision_scores = []
    ndcg_scores = []

    all_recommendations = []

    for user_id, truth in tqdm(
            user_truth.items(),
            total=len(user_truth),
            desc=f"Evaluating Recall/NDCG@{k}",
    ):

        recommendations = (
            model
            .recommend(
                user_id=user_id,
                k=k,
            )
        )


        all_recommendations.append(
            recommendations
        )


        recall_scores.append(
            recall_at_k(
                recommendations,
                truth,
                k,
            )
        )


        precision_scores.append(
            precision_at_k(
                recommendations,
                truth,
                k,
            )
        )


        ndcg_scores.append(
            ndcg_at_k(
                recommendations,
                truth,
                k,
            )
        )


    results = {

        "users_evaluated":
            len(user_truth),

        "k":
            k,

        "recall_at_k":
            sum(recall_scores)
            /
            len(recall_scores),

        "precision_at_k":
            sum(precision_scores)
            /
            len(precision_scores),

        "ndcg_at_k":
            sum(ndcg_scores)
            /
            len(ndcg_scores),

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

    }


    return results