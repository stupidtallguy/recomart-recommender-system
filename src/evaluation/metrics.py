from typing import Dict, List, Set
import math


def recall_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 10,
) -> float:

    if not ground_truth:
        return 0.0

    recommended = set(
        recommendations[:k]
    )

    hits = len(
        recommended.intersection(
            ground_truth
        )
    )

    return hits / len(ground_truth)



def precision_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 10,
) -> float:

    if k == 0:
        return 0.0

    recommended = set(
        recommendations[:k]
    )

    hits = len(
        recommended.intersection(
            ground_truth
        )
    )

    return hits / k



def dcg_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 10,
):

    score = 0.0

    for rank, item in enumerate(
        recommendations[:k],
        start=1,
    ):

        if item in ground_truth:

            score += (
                1 /
                math.log2(rank + 1)
            )

    return score



def ndcg_at_k(
    recommendations: List[int],
    ground_truth: Set[int],
    k: int = 10,
):

    actual_dcg = dcg_at_k(
        recommendations,
        ground_truth,
        k,
    )


    ideal_hits = min(
        len(ground_truth),
        k,
    )


    ideal_dcg = sum(
        1 /
        math.log2(i + 2)
        for i in range(
            ideal_hits
        )
    )


    if ideal_dcg == 0:
        return 0.0


    return actual_dcg / ideal_dcg



def coverage(
    all_recommendations: List[List[int]],
    catalog_size: int,
):

    recommended_items = set()

    for recs in all_recommendations:
        recommended_items.update(recs)


    return (
        len(recommended_items)
        /
        catalog_size
    )