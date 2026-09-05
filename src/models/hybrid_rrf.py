from __future__ import annotations

from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm


class HybridRRFRecommender:
    """
    Weighted Reciprocal Rank Fusion recommender.

    Combines ranked candidate lists from multiple
    recommendation sources without requiring their
    original scores to share the same numerical scale.

    score(item) =
        sum_source(
            source_weight / (rrf_k + rank)
        )
    """

    def __init__(
        self,
        source_weights: dict[str, float],
        rrf_k: int = 60,
        precompute_k: int = 20,
    ) -> None:

        self.source_weights = source_weights
        self.rrf_k = rrf_k
        self.precompute_k = precompute_k

        self.recommendations = None


    def fit(
        self,
        candidates: pd.DataFrame,
    ) -> "HybridRRFRecommender":

        required = {
            "user_id",
            *self.source_weights.keys(),
        }

        missing = (
            required
            - set(candidates.columns)
        )

        if missing:
            raise ValueError(
                "Missing candidate columns: "
                f"{sorted(missing)}"
            )

        recommendations = {}

        for row in tqdm(
            candidates.itertuples(
                index=False
            ),
            total=len(candidates),
            desc="Fusing candidates",
        ):

            user_id = int(
                row.user_id
            )

            scores = defaultdict(float)

            for (
                source,
                weight,
            ) in self.source_weights.items():

                if weight <= 0:
                    continue

                ranked_items = getattr(
                    row,
                    source,
                )

                if ranked_items is None:
                    continue

                for rank, product_id in enumerate(
                    ranked_items,
                    start=1,
                ):

                    scores[
                        int(product_id)
                    ] += (
                        weight
                        /
                        (
                            self.rrf_k
                            + rank
                        )
                    )

            ranked = sorted(
                scores.items(),
                key=lambda x: (
                    -x[1],
                    x[0],
                ),
            )

            recommendations[
                user_id
            ] = [
                product_id
                for product_id, _
                in ranked[
                    :self.precompute_k
                ]
            ]

        self.recommendations = (
            recommendations
        )

        return self


    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:

        if self.recommendations is None:
            raise RuntimeError(
                "Hybrid model must be "
                "fitted first."
            )

        if k > self.precompute_k:
            raise ValueError(
                f"k={k} exceeds "
                f"precomputed limit "
                f"{self.precompute_k}."
            )

        return (
            self.recommendations
            .get(
                int(user_id),
                [],
            )
            [:k]
        )