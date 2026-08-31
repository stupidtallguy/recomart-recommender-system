from pathlib import Path

import pandas as pd


class PopularityRecommender:
    """
    Non-personalized baseline recommender.

    Recommends globally popular products
    based on historical purchase frequency.
    """

    def __init__(self):
        self.popular_items = None


    def fit(
        self,
        interactions: pd.DataFrame,
        top_n: int = 100,
    ):

        popularity = (
            interactions
            .groupby("product_id")
            .size()
            .sort_values(
                ascending=False
            )
        )

        self.popular_items = (
            popularity
            .head(top_n)
            .index
            .tolist()
        )

        return self

    def recommend(
            self,
            user_id: int,
            k: int = 10,
    ):
        """
        Generate recommendations.

        user_id is accepted for compatibility
        with personalized recommenders.
        This baseline ignores user history.
        """

        if self.popular_items is None:
            raise RuntimeError(
                "Model must be fitted first."
            )


        return self.popular_items[:k]