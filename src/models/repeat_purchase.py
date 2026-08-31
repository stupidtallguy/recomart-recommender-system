from collections import defaultdict

import pandas as pd


class RepeatPurchaseRecommender:
    """
    Personalized recommender based on
    user's previous purchase behavior.

    Products are ranked using:
    - purchase frequency
    - reorder behavior
    - recency
    """

    def __init__(self):
        self.user_products = None


    def fit(
        self,
        interactions: pd.DataFrame,
    ):

        print(
            "Building user purchase history..."
        )


        self.user_products = {}


        grouped = (
            interactions
            .groupby(
                [
                    "user_id",
                    "product_id",
                ]
            )
            .agg(
                purchase_count=(
                    "product_id",
                    "count"
                ),

                reorder_rate=(
                    "reordered",
                    "mean"
                ),

                last_order=(
                    "order_number",
                    "max"
                ),
            )
            .reset_index()
        )


        max_order = (
            interactions["order_number"]
            .max()
        )


        grouped["recency"] = (
            grouped["last_order"]
            /
            max_order
        )


        grouped["score"] = (

            grouped["purchase_count"]
            * 0.5

            +

            grouped["reorder_rate"]
            * 0.3

            +

            grouped["recency"]
            * 0.2

        )


        for user_id, user_df in (
            grouped.groupby("user_id")
        ):

            ranked = (
                user_df
                .sort_values(
                    "score",
                    ascending=False
                )
            )


            self.user_products[
                user_id
            ] = (
                ranked["product_id"]
                .tolist()
            )


        return self



    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ):

        if self.user_products is None:

            raise RuntimeError(
                "Model must be fitted first."
            )


        return (
            self.user_products
            .get(user_id, [])
            [:k]
        )