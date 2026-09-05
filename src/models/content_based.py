from __future__ import annotations

from collections import defaultdict
from math import log2

import pandas as pd


class ContentBasedRecommender:
    """
    Metadata-affinity recommender.

    Learns each user's preferred aisles and departments
    from historical purchases, then recommends strong
    products from those preferred categories.

    Item popularity is used only to rank products
    within a category.
    """

    def __init__(
        self,
        top_aisles: int = 5,
        top_departments: int = 3,
        products_per_aisle: int = 50,
        products_per_department: int = 50,
        precompute_k: int = 20,
        department_weight: float = 0.5,
    ) -> None:

        self.top_aisles = top_aisles
        self.top_departments = top_departments

        self.products_per_aisle = (
            products_per_aisle
        )

        self.products_per_department = (
            products_per_department
        )

        self.precompute_k = precompute_k

        # Aisle is more specific than department,
        # therefore department contributes less.
        self.department_weight = (
            department_weight
        )

        self.recommendations = None


    def fit(
        self,
        interactions: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> "ContentBasedRecommender":

        print(
            "Preparing product metadata..."
        )

        product_metadata = (
            item_features[
                [
                    "product_id",
                    "aisle_id",
                    "department_id",
                ]
            ]
            .copy()
        )

        print(
            "Joining user behavior with metadata..."
        )

        behavior = (
            interactions[
                [
                    "user_id",
                    "product_id",
                ]
            ]
            .merge(
                product_metadata,
                on="product_id",
                how="left",
                validate="many_to_one",
            )
        )

        if (
            behavior["aisle_id"].isna().any()
            or
            behavior["department_id"].isna().any()
        ):
            raise ValueError(
                "Missing product metadata found "
                "during content-model fitting."
            )

        # --------------------------------------------------
        # User → aisle affinity
        # --------------------------------------------------

        print(
            "Building user-aisle profiles..."
        )

        aisle_affinity = (
            behavior
            .groupby(
                [
                    "user_id",
                    "aisle_id",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size":
                        "purchase_count"
                }
            )
        )

        aisle_affinity[
            "user_total"
        ] = (
            aisle_affinity
            .groupby("user_id")
            ["purchase_count"]
            .transform("sum")
        )

        aisle_affinity[
            "affinity"
        ] = (
            aisle_affinity[
                "purchase_count"
            ]
            /
            aisle_affinity[
                "user_total"
            ]
        )

        aisle_affinity = (
            aisle_affinity
            .sort_values(
                [
                    "user_id",
                    "purchase_count",
                    "aisle_id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
            .groupby(
                "user_id",
                sort=False,
            )
            .head(
                self.top_aisles
            )
        )

        # --------------------------------------------------
        # User → department affinity
        # --------------------------------------------------

        print(
            "Building user-department profiles..."
        )

        department_affinity = (
            behavior
            .groupby(
                [
                    "user_id",
                    "department_id",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size":
                        "purchase_count"
                }
            )
        )

        department_affinity[
            "user_total"
        ] = (
            department_affinity
            .groupby("user_id")
            ["purchase_count"]
            .transform("sum")
        )

        department_affinity[
            "affinity"
        ] = (
            department_affinity[
                "purchase_count"
            ]
            /
            department_affinity[
                "user_total"
            ]
        )

        department_affinity = (
            department_affinity
            .sort_values(
                [
                    "user_id",
                    "purchase_count",
                    "department_id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
            .groupby(
                "user_id",
                sort=False,
            )
            .head(
                self.top_departments
            )
        )

        del behavior

        # --------------------------------------------------
        # Candidate lookup tables
        # --------------------------------------------------

        print(
            "Building category candidate pools..."
        )

        ranked_products = (
            item_features
            .sort_values(
                [
                    "purchase_count",
                    "reorder_rate",
                    "product_id",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
        )

        global_popular = (
            ranked_products[
                "product_id"
            ]
            .head(500)
            .astype(int)
            .tolist()
        )

        aisle_candidates = {}

        for aisle_id, group in (
            ranked_products
            .groupby(
                "aisle_id",
                sort=False,
            )
        ):

            aisle_candidates[
                int(aisle_id)
            ] = (
                group[
                    "product_id"
                ]
                .head(
                    self.products_per_aisle
                )
                .astype(int)
                .tolist()
            )

        department_candidates = {}

        for department_id, group in (
            ranked_products
            .groupby(
                "department_id",
                sort=False,
            )
        ):

            department_candidates[
                int(department_id)
            ] = (
                group[
                    "product_id"
                ]
                .head(
                    self.products_per_department
                )
                .astype(int)
                .tolist()
            )

        # --------------------------------------------------
        # Convert profiles to lightweight lookup structures
        # --------------------------------------------------

        aisle_profiles = defaultdict(list)

        for row in (
            aisle_affinity
            .itertuples(
                index=False
            )
        ):
            aisle_profiles[
                int(row.user_id)
            ].append(
                (
                    int(row.aisle_id),
                    float(row.affinity),
                )
            )

        department_profiles = defaultdict(list)

        for row in (
            department_affinity
            .itertuples(
                index=False
            )
        ):
            department_profiles[
                int(row.user_id)
            ].append(
                (
                    int(
                        row.department_id
                    ),
                    float(row.affinity),
                )
            )

        users = sorted(
            set(
                aisle_profiles.keys()
            )
            |
            set(
                department_profiles.keys()
            )
        )

        print(
            "Precomputing personalized "
            "content recommendations..."
        )

        recommendations = {}

        for user_id in users:

            scores = defaultdict(float)

            # More specific aisle preferences.
            for (
                aisle_id,
                affinity,
            ) in aisle_profiles.get(
                user_id,
                [],
            ):

                candidates = (
                    aisle_candidates
                    .get(
                        aisle_id,
                        [],
                    )
                )

                for rank, product_id in enumerate(
                    candidates,
                    start=1,
                ):

                    rank_discount = (
                        1.0
                        /
                        log2(
                            rank + 1
                        )
                    )

                    scores[
                        product_id
                    ] += (
                        affinity
                        *
                        rank_discount
                    )

            # Broader department preferences.
            for (
                department_id,
                affinity,
            ) in department_profiles.get(
                user_id,
                [],
            ):

                candidates = (
                    department_candidates
                    .get(
                        department_id,
                        [],
                    )
                )

                for rank, product_id in enumerate(
                    candidates,
                    start=1,
                ):

                    rank_discount = (
                        1.0
                        /
                        log2(
                            rank + 1
                        )
                    )

                    scores[
                        product_id
                    ] += (
                        self.department_weight
                        *
                        affinity
                        *
                        rank_discount
                    )

            ranked = sorted(
                scores.items(),
                key=lambda x: (
                    -x[1],
                    x[0],
                ),
            )

            user_recommendations = [
                product_id
                for product_id, _
                in ranked[
                    :self.precompute_k
                ]
            ]

            # Cold/fallback protection.
            if (
                len(user_recommendations)
                <
                self.precompute_k
            ):

                selected = set(
                    user_recommendations
                )

                for product_id in (
                    global_popular
                ):

                    if (
                        product_id
                        in selected
                    ):
                        continue

                    user_recommendations.append(
                        product_id
                    )

                    selected.add(
                        product_id
                    )

                    if (
                        len(
                            user_recommendations
                        )
                        >=
                        self.precompute_k
                    ):
                        break

            recommendations[
                user_id
            ] = (
                user_recommendations
            )

        self.recommendations = (
            recommendations
        )

        print(
            "Users with recommendations: "
            f"{len(self.recommendations):,}"
        )

        print(
            "Content-based model ready."
        )

        return self


    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:

        if self.recommendations is None:

            raise RuntimeError(
                "Model must be fitted first."
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