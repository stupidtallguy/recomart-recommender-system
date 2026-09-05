from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn


class ItemSimilarityRecommender:
    """
    Item-item collaborative filtering using implicit purchase data.

    Architecture:
        user-item sparse matrix
            -> normalized item vectors
            -> Top-N cosine item similarities
            -> precomputed user candidate scores
            -> remove previously purchased items
            -> cached Top-K recommendations

    Recommendation-time work is only a lookup.
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        candidate_pool_size: int = 100,
        precompute_k: int = 20,
        n_threads: int | None = None,
    ) -> None:

        self.n_neighbors = n_neighbors
        self.candidate_pool_size = candidate_pool_size
        self.precompute_k = precompute_k

        if n_threads is None:
            cpu_count = os.cpu_count() or 2
            n_threads = max(1, cpu_count - 1)

        self.n_threads = n_threads

        self.user_ids: np.ndarray | None = None
        self.product_ids: np.ndarray | None = None

        self.user_to_index: dict[int, int] | None = None
        self.product_to_index: dict[int, int] | None = None

        self.item_similarity = None

        self.precomputed_recommendations: np.ndarray | None = None
        self.recommendation_lengths: np.ndarray | None = None


    def fit(
        self,
        interactions: pd.DataFrame,
    ) -> "ItemSimilarityRecommender":

        required_columns = {
            "user_id",
            "product_id",
        }

        missing_columns = (
            required_columns
            - set(interactions.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        print("Preparing user and product indices...")

        user_values = interactions[
            "user_id"
        ].to_numpy(
            dtype=np.int32,
            copy=False,
        )

        product_values = interactions[
            "product_id"
        ].to_numpy(
            dtype=np.int32,
            copy=False,
        )

        self.user_ids = np.sort(
            np.unique(user_values)
        )

        self.product_ids = np.sort(
            np.unique(product_values)
        )

        user_index = pd.Index(
            self.user_ids
        )

        product_index = pd.Index(
            self.product_ids
        )

        rows = user_index.get_indexer(
            user_values
        ).astype(
            np.int32,
            copy=False,
        )

        cols = product_index.get_indexer(
            product_values
        ).astype(
            np.int32,
            copy=False,
        )

        if (rows < 0).any():
            raise RuntimeError(
                "Failed to map one or more users."
            )

        if (cols < 0).any():
            raise RuntimeError(
                "Failed to map one or more products."
            )

        print(
            "Building binary user-item matrix..."
        )

        data = np.ones(
            len(interactions),
            dtype=np.float32,
        )

        user_item = csr_matrix(
            (
                data,
                (
                    rows,
                    cols,
                ),
            ),
            shape=(
                len(self.user_ids),
                len(self.product_ids),
            ),
            dtype=np.float32,
        )

        # Repeated purchases initially produce values > 1.
        # For this baseline we want binary implicit preference:
        # user has purchased item / has not purchased item.
        user_item.sum_duplicates()
        user_item.data.fill(1.0)
        user_item.sort_indices()

        print(
            "User-item matrix:"
            f" {user_item.shape[0]:,} users ×"
            f" {user_item.shape[1]:,} products"
        )

        print(
            f"Unique user-item relationships:"
            f" {user_item.nnz:,}"
        )

        # We no longer need the original mapping arrays.
        del rows
        del cols
        del data
        del user_values
        del product_values

        gc.collect()

        print(
            "Creating normalized item vectors..."
        )

        item_user = (
            user_item
            .T
            .tocsr()
        )

        item_user = normalize(
            item_user,
            norm="l2",
            axis=1,
            copy=True,
        )

        print(
            "Computing Top-N item similarities..."
        )

        # Each item is naturally most similar to itself,
        # therefore calculate n_neighbors + 1 and remove
        # the diagonal afterward.
        similarity = sp_matmul_topn(
            item_user,
            item_user.T.tocsr(),
            top_n=self.n_neighbors + 1,
            threshold=0.0,
            sort=True,
            n_threads=self.n_threads,
        )

        similarity.setdiag(0.0)
        similarity.eliminate_zeros()

        self.item_similarity = (
            similarity.tocsr()
        )

        print(
            "Stored item similarities:"
            f" {self.item_similarity.nnz:,}"
        )

        del item_user

        gc.collect()

        print(
            "Precomputing user candidate scores..."
        )

        # user_item @ item_similarity:
        #
        # Each user's purchased items vote for their
        # related products.
        #
        # Crucially, we retain only the strongest
        # candidate_pool_size products/user.
        candidate_scores = sp_matmul_topn(
            user_item,
            self.item_similarity,
            top_n=self.candidate_pool_size,
            threshold=0.0,
            sort=True,
            n_threads=self.n_threads,
        )

        print(
            "Filtering previously purchased products..."
        )

        n_users = len(
            self.user_ids
        )

        n_products = len(
            self.product_ids
        )

        recommendations = np.full(
            (
                n_users,
                self.precompute_k,
            ),
            -1,
            dtype=np.int32,
        )

        recommendation_lengths = (
            np.zeros(
                n_users,
                dtype=np.int16,
            )
        )

        # Reusable mask avoids building hundreds of
        # thousands of Python sets.
        seen_mask = np.zeros(
            n_products,
            dtype=bool,
        )

        users_with_short_lists = 0

        for user_idx in range(n_users):

            history_start = (
                user_item.indptr[
                    user_idx
                ]
            )

            history_end = (
                user_item.indptr[
                    user_idx + 1
                ]
            )

            seen_indices = (
                user_item.indices[
                    history_start:
                    history_end
                ]
            )

            seen_mask[
                seen_indices
            ] = True

            candidate_start = (
                candidate_scores.indptr[
                    user_idx
                ]
            )

            candidate_end = (
                candidate_scores.indptr[
                    user_idx + 1
                ]
            )

            candidate_indices = (
                candidate_scores.indices[
                    candidate_start:
                    candidate_end
                ]
            )

            # candidate_scores was generated with
            # sort=True, so ordering already represents
            # descending recommendation score.
            unseen_mask = (
                ~seen_mask[
                    candidate_indices
                ]
            )

            unseen_candidates = (
                candidate_indices[
                    unseen_mask
                ]
            )

            chosen_indices = (
                unseen_candidates[
                    :self.precompute_k
                ]
            )

            chosen_count = len(
                chosen_indices
            )

            if chosen_count:

                recommendations[
                    user_idx,
                    :chosen_count,
                ] = self.product_ids[
                    chosen_indices
                ]

            recommendation_lengths[
                user_idx
            ] = chosen_count

            if (
                chosen_count
                < self.precompute_k
            ):
                users_with_short_lists += 1

            # Reset only positions touched by this user.
            seen_mask[
                seen_indices
            ] = False

        self.precomputed_recommendations = (
            recommendations
        )

        self.recommendation_lengths = (
            recommendation_lengths
        )

        self.user_to_index = {
            int(user_id): idx
            for idx, user_id
            in enumerate(
                self.user_ids
            )
        }

        self.product_to_index = {
            int(product_id): idx
            for idx, product_id
            in enumerate(
                self.product_ids
            )
        }

        print(
            "Users with fewer than "
            f"{self.precompute_k} unseen "
            "recommendations: "
            f"{users_with_short_lists:,}"
        )

        del candidate_scores
        del user_item

        gc.collect()

        print(
            "Item-similarity model ready."
        )

        return self


    def recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> list[int]:

        if (
            self.precomputed_recommendations
            is None
        ):
            raise RuntimeError(
                "Model must be fitted first."
            )

        if k > self.precompute_k:
            raise ValueError(
                f"k={k} exceeds the "
                "precomputed recommendation "
                f"limit of {self.precompute_k}."
            )

        user_idx = (
            self.user_to_index
            .get(int(user_id))
        )

        if user_idx is None:
            return []

        available = int(
            self.recommendation_lengths[
                user_idx
            ]
        )

        count = min(
            k,
            available,
        )

        return (
            self
            .precomputed_recommendations[
                user_idx,
                :count,
            ]
            .tolist()
        )


    def similar_products(
        self,
        product_id: int,
        k: int = 10,
    ) -> list[tuple[int, float]]:

        if self.item_similarity is None:
            raise RuntimeError(
                "Model must be fitted first."
            )

        product_idx = (
            self.product_to_index
            .get(int(product_id))
        )

        if product_idx is None:
            return []

        row = self.item_similarity.getrow(
            product_idx
        )

        order = np.argsort(
            row.data
        )[::-1]

        order = order[:k]

        result = []

        for position in order:

            neighbor_idx = (
                row.indices[position]
            )

            similarity = float(
                row.data[position]
            )

            neighbor_product_id = int(
                self.product_ids[
                    neighbor_idx
                ]
            )

            result.append(
                (
                    neighbor_product_id,
                    similarity,
                )
            )

        return result