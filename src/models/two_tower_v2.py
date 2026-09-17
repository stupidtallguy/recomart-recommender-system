from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HistoryAwareTwoTower(nn.Module):

    def __init__(
        self,
        num_products: int,
        num_aisles: int,
        num_departments: int,
        item_numeric: torch.Tensor,
        item_aisle: torch.Tensor,
        item_department: torch.Tensor,
        embedding_dim: int = 64,
        aisle_embedding_dim: int = 16,
        department_embedding_dim: int = 8,
        history_numeric_dim: int = 4,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim

        # ====================================================
        # Shared embeddings
        # ====================================================

        self.product_embedding = nn.Embedding(
            num_embeddings=num_products,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.aisle_embedding = nn.Embedding(
            num_embeddings=num_aisles,
            embedding_dim=aisle_embedding_dim,
            padding_idx=0,
        )

        self.department_embedding = nn.Embedding(
            num_embeddings=num_departments,
            embedding_dim=department_embedding_dim,
            padding_idx=0,
        )

        # ====================================================
        # Static item feature stores
        # ====================================================

        self.register_buffer(
            "item_numeric",
            item_numeric,
        )

        self.register_buffer(
            "item_aisle",
            item_aisle,
        )

        self.register_buffer(
            "item_department",
            item_department,
        )

        # ====================================================
        # History token encoder
        #
        # A history token contains:
        #
        # product embedding
        # aisle embedding
        # department embedding
        # ====================================================

        history_token_input_dim = (
            embedding_dim
            + aisle_embedding_dim
            + department_embedding_dim
        )

        self.history_token_encoder = nn.Sequential(

            nn.Linear(
                history_token_input_dim,
                96,
            ),

            nn.ReLU(),

            nn.Linear(
                96,
                embedding_dim,
            ),

            nn.ReLU(),
        )

        # ====================================================
        # User tower
        #
        # Weighted history representation
        # +
        # history numerical statistics
        # ====================================================

        self.user_tower = nn.Sequential(

            nn.Linear(
                embedding_dim
                + history_numeric_dim,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                128,
                embedding_dim,
            ),
        )

        # ====================================================
        # Item tower
        # ====================================================

        item_input_dim = (
            embedding_dim
            + aisle_embedding_dim
            + department_embedding_dim
            + item_numeric.shape[1]
        )

        self.item_tower = nn.Sequential(

            nn.Linear(
                item_input_dim,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                128,
                embedding_dim,
            ),
        )

        self._initialize_parameters()


    # ========================================================
    # Initialization
    # ========================================================

    def _initialize_parameters(
        self,
    ) -> None:

        nn.init.normal_(
            self.product_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.aisle_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.department_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        # Padding embeddings must remain zero.

        with torch.no_grad():

            self.product_embedding.weight[
                0
            ].zero_()

            self.aisle_embedding.weight[
                0
            ].zero_()

            self.department_embedding.weight[
                0
            ].zero_()


    # ========================================================
    # History encoder
    # ========================================================

    def encode_history(
        self,
        history_product_ids: torch.Tensor,
        history_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode recent product history.

        Parameters
        ----------
        history_product_ids:
            [batch, history_length]

        history_weights:
            [batch, history_length]

        Returns
        -------
        torch.Tensor
            [batch, embedding_dim]
        """

        # ----------------------------------------------------
        # Product embeddings
        # ----------------------------------------------------

        product_vectors = (
            self.product_embedding(
                history_product_ids
            )
        )

        # ----------------------------------------------------
        # Category IDs
        # ----------------------------------------------------

        aisle_ids = (
            self.item_aisle[
                history_product_ids
            ]
        )

        department_ids = (
            self.item_department[
                history_product_ids
            ]
        )

        aisle_vectors = (
            self.aisle_embedding(
                aisle_ids
            )
        )

        department_vectors = (
            self.department_embedding(
                department_ids
            )
        )

        # ----------------------------------------------------
        # Build history token representation
        # ----------------------------------------------------

        token_features = torch.cat(
            [
                product_vectors,
                aisle_vectors,
                department_vectors,
            ],
            dim=-1,
        )

        token_vectors = (
            self.history_token_encoder(
                token_features
            )
        )

        # ----------------------------------------------------
        # Mask padding product_id = 0
        # ----------------------------------------------------

        mask = (
            history_product_ids
            != 0
        ).float()

        effective_weights = (
            history_weights
            *
            mask
        )

        effective_weights = (
            effective_weights
            .unsqueeze(-1)
        )

        # ----------------------------------------------------
        # Weighted mean
        #
        # Recent orders have higher weights.
        # Repeated products appear multiple times and
        # therefore naturally contribute more mass.
        # ----------------------------------------------------

        weighted_sum = (
            token_vectors
            *
            effective_weights
        ).sum(
            dim=1
        )

        denominator = (
            effective_weights
            .sum(
                dim=1
            )
            .clamp_min(
                1e-6
            )
        )

        history_vector = (
            weighted_sum
            /
            denominator
        )

        return history_vector


    # ========================================================
    # User tower
    # ========================================================

    def encode_users(
        self,
        history_product_ids: torch.Tensor,
        history_weights: torch.Tensor,
        history_numeric: torch.Tensor,
    ) -> torch.Tensor:

        history_vector = (
            self.encode_history(
                history_product_ids=
                    history_product_ids,

                history_weights=
                    history_weights,
            )
        )

        user_input = torch.cat(
            [
                history_vector,
                history_numeric,
            ],
            dim=1,
        )

        user_vector = (
            self.user_tower(
                user_input
            )
        )

        return F.normalize(
            user_vector,
            p=2,
            dim=1,
        )


    # ========================================================
    # Item tower
    # ========================================================

    def encode_items(
        self,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:

        product_vectors = (
            self.product_embedding(
                product_ids
            )
        )

        aisle_ids = (
            self.item_aisle[
                product_ids
            ]
        )

        department_ids = (
            self.item_department[
                product_ids
            ]
        )

        aisle_vectors = (
            self.aisle_embedding(
                aisle_ids
            )
        )

        department_vectors = (
            self.department_embedding(
                department_ids
            )
        )

        numeric = (
            self.item_numeric[
                product_ids
            ]
        )

        item_input = torch.cat(
            [
                product_vectors,
                aisle_vectors,
                department_vectors,
                numeric,
            ],
            dim=1,
        )

        item_vector = (
            self.item_tower(
                item_input
            )
        )

        return F.normalize(
            item_vector,
            p=2,
            dim=1,
        )


    # ========================================================
    # Forward score
    # ========================================================

    def forward(
        self,
        history_product_ids: torch.Tensor,
        history_weights: torch.Tensor,
        history_numeric: torch.Tensor,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:

        user_vectors = (
            self.encode_users(
                history_product_ids=
                    history_product_ids,

                history_weights=
                    history_weights,

                history_numeric=
                    history_numeric,
            )
        )

        item_vectors = (
            self.encode_items(
                product_ids
            )
        )

        scores = (
            user_vectors
            *
            item_vectors
        ).sum(
            dim=1
        )

        return scores