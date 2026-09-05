from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerModel(
    nn.Module
):

    def __init__(
        self,
        num_users: int,
        num_products: int,
        num_aisles: int,
        num_departments: int,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        item_aisle: torch.Tensor,
        item_department: torch.Tensor,
        embedding_dim: int = 64,
    ):

        super().__init__()

        self.embedding_dim = (
            embedding_dim
        )

        # --------------------------------------------------
        # Embeddings
        # --------------------------------------------------

        self.user_embedding = (
            nn.Embedding(
                num_users,
                embedding_dim,
                padding_idx=0,
            )
        )

        self.product_embedding = (
            nn.Embedding(
                num_products,
                embedding_dim,
                padding_idx=0,
            )
        )

        self.aisle_embedding = (
            nn.Embedding(
                num_aisles,
                16,
                padding_idx=0,
            )
        )

        self.department_embedding = (
            nn.Embedding(
                num_departments,
                8,
                padding_idx=0,
            )
        )

        # --------------------------------------------------
        # Static feature stores
        # --------------------------------------------------

        self.register_buffer(
            "user_features",
            user_features,
        )

        self.register_buffer(
            "item_features",
            item_features,
        )

        self.register_buffer(
            "item_aisle",
            item_aisle,
        )

        self.register_buffer(
            "item_department",
            item_department,
        )

        # --------------------------------------------------
        # Towers
        # --------------------------------------------------

        self.user_tower = (
            nn.Sequential(
                nn.Linear(
                    embedding_dim + 3,
                    128,
                ),
                nn.ReLU(),
                nn.Dropout(0.10),

                nn.Linear(
                    128,
                    embedding_dim,
                ),
            )
        )

        self.item_tower = (
            nn.Sequential(
                nn.Linear(
                    embedding_dim
                    + 16
                    + 8
                    + 2,
                    128,
                ),
                nn.ReLU(),
                nn.Dropout(0.10),

                nn.Linear(
                    128,
                    embedding_dim,
                ),
            )
        )

        self._initialize()


    def _initialize(
        self,
    ):

        nn.init.normal_(
            self.user_embedding.weight,
            std=0.02,
        )

        nn.init.normal_(
            self.product_embedding.weight,
            std=0.02,
        )

        nn.init.normal_(
            self.aisle_embedding.weight,
            std=0.02,
        )

        nn.init.normal_(
            self.department_embedding.weight,
            std=0.02,
        )

        with torch.no_grad():

            self.user_embedding.weight[
                0
            ].zero_()

            self.product_embedding.weight[
                0
            ].zero_()

            self.aisle_embedding.weight[
                0
            ].zero_()

            self.department_embedding.weight[
                0
            ].zero_()


    def encode_users(
        self,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:

        user_emb = (
            self.user_embedding(
                user_ids
            )
        )

        numeric = (
            self.user_features[
                user_ids
            ]
        )

        x = torch.cat(
            [
                user_emb,
                numeric,
            ],
            dim=1,
        )

        x = self.user_tower(
            x
        )

        return F.normalize(
            x,
            p=2,
            dim=1,
        )


    def encode_items(
        self,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:

        product_emb = (
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

        aisle_emb = (
            self.aisle_embedding(
                aisle_ids
            )
        )

        department_emb = (
            self.department_embedding(
                department_ids
            )
        )

        numeric = (
            self.item_features[
                product_ids
            ]
        )

        x = torch.cat(
            [
                product_emb,
                aisle_emb,
                department_emb,
                numeric,
            ],
            dim=1,
        )

        x = self.item_tower(
            x
        )

        return F.normalize(
            x,
            p=2,
            dim=1,
        )


    def forward(
        self,
        user_ids: torch.Tensor,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:

        users = self.encode_users(
            user_ids
        )

        items = self.encode_items(
            product_ids
        )

        return (
            users
            * items
        ).sum(
            dim=1
        )