from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from scipy.sparse import (
    csr_matrix,
)

from torch.nn import functional as F

from src.models.two_tower import (
    TwoTowerModel,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "two_tower"
)

PAIR_PATH = (
    DATA_DIR
    / "positive_pairs.parquet"
)

FEATURE_PATH = (
    DATA_DIR
    / "feature_arrays.npz"
)

METADATA_PATH = (
    DATA_DIR
    / "metadata.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_tower_v1.pt"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "two_tower_training_v1.json"
)


SEED = 42

EMBEDDING_DIM = 64

BATCH_SIZE = 8192

EPOCHS = 5

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

SCORE_SCALE = 10.0


def sample_negatives(
    rng: np.random.Generator,
    users: np.ndarray,
    observed_products: np.ndarray,
    seen_matrix,
) -> np.ndarray:

    batch_size = len(
        users
    )

    negatives = (
        observed_products[
            rng.integers(
                0,
                len(
                    observed_products
                ),
                size=batch_size,
            )
        ]
    ).astype(
        np.int32,
        copy=False,
    )

    # Reject products already purchased
    # by the corresponding user.
    while True:

        seen = (
            seen_matrix[
                users,
                negatives,
            ]
            .A1
            .astype(
                bool,
                copy=False,
            )
        )

        if not seen.any():
            break

        count = int(
            seen.sum()
        )

        negatives[
            seen
        ] = (
            observed_products[
                rng.integers(
                    0,
                    len(
                        observed_products
                    ),
                    size=count,
                )
            ]
        )

    return negatives


def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Two-Tower GPU training "
            "cannot start."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "Device:",
        torch.cuda.get_device_name(
            0
        ),
    )

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        True
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    rng = (
        np.random.default_rng(
            SEED
        )
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print(
        "\nLoading positive pairs..."
    )

    pairs = pd.read_parquet(
        PAIR_PATH,
        columns=[
            "user_id",
            "product_id",
            "sample_weight",
        ],
    )

    users = (
        pairs[
            "user_id"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    positive_products = (
        pairs[
            "product_id"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    sample_weights = (
        pairs[
            "sample_weight"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    del pairs

    print(
        f"Training pairs: "
        f"{len(users):,}"
    )

    print(
        "Loading feature arrays..."
    )

    arrays = np.load(
        FEATURE_PATH
    )

    user_features_np = (
        arrays[
            "user_features"
        ]
        .astype(
            np.float32
        )
    )

    item_features_np = (
        arrays[
            "item_features"
        ]
        .astype(
            np.float32
        )
    )

    item_aisle_np = (
        arrays[
            "item_aisle"
        ]
        .astype(
            np.int64
        )
    )

    item_department_np = (
        arrays[
            "item_department"
        ]
        .astype(
            np.int64
        )
    )

    observed_products = (
        arrays[
            "observed_product_ids"
        ]
        .astype(
            np.int32
        )
    )

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------
    # Seen interaction matrix
    # --------------------------------------------------

    print(
        "Building sparse seen-item matrix..."
    )

    seen_matrix = csr_matrix(
        (
            np.ones(
                len(users),
                dtype=np.uint8,
            ),
            (
                users,
                positive_products,
            ),
        ),
        shape=(
            metadata[
                "max_user_id"
            ] + 1,

            metadata[
                "max_product_id"
            ] + 1,
        ),
        dtype=np.uint8,
    )

    seen_matrix.sort_indices()

    # --------------------------------------------------
    # Torch feature stores
    # --------------------------------------------------

    user_feature_tensor = (
        torch.from_numpy(
            user_features_np
        )
    )

    item_feature_tensor = (
        torch.from_numpy(
            item_features_np
        )
    )

    item_aisle_tensor = (
        torch.from_numpy(
            item_aisle_np
        )
    )

    item_department_tensor = (
        torch.from_numpy(
            item_department_np
        )
    )

    model = TwoTowerModel(

        num_users=
            metadata[
                "max_user_id"
            ] + 1,

        num_products=
            metadata[
                "max_product_id"
            ] + 1,

        num_aisles=
            metadata[
                "max_aisle_id"
            ] + 1,

        num_departments=
            metadata[
                "max_department_id"
            ] + 1,

        user_features=
            user_feature_tensor,

        item_features=
            item_feature_tensor,

        item_aisle=
            item_aisle_tensor,

        item_department=
            item_department_tensor,

        embedding_dim=
            EMBEDDING_DIM,
    )

    model = model.to(
        device
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    print(
        "\nModel parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    history = []

    total_start = perf_counter()

    n = len(
        users
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = (
            perf_counter()
        )

        model.train()

        permutation = (
            rng.permutation(
                n
            )
            .astype(
                np.int32,
                copy=False,
            )
        )

        epoch_loss_sum = 0.0

        epoch_weight_sum = 0.0

        batches = 0

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):

            index = permutation[
                start:
                start + BATCH_SIZE
            ]

            batch_users_np = (
                users[
                    index
                ]
            )

            batch_positive_np = (
                positive_products[
                    index
                ]
            )

            batch_weight_np = (
                sample_weights[
                    index
                ]
            )

            batch_negative_np = (
                sample_negatives(
                    rng=rng,

                    users=
                        batch_users_np,

                    observed_products=
                        observed_products,

                    seen_matrix=
                        seen_matrix,
                )
            )

            batch_users = (
                torch.from_numpy(
                    batch_users_np
                )
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            batch_positive = (
                torch.from_numpy(
                    batch_positive_np
                )
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            batch_negative = (
                torch.from_numpy(
                    batch_negative_np
                )
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            batch_weight = (
                torch.from_numpy(
                    batch_weight_np
                )
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            user_vectors = (
                model.encode_users(
                    batch_users
                )
            )

            positive_vectors = (
                model.encode_items(
                    batch_positive
                )
            )

            negative_vectors = (
                model.encode_items(
                    batch_negative
                )
            )

            positive_scores = (
                user_vectors
                * positive_vectors
            ).sum(
                dim=1
            )

            negative_scores = (
                user_vectors
                * negative_vectors
            ).sum(
                dim=1
            )

            pair_margin = (
                SCORE_SCALE
                *
                (
                    positive_scores
                    -
                    negative_scores
                )
            )

            losses = (
                -F.logsigmoid(
                    pair_margin
                )
            )

            loss = (
                losses
                * batch_weight
            ).sum() / (
                batch_weight.sum()
                + 1e-8
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            weight_sum = float(
                batch_weight.sum().item()
            )

            epoch_loss_sum += (
                float(
                    loss.item()
                )
                * weight_sum
            )

            epoch_weight_sum += (
                weight_sum
            )

            batches += 1

            if (
                batches % 250
                == 0
            ):

                print(
                    f"Epoch "
                    f"{epoch}/{EPOCHS} "
                    f"| batch "
                    f"{batches:,} "
                    f"| loss "
                    f"{loss.item():.5f}"
                )

        average_loss = (
            epoch_loss_sum
            /
            epoch_weight_sum
        )

        epoch_seconds = (
            perf_counter()
            -
            epoch_start
        )

        gpu_memory = (
            torch.cuda
            .max_memory_allocated()
            /
            1024**3
        )

        epoch_result = {

            "epoch":
                epoch,

            "loss":
                float(
                    average_loss
                ),

            "seconds":
                float(
                    epoch_seconds
                ),

            "max_gpu_memory_gb":
                float(
                    gpu_memory
                ),
        }

        history.append(
            epoch_result
        )

        print(
            "\n"
            f"Epoch {epoch} complete"
        )

        print(
            f"Average loss: "
            f"{average_loss:.6f}"
        )

        print(
            f"Seconds: "
            f"{epoch_seconds:.2f}"
        )

        print(
            f"Max GPU memory: "
            f"{gpu_memory:.2f} GB"
        )

    total_seconds = (
        perf_counter()
        -
        total_start
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "config": {

            "embedding_dim":
                EMBEDDING_DIM,

            "max_user_id":
                metadata[
                    "max_user_id"
                ],

            "max_product_id":
                metadata[
                    "max_product_id"
                ],

            "max_aisle_id":
                metadata[
                    "max_aisle_id"
                ],

            "max_department_id":
                metadata[
                    "max_department_id"
                ],
        },

        "training": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "score_scale":
                SCORE_SCALE,

            "seed":
                SEED,

            "total_seconds":
                total_seconds,

            "history":
                history,
        },
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            checkpoint[
                "training"
            ],
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TWO-TOWER TRAINING COMPLETE"
    )

    print(
        f"Total seconds: "
        f"{total_seconds:.2f}"
    )

    print(
        "\nSaved model:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()