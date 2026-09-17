from __future__ import annotations

import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.models.two_tower_v2 import (
    HistoryAwareTwoTower,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "two_tower_v2"
)

CONTEXT_PATH = (
    DATA_DIR
    / "contexts.npz"
)

PAIR_PATH = (
    DATA_DIR
    / "positive_pairs.parquet"
)

ITEM_FEATURE_PATH = (
    DATA_DIR
    / "item_features.npz"
)

METADATA_PATH = (
    DATA_DIR
    / "metadata.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_tower_v2.pt"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "two_tower_v2_training.json"
)


# ============================================================
# Training configuration
# ============================================================

SEED = 42

EMBEDDING_DIM = 64

# Safe initial value for RTX 3050 4GB.
BATCH_SIZE = 4096

MAX_EPOCHS = 8

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

SCORE_SCALE = 10.0

EARLY_STOPPING_PATIENCE = 2

# Internal temporal validation.
VALIDATION_SAMPLE_CONTEXTS = 20_000

VALIDATION_RECALL_K = 50

VALIDATION_USER_BATCH = 512


# ============================================================
# Negative sampling
# ============================================================

def sample_negatives(
    rng: np.random.Generator,
    context_ids: np.ndarray,
    observed_product_ids: np.ndarray,
    probabilities: np.ndarray,
    target_matrix: csr_matrix,
) -> np.ndarray:

    batch_size = len(
        context_ids
    )

    negatives = rng.choice(
        observed_product_ids,
        size=batch_size,
        replace=True,
        p=probabilities,
    ).astype(
        np.int32,
        copy=False,
    )

    # --------------------------------------------------------
    # Never treat another product from the same target basket
    # as a negative.
    # --------------------------------------------------------

    while True:

        invalid = (
            target_matrix[
                context_ids,
                negatives,
            ]
            .A1
            .astype(
                bool,
                copy=False,
            )
        )

        if not invalid.any():
            break

        count = int(
            invalid.sum()
        )

        negatives[
            invalid
        ] = rng.choice(
            observed_product_ids,
            size=count,
            replace=True,
            p=probabilities,
        )

    return negatives


# ============================================================
# Internal temporal retrieval evaluation
# ============================================================

def evaluate_recall_at_k(
    model: HistoryAwareTwoTower,
    device: torch.device,
    context_ids: np.ndarray,
    truth: dict[int, set[int]],
    history_product_ids: np.ndarray,
    history_weights: np.ndarray,
    history_numeric: np.ndarray,
    observed_product_ids: np.ndarray,
    k: int,
) -> float:

    model.eval()

    # --------------------------------------------------------
    # Encode all products once
    # --------------------------------------------------------

    product_tensor = (
        torch.from_numpy(
            observed_product_ids
        )
        .long()
        .to(device)
    )

    with torch.inference_mode():

        item_vectors = (
            model.encode_items(
                product_tensor
            )
        )

    recalls = []

    with torch.inference_mode():

        for start in range(
            0,
            len(context_ids),
            VALIDATION_USER_BATCH,
        ):

            batch_context_ids = (
                context_ids[
                    start:
                    start
                    +
                    VALIDATION_USER_BATCH
                ]
            )

            history_ids = (
                torch.from_numpy(
                    history_product_ids[
                        batch_context_ids
                    ]
                )
                .long()
                .to(device)
            )

            weights = (
                torch.from_numpy(
                    history_weights[
                        batch_context_ids
                    ]
                    .astype(
                        np.float32
                    )
                )
                .to(device)
            )

            numeric = (
                torch.from_numpy(
                    history_numeric[
                        batch_context_ids
                    ]
                )
                .float()
                .to(device)
            )

            user_vectors = (
                model.encode_users(
                    history_product_ids=
                        history_ids,

                    history_weights=
                        weights,

                    history_numeric=
                        numeric,
                )
            )

            similarity = (
                user_vectors
                @
                item_vectors.T
            )

            _, indices = torch.topk(
                similarity,
                k=k,
                dim=1,
                largest=True,
                sorted=False,
            )

            predictions = (
                product_tensor[
                    indices
                ]
                .cpu()
                .numpy()
            )

            for row_index, context_id in enumerate(
                batch_context_ids
            ):

                relevant = truth[
                    int(
                        context_id
                    )
                ]

                predicted = set(
                    int(x)
                    for x
                    in predictions[
                        row_index
                    ]
                )

                recall = (
                    len(
                        predicted
                        &
                        relevant
                    )
                    /
                    len(
                        relevant
                    )
                )

                recalls.append(
                    recall
                )

            del similarity
            del indices
            del user_vectors

    return float(
        np.mean(
            recalls
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable."
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

    rng = np.random.default_rng(
        SEED
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        True
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        "\nLoading temporal contexts..."
    )

    contexts = np.load(
        CONTEXT_PATH
    )

    context_history_products = (
        contexts[
            "history_product_ids"
        ]
        .astype(
            np.int32
        )
    )

    context_history_weights = (
        contexts[
            "history_weights"
        ]
        # Keep compressed source as float16.
    )

    context_numeric = (
        contexts[
            "numeric_features"
        ]
        .astype(
            np.float32
        )
    )

    context_split = (
        contexts[
            "split"
        ]
        .astype(
            np.int8
        )
    )

    print(
        f"Contexts: "
        f"{len(context_split):,}"
    )

    # --------------------------------------------------------
    # Positive target pairs
    # --------------------------------------------------------

    print(
        "Loading temporal target pairs..."
    )

    pairs = pd.read_parquet(
        PAIR_PATH,
        columns=[
            "context_id",
            "product_id",
            "split",
        ],
    )

    pairs[
        "context_id"
    ] = (
        pairs[
            "context_id"
        ]
        .astype(
            np.int32
        )
    )

    pairs[
        "product_id"
    ] = (
        pairs[
            "product_id"
        ]
        .astype(
            np.int32
        )
    )

    pairs[
        "split"
    ] = (
        pairs[
            "split"
        ]
        .astype(
            np.int8
        )
    )

    train_pairs = (
        pairs[
            pairs[
                "split"
            ]
            == 0
        ]
    )

    validation_pairs = (
        pairs[
            pairs[
                "split"
            ]
            == 1
        ]
    )

    train_context_ids = (
        train_pairs[
            "context_id"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    train_positive_products = (
        train_pairs[
            "product_id"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    print(
        f"Train positive pairs: "
        f"{len(train_context_ids):,}"
    )

    print(
        f"Validation positive pairs: "
        f"{len(validation_pairs):,}"
    )

    # --------------------------------------------------------
    # Item arrays
    # --------------------------------------------------------

    print(
        "Loading item features..."
    )

    item_arrays = np.load(
        ITEM_FEATURE_PATH
    )

    item_numeric_np = (
        item_arrays[
            "item_numeric"
        ]
        .astype(
            np.float32
        )
    )

    item_aisle_np = (
        item_arrays[
            "item_aisle"
        ]
        .astype(
            np.int64
        )
    )

    item_department_np = (
        item_arrays[
            "item_department"
        ]
        .astype(
            np.int64
        )
    )

    observed_product_ids = (
        item_arrays[
            "observed_product_ids"
        ]
        .astype(
            np.int32
        )
    )

    negative_probabilities = (
        item_arrays[
            "negative_sampling_probabilities"
        ]
        .astype(
            np.float64
        )
    )

    # Ensure normalization survives serialization.
    negative_probabilities = (
        negative_probabilities
        /
        negative_probabilities.sum()
    )

    # --------------------------------------------------------
    # Target sparse matrix
    #
    # Used only to ensure that products in the actual target
    # basket aren't sampled as negatives.
    # --------------------------------------------------------

    print(
        "Building context-target sparse matrix..."
    )

    target_matrix = csr_matrix(
        (
            np.ones(
                len(pairs),
                dtype=np.uint8,
            ),

            (
                pairs[
                    "context_id"
                ]
                .to_numpy(
                    dtype=np.int32
                ),

                pairs[
                    "product_id"
                ]
                .to_numpy(
                    dtype=np.int32
                ),
            ),
        ),

        shape=(
            metadata[
                "temporal_contexts"
            ],

            metadata[
                "max_product_id"
            ]
            + 1,
        ),

        dtype=np.uint8,
    )

    target_matrix.sort_indices()

    # --------------------------------------------------------
    # Internal temporal validation sample
    # --------------------------------------------------------

    validation_context_ids = np.sort(
        validation_pairs[
            "context_id"
        ]
        .unique()
        .astype(
            np.int32
        )
    )

    validation_sample_size = min(
        VALIDATION_SAMPLE_CONTEXTS,
        len(
            validation_context_ids
        ),
    )

    validation_sample = np.sort(
        rng.choice(
            validation_context_ids,
            size=
                validation_sample_size,
            replace=False,
        )
    )

    validation_sample_set = set(
        int(x)
        for x
        in validation_sample
    )

    sampled_truth_frame = (
        validation_pairs[
            validation_pairs[
                "context_id"
            ]
            .isin(
                validation_sample_set
            )
        ]
    )

    validation_truth = (
        sampled_truth_frame
        .groupby(
            "context_id"
        )[
            "product_id"
        ]
        .apply(
            lambda values:
                set(
                    int(x)
                    for x
                    in values
                )
        )
        .to_dict()
    )

    print(
        "Internal validation contexts sampled: "
        f"{len(validation_sample):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HistoryAwareTwoTower(

        num_products=
            metadata[
                "max_product_id"
            ]
            + 1,

        num_aisles=
            metadata[
                "max_aisle_id"
            ]
            + 1,

        num_departments=
            metadata[
                "max_department_id"
            ]
            + 1,

        item_numeric=
            torch.from_numpy(
                item_numeric_np
            ),

        item_aisle=
            torch.from_numpy(
                item_aisle_np
            ),

        item_department=
            torch.from_numpy(
                item_department_np
            ),

        embedding_dim=
            EMBEDDING_DIM,
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "\nModel parameters:",
        f"{parameter_count:,}"
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=
            LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_recall = float(
        "-inf"
    )

    best_epoch = 0

    best_state = None

    epochs_without_improvement = 0

    history = []

    n_train = len(
        train_context_ids
    )

    total_start = perf_counter()

    # ========================================================
    # Train
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = perf_counter()

        model.train()

        permutation = rng.permutation(
            n_train
        )

        loss_sum = 0.0

        example_count = 0

        batch_count = 0

        for start in range(
            0,
            n_train,
            BATCH_SIZE,
        ):

            indices = permutation[
                start:
                start
                +
                BATCH_SIZE
            ]

            batch_context_ids_np = (
                train_context_ids[
                    indices
                ]
            )

            batch_positive_np = (
                train_positive_products[
                    indices
                ]
            )

            batch_negative_np = (
                sample_negatives(
                    rng=rng,

                    context_ids=
                        batch_context_ids_np,

                    observed_product_ids=
                        observed_product_ids,

                    probabilities=
                        negative_probabilities,

                    target_matrix=
                        target_matrix,
                )
            )

            # ------------------------------------------------
            # Context tensors
            # ------------------------------------------------

            history_ids = (
                torch.from_numpy(
                    context_history_products[
                        batch_context_ids_np
                    ]
                )
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            history_weights = (
                torch.from_numpy(
                    context_history_weights[
                        batch_context_ids_np
                    ]
                    .astype(
                        np.float32
                    )
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            numeric = (
                torch.from_numpy(
                    context_numeric[
                        batch_context_ids_np
                    ]
                )
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            positive_ids = (
                torch.from_numpy(
                    batch_positive_np
                )
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            negative_ids = (
                torch.from_numpy(
                    batch_negative_np
                )
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            user_vectors = (
                model.encode_users(
                    history_product_ids=
                        history_ids,

                    history_weights=
                        history_weights,

                    history_numeric=
                        numeric,
                )
            )

            positive_vectors = (
                model.encode_items(
                    positive_ids
                )
            )

            negative_vectors = (
                model.encode_items(
                    negative_ids
                )
            )

            positive_scores = (
                user_vectors
                *
                positive_vectors
            ).sum(
                dim=1
            )

            negative_scores = (
                user_vectors
                *
                negative_vectors
            ).sum(
                dim=1
            )

            margin = (
                SCORE_SCALE
                *
                (
                    positive_scores
                    -
                    negative_scores
                )
            )

            loss = (
                -F.logsigmoid(
                    margin
                )
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            # ------------------------------------------------
            # Stats
            # ------------------------------------------------

            batch_size_actual = len(
                indices
            )

            loss_sum += (
                float(
                    loss.item()
                )
                *
                batch_size_actual
            )

            example_count += (
                batch_size_actual
            )

            batch_count += 1

            if (
                batch_count
                % 250
                == 0
            ):

                print(
                    f"Epoch "
                    f"{epoch}/{MAX_EPOCHS} "
                    f"| batch "
                    f"{batch_count:,} "
                    f"| loss "
                    f"{loss.item():.5f}"
                )

        average_loss = (
            loss_sum
            /
            example_count
        )

        # ----------------------------------------------------
        # Internal temporal validation
        # ----------------------------------------------------

        print(
            "\nEvaluating internal "
            "temporal Recall@50..."
        )

        validation_recall = (
            evaluate_recall_at_k(

                model=model,

                device=device,

                context_ids=
                    validation_sample,

                truth=
                    validation_truth,

                history_product_ids=
                    context_history_products,

                history_weights=
                    context_history_weights,

                history_numeric=
                    context_numeric,

                observed_product_ids=
                    observed_product_ids,

                k=
                    VALIDATION_RECALL_K,
            )
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

            "internal_recall_at_50":
                float(
                    validation_recall
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
            + "-" * 60
        )

        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Average loss: "
            f"{average_loss:.6f}"
        )

        print(
            "Internal Recall@50: "
            f"{validation_recall:.6f}"
        )

        print(
            f"Seconds: "
            f"{epoch_seconds:.2f}"
        )

        print(
            "Max GPU memory: "
            f"{gpu_memory:.2f} GB"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            validation_recall
            >
            best_recall
            + 1e-5
        ):

            best_recall = (
                validation_recall
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                {
                    key:
                        value.detach()
                        .cpu()
                        .clone()

                    for key, value
                    in model.state_dict().items()
                }
            )

            epochs_without_improvement = (
                0
            )

            print(
                "New best checkpoint."
            )

        else:

            epochs_without_improvement += (
                1
            )

            print(
                "No validation improvement. "
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # Restore best model
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "No valid model checkpoint found."
        )

    model.load_state_dict(
        best_state
    )

    total_seconds = (
        perf_counter()
        -
        total_start
    )

    # ========================================================
    # Save
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {

        "model_state_dict":
            best_state,

        "config": {

            "embedding_dim":
                EMBEDDING_DIM,

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

            "history_numeric_dim":
                4,

            "max_history_products":
                metadata[
                    "max_history_products"
                ],

            "history_orders":
                metadata[
                    "history_orders"
                ],
        },

        "training": {

            "best_epoch":
                best_epoch,

            "best_internal_recall_at_50":
                best_recall,

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

    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TWO-TOWER V2 TRAINING COMPLETE"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        "Best internal Recall@50: "
        f"{best_recall:.6f}"
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

    print(
        "\nSaved training results:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()