from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
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

SERVING_CONTEXT_PATH = (
    DATA_DIR
    / "serving_contexts.npz"
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidates"
    / "two_tower_v2_top50.parquet"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "two_tower_v2_retrieval.json"
)


# ============================================================
# Configuration
# ============================================================

TOP_K = 50

# Safer than 2048 because the v2 user tower also encodes
# up to 80 historical products/user.
USER_BATCH_SIZE = 1024

WRITE_CHUNK_USERS = 20_000


# ============================================================
# Streaming Parquet writer
# ============================================================

class CandidateWriter:

    def __init__(
        self,
        output_path: Path,
    ) -> None:

        self.output_path = (
            output_path
        )

        self.tmp_path = (
            output_path.with_suffix(
                output_path.suffix
                + ".tmp"
            )
        )

        self.writer = None

        self.user_ids = []
        self.recommendations = []
        self.scores = []

        if self.tmp_path.exists():
            self.tmp_path.unlink()


    def append_batch(
        self,
        user_ids: np.ndarray,
        product_ids: np.ndarray,
        scores: np.ndarray,
    ) -> None:

        for row_index, user_id in enumerate(
            user_ids
        ):

            self.user_ids.append(
                int(user_id)
            )

            self.recommendations.append(
                product_ids[
                    row_index
                ].astype(
                    np.int32,
                    copy=False,
                )
                .tolist()
            )

            self.scores.append(
                scores[
                    row_index
                ].astype(
                    np.float32,
                    copy=False,
                )
                .tolist()
            )

        if (
            len(
                self.user_ids
            )
            >=
            WRITE_CHUNK_USERS
        ):
            self.flush()


    def flush(
        self,
    ) -> None:

        if not self.user_ids:
            return

        table = pa.Table.from_arrays(
            [
                pa.array(
                    self.user_ids,
                    type=pa.int32(),
                ),

                pa.array(
                    self.recommendations,
                    type=pa.list_(
                        pa.int32()
                    ),
                ),

                pa.array(
                    self.scores,
                    type=pa.list_(
                        pa.float32()
                    ),
                ),
            ],
            names=[
                "user_id",
                "recommendations",
                "scores",
            ],
        )

        if self.writer is None:

            self.writer = (
                pq.ParquetWriter(
                    self.tmp_path,
                    table.schema,
                    compression="snappy",
                )
            )

        self.writer.write_table(
            table
        )

        self.user_ids.clear()
        self.recommendations.clear()
        self.scores.clear()


    def close(
        self,
    ) -> None:

        self.flush()

        if self.writer is None:

            raise RuntimeError(
                "No Two-Tower v2 "
                "candidates were written."
            )

        self.writer.close()

        self.writer = None

        if self.output_path.exists():
            self.output_path.unlink()

        self.tmp_path.replace(
            self.output_path
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
            "CUDA is required for "
            "Two-Tower v2 retrieval."
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

    torch.set_float32_matmul_precision(
        "high"
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        True
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # Serving contexts
    # --------------------------------------------------------

    print(
        "\nLoading serving contexts..."
    )

    contexts = np.load(
        SERVING_CONTEXT_PATH
    )

    user_ids = (
        contexts[
            "user_ids"
        ]
        .astype(
            np.int32
        )
    )

    history_product_ids = (
        contexts[
            "history_product_ids"
        ]
        .astype(
            np.int32
        )
    )

    history_weights = (
        contexts[
            "history_weights"
        ]
    )

    history_numeric = (
        contexts[
            "numeric_features"
        ]
        .astype(
            np.float32
        )
    )

    print(
        f"Users: "
        f"{len(user_ids):,}"
    )

    # --------------------------------------------------------
    # Item features
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

    print(
        f"Candidate products: "
        f"{len(observed_product_ids):,}"
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    print(
        "Loading best Two-Tower v2 checkpoint..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config = (
        checkpoint[
            "config"
        ]
    )

    training_info = (
        checkpoint[
            "training"
        ]
    )

    print(
        "Checkpoint best epoch:",
        training_info[
            "best_epoch"
        ],
    )

    print(
        "Checkpoint internal Recall@50:",
        training_info[
            "best_internal_recall_at_50"
        ],
    )

    # --------------------------------------------------------
    # Recreate model
    # --------------------------------------------------------

    model = HistoryAwareTwoTower(

        num_products=
            config[
                "max_product_id"
            ]
            + 1,

        num_aisles=
            config[
                "max_aisle_id"
            ]
            + 1,

        num_departments=
            config[
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
            config[
                "embedding_dim"
            ],

        history_numeric_dim=
            config[
                "history_numeric_dim"
            ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # Encode all products ONCE
    # --------------------------------------------------------

    print(
        "\nEncoding all candidate products..."
    )

    product_tensor = (
        torch.from_numpy(
            observed_product_ids
        )
        .long()
        .to(
            device
        )
    )

    item_start = (
        perf_counter()
    )

    with torch.inference_mode():

        item_vectors = (
            model.encode_items(
                product_tensor
            )
        )

    item_seconds = (
        perf_counter()
        -
        item_start
    )

    print(
        "Item vector shape:",
        tuple(
            item_vectors.shape
        ),
    )

    print(
        f"Item encoding seconds: "
        f"{item_seconds:.3f}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = CandidateWriter(
        OUTPUT_PATH
    )

    retrieval_start = (
        perf_counter()
    )

    print(
        "\nRunning exact "
        "history-aware Top-50 retrieval..."
    )

    with torch.inference_mode():

        for start in tqdm(
            range(
                0,
                len(user_ids),
                USER_BATCH_SIZE,
            ),
            desc="Retrieving Two-Tower v2 Top-50",
        ):

            end = min(
                start
                +
                USER_BATCH_SIZE,

                len(
                    user_ids
                ),
            )

            batch_user_ids = (
                user_ids[
                    start:end
                ]
            )

            # ----------------------------------------------
            # User history
            # ----------------------------------------------

            batch_history_ids = (
                torch.from_numpy(
                    history_product_ids[
                        start:end
                    ]
                )
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            batch_history_weights = (
                torch.from_numpy(
                    history_weights[
                        start:end
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

            batch_numeric = (
                torch.from_numpy(
                    history_numeric[
                        start:end
                    ]
                )
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # ----------------------------------------------
            # Encode users
            # ----------------------------------------------

            user_vectors = (
                model.encode_users(
                    history_product_ids=
                        batch_history_ids,

                    history_weights=
                        batch_history_weights,

                    history_numeric=
                        batch_numeric,
                )
            )

            # ----------------------------------------------
            # Exact dot-product retrieval
            #
            # Both user/item vectors are normalized,
            # therefore this is cosine similarity.
            # ----------------------------------------------

            similarity = (
                user_vectors
                @
                item_vectors.T
            )

            (
                top_scores,
                top_indices,
            ) = torch.topk(
                similarity,
                k=
                    TOP_K,
                dim=1,
                largest=True,
                sorted=True,
            )

            top_products = (
                product_tensor[
                    top_indices
                ]
            )

            # ----------------------------------------------
            # CPU arrays
            # ----------------------------------------------

            top_products_np = (
                top_products
                .cpu()
                .numpy()
                .astype(
                    np.int32,
                    copy=False,
                )
            )

            top_scores_np = (
                top_scores
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            writer.append_batch(
                user_ids=
                    batch_user_ids,

                product_ids=
                    top_products_np,

                scores=
                    top_scores_np,
            )

            del batch_history_ids
            del batch_history_weights
            del batch_numeric
            del user_vectors
            del similarity
            del top_scores
            del top_indices
            del top_products

    writer.close()

    retrieval_seconds = (
        perf_counter()
        -
        retrieval_start
    )

    max_gpu_memory = (
        torch.cuda
        .max_memory_allocated()
        /
        1024**3
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {

        "model":
            "two_tower_v2_history_aware",

        "checkpoint_epoch":
            int(
                training_info[
                    "best_epoch"
                ]
            ),

        "internal_recall_at_50":
            float(
                training_info[
                    "best_internal_recall_at_50"
                ]
            ),

        "users":
            int(
                len(
                    user_ids
                )
            ),

        "candidate_products":
            int(
                len(
                    observed_product_ids
                )
            ),

        "top_k":
            TOP_K,

        "embedding_dim":
            int(
                config[
                    "embedding_dim"
                ]
            ),

        "user_batch_size":
            USER_BATCH_SIZE,

        "item_embedding_seconds":
            float(
                item_seconds
            ),

        "retrieval_seconds":
            float(
                retrieval_seconds
            ),

        "max_gpu_memory_gb":
            float(
                max_gpu_memory
            ),
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TWO-TOWER V2 CANDIDATES COMPLETE"
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()