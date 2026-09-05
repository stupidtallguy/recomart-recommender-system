from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from src.models.two_tower import TwoTowerModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TWO_TOWER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "two_tower"
)

PAIR_PATH = (
    TWO_TOWER_DIR
    / "positive_pairs.parquet"
)

FEATURE_PATH = (
    TWO_TOWER_DIR
    / "feature_arrays.npz"
)

METADATA_PATH = (
    TWO_TOWER_DIR
    / "metadata.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "two_tower_v1.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidates"
    / "two_tower_v1_top50.parquet"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "two_tower_retrieval_v1.json"
)


TOP_K = 50

# Safe for your RTX 3050 4GB.
USER_BATCH_SIZE = 2048


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for "
            "Two-Tower retrieval."
        )

    device = torch.device("cuda")

    print(
        "Device:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print(
        "\nLoading feature arrays..."
    )

    arrays = np.load(
        FEATURE_PATH
    )

    user_features_np = (
        arrays["user_features"]
        .astype(
            np.float32
        )
    )

    item_features_np = (
        arrays["item_features"]
        .astype(
            np.float32
        )
    )

    item_aisle_np = (
        arrays["item_aisle"]
        .astype(
            np.int64
        )
    )

    item_department_np = (
        arrays["item_department"]
        .astype(
            np.int64
        )
    )

    observed_product_ids = (
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

    print(
        "Loading user IDs..."
    )

    pairs = pd.read_parquet(
        PAIR_PATH,
        columns=["user_id"],
    )

    user_ids = np.sort(
        pairs[
            "user_id"
        ]
        .unique()
        .astype(
            np.int32
        )
    )

    del pairs

    print(
        f"Users: "
        f"{len(user_ids):,}"
    )

    print(
        f"Candidate products: "
        f"{len(observed_product_ids):,}"
    )

    # --------------------------------------------------
    # Recreate model
    # --------------------------------------------------

    print(
        "Loading Two-Tower model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint[
        "config"
    ]

    model = TwoTowerModel(

        num_users=
            config[
                "max_user_id"
            ] + 1,

        num_products=
            config[
                "max_product_id"
            ] + 1,

        num_aisles=
            config[
                "max_aisle_id"
            ] + 1,

        num_departments=
            config[
                "max_department_id"
            ] + 1,

        user_features=
            torch.from_numpy(
                user_features_np
            ),

        item_features=
            torch.from_numpy(
                item_features_np
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

    # --------------------------------------------------
    # Encode all candidate products ONCE
    # --------------------------------------------------

    print(
        "\nEncoding all products..."
    )

    product_tensor = (
        torch.from_numpy(
            observed_product_ids
        )
        .long()
        .to(device)
    )

    start = perf_counter()

    with torch.inference_mode():

        item_embeddings = (
            model.encode_items(
                product_tensor
            )
        )

    item_seconds = (
        perf_counter()
        - start
    )

    print(
        f"Item embedding seconds: "
        f"{item_seconds:.2f}"
    )

    print(
        "Item embedding shape:",
        tuple(
            item_embeddings.shape
        ),
    )

    # --------------------------------------------------
    # Exact Top-K retrieval in user batches
    # --------------------------------------------------

    all_user_ids = []
    all_recommendations = []
    all_scores = []

    print(
        "\nRunning exact GPU retrieval..."
    )

    retrieval_start = (
        perf_counter()
    )

    with torch.inference_mode():

        for start_index in tqdm(
            range(
                0,
                len(user_ids),
                USER_BATCH_SIZE,
            ),
            desc="Retrieving Top-50",
        ):

            batch_user_ids_np = (
                user_ids[
                    start_index:
                    start_index
                    + USER_BATCH_SIZE
                ]
            )

            batch_user_ids = (
                torch.from_numpy(
                    batch_user_ids_np
                )
                .long()
                .to(device)
            )

            user_embeddings = (
                model.encode_users(
                    batch_user_ids
                )
            )

            # Since both towers output normalized
            # vectors, this dot product is cosine
            # similarity.
            similarity = (
                user_embeddings
                @
                item_embeddings.T
            )

            top_scores, top_indices = (
                torch.topk(
                    similarity,
                    k=TOP_K,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
            )

            top_products = (
                product_tensor[
                    top_indices
                ]
            )

            top_products_np = (
                top_products
                .cpu()
                .numpy()
            )

            top_scores_np = (
                top_scores
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            for i, user_id in enumerate(
                batch_user_ids_np
            ):

                all_user_ids.append(
                    int(user_id)
                )

                all_recommendations.append(
                    [
                        int(x)
                        for x
                        in top_products_np[
                            i
                        ]
                    ]
                )

                all_scores.append(
                    [
                        float(x)
                        for x
                        in top_scores_np[
                            i
                        ]
                    ]
                )

            del similarity
            del top_scores
            del top_indices
            del user_embeddings

    retrieval_seconds = (
        perf_counter()
        -
        retrieval_start
    )

    # --------------------------------------------------
    # Save candidate store
    # --------------------------------------------------

    print(
        "\nWriting candidate file..."
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pydict(
        {
            "user_id":
                all_user_ids,

            "recommendations":
                all_recommendations,

            # Keep raw Two-Tower cosine scores.
            # These can later become ranking features.
            "scores":
                all_scores,
        }
    )

    pq.write_table(
        table,
        OUTPUT_PATH,
        compression="snappy",
    )

    result = {

        "model":
            "two_tower_v1",

        "users":
            len(
                all_user_ids
            ),

        "candidate_products":
            len(
                observed_product_ids
            ),

        "top_k":
            TOP_K,

        "embedding_dim":
            config[
                "embedding_dim"
            ],

        "user_batch_size":
            USER_BATCH_SIZE,

        "item_embedding_seconds":
            item_seconds,

        "retrieval_seconds":
            retrieval_seconds,

        "max_gpu_memory_gb":
            (
                torch.cuda
                .max_memory_allocated()
                /
                1024**3
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
        "TWO-TOWER CANDIDATES COMPLETE"
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