from __future__ import annotations

import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CANDIDATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidates"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_interactions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "candidate_diagnostics.json"
)


SOURCE_FILES = {
    "repeat":
        "repeat_purchase_top50.parquet",

    "als":
        "als_v2_top50.parquet",

    "content":
        "content_v1_top50.parquet",
}


def load_candidates():

    frames = []

    for source, filename in (
        SOURCE_FILES.items()
    ):

        frame = pd.read_parquet(
            CANDIDATE_DIR / filename,
            columns=[
                "user_id",
                "recommendations",
            ],
        )

        frame = frame.rename(
            columns={
                "recommendations": source
            }
        )

        frames.append(frame)

    return reduce(
        lambda left, right:
            left.merge(
                right,
                on="user_id",
                how="inner",
                validate="one_to_one",
            ),
        frames,
    )


def recall(
    candidates,
    truth,
):

    if not truth:
        return 0.0

    return (
        len(
            set(candidates)
            & truth
        )
        /
        len(truth)
    )


def main():

    print(
        "Loading candidate lists..."
    )

    candidates = load_candidates()

    print(
        "Loading validation truth..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    truth_lookup = (
        validation
        .groupby("user_id")
        ["product_id"]
        .apply(set)
        .to_dict()
    )

    stats = {

        "repeat_recall_50": [],
        "als_recall_50": [],
        "content_recall_50": [],
        "union_recall": [],

        "union_size": [],

        "repeat_als_overlap": [],
        "repeat_content_overlap": [],
        "als_content_overlap": [],

        "als_incremental_hits": [],
        "content_incremental_hits": [],

        "union_positive_count": [],
    }


    for row in tqdm(
        candidates.itertuples(
            index=False
        ),
        total=len(candidates),
        desc="Diagnosing candidates",
    ):

        truth = truth_lookup[
            int(row.user_id)
        ]

        repeat = list(row.repeat)
        als = list(row.als)
        content = list(row.content)

        repeat_set = set(repeat)
        als_set = set(als)
        content_set = set(content)

        union = (
            repeat_set
            | als_set
            | content_set
        )

        stats[
            "repeat_recall_50"
        ].append(
            recall(
                repeat,
                truth,
            )
        )

        stats[
            "als_recall_50"
        ].append(
            recall(
                als,
                truth,
            )
        )

        stats[
            "content_recall_50"
        ].append(
            recall(
                content,
                truth,
            )
        )

        stats[
            "union_recall"
        ].append(
            recall(
                union,
                truth,
            )
        )

        stats[
            "union_size"
        ].append(
            len(union)
        )

        # Jaccard overlaps
        repeat_als_union = (
            repeat_set
            | als_set
        )

        repeat_content_union = (
            repeat_set
            | content_set
        )

        als_content_union = (
            als_set
            | content_set
        )

        stats[
            "repeat_als_overlap"
        ].append(
            len(
                repeat_set
                & als_set
            )
            /
            max(
                1,
                len(
                    repeat_als_union
                ),
            )
        )

        stats[
            "repeat_content_overlap"
        ].append(
            len(
                repeat_set
                & content_set
            )
            /
            max(
                1,
                len(
                    repeat_content_union
                ),
            )
        )

        stats[
            "als_content_overlap"
        ].append(
            len(
                als_set
                & content_set
            )
            /
            max(
                1,
                len(
                    als_content_union
                ),
            )
        )

        # Relevant products contributed by ALS
        # that Repeat does NOT contain.
        als_incremental = (
            (
                als_set
                - repeat_set
            )
            & truth
        )

        content_incremental = (
            (
                content_set
                - repeat_set
            )
            & truth
        )

        stats[
            "als_incremental_hits"
        ].append(
            len(
                als_incremental
            )
        )

        stats[
            "content_incremental_hits"
        ].append(
            len(
                content_incremental
            )
        )

        stats[
            "union_positive_count"
        ].append(
            len(
                union
                & truth
            )
        )


    result = {

        "users":
            len(candidates),

        "mean_candidate_pool_size":
            float(
                np.mean(
                    stats[
                        "union_size"
                    ]
                )
            ),

        "recall_at_candidate_stage": {

            "repeat_top50":
                float(
                    np.mean(
                        stats[
                            "repeat_recall_50"
                        ]
                    )
                ),

            "als_top50":
                float(
                    np.mean(
                        stats[
                            "als_recall_50"
                        ]
                    )
                ),

            "content_top50":
                float(
                    np.mean(
                        stats[
                            "content_recall_50"
                        ]
                    )
                ),

            "union":
                float(
                    np.mean(
                        stats[
                            "union_recall"
                        ]
                    )
                ),
        },

        "average_jaccard_overlap": {

            "repeat_vs_als":
                float(
                    np.mean(
                        stats[
                            "repeat_als_overlap"
                        ]
                    )
                ),

            "repeat_vs_content":
                float(
                    np.mean(
                        stats[
                            "repeat_content_overlap"
                        ]
                    )
                ),

            "als_vs_content":
                float(
                    np.mean(
                        stats[
                            "als_content_overlap"
                        ]
                    )
                ),
        },

        "incremental_relevant_items_per_user": {

            "als_beyond_repeat":
                float(
                    np.mean(
                        stats[
                            "als_incremental_hits"
                        ]
                    )
                ),

            "content_beyond_repeat":
                float(
                    np.mean(
                        stats[
                            "content_incremental_hits"
                        ]
                    )
                ),
        },

        "average_relevant_items_in_union":
            float(
                np.mean(
                    stats[
                        "union_positive_count"
                    ]
                )
            ),
    }


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=4,
        ),
        encoding="utf-8",
    )


    print(
        "\nCandidate Diagnostics"
    )

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()