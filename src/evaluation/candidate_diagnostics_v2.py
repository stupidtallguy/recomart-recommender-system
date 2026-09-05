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
    / "candidate_diagnostics_v2.json"
)


SOURCE_FILES = {

    "repeat":
        "repeat_purchase_top50.parquet",

    "als":
        "als_v2_top50.parquet",

    "content":
        "content_v1_top50.parquet",

    "two_tower":
        "two_tower_v1_top50.parquet",
}


def load_candidates():

    frames = []

    for source, filename in (
        SOURCE_FILES.items()
    ):

        print(
            f"Loading {source}..."
        )

        frame = pd.read_parquet(
            CANDIDATE_DIR / filename,
            columns=[
                "user_id",
                "recommendations",
            ],
        )

        frame = frame.rename(
            columns={
                "recommendations":
                    source
            }
        )

        frames.append(
            frame
        )

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
            candidates
            & truth
        )
        /
        len(truth)
    )


def jaccard(
    left,
    right,
):

    union = (
        left
        | right
    )

    if not union:
        return 0.0

    return (
        len(
            left
            & right
        )
        /
        len(union)
    )


def main():

    print(
        "Loading candidate stores..."
    )

    candidates = (
        load_candidates()
    )

    print(
        f"Users: "
        f"{len(candidates):,}"
    )

    print(
        "\nLoading validation truth..."
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
        .groupby(
            "user_id"
        )["product_id"]
        .apply(set)
        .to_dict()
    )

    repeat_recall = []
    als_recall = []
    content_recall = []
    two_tower_recall = []

    old_union_recall = []
    new_union_recall = []

    old_pool_sizes = []
    new_pool_sizes = []

    tt_unique_candidates = []
    tt_incremental_hits = []
    tt_added_hit_users = []

    repeat_tt_overlap = []
    als_tt_overlap = []
    content_tt_overlap = []


    for row in tqdm(
        candidates.itertuples(
            index=False
        ),
        total=len(candidates),
        desc="Diagnosing 4-source retrieval",
    ):

        user_id = int(
            row.user_id
        )

        truth = (
            truth_lookup[
                user_id
            ]
        )

        repeat = set(
            int(x)
            for x
            in row.repeat
        )

        als = set(
            int(x)
            for x
            in row.als
        )

        content = set(
            int(x)
            for x
            in row.content
        )

        two_tower = set(
            int(x)
            for x
            in row.two_tower
        )

        old_union = (
            repeat
            | als
            | content
        )

        new_union = (
            old_union
            | two_tower
        )

        repeat_recall.append(
            recall(
                repeat,
                truth,
            )
        )

        als_recall.append(
            recall(
                als,
                truth,
            )
        )

        content_recall.append(
            recall(
                content,
                truth,
            )
        )

        two_tower_recall.append(
            recall(
                two_tower,
                truth,
            )
        )

        old_union_recall.append(
            recall(
                old_union,
                truth,
            )
        )

        new_union_recall.append(
            recall(
                new_union,
                truth,
            )
        )

        old_pool_sizes.append(
            len(
                old_union
            )
        )

        new_pool_sizes.append(
            len(
                new_union
            )
        )

        unique_tt = (
            two_tower
            -
            old_union
        )

        incremental_hits = (
            unique_tt
            & truth
        )

        tt_unique_candidates.append(
            len(
                unique_tt
            )
        )

        tt_incremental_hits.append(
            len(
                incremental_hits
            )
        )

        tt_added_hit_users.append(
            int(
                len(
                    incremental_hits
                )
                > 0
            )
        )

        repeat_tt_overlap.append(
            jaccard(
                repeat,
                two_tower,
            )
        )

        als_tt_overlap.append(
            jaccard(
                als,
                two_tower,
            )
        )

        content_tt_overlap.append(
            jaccard(
                content,
                two_tower,
            )
        )


    old_recall = float(
        np.mean(
            old_union_recall
        )
    )

    new_recall = float(
        np.mean(
            new_union_recall
        )
    )

    result = {

        "users":
            len(candidates),

        "individual_candidate_recall_at_50": {

            "repeat":
                float(
                    np.mean(
                        repeat_recall
                    )
                ),

            "als":
                float(
                    np.mean(
                        als_recall
                    )
                ),

            "content":
                float(
                    np.mean(
                        content_recall
                    )
                ),

            "two_tower":
                float(
                    np.mean(
                        two_tower_recall
                    )
                ),
        },

        "candidate_union": {

            "three_source_recall":
                old_recall,

            "four_source_recall":
                new_recall,

            "absolute_recall_gain":
                (
                    new_recall
                    -
                    old_recall
                ),

            "relative_recall_gain":
                (
                    (
                        new_recall
                        -
                        old_recall
                    )
                    /
                    old_recall
                    if old_recall > 0
                    else 0.0
                ),

            "mean_three_source_pool_size":
                float(
                    np.mean(
                        old_pool_sizes
                    )
                ),

            "mean_four_source_pool_size":
                float(
                    np.mean(
                        new_pool_sizes
                    )
                ),
        },

        "two_tower_incremental_value": {

            "mean_unique_candidates_beyond_existing_union":
                float(
                    np.mean(
                        tt_unique_candidates
                    )
                ),

            "mean_incremental_relevant_items":
                float(
                    np.mean(
                        tt_incremental_hits
                    )
                ),

            "fraction_users_with_incremental_relevant_item":
                float(
                    np.mean(
                        tt_added_hit_users
                    )
                ),
        },

        "two_tower_average_jaccard_overlap": {

            "vs_repeat":
                float(
                    np.mean(
                        repeat_tt_overlap
                    )
                ),

            "vs_als":
                float(
                    np.mean(
                        als_tt_overlap
                    )
                ),

            "vs_content":
                float(
                    np.mean(
                        content_tt_overlap
                    )
                ),
        },
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
        "\n"
        + "=" * 70
    )

    print(
        "FOUR-SOURCE CANDIDATE DIAGNOSTICS"
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