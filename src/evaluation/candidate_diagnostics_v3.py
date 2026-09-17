from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CANDIDATE_DIR = (
    PROCESSED_DIR
    / "candidates"
)

VALIDATION_PATH = (
    PROCESSED_DIR
    / "validation_interactions.parquet"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmarks"
    / "candidate_diagnostics_v3.json"
)


# ============================================================
# Candidate files
# ============================================================

SOURCE_PATHS = {

    "repeat":
        CANDIDATE_DIR
        / "repeat_purchase_top50.parquet",

    "als":
        CANDIDATE_DIR
        / "als_v2_top50.parquet",

    "content":
        CANDIDATE_DIR
        / "content_v1_top50.parquet",

    "two_tower_v1":
        CANDIDATE_DIR
        / "two_tower_v1_top50.parquet",

    "two_tower_v2":
        CANDIDATE_DIR
        / "two_tower_v2_top50.parquet",
}


# ============================================================
# Loading
# ============================================================

def load_source(
    path: Path,
    name: str,
) -> dict[int, list[int]]:

    print(
        f"Loading {name}:"
    )

    print(
        path
    )

    df = pd.read_parquet(
        path,
        columns=[
            "user_id",
            "recommendations",
        ],
    )

    result = {}

    for row in df.itertuples(
        index=False
    ):

        result[
            int(
                row.user_id
            )
        ] = [
            int(x)
            for x
            in row.recommendations
        ]

    print(
        f"Users: {len(result):,}"
    )

    return result


def load_truth() -> dict[int, set[int]]:

    print(
        "\nLoading official validation truth..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            "user_id",
            "product_id",
        ],
    )

    truth = (
        validation
        .groupby(
            "user_id"
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

    truth = {
        int(user_id):
            products

        for user_id, products
        in truth.items()
    }

    print(
        f"Validation users: "
        f"{len(truth):,}"
    )

    return truth


# ============================================================
# Metrics
# ============================================================

def candidate_recall(
    candidates: set[int],
    truth: set[int],
) -> float:

    if not truth:
        return 0.0

    return (
        len(
            candidates
            &
            truth
        )
        /
        len(
            truth
        )
    )


def jaccard(
    a: set[int],
    b: set[int],
) -> float:

    union = (
        a
        |
        b
    )

    if not union:
        return 0.0

    return (
        len(
            a
            &
            b
        )
        /
        len(
            union
        )
    )


# ============================================================
# Main diagnostic
# ============================================================

def main():

    # --------------------------------------------------------
    # Load all sources
    # --------------------------------------------------------

    sources = {

        name:
            load_source(
                path,
                name,
            )

        for name, path
        in SOURCE_PATHS.items()
    }

    truth = load_truth()

    users = sorted(
        truth.keys()
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    individual_recall_sum = {
        name: 0.0
        for name
        in SOURCE_PATHS
    }

    individual_relevant_sum = {
        name: 0
        for name
        in SOURCE_PATHS
    }

    jaccard_sum = {
        (
            "two_tower_v2",
            "repeat",
        ): 0.0,

        (
            "two_tower_v2",
            "als",
        ): 0.0,

        (
            "two_tower_v2",
            "content",
        ): 0.0,

        (
            "two_tower_v2",
            "two_tower_v1",
        ): 0.0,
    }

    union_recall_sum = {
        "three_source": 0.0,
        "three_source_plus_tt_v1": 0.0,
        "three_source_plus_tt_v2": 0.0,
        "five_source": 0.0,
    }

    union_size_sum = {
        "three_source": 0,
        "three_source_plus_tt_v1": 0,
        "three_source_plus_tt_v2": 0,
        "five_source": 0,
    }

    # TT-v2 beyond old 3-source system
    tt2_unique_beyond_three_sum = 0
    tt2_relevant_beyond_three_sum = 0
    users_tt2_relevant_beyond_three = 0

    # TT-v2 beyond 3-source + TT-v1
    tt2_unique_beyond_tt1_union_sum = 0
    tt2_relevant_beyond_tt1_union_sum = 0
    users_tt2_relevant_beyond_tt1_union = 0

    # TT-v1 incremental for reference
    tt1_unique_beyond_three_sum = 0
    tt1_relevant_beyond_three_sum = 0
    users_tt1_relevant_beyond_three = 0

    # --------------------------------------------------------
    # Per-user analysis
    # --------------------------------------------------------

    print(
        "\nRunning five-source diagnostics..."
    )

    for user_id in tqdm(
        users,
        desc="Candidate diagnostics",
    ):

        relevant = truth[
            user_id
        ]

        repeat = set(
            sources[
                "repeat"
            ].get(
                user_id,
                [],
            )
        )

        als = set(
            sources[
                "als"
            ].get(
                user_id,
                [],
            )
        )

        content = set(
            sources[
                "content"
            ].get(
                user_id,
                [],
            )
        )

        tt1 = set(
            sources[
                "two_tower_v1"
            ].get(
                user_id,
                [],
            )
        )

        tt2 = set(
            sources[
                "two_tower_v2"
            ].get(
                user_id,
                [],
            )
        )

        candidate_sets = {
            "repeat": repeat,
            "als": als,
            "content": content,
            "two_tower_v1": tt1,
            "two_tower_v2": tt2,
        }

        # ----------------------------------------------------
        # Individual source recall
        # ----------------------------------------------------

        for name, candidates in (
            candidate_sets.items()
        ):

            hits = len(
                candidates
                &
                relevant
            )

            individual_relevant_sum[
                name
            ] += hits

            individual_recall_sum[
                name
            ] += candidate_recall(
                candidates,
                relevant,
            )

        # ----------------------------------------------------
        # TT-v2 overlap
        # ----------------------------------------------------

        jaccard_sum[
            (
                "two_tower_v2",
                "repeat",
            )
        ] += jaccard(
            tt2,
            repeat,
        )

        jaccard_sum[
            (
                "two_tower_v2",
                "als",
            )
        ] += jaccard(
            tt2,
            als,
        )

        jaccard_sum[
            (
                "two_tower_v2",
                "content",
            )
        ] += jaccard(
            tt2,
            content,
        )

        jaccard_sum[
            (
                "two_tower_v2",
                "two_tower_v1",
            )
        ] += jaccard(
            tt2,
            tt1,
        )

        # ----------------------------------------------------
        # Unions
        # ----------------------------------------------------

        three_source = (
            repeat
            |
            als
            |
            content
        )

        three_plus_tt1 = (
            three_source
            |
            tt1
        )

        three_plus_tt2 = (
            three_source
            |
            tt2
        )

        five_source = (
            three_source
            |
            tt1
            |
            tt2
        )

        unions = {

            "three_source":
                three_source,

            "three_source_plus_tt_v1":
                three_plus_tt1,

            "three_source_plus_tt_v2":
                three_plus_tt2,

            "five_source":
                five_source,
        }

        for name, union in (
            unions.items()
        ):

            union_recall_sum[
                name
            ] += candidate_recall(
                union,
                relevant,
            )

            union_size_sum[
                name
            ] += len(
                union
            )

        # ----------------------------------------------------
        # TT-v1 incremental over 3-source
        # ----------------------------------------------------

        tt1_unique = (
            tt1
            -
            three_source
        )

        tt1_incremental_relevant = (
            tt1_unique
            &
            relevant
        )

        tt1_unique_beyond_three_sum += (
            len(
                tt1_unique
            )
        )

        tt1_relevant_beyond_three_sum += (
            len(
                tt1_incremental_relevant
            )
        )

        if tt1_incremental_relevant:

            users_tt1_relevant_beyond_three += 1

        # ----------------------------------------------------
        # TT-v2 incremental over 3-source
        # ----------------------------------------------------

        tt2_unique_three = (
            tt2
            -
            three_source
        )

        tt2_relevant_three = (
            tt2_unique_three
            &
            relevant
        )

        tt2_unique_beyond_three_sum += (
            len(
                tt2_unique_three
            )
        )

        tt2_relevant_beyond_three_sum += (
            len(
                tt2_relevant_three
            )
        )

        if tt2_relevant_three:

            users_tt2_relevant_beyond_three += 1

        # ----------------------------------------------------
        # MOST IMPORTANT:
        #
        # TT-v2 incremental after TT-v1 already exists
        # ----------------------------------------------------

        tt2_unique_five = (
            tt2
            -
            three_plus_tt1
        )

        tt2_relevant_five = (
            tt2_unique_five
            &
            relevant
        )

        tt2_unique_beyond_tt1_union_sum += (
            len(
                tt2_unique_five
            )
        )

        tt2_relevant_beyond_tt1_union_sum += (
            len(
                tt2_relevant_five
            )
        )

        if tt2_relevant_five:

            users_tt2_relevant_beyond_tt1_union += (
                1
            )

    # --------------------------------------------------------
    # Final calculations
    # --------------------------------------------------------

    n_users = len(
        users
    )

    individual = {}

    for name in SOURCE_PATHS:

        individual[
            name
        ] = {

            "candidate_recall_at_50":
                individual_recall_sum[
                    name
                ]
                /
                n_users,

            "mean_relevant_items_per_user":
                individual_relevant_sum[
                    name
                ]
                /
                n_users,
        }

    overlap = {

        f"{a}_vs_{b}":
            jaccard_sum[
                (
                    a,
                    b,
                )
            ]
            /
            n_users

        for a, b
        in jaccard_sum
    }

    unions_result = {}

    for name in union_recall_sum:

        unions_result[
            name
        ] = {

            "candidate_recall":
                union_recall_sum[
                    name
                ]
                /
                n_users,

            "mean_pool_size":
                union_size_sum[
                    name
                ]
                /
                n_users,
        }

    three_recall = (
        unions_result[
            "three_source"
        ][
            "candidate_recall"
        ]
    )

    tt1_union_recall = (
        unions_result[
            "three_source_plus_tt_v1"
        ][
            "candidate_recall"
        ]
    )

    tt2_union_recall = (
        unions_result[
            "three_source_plus_tt_v2"
        ][
            "candidate_recall"
        ]
    )

    five_recall = (
        unions_result[
            "five_source"
        ][
            "candidate_recall"
        ]
    )

    result = {

        "users_evaluated":
            n_users,

        "individual_sources":
            individual,

        "average_jaccard":
            overlap,

        "candidate_unions":
            unions_result,

        "tt_v1_incremental_beyond_three_source": {

            "mean_unique_candidates_per_user":
                tt1_unique_beyond_three_sum
                /
                n_users,

            "mean_incremental_relevant_items_per_user":
                tt1_relevant_beyond_three_sum
                /
                n_users,

            "fraction_users_with_incremental_relevant":
                users_tt1_relevant_beyond_three
                /
                n_users,

            "candidate_recall_absolute_gain":
                tt1_union_recall
                -
                three_recall,
        },

        "tt_v2_incremental_beyond_three_source": {

            "mean_unique_candidates_per_user":
                tt2_unique_beyond_three_sum
                /
                n_users,

            "mean_incremental_relevant_items_per_user":
                tt2_relevant_beyond_three_sum
                /
                n_users,

            "fraction_users_with_incremental_relevant":
                users_tt2_relevant_beyond_three
                /
                n_users,

            "candidate_recall_absolute_gain":
                tt2_union_recall
                -
                three_recall,
        },

        "tt_v2_incremental_after_tt_v1": {

            "mean_unique_candidates_per_user":
                tt2_unique_beyond_tt1_union_sum
                /
                n_users,

            "mean_incremental_relevant_items_per_user":
                tt2_relevant_beyond_tt1_union_sum
                /
                n_users,

            "fraction_users_with_incremental_relevant":
                users_tt2_relevant_beyond_tt1_union
                /
                n_users,

            "candidate_recall_absolute_gain":
                five_recall
                -
                tt1_union_recall,
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FIVE-SOURCE CANDIDATE DIAGNOSTICS"
    )

    print(
        "=" * 75
    )

    print(
        "\nINDIVIDUAL RECALL@50"
    )

    for name, metrics in (
        individual.items()
    ):

        print(
            f"{name:20s}: "
            f"{metrics['candidate_recall_at_50']:.6f}"
        )

    print(
        "\nTT-V2 AVERAGE JACCARD"
    )

    for name, value in (
        overlap.items()
    ):

        print(
            f"{name:35s}: "
            f"{value:.6f}"
        )

    print(
        "\nUNION RECALL"
    )

    for name, metrics in (
        unions_result.items()
    ):

        print(
            f"{name:30s}: "
            f"Recall={metrics['candidate_recall']:.6f} "
            f"| Pool={metrics['mean_pool_size']:.2f}"
        )

    print(
        "\nTT-V2 BEYOND THREE-SOURCE"
    )

    section = result[
        "tt_v2_incremental_beyond_three_source"
    ]

    print(
        "Unique candidates/user: "
        f"{section['mean_unique_candidates_per_user']:.4f}"
    )

    print(
        "Relevant additions/user: "
        f"{section['mean_incremental_relevant_items_per_user']:.6f}"
    )

    print(
        "Users helped: "
        f"{section['fraction_users_with_incremental_relevant']:.4%}"
    )

    print(
        "Recall gain: "
        f"{section['candidate_recall_absolute_gain']:+.6f}"
    )

    print(
        "\nTT-V2 AFTER THREE-SOURCE + TT-V1"
    )

    section = result[
        "tt_v2_incremental_after_tt_v1"
    ]

    print(
        "Unique candidates/user: "
        f"{section['mean_unique_candidates_per_user']:.4f}"
    )

    print(
        "Relevant additions/user: "
        f"{section['mean_incremental_relevant_items_per_user']:.6f}"
    )

    print(
        "Users helped: "
        f"{section['fraction_users_with_incremental_relevant']:.4%}"
    )

    print(
        "Recall gain: "
        f"{section['candidate_recall_absolute_gain']:+.6f}"
    )

    print(
        "\nSaved:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()