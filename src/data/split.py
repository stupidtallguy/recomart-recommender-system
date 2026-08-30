from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)


INPUT_FILE = (
    PROCESSED_DIR /
    "interactions.parquet"
)


REPORT_FILE = (
    PROCESSED_DIR /
    "split_report.json"
)


MIN_ORDERS = 3


def create_temporal_split(
    interactions: pd.DataFrame,
):

    print("Calculating user order history...")


    user_order_counts = (
        interactions
        .groupby("user_id")
        ["order_number"]
        .nunique()
    )


    eligible_users = (
        user_order_counts[
            user_order_counts >= MIN_ORDERS
        ]
        .index
    )


    filtered = interactions[
        interactions["user_id"]
        .isin(eligible_users)
    ].copy()


    print(
        f"Eligible users: "
        f"{len(eligible_users):,}"
    )


    user_orders = (
        filtered[
            [
                "user_id",
                "order_id",
                "order_number",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "user_id",
                "order_number",
            ]
        )
    )


    order_rank = (
        user_orders
        .groupby("user_id")
        ["order_number"]
        .rank(
            method="first",
            ascending=True
        )
    )


    max_rank = (
        user_orders
        .groupby("user_id")
        ["order_number"]
        .transform("count")
    )


    user_orders["split"] = "train"


    user_orders.loc[
        order_rank == max_rank,
        "split"
    ] = "test"


    user_orders.loc[
        order_rank == max_rank - 1,
        "split"
    ] = "validation"


    split_labels = (
        user_orders
        [
            [
                "order_id",
                "split"
            ]
        ]
    )


    labeled = (
        filtered
        .merge(
            split_labels,
            on="order_id",
            how="left",
            validate="many_to_one"
        )
    )


    train = labeled[
        labeled["split"] == "train"
    ].drop(
        columns="split"
    )


    validation = labeled[
        labeled["split"] == "validation"
    ].drop(
        columns="split"
    )


    test = labeled[
        labeled["split"] == "test"
    ].drop(
        columns="split"
    )


    return train, validation, test



def save_outputs(
    train,
    validation,
    test
):

    outputs = {
        "train":
            train,

        "validation":
            validation,

        "test":
            test,
    }


    for name, df in outputs.items():

        path = (
            PROCESSED_DIR /
            f"{name}_interactions.parquet"
        )

        print(
            f"Saving {name}: "
            f"{len(df):,} rows"
        )


        df.to_parquet(
            path,
            index=False,
            compression="snappy"
        )



    report = {

        "users": {

            "total_users":
                int(
                    pd.concat(outputs.values())
                    ["user_id"]
                    .nunique()
                ),

        },


        "rows": {

            name:
                int(len(df))

            for name, df
            in outputs.items()

        },


        "unique_orders": {

            name:
                int(
                    df["order_id"]
                    .nunique()
                )

            for name, df
            in outputs.items()

        },

        "unique_users": {

            name:
                int(
                    df["user_id"]
                    .nunique()
                )

            for name, df
            in outputs.items()

        }

    }


    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )


    print("\nSplit report saved.")



def main():

    print(
        "Loading interactions..."
    )


    interactions = pd.read_parquet(
        INPUT_FILE
    )


    train, validation, test = (
        create_temporal_split(
            interactions
        )
    )


    save_outputs(
        train,
        validation,
        test
    )


    print(
        "\nTemporal split completed."
    )



if __name__ == "__main__":
    main()