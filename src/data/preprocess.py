from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"


ORDER_DTYPES = {
    "order_id": "int32",
    "user_id": "int32",
    "eval_set": "category",
    "order_number": "int16",
    "order_dow": "int8",
    "order_hour_of_day": "int8",
    "days_since_prior_order": "float32",
}

ORDER_PRODUCT_DTYPES = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int16",
    "reordered": "int8",
}

PRODUCT_DTYPES = {
    "product_id": "int32",
    "product_name": "string",
    "aisle_id": "int16",
    "department_id": "int8",
}

AISLE_DTYPES = {
    "aisle_id": "int16",
    "aisle": "string",
}

DEPARTMENT_DTYPES = {
    "department_id": "int8",
    "department": "string",
}


INTERACTION_COLUMNS = [
    "user_id",
    "order_id",
    "order_number",
    "product_id",
    "add_to_cart_order",
    "reordered",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
    "eval_set",
]


STANDARD_DATASET_ROWS = {
    "orders": 3_421_083,
    "order_products__prior": 32_434_489,
    "order_products__train": 1_384_617,
    "products": 49_688,
}


def atomic_to_parquet(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a DataFrame to a temporary parquet file first.

    The final output replaces the temporary file only after
    the write completes successfully.
    """
    tmp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if tmp_path.exists():
        tmp_path.unlink()

    df.to_parquet(
        tmp_path,
        index=False,
        compression="snappy",
    )

    tmp_path.replace(output_path)


def require_unique(
    df: pd.DataFrame,
    column: str,
    table_name: str,
) -> None:
    duplicate_count = int(
        df[column].duplicated().sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{table_name}.{column} contains "
            f"{duplicate_count:,} duplicate values."
        )


def validate_orders(
    orders: pd.DataFrame,
) -> dict:

    require_unique(
        orders,
        "order_id",
        "orders",
    )

    allowed_eval_sets = {
        "prior",
        "train",
        "test",
    }

    actual_eval_sets = set(
        orders["eval_set"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    unexpected_eval_sets = (
        actual_eval_sets - allowed_eval_sets
    )

    if unexpected_eval_sets:
        raise ValueError(
            "Unexpected eval_set values: "
            f"{sorted(unexpected_eval_sets)}"
        )

    invalid_order_number = int(
        (orders["order_number"] < 1).sum()
    )

    invalid_dow = int(
        (~orders["order_dow"].between(0, 6)).sum()
    )

    invalid_hour = int(
        (
            ~orders["order_hour_of_day"]
            .between(0, 23)
        ).sum()
    )

    invalid_days = int(
        (
            orders["days_since_prior_order"]
            .dropna()
            < 0
        ).sum()
    )

    first_orders = (
        orders["order_number"] == 1
    )

    first_order_days_not_null = int(
        orders.loc[
            first_orders,
            "days_since_prior_order",
        ]
        .notna()
        .sum()
    )

    problems = {
        "invalid_order_number":
            invalid_order_number,

        "invalid_order_dow":
            invalid_dow,

        "invalid_order_hour":
            invalid_hour,

        "negative_days_since_prior_order":
            invalid_days,

        "first_orders_with_non_null_days":
            first_order_days_not_null,
    }

    if any(problems.values()):
        raise ValueError(
            "orders.csv failed data-quality "
            f"checks: {problems}"
        )

    return {
        "rows": int(len(orders)),

        "users": int(
            orders["user_id"].nunique()
        ),

        "orders_by_eval_set": {
            str(key): int(value)
            for key, value
            in orders["eval_set"]
            .value_counts()
            .to_dict()
            .items()
        },
    }


def build_products() -> tuple[pd.DataFrame, dict]:

    products = pd.read_csv(
        RAW_DIR / "products.csv",
        dtype=PRODUCT_DTYPES,
    )

    aisles = pd.read_csv(
        RAW_DIR / "aisles.csv",
        dtype=AISLE_DTYPES,
    )

    departments = pd.read_csv(
        RAW_DIR / "departments.csv",
        dtype=DEPARTMENT_DTYPES,
    )

    require_unique(
        products,
        "product_id",
        "products",
    )

    require_unique(
        aisles,
        "aisle_id",
        "aisles",
    )

    require_unique(
        departments,
        "department_id",
        "departments",
    )

    invalid_aisles = int(
        (
            ~products["aisle_id"]
            .isin(aisles["aisle_id"])
        ).sum()
    )

    invalid_departments = int(
        (
            ~products["department_id"]
            .isin(
                departments["department_id"]
            )
        ).sum()
    )

    if invalid_aisles or invalid_departments:
        raise ValueError(
            "Product metadata contains "
            "invalid foreign keys: "
            f"invalid_aisles={invalid_aisles:,}, "
            f"invalid_departments="
            f"{invalid_departments:,}"
        )

    product_table = (
        products
        .merge(
            aisles,
            on="aisle_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            departments,
            on="department_id",
            how="left",
            validate="many_to_one",
        )
        [
            [
                "product_id",
                "product_name",
                "aisle_id",
                "aisle",
                "department_id",
                "department",
            ]
        ]
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    atomic_to_parquet(
        product_table,
        PROCESSED_DIR / "products.parquet",
    )

    report = {
        "rows": int(
            len(product_table)
        ),

        "unique_products": int(
            product_table[
                "product_id"
            ].nunique()
        ),

        "unique_aisles": int(
            product_table[
                "aisle_id"
            ].nunique()
        ),

        "unique_departments": int(
            product_table[
                "department_id"
            ].nunique()
        ),

        "missing_product_names": int(
            product_table[
                "product_name"
            ].isna().sum()
        ),

        "missing_aisle_names": int(
            product_table[
                "aisle"
            ].isna().sum()
        ),

        "missing_department_names": int(
            product_table[
                "department"
            ].isna().sum()
        ),
    }

    return product_table, report


def write_interactions(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    chunk_size: int,
) -> dict:

    output_path = (
        PROCESSED_DIR
        / "interactions.parquet"
    )

    tmp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if tmp_path.exists():
        tmp_path.unlink()

    order_lookup = (
        orders
        .set_index("order_id")
        [
            [
                "user_id",
                "eval_set",
                "order_number",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order",
            ]
        ]
    )

    valid_product_ids = pd.Index(
        products["product_id"]
    )

    source_files = {
        "prior":
            RAW_DIR
            / "order_products__prior.csv",

        "train":
            RAW_DIR
            / "order_products__train.csv",
    }

    writer: pq.ParquetWriter | None = None

    report = {
        "rows": 0,
        "rows_by_source": {},
        "invalid_reordered_values": 0,
        "unknown_order_ids": 0,
        "unknown_product_ids": 0,
        "source_eval_set_mismatches": 0,
        "within_chunk_duplicate_order_product_pairs": 0,
    }

    try:

        for (
            expected_eval_set,
            csv_path,
        ) in source_files.items():

            source_rows = 0

            reader = pd.read_csv(
                csv_path,
                dtype=ORDER_PRODUCT_DTYPES,
                chunksize=chunk_size,
            )

            for chunk_number, chunk in enumerate(
                reader,
                start=1,
            ):

                invalid_reordered = int(
                    (
                        ~chunk["reordered"]
                        .isin([0, 1])
                    ).sum()
                )

                unknown_products = int(
                    (
                        ~chunk["product_id"]
                        .isin(valid_product_ids)
                    ).sum()
                )

                duplicate_pairs = int(
                    chunk
                    .duplicated(
                        [
                            "order_id",
                            "product_id",
                        ]
                    )
                    .sum()
                )

                enriched = chunk.join(
                    order_lookup,
                    on="order_id",
                    how="left",
                )

                unknown_orders = int(
                    enriched[
                        "user_id"
                    ].isna().sum()
                )

                source_mismatches = int(
                    (
                        enriched[
                            "eval_set"
                        ].notna()
                        &
                        enriched[
                            "eval_set"
                        ].astype(str).ne(
                            expected_eval_set
                        )
                    ).sum()
                )

                report[
                    "invalid_reordered_values"
                ] += invalid_reordered

                report[
                    "unknown_order_ids"
                ] += unknown_orders

                report[
                    "unknown_product_ids"
                ] += unknown_products

                report[
                    "source_eval_set_mismatches"
                ] += source_mismatches

                report[
                    "within_chunk_duplicate_order_product_pairs"
                ] += duplicate_pairs

                if any(
                    [
                        invalid_reordered,
                        unknown_products,
                        unknown_orders,
                        source_mismatches,
                        duplicate_pairs,
                    ]
                ):
                    raise ValueError(
                        "Data-quality failure in "
                        f"{csv_path.name}, "
                        f"chunk {chunk_number}: "
                        f"invalid_reordered="
                        f"{invalid_reordered:,}, "
                        f"unknown_products="
                        f"{unknown_products:,}, "
                        f"unknown_orders="
                        f"{unknown_orders:,}, "
                        f"source_mismatches="
                        f"{source_mismatches:,}, "
                        f"duplicate_pairs="
                        f"{duplicate_pairs:,}"
                    )

                enriched = enriched[
                    INTERACTION_COLUMNS
                ]

                enriched = enriched.astype(
                    {
                        "user_id": "int32",
                        "order_id": "int32",
                        "order_number": "int16",
                        "product_id": "int32",
                        "add_to_cart_order": "int16",
                        "reordered": "int8",
                        "order_dow": "int8",
                        "order_hour_of_day": "int8",
                        "days_since_prior_order":
                            "float32",
                        "eval_set": "string",
                    }
                )

                table = pa.Table.from_pandas(
                    enriched,
                    preserve_index=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        tmp_path,
                        table.schema,
                        compression="snappy",
                        use_dictionary=True,
                    )

                writer.write_table(table)

                rows = len(enriched)

                source_rows += rows
                report["rows"] += rows

                print(
                    f"[{expected_eval_set.upper():5}] "
                    f"chunk={chunk_number:02d} "
                    f"rows={rows:>9,} "
                    f"source_total="
                    f"{source_rows:>11,}"
                )

            report[
                "rows_by_source"
            ][expected_eval_set] = int(
                source_rows
            )

    except Exception:

        if writer is not None:
            writer.close()

        if tmp_path.exists():
            tmp_path.unlink()

        raise

    if writer is None:
        raise RuntimeError(
            "No interaction rows were written."
        )

    writer.close()

    tmp_path.replace(output_path)

    return report


def add_standard_dataset_sanity_checks(
    report: dict,
) -> None:

    observed = {
        "orders":
            report["orders"]["rows"],

        "order_products__prior":
            report["interactions"]
            ["rows_by_source"]
            ["prior"],

        "order_products__train":
            report["interactions"]
            ["rows_by_source"]
            ["train"],

        "products":
            report["products"]["rows"],
    }

    report[
        "standard_dataset_row_check"
    ] = {
        name: {
            "expected": int(expected),
            "observed": int(
                observed[name]
            ),
            "matches": bool(
                observed[name] == expected
            ),
        }
        for name, expected
        in STANDARD_DATASET_ROWS.items()
    }


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build RecoMart processed "
            "Instacart tables."
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help=(
            "Number of order-product rows "
            "processed at a time."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing processed "
            "outputs."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be "
            "greater than zero."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = [
        PROCESSED_DIR
        / "orders.parquet",

        PROCESSED_DIR
        / "products.parquet",

        PROCESSED_DIR
        / "interactions.parquet",

        REPORT_PATH,
    ]

    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing and not args.overwrite:

        names = ", ".join(
            path.name
            for path in existing
        )

        raise FileExistsError(
            "Processed outputs already "
            f"exist: {names}. "
            "Use --overwrite if you "
            "intentionally want to rebuild."
        )

    print(
        "Loading and validating orders..."
    )

    orders = pd.read_csv(
        RAW_DIR / "orders.csv",
        dtype=ORDER_DTYPES,
    )

    orders_report = validate_orders(
        orders
    )

    atomic_to_parquet(
        orders,
        PROCESSED_DIR
        / "orders.parquet",
    )

    print(
        "Building product dimension..."
    )

    products, products_report = (
        build_products()
    )

    print(
        "\nBuilding interaction table "
        "in chunks "
        f"(chunk_size={args.chunk_size:,})..."
    )

    interactions_report = (
        write_interactions(
            orders=orders,
            products=products,
            chunk_size=args.chunk_size,
        )
    )

    report = {
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "orders":
            orders_report,

        "products":
            products_report,

        "interactions":
            interactions_report,
    }

    add_standard_dataset_sanity_checks(
        report
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nProcessed dataset "
        "created successfully."
    )

    print(
        "orders.parquet       : "
        f"{orders_report['rows']:,} rows"
    )

    print(
        "products.parquet     : "
        f"{products_report['rows']:,} rows"
    )

    print(
        "interactions.parquet : "
        f"{interactions_report['rows']:,} rows"
    )

    print(
        "quality report       : "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()