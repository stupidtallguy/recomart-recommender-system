from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")


EXPECTED_FILES = {
    "orders.csv": {
        "order_id",
        "user_id",
        "eval_set",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    },
    "order_products__prior.csv": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    },
    "order_products__train.csv": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    },
    "products.csv": {
        "product_id",
        "product_name",
        "aisle_id",
        "department_id",
    },
    "aisles.csv": {
        "aisle_id",
        "aisle",
    },
    "departments.csv": {
        "department_id",
        "department",
    },
}


def validate_file_exists(filename: str) -> Path:
    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required dataset file: {path}"
        )

    return path


def validate_columns(path: Path, expected_columns: set[str]) -> None:
    df = pd.read_csv(path, nrows=5)

    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def validate_raw_dataset() -> None:
    print("Validating Instacart raw dataset...\n")

    for filename, columns in EXPECTED_FILES.items():

        path = validate_file_exists(filename)

        validate_columns(
            path=path,
            expected_columns=columns,
        )

        size_mb = path.stat().st_size / (1024 * 1024)

        print(
            f"[OK] {filename:<32} "
            f"{size_mb:>8.2f} MB"
        )

    print("\nRaw dataset validation completed successfully.")


if __name__ == "__main__":
    validate_raw_dataset()