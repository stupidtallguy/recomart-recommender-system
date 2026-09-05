from __future__ import annotations

import shutil
from pathlib import Path


def _arrow_type(type_name: str):
    import pyarrow as pa

    mapping = {
        "tinyint": pa.int8(),
        "smallint": pa.int16(),
        "int": pa.int32(),
        "bigint": pa.int64(),
        "float": pa.float32(),
        "double": pa.float64(),
        "boolean": pa.bool_(),
        "string": pa.string(),
    }

    if type_name not in mapping:
        raise TypeError(
            f"Unsupported Spark type for local "
            f"Parquet writer: {type_name}"
        )

    return mapping[type_name]


def write_spark_df_local_parquet(
    df,
    output_dir: Path,
    num_partitions: int,
    partition_column: str = "user_id",
    batch_size: int = 50_000,
) -> int:
    """
    Write a Spark DataFrame to local Parquet files
    using PyArrow instead of Hadoop's Windows
    filesystem writer.

    Intended for local Windows development where
    winutils.exe is unavailable.

    Returns
    -------
    int
        Number of rows written.
    """

    output_dir = Path(
        output_dir
    ).resolve()

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema_spec = [
        (
            field.name,
            field.dataType.simpleString(),
        )
        for field
        in df.schema.fields
    ]

    unsupported = [
        type_name
        for _, type_name
        in schema_spec
        if type_name not in {
            "tinyint",
            "smallint",
            "int",
            "bigint",
            "float",
            "double",
            "boolean",
            "string",
        }
    ]

    if unsupported:
        raise TypeError(
            "Unsupported Spark types: "
            f"{sorted(set(unsupported))}"
        )

    output_dir_string = str(
        output_dir
    )

    prepared = df.repartition(
        num_partitions,
        partition_column,
    )

    def write_partition(
        partition_id,
        rows,
    ):
        import os

        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(
            output_dir_string,
            exist_ok=True,
        )

        arrow_schema = pa.schema(
            [
                (
                    name,
                    _arrow_type(
                        type_name
                    ),
                )
                for name, type_name
                in schema_spec
            ]
        )

        final_path = Path(
            output_dir_string
        ) / (
            f"part-{partition_id:05d}.parquet"
        )

        tmp_path = Path(
            str(final_path)
            + ".tmp"
        )

        if tmp_path.exists():
            tmp_path.unlink()

        writer = None
        batch = []
        row_count = 0

        try:

            for row in rows:

                batch.append(
                    row.asDict(
                        recursive=False
                    )
                )

                if (
                    len(batch)
                    >= batch_size
                ):

                    table = (
                        pa.Table
                        .from_pylist(
                            batch,
                            schema=
                                arrow_schema,
                        )
                    )

                    if writer is None:

                        writer = (
                            pq.ParquetWriter(
                                tmp_path,
                                arrow_schema,
                                compression=
                                    "snappy",
                            )
                        )

                    writer.write_table(
                        table
                    )

                    row_count += len(
                        batch
                    )

                    batch.clear()

            if batch:

                table = (
                    pa.Table
                    .from_pylist(
                        batch,
                        schema=
                            arrow_schema,
                    )
                )

                if writer is None:

                    writer = (
                        pq.ParquetWriter(
                            tmp_path,
                            arrow_schema,
                            compression=
                                "snappy",
                        )
                    )

                writer.write_table(
                    table
                )

                row_count += len(
                    batch
                )

                batch.clear()

            if writer is not None:

                writer.close()
                writer = None

                tmp_path.replace(
                    final_path
                )

        except Exception:

            if writer is not None:
                writer.close()

            if tmp_path.exists():
                tmp_path.unlink()

            raise

        return iter(
            [row_count]
        )

    counts = (
        prepared.rdd
        .mapPartitionsWithIndex(
            write_partition
        )
        .collect()
    )

    total_rows = int(
        sum(counts)
    )

    return total_rows