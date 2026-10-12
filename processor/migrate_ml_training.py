"""
One-time migration: ML-ready -> partitioned ML training layout.

Reads every row of s3://my-bucket/ml-ready/, adds the deterministic `split`
column and writes it to s3://my-bucket/ml/training/articles/ partitioned by
split and then _ingest_date (one file per partition).

The source ML-ready path is only read, never modified. The script refuses to
run if the training path already holds data, so it cannot double-write.
Run it once, before the next pipeline run (which appends new batches to the
training layout itself).

Run from processor/ with:
    PYSPARK_PYTHON=.venv/bin/python PYSPARK_DRIVER_PYTHON=.venv/bin/python \
        .venv/bin/python migrate_ml_training.py
"""

import boto3

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_ENDPOINT_URL,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    ML_READY_PATH,
    ML_TRAINING_PATH,
)
from src.pipeline import create_spark
from src.transformations import assign_split, write_training_dataset


def s3a(path):
    return path.replace("s3://", "s3a://", 1)


def training_path_has_data():
    bucket, _, prefix = ML_TRAINING_PATH.replace("s3://", "", 1).partition("/")
    s3 = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def main():

    if training_path_has_data():
        raise RuntimeError(
            f"{ML_TRAINING_PATH} already contains data; refusing to migrate "
            f"again (it would duplicate rows)."
        )

    spark = create_spark()
    spark.sparkContext.setLogLevel("ERROR")

    try:

        ml_ready = spark.read.parquet(s3a(ML_READY_PATH))
        source_rows = ml_ready.count()
        print(f"ML-ready source rows: {source_rows}")

        training_df = assign_split(ml_ready)

        write_training_dataset(training_df, s3a(ML_TRAINING_PATH))

        written = spark.read.parquet(s3a(ML_TRAINING_PATH))
        written_rows = written.count()

        print(f"Training rows written: {written_rows}")
        print("Rows by split:")
        written.groupBy("split").count().orderBy("split").show()
        print("Rows by split and _ingest_date:")
        (
            written.groupBy("split", "_ingest_date")
            .count()
            .orderBy("split", "_ingest_date")
            .show(50)
        )

        if written_rows != source_rows:
            raise RuntimeError(
                f"Row count mismatch: source={source_rows}, "
                f"written={written_rows}"
            )

        print("MIGRATION COMPLETED SUCCESSFULLY")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
