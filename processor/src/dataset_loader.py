"""
Spark-free loader for the ML training layout.

Reads s3://my-bucket/ml/training/articles/ (partitioned by split and then
_ingest_date) with PyArrow and streams record batches. Only the requested
split / dates are listed and only the requested columns are read, so a
training job never scans the whole dataset.

This is the shape a PyTorch IterableDataset or a Hugging Face
`Dataset.from_generator` needs, e.g.:

    for batch in iter_batches("train", batch_size=256):
        texts = batch.column("body_text").to_pylist()
        ...

Demo (from processor/):
    .venv/bin/python -m src.dataset_loader
"""

import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs

from .config import (
    AWS_ACCESS_KEY_ID,
    AWS_ENDPOINT_URL,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    ML_TRAINING_PATH,
)


SPLITS = ("train", "validation", "test")

DEFAULT_COLUMNS = [
    "article_id",
    "title",
    "body_text",
    "extractive_summary",
    "word_count",
    "summary_word_count",
    "compression_ratio",
]

PARTITIONING = ds.partitioning(
    pa.schema(
        [
            ("split", pa.string()),
            ("_ingest_date", pa.string()),
        ]
    ),
    flavor="hive",
)


def s3_filesystem():
    scheme, _, endpoint = AWS_ENDPOINT_URL.partition("://")
    return fs.S3FileSystem(
        access_key=AWS_ACCESS_KEY_ID,
        secret_key=AWS_SECRET_ACCESS_KEY,
        region=AWS_REGION,
        endpoint_override=endpoint,
        scheme=scheme,
    )


def training_dataset():
    path = ML_TRAINING_PATH.replace("s3://", "", 1).rstrip("/")
    return ds.dataset(
        path,
        format="parquet",
        filesystem=s3_filesystem(),
        partitioning=PARTITIONING,
        # Default ignore_prefixes is [".", "_"], which would also skip the
        # "_ingest_date=..." directories; only skip Spark marker files.
        ignore_prefixes=[".", "_SUCCESS", "_committed", "_started"],
    )


def split_filter(split, since=None, until=None):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    expression = ds.field("split") == split
    if since:
        expression = expression & (ds.field("_ingest_date") >= since)
    if until:
        expression = expression & (ds.field("_ingest_date") <= until)
    return expression


def iter_batches(
    split,
    since=None,
    until=None,
    columns=None,
    batch_size=256,
):
    """
    Yield pyarrow.RecordBatch objects for one split.

    since / until are inclusive ISO dates ("YYYY-MM-DD") on _ingest_date,
    e.g. since="2026-10-11" streams only articles ingested from that day on.
    """
    scanner = training_dataset().scanner(
        columns=columns or DEFAULT_COLUMNS,
        filter=split_filter(split, since, until),
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        if batch.num_rows:
            yield batch


def count_rows(split, since=None, until=None):
    return training_dataset().count_rows(
        filter=split_filter(split, since, until)
    )


def main():

    print("=" * 70)
    print(f"ML TRAINING DATASET LOADER (PyArrow) -- {ML_TRAINING_PATH}")
    print("=" * 70)

    for split in SPLITS:
        print(f"{split:<11} rows: {count_rows(split)}")

    first = next(iter_batches("train", batch_size=4))
    print(
        f"\nFirst train batch: {first.num_rows} rows x "
        f"{first.num_columns} columns {first.schema.names}"
    )
    for row in first.to_pylist()[:2]:
        print(
            f"- {row['article_id']} | words={row['word_count']} | "
            f"summary_words={row['summary_word_count']} | "
            f"summary: {row['extractive_summary'][:80]}..."
        )

    fragments = list(
        training_dataset().get_fragments(filter=split_filter("train"))
    )
    print(f"\nTrain files read for split='train': {len(fragments)}")


if __name__ == "__main__":
    main()
