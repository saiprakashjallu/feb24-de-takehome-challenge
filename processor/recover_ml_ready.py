"""
One-off recovery for the 2026-10-12 batch that reached Gold but not ML-ready.

The failed run saved the Kinesis checkpoint after Bronze, then crashed in
Step 10 (PYTHON_VERSION_MISMATCH). This script re-runs Steps 8-12 only for
Gold rows whose _source_sequence_number is after the previous checkpoint,
appends them to ML-ready, and writes monitoring + MLflow records.

It does not read Kinesis and does not touch the checkpoint.

Run from processor/ with:
    PYSPARK_PYTHON=.venv/bin/python PYSPARK_DRIVER_PYTHON=.venv/bin/python \
        .venv/bin/python recover_ml_ready.py
"""

import time

from pyspark.sql import functions as F

from src.config import BRONZE_PATH, ML_READY_PATH, S3_BUCKET
from src.mlflow_tracking import log_pipeline_run
from src.monitoring import record_pipeline_metrics
from src.pipeline import append_batch, create_spark
from src.summarization import add_extractive_summary
from src.transformations import enrich_articles


# Checkpoint value before the failed run (shardId-000000000000).
PREVIOUS_CHECKPOINT = (
    "49679031755053292072256664294935283924259887041950515202"
)


def after_previous_checkpoint(df):
    # Sequence numbers exceed bigint/decimal(38), so compare as strings
    # ordered by length first, then lexicographically.
    seq = F.col("_source_sequence_number")
    return df.filter(
        (F.length(seq) > len(PREVIOUS_CHECKPOINT))
        | (
            (F.length(seq) == len(PREVIOUS_CHECKPOINT))
            & (seq > F.lit(PREVIOUS_CHECKPOINT))
        )
    )


def s3a(path):
    return path.replace("s3://", "s3a://")


def main():

    start_time = time.time()
    spark = create_spark()

    try:

        export_path = s3a(ML_READY_PATH)
        gold_path = f"s3a://{S3_BUCKET}/gold/articles/"
        silver_path = f"s3a://{S3_BUCKET}/silver/articles/"

        ml_ready_before = spark.read.parquet(export_path).count()
        print(f"ML-ready rows BEFORE recovery: {ml_ready_before}")

        bronze_batch = after_previous_checkpoint(
            spark.read.parquet(s3a(BRONZE_PATH))
        )
        silver_batch = after_previous_checkpoint(
            spark.read.parquet(silver_path)
        )
        gold_batch_df = after_previous_checkpoint(
            spark.read.parquet(gold_path)
        )

        records_ingested = bronze_batch.count()
        valid_count = silver_batch.count()
        duplicate_count = silver_batch.filter(
            F.col("is_duplicate") == True
        ).count()
        group_count = (
            silver_batch.select("canonical_article_id").distinct().count()
        )
        gold_count = gold_batch_df.count()

        print(f"Bronze rows in batch: {records_ingested}")
        print(f"Silver rows in batch: {valid_count}")
        print(f"Gold rows in batch:   {gold_count}")

        # Guard: refuse to append if this batch is already in ML-ready.
        already_exported = after_previous_checkpoint(
            spark.read.parquet(export_path)
        ).count()
        if already_exported > 0:
            raise RuntimeError(
                f"{already_exported} rows from this batch are already in "
                f"ML-ready; refusing to append duplicates."
            )

        if gold_count == 0:
            raise RuntimeError("No Gold rows found after the checkpoint.")

        # Step 8 - Feature engineering
        enriched_df = enrich_articles(gold_batch_df)

        # Step 9 - Extractive summarization (same logic as pipeline.py)
        enriched_df = add_extractive_summary(enriched_df)

        enriched_df = enriched_df.withColumn(
            "summary_word_count",
            F.when(
                F.col("extractive_summary").isNotNull(),
                F.size(
                    F.split(F.trim(F.col("extractive_summary")), r"\s+")
                ),
            ).otherwise(0),
        )

        enriched_df = enriched_df.withColumn(
            "compression_ratio",
            F.when(
                F.col("word_count") > 0,
                F.col("summary_word_count") / F.col("word_count"),
            ).otherwise(0.0),
        )

        # Step 10 - Statistics
        statistics = enriched_df.select(
            F.avg("word_count").alias("average_word_count"),
            F.avg("summary_word_count").alias("average_summary_word_count"),
            F.avg("compression_ratio").alias("average_compression_ratio"),
        ).collect()[0]

        summaries_generated = enriched_df.filter(
            F.col("extractive_summary").isNotNull()
        ).count()

        # Step 11 - ML-ready append
        append_batch(new_df=enriched_df, output_path=export_path)

        ml_ready_after = spark.read.parquet(export_path).count()
        print(f"ML-ready rows AFTER recovery:  {ml_ready_after}")
        print(f"ML-ready rows added:           {ml_ready_after - ml_ready_before}")

        # Step 12 - Monitoring + MLflow
        def as_float(value):
            return float(value) if value is not None else 0.0

        metrics = {
            "records_ingested": records_ingested,
            "records_valid": valid_count,
            "records_quarantined": records_ingested - valid_count,
            "semantic_duplicates": duplicate_count,
            "canonical_articles": gold_count,
            "duplicate_groups": group_count,
            "data_quality_rate": (
                valid_count / records_ingested if records_ingested else 0.0
            ),
            "average_word_count": as_float(statistics["average_word_count"]),
            "average_summary_word_count": as_float(
                statistics["average_summary_word_count"]
            ),
            "average_compression_ratio": as_float(
                statistics["average_compression_ratio"]
            ),
            "summaries_generated": summaries_generated,
            "summary_generation_rate": (
                summaries_generated / gold_count if gold_count else 0.0
            ),
            "processing_time_seconds": round(time.time() - start_time, 3),
            "pipeline_version": "1.2.0",
            "run_type": "recovery",
        }

        print("\nRecovery metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value}")

        record_pipeline_metrics(metrics)
        log_pipeline_run(metrics)

        print("\nRECOVERY COMPLETED SUCCESSFULLY")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
