import json
import tempfile
import time

from pyspark.sql import functions as F


try:
    from .kinesis_reader import (
        read_all_articles,
        save_malformed_records,
    )
    from .deduplication import deduplicate_articles
    from .transformations import (
        assign_split,
        enrich_articles,
        write_training_dataset,
    )
    from .summarization import add_extractive_summary
    from .monitoring import record_pipeline_metrics
    from .mlflow_tracking import log_pipeline_run
    from .data_quality import (
        validate_and_clean_articles,
        split_valid_and_quarantine,
        print_quality_summary,
    )
    from .checkpoint import load_checkpoints, save_checkpoints
    from .config import (
        ML_READY_PATH,
        ML_TRAINING_PATH,
        BRONZE_PATH,
        S3_BUCKET,
    )

except ImportError:
    from kinesis_reader import (
        read_all_articles,
        save_malformed_records,
    )
    from deduplication import deduplicate_articles
    from transformations import (
        assign_split,
        enrich_articles,
        write_training_dataset,
    )
    from summarization import add_extractive_summary
    from monitoring import record_pipeline_metrics
    from mlflow_tracking import log_pipeline_run
    from data_quality import (
        validate_and_clean_articles,
        split_valid_and_quarantine,
        print_quality_summary,
    )
    from checkpoint import load_checkpoints, save_checkpoints
    from config import (
        ML_READY_PATH,
        ML_TRAINING_PATH,
        BRONZE_PATH,
        S3_BUCKET,
    )


# -------------------------------------------------------
# Spark session
# -------------------------------------------------------

def create_spark():

    from pyspark.sql import SparkSession

    try:
        from .config import (
            AWS_ENDPOINT_URL,
            AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY,
        )

    except ImportError:
        from config import (
            AWS_ENDPOINT_URL,
            AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY,
        )

    return (
        SparkSession.builder
        .appName("NewsETLPipeline")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            AWS_ENDPOINT_URL,
        )
        .config(
            "spark.hadoop.fs.s3a.access.key",
            AWS_ACCESS_KEY_ID,
        )
        .config(
            "spark.hadoop.fs.s3a.secret.key",
            AWS_SECRET_ACCESS_KEY,
        )
        .config(
            "spark.hadoop.fs.s3a.path.style.access",
            "true",
        )
        .config(
            "spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .config(
            "spark.hadoop.fs.s3.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem",
        )
        .getOrCreate()
    )


# -------------------------------------------------------
# Convert article dictionaries to Spark DataFrame
# -------------------------------------------------------

def articles_to_dataframe(spark, articles):

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
    ) as f:

        for article in articles:

            serializable_article = {}

            for key, value in article.items():

                if hasattr(value, "isoformat"):
                    serializable_article[key] = value.isoformat()
                else:
                    serializable_article[key] = value

            f.write(
                json.dumps(
                    serializable_article,
                    ensure_ascii=False,
                )
                + "\n"
            )

        temp_json_path = f.name

    return spark.read.json(temp_json_path)


# -------------------------------------------------------
# Append-only data lake writer
# -------------------------------------------------------

def append_batch(new_df, output_path):
    """
    Append the current pipeline batch to an immutable Parquet dataset.

    We intentionally do not read and overwrite the same Parquet path
    during one Spark job.

    This avoids SparkFileNotFoundException caused by Spark lazy
    evaluation when an input dataset is overwritten while it is
    still referenced by the execution plan.

    For production exactly-once/idempotent upserts, a transactional
    table format such as Delta Lake, Apache Iceberg, or Apache Hudi
    would be preferable.
    """

    record_count = new_df.count()

    print(
        f"Appending {record_count} records to: "
        f"{output_path}"
    )

    (
        new_df
        .write
        .mode("append")
        .parquet(output_path)
    )

    print(
        f"Batch appended successfully: "
        f"{output_path}"
    )

    return new_df


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------

def run_pipeline():

    pipeline_start_time = time.time()

    spark = create_spark()

    try:

        # =================================================
        # 1. Read from Kinesis
        # =================================================

        print("\n" + "=" * 70)
        print("STEP 1 - READING FROM KINESIS")
        print("=" * 70)

        checkpoints = load_checkpoints()

        print(
            f"Current checkpoints: "
            f"{checkpoints}"
        )

        (
            articles,
            updated_checkpoints,
            malformed_records,
        ) = read_all_articles(checkpoints)

        print(
            f"Retrieved {len(articles)} new articles"
        )

        print(
            f"Malformed records: "
            f"{len(malformed_records)}"
        )

        # Malformed records go to the dead-letter area now.
        # Their keys come from the sequence number, so a replay
        # overwrites rather than duplicates them.
        save_malformed_records(malformed_records)

        if not articles:

            print(
                "No articles found in Kinesis."
            )

            if malformed_records:

                # Only malformed records in this batch: they are
                # stored, so advance past them.
                save_checkpoints(updated_checkpoints)

            return

        # =================================================
        # 2. Bronze
        # =================================================

        print("\n" + "=" * 70)
        print("STEP 2 - BRONZE DATASET")
        print("=" * 70)

        bronze_df = articles_to_dataframe(
            spark,
            articles,
        )

        bronze_record_count = bronze_df.count()

        print(
            f"Bronze records: "
            f"{bronze_record_count}"
        )

        print("\nBronze schema:")
        bronze_df.printSchema()

        bronze_df = bronze_df.withColumn(
            "_ingest_date",
            F.to_date(
                F.to_timestamp(
                    " _ingest_timestamp".strip()
                )
            ),
        )

        bronze_path = BRONZE_PATH.replace(
            "s3://",
            "s3a://",
        )

        (
            bronze_df
            .write
            .mode("append")
            .partitionBy("_ingest_date")
            .parquet(bronze_path)
        )

        print(
            f"Bronze dataset written to: "
            f"{BRONZE_PATH}"
        )

        # Checkpoint is saved only after ML-ready is written (Step 11),
        # so a failure in any downstream step replays this batch.

        # =================================================
        # 3. Data Quality
        # =================================================

        print("\n" + "=" * 70)
        print("STEP 3 - DATA QUALITY")
        print("=" * 70)

        cleaned_df = validate_and_clean_articles(
            bronze_df
        )

        print_quality_summary(cleaned_df)

        (
            valid_df,
            quarantine_df,
        ) = split_valid_and_quarantine(
            cleaned_df
        )

        valid_count = valid_df.count()
        quarantine_count = quarantine_df.count()

        print(
            f"Valid records: "
            f"{valid_count}"
        )

        print(
            f"Quarantine records: "
            f"{quarantine_count}"
        )

        # =================================================
        # 4. Quarantine
        # =================================================

        quarantine_path = (
            f"s3a://{S3_BUCKET}/"
            f"quarantine/articles/"
        )

        if quarantine_count > 0:

            print(
                "\nWriting quarantine records..."
            )

            (
                quarantine_df
                .write
                .mode("append")
                .parquet(quarantine_path)
            )

            print(
                f"Quarantine written to: "
                f"{quarantine_path}"
            )

        # =================================================
        # 5. Prepare valid records
        # =================================================

        print("\n" + "=" * 70)
        print(
            "STEP 4 - "
            "PREPARING FOR SEMANTIC DEDUPLICATION"
        )
        print("=" * 70)

        valid_articles = [
            row.asDict(recursive=True)
            for row in valid_df.collect()
        ]

        for article in valid_articles:

            article.pop(
                "parsed_published_at",
                None,
            )

            article.pop(
                "data_quality_issue",
                None,
            )

            article.pop(
                "duplicate_count",
                None,
            )

            article.pop(
                "is_exact_duplicate",
                None,
            )

            article.pop(
                "_duplicate_rank",
                None,
            )

            article.pop(
                "is_valid",
                None,
            )

        # =================================================
        # 6. MinHash Deduplication
        # =================================================

        print("\n" + "=" * 70)
        print(
            "STEP 5 - "
            "SEMANTIC DEDUPLICATION"
        )
        print("=" * 70)

        (
            processed_articles,
            groups,
        ) = deduplicate_articles(
            valid_articles
        )

        duplicate_count = sum(
            1
            for article in processed_articles
            if article.get(
                "is_duplicate",
                False,
            )
        )

        canonical_count = (
            len(processed_articles)
            - duplicate_count
        )

        print(
            f"Processed valid articles: "
            f"{len(processed_articles)}"
        )

        print(
            f"Canonical articles: "
            f"{canonical_count}"
        )

        print(
            f"Semantic duplicates: "
            f"{duplicate_count}"
        )

        print(
            f"Duplicate groups: "
            f"{len(groups)}"
        )

        # =================================================
        # 7. Silver
        # =================================================

        print("\n" + "=" * 70)
        print("STEP 6 - SILVER DATASET")
        print("=" * 70)

        silver_batch_df = articles_to_dataframe(
            spark,
            processed_articles,
        )

        silver_path = (
            f"s3a://{S3_BUCKET}/"
            f"silver/articles/"
        )

        silver_df = append_batch(
            new_df=silver_batch_df,
            output_path=silver_path,
        )

        print(
            "Silver dataset updated:"
        )

        print(silver_path)

        # =================================================
        # 8. Gold
        # =================================================

        print("\n" + "=" * 70)
        print("STEP 7 - GOLD DATASET")
        print("=" * 70)

        # Only canonical records enter Gold.
        gold_batch_df = silver_batch_df.filter(
            F.col("is_duplicate") == False
        )

        gold_count = gold_batch_df.count()

        print(
            f"Gold canonical records "
            f"in current batch: "
            f"{gold_count}"
        )

        gold_path = (
            f"s3a://{S3_BUCKET}/"
            f"gold/articles/"
        )

        gold_df = append_batch(
            new_df=gold_batch_df,
            output_path=gold_path,
        )

        print(
            "Gold dataset updated:"
        )

        print(gold_path)

        # =================================================
        # 9. Feature Engineering
        # =================================================

        print("\n" + "=" * 70)
        print("STEP 8 - FEATURE ENGINEERING")
        print("=" * 70)

        enriched_df = enrich_articles(
            gold_batch_df
        )

        # =================================================
        # 10. Extractive Summarization
        # =================================================

        print("\n" + "=" * 70)
        print(
            "STEP 9 - "
            "EXTRACTIVE SUMMARIZATION"
        )
        print("=" * 70)

        enriched_df = add_extractive_summary(
            enriched_df
        )

        enriched_df = enriched_df.withColumn(
            "summary_word_count",
            F.when(
                F.col(
                    "extractive_summary"
                ).isNotNull(),
                F.size(
                    F.split(
                        F.trim(
                            F.col(
                                "extractive_summary"
                            )
                        ),
                        r"\s+",
                    )
                ),
            ).otherwise(0),
        )

        enriched_df = enriched_df.withColumn(
            "compression_ratio",
            F.when(
                F.col("word_count") > 0,
                F.col("summary_word_count")
                / F.col("word_count"),
            ).otherwise(0.0),
        )

        # Persist so statistics, ML-ready and the training layout share one
        # computation (same summaries and same processed_at values).
        enriched_df = enriched_df.persist()

        # =================================================
        # 11. Dataset Statistics
        # =================================================

        print("\n" + "=" * 70)
        print("STEP 10 - DATASET STATISTICS")
        print("=" * 70)

        statistics = (
            enriched_df
            .select(
                F.avg("word_count").alias(
                    "average_word_count"
                ),
                F.avg("summary_word_count").alias(
                    "average_summary_word_count"
                ),
                F.avg("compression_ratio").alias(
                    "average_compression_ratio"
                ),
            )
            .collect()[0]
        )

        average_word_count = (
            statistics["average_word_count"]
        )

        average_summary_word_count = (
            statistics["average_summary_word_count"]
        )

        average_compression_ratio = (
            statistics["average_compression_ratio"]
        )

        summaries_generated = (
            enriched_df
            .filter(
                F.col(
                    "extractive_summary"
                ).isNotNull()
            )
            .count()
        )

        print(
            f"Average word count: "
            f"{average_word_count}"
        )

        print(
            f"Average summary word count: "
            f"{average_summary_word_count}"
        )

        print(
            f"Average compression ratio: "
            f"{average_compression_ratio}"
        )

        print(
            f"Summaries generated: "
            f"{summaries_generated}"
        )

        # =================================================
        # 12. ML-ready dataset
        # =================================================

        export_path = ML_READY_PATH.replace(
            "s3://",
            "s3a://",
        )

        print("\n" + "=" * 70)
        print(
            "STEP 11 - "
            "ML-READY DATASET"
        )
        print("=" * 70)

        ml_ready_df = append_batch(
            new_df=enriched_df,
            output_path=export_path,
        )

        print(
            "ML-ready data updated:"
        )

        print(ML_READY_PATH)

        # =================================================
        # 12b. ML training layout (split / _ingest_date)
        # =================================================

        print("\n" + "=" * 70)
        print(
            "STEP 11b - "
            "ML TRAINING DATASET"
        )
        print("=" * 70)

        training_df = assign_split(
            enriched_df
        )

        write_training_dataset(
            training_df,
            ML_TRAINING_PATH.replace(
                "s3://",
                "s3a://",
            ),
        )

        split_counts = {
            row["split"]: row["count"]
            for row in (
                training_df
                .groupBy("split")
                .count()
                .collect()
            )
        }

        print(
            f"Training rows by split: "
            f"{split_counts}"
        )

        print(
            "ML training data updated:"
        )

        print(ML_TRAINING_PATH)

        # Every layer for this batch is written; advance the checkpoint.
        save_checkpoints(updated_checkpoints)

        # =================================================
        # 13. Monitoring
        # =================================================

        print("\n" + "=" * 70)
        print(
            "STEP 12 - "
            "PIPELINE MONITORING"
        )
        print("=" * 70)

        processing_time = (
            time.time()
            - pipeline_start_time
        )

        quality_rate = (
            valid_count / len(articles)
            if len(articles) > 0
            else 0.0
        )

        summary_generation_rate = (
            summaries_generated / gold_count
            if gold_count > 0
            else 0.0
        )

        metrics = {

            "records_ingested":
                len(articles),

            "records_malformed":
                len(malformed_records),

            "records_valid":
                valid_count,

            "records_quarantined":
                quarantine_count,

            "semantic_duplicates":
                duplicate_count,

            "canonical_articles":
                canonical_count,

            "duplicate_groups":
                len(groups),

            "data_quality_rate":
                quality_rate,

            "average_word_count":
                float(
                    average_word_count
                )
                if average_word_count is not None
                else 0.0,

            "average_summary_word_count":
                float(
                    average_summary_word_count
                )
                if average_summary_word_count is not None
                else 0.0,

            "average_compression_ratio":
                float(
                    average_compression_ratio
                )
                if average_compression_ratio is not None
                else 0.0,

            "summaries_generated":
                summaries_generated,

            "summary_generation_rate":
                summary_generation_rate,

            "processing_time_seconds":
                round(
                    processing_time,
                    3,
                ),

            "pipeline_version":
                "1.2.0",
        }

        print("\nPipeline metrics:")

        for metric, value in metrics.items():

            print(
                f"{metric}: "
                f"{value}"
            )

        # -------------------------------------------------
        # Save monitoring metrics
        # -------------------------------------------------

        record_pipeline_metrics(
            metrics
        )

        # -------------------------------------------------
        # Log pipeline metrics to MLflow
        # -------------------------------------------------

        log_pipeline_run(
            metrics
        )

        # =================================================
        # 14. Completion
        # =================================================

        print("\n" + "=" * 70)
        print(
            "PIPELINE COMPLETED SUCCESSFULLY"
        )
        print("=" * 70)

    finally:

        spark.stop()


if __name__ == "__main__":
    run_pipeline()