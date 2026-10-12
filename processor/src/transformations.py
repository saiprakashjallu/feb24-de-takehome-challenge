from pyspark.sql import functions as F


PIPELINE_VERSION = "1.0.0"

# Train / validation / test split, assigned from a hash of article_id so an
# article keeps the same split no matter how much data is added later.
SPLIT_BUCKETS = 100
TRAIN_BUCKETS = 80       # buckets  0-79 -> train      (80%)
VALIDATION_BUCKETS = 10  # buckets 80-89 -> validation (10%), 90-99 -> test


def split_bucket_column():
    return F.pmod(
        F.xxhash64(F.col("article_id")),
        F.lit(SPLIT_BUCKETS),
    )


def assign_split(df):
    """
    Add a deterministic `split` column (train / validation / test).
    """

    bucket = split_bucket_column()

    return df.withColumn(
        "split",
        F.when(
            bucket < TRAIN_BUCKETS,
            F.lit("train"),
        ).when(
            bucket < TRAIN_BUCKETS + VALIDATION_BUCKETS,
            F.lit("validation"),
        ).otherwise(
            F.lit("test")
        ),
    )


def write_training_dataset(df, output_path):
    """
    Append a batch to the ML training layout, partitioned by
    split and then _ingest_date, with one file per partition.
    """

    (
        df
        .repartition("split", "_ingest_date")
        .write
        .mode("append")
        .partitionBy("split", "_ingest_date")
        .parquet(output_path)
    )


def enrich_articles(df):

    # =========================================================
    # BASIC TEXT FEATURES
    # =========================================================

    df = df.withColumn(
        "word_count",
        F.when(
            F.col("body_text").isNull()
            | (F.trim(F.col("body_text")) == ""),
            F.lit(0),
        ).otherwise(
            F.size(
                F.split(
                    F.trim(F.col("body_text")),
                    r"\s+",
                )
            )
        ),
    )

    df = df.withColumn(
        "character_count",
        F.coalesce(
            F.length(F.col("body_text")),
            F.lit(0),
        ),
    )

    df = df.withColumn(
        "non_whitespace_character_count",
        F.length(
            F.regexp_replace(
                F.coalesce(
                    F.col("body_text"),
                    F.lit(""),
                ),
                r"\s+",
                "",
            )
        ),
    )

    df = df.withColumn(
        "title_word_count",
        F.when(
            F.col("title").isNull()
            | (F.trim(F.col("title")) == ""),
            F.lit(0),
        ).otherwise(
            F.size(
                F.split(
                    F.trim(F.col("title")),
                    r"\s+",
                )
            )
        ),
    )

    df = df.withColumn(
        "sentence_count",
        F.when(
            F.col("body_text").isNull()
            | (F.trim(F.col("body_text")) == ""),
            F.lit(0),
        ).otherwise(
            F.size(
                F.split(
                    F.trim(F.col("body_text")),
                    r"[.!?]+",
                )
            )
        ),
    )

    df = df.withColumn(
        "avg_word_length",
        F.when(
            F.col("word_count") > 0,
            F.col("non_whitespace_character_count")
            / F.col("word_count"),
        ).otherwise(0.0),
    )

    # =========================================================
    # CONTENT VALIDATION
    # =========================================================

    df = df.withColumn(
        "has_content",
        (
            F.col("body_text").isNotNull()
            &
            (
                F.length(
                    F.trim(F.col("body_text"))
                ) > 0
            )
        ),
    )

    # =========================================================
    # ARTICLE FRESHNESS
    # =========================================================

    df = df.withColumn(
        "published_timestamp",
        F.to_timestamp(
            F.col("published_at")
        ),
    )

    df = df.withColumn(
        "article_age_hours",
        (
            F.unix_timestamp(
                F.current_timestamp()
            )
            -
            F.unix_timestamp(
                F.col("published_timestamp")
            )
        ) / 3600.0,
    )

    # =========================================================
    # PIPELINE METADATA
    # =========================================================

    df = df.withColumn(
        "processed_at",
        F.current_timestamp(),
    )

    df = df.withColumn(
        "pipeline_version",
        F.lit(PIPELINE_VERSION),
    )

    return df