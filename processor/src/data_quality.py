from pyspark.sql import functions as F
from pyspark.sql.window import Window


# -------------------------------------------------------
# Required article fields
# -------------------------------------------------------

REQUIRED_COLUMNS = [
    "article_id",
    "source_id",
    "title",
    "body_text",
    "published_at",
]


# -------------------------------------------------------
# Validate and clean articles
# -------------------------------------------------------

def validate_and_clean_articles(df):

    # Check required columns exist
    for column in REQUIRED_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    # ---------------------------------------------------
    # Trim string columns
    # ---------------------------------------------------

    string_columns = [
    "article_id",
    "source_id",
    "source_name",
    "domain_name",
    "title",
    "subheading_text",
    "body_text",
    "url",
    "published_at",
    "language",
    "author",
    "primary_category",
]

    for column in string_columns:

        if column in df.columns:

            df = df.withColumn(
                column,
                F.trim(
                    F.col(column).cast("string")
                ),
            )

    # ---------------------------------------------------
    # Parse published timestamp
    # ---------------------------------------------------

    df = df.withColumn(
        "parsed_published_at",
        F.to_timestamp(
            F.col("published_at")
        ),
    )

    # ---------------------------------------------------
    # Identify data quality problems
    # ---------------------------------------------------

    df = df.withColumn(
        "data_quality_issue",

        F.when(
            F.col("article_id").isNull()
            |
            (F.col("article_id") == ""),
            "missing_article_id",
        )

        .when(
            F.col("source_id").isNull()
            |
            (F.col("source_id") == ""),
            "missing_source_id",
        )

        .when(
            F.col("title").isNull()
            |
            (F.col("title") == ""),
            "missing_title",
        )

        .when(
            F.col("body_text").isNull()
            |
            (F.col("body_text") == ""),
            "missing_body_text",
        )

        .when(
            F.col("parsed_published_at").isNull(),
            "invalid_published_at",
        )

        .when(
            F.length(
                F.col("body_text")
            ) < 10,
            "body_text_too_short",
        )

        .otherwise(None)
    )

    # ---------------------------------------------------
    # Detect exact duplicate article IDs
    # ---------------------------------------------------

    duplicate_window = Window.partitionBy(
        "article_id"
    )

    df = df.withColumn(
        "duplicate_count",
        F.count("*").over(
            duplicate_window
        ),
    )

    # Records without an article_id are already invalid
    # (missing_article_id); they must not also be counted as
    # duplicates of each other.
    df = df.withColumn(
        "is_exact_duplicate",
        (F.col("duplicate_count") > 1)
        & F.col("article_id").isNotNull()
        & (F.col("article_id") != ""),
    )

    # ---------------------------------------------------
    # Determine which duplicate is retained
    # ---------------------------------------------------

    first_record_window = (
        Window
        .partitionBy(
            "article_id"
        )
        .orderBy(
            F.col(
                "published_at"
            ).asc_nulls_last()
        )
    )

    df = df.withColumn(
        "_duplicate_rank",
        F.row_number().over(
            first_record_window
        ),
    )

    # ---------------------------------------------------
    # Overall validity
    # ---------------------------------------------------

    df = df.withColumn(
        "is_valid",
        F.col(
            "data_quality_issue"
        ).isNull(),
    )

    return df


# -------------------------------------------------------
# Quarantine reason for exact duplicates
# -------------------------------------------------------

DUPLICATE_ARTICLE_ID_ISSUE = "duplicate_article_id"


def with_duplicate_issue(df):
    """
    Give every non-first copy of an exact duplicate a
    data_quality_issue, so each quarantined row says why.

    Rows that already have an issue (e.g. missing_article_id)
    keep it. is_valid is not changed, so the valid/invalid
    counts stay the same.
    """

    return df.withColumn(
        "data_quality_issue",
        F.coalesce(
            F.col("data_quality_issue"),
            F.when(
                (F.col("is_exact_duplicate") == True)
                & (F.col("_duplicate_rank") > 1),
                F.lit(DUPLICATE_ARTICLE_ID_ISSUE),
            ),
        ),
    )


# -------------------------------------------------------
# Split valid records and quarantine records
# -------------------------------------------------------

def split_valid_and_quarantine(df):

    # Keep the first occurrence of an exact duplicate.
    # Invalid records are not sent to the ML pipeline.

    valid_df = df.filter(
        (F.col("is_valid") == True)
        &
        (
            (F.col("is_exact_duplicate") == False)
            |
            (F.col("_duplicate_rank") == 1)
        )
    )

    quarantine_df = with_duplicate_issue(df).filter(
        (F.col("is_valid") == False)
        |
        (
            (F.col("is_exact_duplicate") == True)
            &
            (F.col("_duplicate_rank") > 1)
        )
    )

    return (
        valid_df,
        quarantine_df
    )


# -------------------------------------------------------
# Print data quality summary
# -------------------------------------------------------

def print_quality_summary(df):

    total = df.count()

    invalid = (
        df
        .filter(
            F.col("is_valid") == False
        )
        .count()
    )

    exact_duplicates = (
        df
        .filter(
            (F.col("is_exact_duplicate") == True)
            &
            (
                F.col("_duplicate_rank") > 1
            )
        )
        .count()
    )

    valid = (
        total
        - invalid
        - exact_duplicates
    )

    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)

    print(
        f"Total records:           {total}"
    )

    print(
        f"Invalid records:         {invalid}"
    )

    print(
        f"Exact duplicates:        {exact_duplicates}"
    )

    print(
        f"Valid records:           {valid}"
    )

    print("=" * 60)

    print("\nIssues by type:")

    (
        with_duplicate_issue(df)
        .filter(
            F.col(
                "data_quality_issue"
            ).isNotNull()
        )
        .groupBy(
            "data_quality_issue"
        )
        .count()
        .orderBy(
            F.desc("count")
        )
        .show(
            truncate=False
        )
    )