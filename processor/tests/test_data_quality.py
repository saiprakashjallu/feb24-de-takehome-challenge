import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from pyspark.sql import SparkSession

from src.data_quality import (
    validate_and_clean_articles,
    split_valid_and_quarantine,
)


def create_spark():

    return (
        SparkSession.builder
        .master("local[2]")
        .appName("DataQualityTest")
        .getOrCreate()
    )


def test_data_quality():

    spark = create_spark()

    data = [
        {
            "article_id": "article-001",
            "source_id": "source-tech",
            "source_name": "Tech Daily",
            "title": "AI technology article",
            "body_text": (
                "This is a valid article "
                "with enough content."
            ),
            "published_at": "2026-10-01T10:00:00",
            "language": "en",
        },
        {
            "article_id": "",
            "source_id": "source-tech",
            "source_name": "Tech Daily",
            "title": "Invalid article",
            "body_text": "Some content here.",
            "published_at": "2026-10-01T10:00:00",
            "language": "en",
        },
        {
            "article_id": "article-003",
            "source_id": "source-tech",
            "source_name": "Tech Daily",
            "title": "Missing body",
            "body_text": "",
            "published_at": "2026-10-01T10:00:00",
            "language": "en",
        },
    ]

    df = spark.createDataFrame(data)

    cleaned_df = validate_and_clean_articles(
        df
    )

    valid_df, quarantine_df = (
        split_valid_and_quarantine(
            cleaned_df
        )
    )

    print("\nVALID RECORDS:")

    valid_df.select(
        "article_id",
        "is_valid",
        "data_quality_issue",
    ).show(
        truncate=False
    )

    print("\nQUARANTINE RECORDS:")

    quarantine_df.select(
        "article_id",
        "is_valid",
        "data_quality_issue",
    ).show(
        truncate=False
    )

    assert (
        valid_df.count() == 1
    )

    assert (
        quarantine_df.count() == 2
    )

    spark.stop()


def test_exact_duplicate_has_quarantine_reason():

    spark = create_spark()

    base = {
        "article_id": "article-dup",
        "source_id": "source-tech",
        "source_name": "Tech Daily",
        "title": "Duplicated article",
        "body_text": "This article is published twice.",
        "language": "en",
    }

    df = spark.createDataFrame(
        [
            {**base, "published_at": "2026-10-01T10:00:00"},
            {**base, "published_at": "2026-10-01T11:00:00"},
        ]
    )

    valid_df, quarantine_df = split_valid_and_quarantine(
        validate_and_clean_articles(df)
    )

    quarantined = quarantine_df.collect()

    assert valid_df.count() == 1
    assert len(quarantined) == 1
    assert quarantined[0]["data_quality_issue"] == "duplicate_article_id"
    assert quarantined[0]["published_at"] == "2026-10-01T11:00:00"

    spark.stop()



def test_missing_ids_are_not_counted_as_duplicates():

    spark = create_spark()

    base = {
        "source_id": "source-tech",
        "source_name": "Tech Daily",
        "title": "Article without an id",
        "body_text": "This article has no article_id at all.",
        "published_at": "2026-10-01T10:00:00",
        "language": "en",
    }

    df = spark.createDataFrame(
        [
            {**base, "article_id": None},
            {**base, "article_id": None},
            {**base, "article_id": ""},
            {**base, "article_id": "article-ok"},
        ]
    )

    cleaned = validate_and_clean_articles(df)
    valid_df, quarantine_df = split_valid_and_quarantine(cleaned)

    assert cleaned.filter("is_exact_duplicate").count() == 0
    assert valid_df.count() == 1
    assert quarantine_df.count() == 3
    assert {
        row["data_quality_issue"] for row in quarantine_df.collect()
    } == {"missing_article_id"}

    spark.stop()


if __name__ == "__main__":
    test_data_quality()