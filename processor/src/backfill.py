from pyspark.sql import SparkSession

try:
    from .config import (
        BRONZE_PATH,
        S3_BUCKET,
        AWS_ENDPOINT_URL,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
    )
except ImportError:
    from config import (
        BRONZE_PATH,
        S3_BUCKET,
        AWS_ENDPOINT_URL,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
    )


def create_spark():
    return (
        SparkSession.builder
        .appName("NewsETLBackfill")
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


def backfill_bronze(
    spark,
    start_date=None,
    end_date=None,
):
    """
    Read historical Bronze articles.

    Optional date filtering allows selective
    historical reprocessing.
    """

    bronze_path = BRONZE_PATH.replace(
        "s3://",
        "s3a://",
    )

    print("=" * 70)
    print("BACKFILL - READING BRONZE DATA")
    print("=" * 70)

    df = spark.read.parquet(
        bronze_path
    )

    print(
        f"Total Bronze records: "
        f"{df.count()}"
    )

    # ---------------------------------------------
    # Filter historical range
    # ---------------------------------------------

    if start_date:

        df = df.filter(
            df["_ingest_date"]
            >= start_date
        )

    if end_date:

        df = df.filter(
            df["_ingest_date"]
            <= end_date
        )

    print(
        f"Backfill records: "
        f"{df.count()}"
    )

    return df


def main():

    spark = create_spark()

    try:

        df = backfill_bronze(
            spark
        )

        print(
            "\nBackfill schema:"
        )

        df.printSchema()

        print(
            "\nSample records:"
        )

        df.select(
            "article_id",
            "source_name",
            "title",
            "published_at",
            "_ingest_timestamp",
        ).show(
            10,
            truncate=False,
        )

    finally:

        spark.stop()


if __name__ == "__main__":
    main()