from pyspark.sql import functions as F
from pyspark.sql import SparkSession

try:
    from .mlflow_tracking import (
        log_summarization_baseline
    )
except ImportError:
    from mlflow_tracking import (
        log_summarization_baseline
    )

try:
    from .config import ML_READY_PATH
except ImportError:
    from config import ML_READY_PATH


def create_spark():

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
        .appName("NewsSummarizationEvaluation")
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
        .getOrCreate()
    )


def evaluate_baseline(df):

    total_records = df.count()

    valid_summaries = (
        df
        .filter(
            F.col("extractive_summary").isNotNull()
            & (
                F.trim(
                    F.col("extractive_summary")
                )
                != ""
            )
        )
        .count()
    )

    summary_success_rate = (
        valid_summaries / total_records
        if total_records > 0
        else 0.0
    )

    empty_summary_rate = (
        1
        - (valid_summaries / total_records)
        if total_records > 0
        else 0.0
    )

    avg_article_words = (
        df
        .select(
            F.avg("word_count")
        )
        .collect()[0][0]
    )

    avg_summary_words = (
        df
        .select(
            F.avg("summary_word_count")
        )
        .collect()[0][0]
    )

    avg_compression_ratio = (
        df
        .select(
            F.avg("compression_ratio")
        )
        .collect()[0][0]
    )

    min_compression_ratio = (
        df
        .select(
            F.min("compression_ratio")
        )
        .collect()[0][0]
    )

    max_compression_ratio = (
        df
        .select(
            F.max("compression_ratio")
        )
        .collect()[0][0]
    )

    average_word_reduction = (
        (1 - avg_compression_ratio) * 100
        if avg_compression_ratio is not None
        else 0.0
    )

    metrics = {
        "articles_evaluated": total_records,
        "valid_summaries": valid_summaries,
        "summary_success_rate": summary_success_rate,
        "average_summary_words": avg_summary_words,
        "average_compression_ratio": avg_compression_ratio,
        "average_word_reduction": average_word_reduction,
        "min_compression_ratio": min_compression_ratio,
        "max_compression_ratio": max_compression_ratio,
    }

    log_summarization_baseline(
        metrics
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXTRACTIVE SUMMARIZATION BASELINE EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Total articles: "
        f"{total_records}"
    )

    print(
        f"Valid summaries: "
        f"{valid_summaries}"
    )

    print(
        f"Empty summary rate: "
        f"{empty_summary_rate:.4f}"
    )

    print(
        f"Average article words: "
        f"{avg_article_words:.2f}"
    )

    print(
        f"Average summary words: "
        f"{avg_summary_words:.2f}"
    )

    print(
        f"Average compression ratio: "
        f"{avg_compression_ratio:.4f}"
    )

    print(
        f"Minimum compression ratio: "
        f"{min_compression_ratio:.4f}"
    )

    print(
        f"Maximum compression ratio: "
        f"{max_compression_ratio:.4f}"
    )

    if avg_compression_ratio is not None:

        reduction = (
            1 - avg_compression_ratio
        ) * 100

        print(
            f"Average word reduction: "
            f"{reduction:.2f}%"
        )

    print(
        "\nEvaluation limitation:"
    )

    print(
        "ROUGE cannot currently be calculated because "
        "the source dataset does not contain human-authored "
        "reference summaries."
    )

    print(
        "=" * 70
    )


def run_evaluation():

    spark = create_spark()

    try:

        path = ML_READY_PATH.replace(
            "s3://",
            "s3a://",
        )

        print(
            f"Loading ML-ready dataset from: "
            f"{ML_READY_PATH}"
        )

        df = spark.read.parquet(
            path
        )

        evaluate_baseline(
            df
        )

    finally:

        spark.stop()


if __name__ == "__main__":

    run_evaluation()
