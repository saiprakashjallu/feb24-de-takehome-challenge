import hashlib

from pyspark.sql import functions as F
from pyspark.sql import SparkSession

try:
    from .config import ML_TRAINING_PATH
except ImportError:
    from config import ML_TRAINING_PATH


EXPERIMENT_NAME = "news-summarization"


def create_spark():
    """
    Create a Spark session for the training workflow.
    """

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
        .appName("NewsSummarizationTraining")
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


def load_training_data(spark):
    """
    Load the partitioned ML training dataset
    (split / _ingest_date).
    """

    training_path = ML_TRAINING_PATH.replace(
        "s3://",
        "s3a://",
    )

    print(
        f"Loading ML training dataset from: "
        f"{ML_TRAINING_PATH}"
    )

    df = spark.read.parquet(
        training_path
    )

    return df


def validate_training_data(df):
    """
    Validate that the ML-ready dataset contains
    the fields required for summarization.

    The current source does not contain human-written
    reference summaries, so this function validates
    the extractive baseline dataset rather than
    pretending that supervised labels exist.
    """

    required_columns = [
        "article_id",
        "body_text",
        "extractive_summary",
        "word_count",
        "summary_word_count",
        "compression_ratio",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required training columns: "
            + ", ".join(missing_columns)
        )

    validated_df = (
        df
        .filter(
            F.col("body_text").isNotNull()
            & (F.trim(F.col("body_text")) != "")
        )
        .filter(
            F.col("extractive_summary").isNotNull()
            & (
                F.trim(
                    F.col("extractive_summary")
                )
                != ""
            )
        )
        .filter(
            F.col("word_count") > 0
        )
        .filter(
            F.col("summary_word_count") > 0
        )
    )

    return validated_df


def create_training_splits(df):
    """
    Return train/validation/test DataFrames.

    The split is stored per article (a hash of article_id, assigned
    when the row is written), so an article never moves between
    splits as the dataset grows. Filtering on the `split` partition
    column only reads that split's files.

    These splits are intended for the future supervised
    summarization model once reference summaries are available.
    """

    train_df = df.filter(F.col("split") == "train")
    validation_df = df.filter(F.col("split") == "validation")
    test_df = df.filter(F.col("split") == "test")

    return (
        train_df,
        validation_df,
        test_df,
    )


def log_dataset_version(
    df,
    train_df,
    validation_df,
    test_df,
):
    """
    Log which data a training run used to MLflow: rows per split,
    the _ingest_date range and a hash of the schema.
    """

    import mlflow

    date_range = df.select(
        F.min("_ingest_date").alias("first"),
        F.max("_ingest_date").alias("last"),
    ).collect()[0]

    schema_hash = hashlib.sha256(
        df.schema.json().encode("utf-8")
    ).hexdigest()[:12]

    counts = {
        "rows_total": df.count(),
        "rows_train": train_df.count(),
        "rows_validation": validation_df.count(),
        "rows_test": test_df.count(),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="training-dataset"):

        mlflow.set_tag("run_type", "training_dataset")

        mlflow.log_params(
            {
                "dataset_path": ML_TRAINING_PATH,
                "partitioning": "split/_ingest_date",
                "split_method": "xxhash64(article_id) % 100: 0-79 train, 80-89 validation, 90-99 test",
                "ingest_date_first": str(date_range["first"]),
                "ingest_date_last": str(date_range["last"]),
                "schema_hash": schema_hash,
            }
        )

        for name, value in counts.items():
            mlflow.log_metric(name, value)

    print(
        f"\nDataset version logged to MLflow: {counts}, "
        f"ingest dates {date_range['first']}..{date_range['last']}, "
        f"schema {schema_hash}"
    )


def print_dataset_summary(
    df,
    train_df,
    validation_df,
    test_df,
):
    """
    Print dataset statistics for the training workflow.
    """

    total_count = df.count()

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING DATASET SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total usable records: "
        f"{total_count}"
    )

    print(
        f"Training records: "
        f"{train_df.count()}"
    )

    print(
        f"Validation records: "
        f"{validation_df.count()}"
    )

    print(
        f"Test records: "
        f"{test_df.count()}"
    )

    print(
        "\nColumns available for summarization:"
    )

    for column in [
        "article_id",
        "body_text",
        "extractive_summary",
        "word_count",
        "summary_word_count",
        "compression_ratio",
    ]:
        print(
            f"  - {column}"
        )

    print(
        "\nImportant:"
    )

    print(
        "The current dataset does not contain "
        "human-authored reference summaries."
    )

    print(
        "The extractive_summary column is therefore "
        "treated as a baseline output, not as a "
        "ground-truth supervised training label."
    )

    print(
        "=" * 70
    )


def run_training_workflow():

    spark = create_spark()

    try:

        print(
            "\n" + "=" * 70
        )

        print(
            "NEWS SUMMARIZATION TRAINING WORKFLOW"
        )

        print(
            "=" * 70
        )

        # -------------------------------------------------
        # 1. Load ML-ready data
        # -------------------------------------------------

        df = load_training_data(
            spark
        )

        print(
            f"Loaded records: "
            f"{df.count()}"
        )

        # -------------------------------------------------
        # 2. Validate dataset
        # -------------------------------------------------

        df = validate_training_data(
            df
        )

        print(
            f"Usable records after validation: "
            f"{df.count()}"
        )

        # -------------------------------------------------
        # 3. Select the stored, deterministic splits
        # -------------------------------------------------

        (
            train_df,
            validation_df,
            test_df,
        ) = create_training_splits(
            df
        )

        # -------------------------------------------------
        # 4. Print dataset summary
        # -------------------------------------------------

        print_dataset_summary(
            df,
            train_df,
            validation_df,
            test_df,
        )

        # -------------------------------------------------
        # 4b. Log the dataset version to MLflow
        # -------------------------------------------------

        log_dataset_version(
            df,
            train_df,
            validation_df,
            test_df,
        )

        # -------------------------------------------------
        # 5. Display sample records
        # -------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "SAMPLE TRAINING RECORDS"
        )

        print(
            "=" * 70
        )

        (
            train_df
            .select(
                "article_id",
                "word_count",
                "summary_word_count",
                "compression_ratio",
                "body_text",
                "extractive_summary",
            )
            .show(
                5,
                truncate=120,
            )
        )

        # -------------------------------------------------
        # 6. Training readiness status
        # -------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "TRAINING READINESS"
        )

        print(
            "=" * 70
        )

        print(
            "Status: BASELINE READY"
        )

        print(
            "Supervised model training: "
            "WAITING FOR REFERENCE SUMMARIES"
        )

        print(
            "Current summarization approach: "
            "EXTRACTIVE BASELINE"
        )

        print(
            "Future evaluation: "
            "ROUGE-1 / ROUGE-2 / ROUGE-L"
        )

        print(
            "=" * 70
        )

    finally:

        spark.stop()


if __name__ == "__main__":

    run_training_workflow()
