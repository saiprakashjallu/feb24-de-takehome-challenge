"""
End-to-end pipeline validator (read-only).

Checks infrastructure, every medallion layer, cross-layer consistency,
feature ranges, the partitioned ML training layout, monitoring and MLflow. Prints one PASS/FAIL line per check,
then a summary. Exits non-zero if any check fails.

Nothing is written to S3, Kinesis or the checkpoint.

Run from processor/:
    export PYSPARK_PYTHON="$PWD/.venv/bin/python"
    export PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python"
    .venv/bin/python -m src.validate_pipeline
"""

import json
import sys
import urllib.request

import boto3
from pyspark.sql import functions as F

from .config import (
    AWS_ACCESS_KEY_ID,
    AWS_ENDPOINT_URL,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    BRONZE_PATH,
    CHECKPOINT_PATH,
    KINESIS_STREAM,
    MALFORMED_PATH,
    ML_READY_PATH,
    ML_TRAINING_PATH,
    MONITORING_PATH,
    S3_BUCKET,
)
from .transformations import (
    SPLIT_BUCKETS,
    TRAIN_BUCKETS,
    VALIDATION_BUCKETS,
    assign_split,
)


EXPERIMENT_NAME = "news-summarization"

SILVER_PATH = f"s3://{S3_BUCKET}/silver/articles/"
GOLD_PATH = f"s3://{S3_BUCKET}/gold/articles/"
QUARANTINE_PATH = f"s3://{S3_BUCKET}/quarantine/articles/"

LINEAGE_COLUMNS = [
    "_ingest_timestamp",
    "_source_shard_id",
    "_source_sequence_number",
    "_source_timestamp",
    "_record_key",
    "_pipeline_version",
    "_ingest_date",
]

ARTICLE_COLUMNS = [
    "article_id",
    "source_id",
    "source_name",
    "title",
    "author",
    "body_text",
    "url",
    "published_at",
    "language",
]

DEDUP_COLUMNS = [
    "is_duplicate",
    "canonical_article_id",
    "duplicate_group_id",
]

FEATURE_COLUMNS = [
    "word_count",
    "character_count",
    "sentence_count",
    "avg_word_length",
    "has_content",
    "published_timestamp",
    "article_age_hours",
    "processed_at",
    "pipeline_version",
    "extractive_summary",
    "summary_word_count",
    "compression_ratio",
]

EXPECTED_COLUMNS = {
    "Bronze": ARTICLE_COLUMNS + LINEAGE_COLUMNS,
    "Silver": ARTICLE_COLUMNS + LINEAGE_COLUMNS + DEDUP_COLUMNS,
    "Gold": ARTICLE_COLUMNS + LINEAGE_COLUMNS + DEDUP_COLUMNS,
    "ML-ready": (
        ARTICLE_COLUMNS + LINEAGE_COLUMNS + DEDUP_COLUMNS + FEATURE_COLUMNS
    ),
    "ML training": (
        ARTICLE_COLUMNS + LINEAGE_COLUMNS + DEDUP_COLUMNS + FEATURE_COLUMNS
        + ["split"]
    ),
}

MEDALLION_LAYERS = ("Bronze", "Silver", "Gold", "ML-ready")

# Allowed deviation (in percentage points) from the 80/10/10 split.
SPLIT_TOLERANCE_PP = 2.0


# -------------------------------------------------------
# Result tracking
# -------------------------------------------------------

results = []


def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    line = f"[{status}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line, flush=True)
    return passed


def info(message):
    print(f"[INFO] {message}", flush=True)


def check(name, fn):
    """Run fn() -> (passed, detail); an exception counts as FAIL."""
    try:
        passed, detail = fn()
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc) else ""
        passed, detail = False, f"{type(exc).__name__}: {first_line}"
    return record(name, passed, detail)


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def aws_client(service):
    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def split_s3(path):
    bucket, _, key = path.replace("s3://", "", 1).partition("/")
    return bucket, key


def s3a(path):
    return path.replace("s3://", "s3a://", 1)


def prefix_exists(s3, path):
    bucket, prefix = split_s3(path)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


# -------------------------------------------------------
# Checks
# -------------------------------------------------------

def check_infrastructure(s3):

    def localstack():
        url = AWS_ENDPOINT_URL.rstrip("/") + "/_localstack/health"
        with urllib.request.urlopen(url, timeout=5) as response:
            services = json.loads(response.read()).get("services", {})
        states = {s: services.get(s) for s in ("kinesis", "s3")}
        ok = all(state in ("running", "available") for state in states.values())
        return ok, f"{AWS_ENDPOINT_URL} {states}"

    def stream():
        description = aws_client("kinesis").describe_stream(
            StreamName=KINESIS_STREAM
        )["StreamDescription"]
        status = description["StreamStatus"]
        shards = len(description["Shards"])
        return status == "ACTIVE", f"{KINESIS_STREAM} {status}, {shards} shard(s)"

    def bucket():
        s3.head_bucket(Bucket=S3_BUCKET)
        return True, S3_BUCKET

    def checkpoint():
        bucket_name, prefix = split_s3(CHECKPOINT_PATH)
        key = prefix.rstrip("/") + "/kinesis_checkpoint.json"
        body = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
        checkpoints = json.loads(body)
        ok = isinstance(checkpoints, dict) and len(checkpoints) > 0
        return ok, f"s3://{bucket_name}/{key} -> {len(checkpoints)} shard(s)"

    reachable = check("LocalStack reachable (kinesis, s3)", localstack)
    if not reachable:
        return False

    check("Kinesis stream ACTIVE", stream)
    bucket_ok = check("S3 bucket exists", bucket)
    check("Kinesis checkpoint JSON exists", checkpoint)
    return bucket_ok


def load_layers(spark):

    paths = {
        "Bronze": BRONZE_PATH,
        "Silver": SILVER_PATH,
        "Gold": GOLD_PATH,
        "ML-ready": ML_READY_PATH,
        "ML training": ML_TRAINING_PATH,
    }

    layers = {}

    for name, path in paths.items():

        def layer_schema(name=name, path=path):
            df = spark.read.parquet(s3a(path))
            missing = sorted(set(EXPECTED_COLUMNS[name]) - set(df.columns))
            layers[name] = df
            if missing:
                return False, f"{path} missing columns: {missing}"
            return True, f"{path} ({len(df.columns)} columns)"

        check(f"{name} exists with expected columns", layer_schema)

    return layers


def check_consistency(spark, s3, layers):

    if not all(name in layers for name in MEDALLION_LAYERS):
        record(
            "Layer consistency",
            False,
            "skipped: one or more layers could not be read",
        )
        return

    bronze = layers["Bronze"]
    silver = layers["Silver"]
    gold = layers["Gold"]
    ml_ready = layers["ML-ready"]

    seq = "_source_sequence_number"

    bronze_rows = bronze.count()
    bronze_unique = bronze.select(seq).distinct().count()
    silver_rows = silver.count()
    gold_rows = gold.count()
    ml_rows = ml_ready.count()

    if prefix_exists(s3, QUARANTINE_PATH):
        quarantine_rows = spark.read.parquet(s3a(QUARANTINE_PATH)).count()
    else:
        quarantine_rows = 0

    info(
        f"Row counts: Bronze={bronze_rows} "
        f"(unique Kinesis records={bronze_unique}), "
        f"Quarantine={quarantine_rows}, Silver={silver_rows}, "
        f"Gold={gold_rows}, ML-ready={ml_rows}"
    )

    malformed_bucket, malformed_prefix = split_s3(MALFORMED_PATH)
    malformed_count = sum(
        page.get("KeyCount", 0)
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=malformed_bucket, Prefix=malformed_prefix
        )
    )
    info(
        f"Malformed (non-JSON) records in dead-letter: {malformed_count} "
        f"under {MALFORMED_PATH} (never reach Bronze)"
    )

    if bronze_rows != bronze_unique:
        info(
            f"Bronze holds {bronze_rows - bronze_unique} replayed rows "
            f"(same Kinesis sequence number ingested more than once); "
            f"consistency checks use unique records."
        )

    def silver_matches_bronze_valid():
        orphans = silver.join(bronze, on=seq, how="left_anti").count()
        silver_unique = silver.select(seq).distinct().count()
        ok = orphans == 0 and silver_unique == silver_rows
        return ok, (
            f"Silver={silver_rows}, Silver rows not traceable to Bronze="
            f"{orphans}, duplicate Silver records="
            f"{silver_rows - silver_unique}"
        )

    def bronze_equals_valid_plus_quarantine():
        expected = silver_rows + quarantine_rows
        return bronze_unique == expected, (
            f"Bronze unique={bronze_unique}, "
            f"valid({silver_rows}) + quarantined({quarantine_rows})={expected}"
        )

    def gold_equals_silver_canonical():
        canonical = silver.filter(F.col("is_duplicate") == False).count()
        non_canonical_in_gold = gold.filter(
            F.col("is_duplicate") != False
        ).count()
        ok = gold_rows == canonical and non_canonical_in_gold == 0
        return ok, (
            f"Gold={gold_rows}, Silver canonical={canonical}, "
            f"duplicates in Gold={non_canonical_in_gold}"
        )

    def ml_ready_equals_gold():
        gold_ids = gold.select("article_id")
        ml_ids = ml_ready.select("article_id")
        missing = gold_ids.join(ml_ids, "article_id", "left_anti").count()
        extra = ml_ids.join(gold_ids, "article_id", "left_anti").count()
        ok = ml_rows == gold_rows and missing == 0 and extra == 0
        return ok, (
            f"ML-ready={ml_rows}, Gold={gold_rows}, "
            f"Gold ids missing from ML-ready={missing}, "
            f"ML-ready ids not in Gold={extra}"
        )

    check("Silver = Bronze valid rows", silver_matches_bronze_valid)
    check("Bronze = valid + quarantined", bronze_equals_valid_plus_quarantine)
    check("Gold = Silver canonical rows", gold_equals_silver_canonical)
    check("ML-ready = Gold", ml_ready_equals_gold)


def check_article_ids(layers):

    for name in ("Gold", "ML-ready"):

        if name not in layers:
            record(f"{name} article_id not null and unique", False, "layer unreadable")
            continue

        def ids(df=layers[name]):
            total = df.count()
            nulls = df.filter(
                F.col("article_id").isNull() | (F.trim("article_id") == "")
            ).count()
            distinct = df.select("article_id").distinct().count()
            duplicates = total - distinct
            ok = nulls == 0 and duplicates == 0
            return ok, f"rows={total}, null/empty={nulls}, duplicates={duplicates}"

        check(f"{name} article_id not null and unique", ids)


def check_feature_ranges(layers):

    if "ML-ready" not in layers:
        record("ML-ready feature ranges", False, "layer unreadable")
        return

    df = layers["ML-ready"]

    def word_count_positive():
        bad = df.filter(
            F.col("word_count").isNull() | (F.col("word_count") <= 0)
        ).count()
        return bad == 0, f"rows with word_count <= 0 or null: {bad}"

    def compression_ratio_range():
        bad = df.filter(
            F.col("compression_ratio").isNull()
            | (F.col("compression_ratio") <= 0)
            | (F.col("compression_ratio") >= 1)
        ).count()
        stats = df.select(
            F.min("compression_ratio").alias("min"),
            F.max("compression_ratio").alias("max"),
        ).collect()[0]
        return bad == 0, (
            f"rows outside (0, 1): {bad}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}"
        )

    def summary_not_longer():
        bad = df.filter(
            F.col("summary_word_count").isNull()
            | (F.col("summary_word_count") > F.col("word_count"))
        ).count()
        return bad == 0, f"rows with summary_word_count > word_count: {bad}"

    check("word_count > 0", word_count_positive)
    check("0 < compression_ratio < 1", compression_ratio_range)
    check("summary_word_count <= word_count", summary_not_longer)

    averages = df.select(
        F.avg("word_count").alias("words"),
        F.avg("summary_word_count").alias("summary_words"),
        F.avg("compression_ratio").alias("ratio"),
    ).collect()[0]

    info(
        f"Average word_count={averages['words']:.2f}, "
        f"average summary_word_count={averages['summary_words']:.2f}, "
        f"average compression_ratio={averages['ratio']:.4f}"
    )


def check_training_layout(layers):

    names = ("ML-ready", "ML training")

    if not all(name in layers for name in names):
        record("ML training layout", False, "skipped: layer unreadable")
        return

    ml_ready = layers["ML-ready"]
    training = layers["ML training"]

    split_counts = {
        row["split"]: row["count"]
        for row in training.groupBy("split").count().collect()
    }
    partitions = training.select("split", "_ingest_date").distinct().count()

    info(
        f"ML training rows by split: {split_counts}, "
        f"{partitions} split/_ingest_date partitions"
    )

    def matches_ml_ready():
        ml_ids = ml_ready.select("article_id")
        training_ids = training.select("article_id")
        ml_rows = ml_ready.count()
        training_rows = training.count()
        missing = ml_ids.join(training_ids, "article_id", "left_anti").count()
        extra = training_ids.join(ml_ids, "article_id", "left_anti").count()
        ok = ml_rows == training_rows and missing == 0 and extra == 0
        return ok, (
            f"ML training={training_rows}, ML-ready={ml_rows}, "
            f"ML-ready ids missing from training={missing}, "
            f"training ids not in ML-ready={extra}"
        )

    def split_matches_hash():
        expected = assign_split(
            training.select("article_id", F.col("split").alias("stored"))
        )
        mismatched = expected.filter(F.col("split") != F.col("stored")).count()
        return mismatched == 0, (
            f"rows whose stored split differs from the article_id hash: "
            f"{mismatched}"
        )

    def one_split_per_article():
        per_article = training.groupBy("article_id").agg(
            F.count("*").alias("rows"),
            F.countDistinct("split").alias("splits"),
        )
        leaked = per_article.filter(F.col("splits") > 1).count()
        duplicated = per_article.filter(F.col("rows") > 1).count()
        ok = leaked == 0 and duplicated == 0
        return ok, (
            f"articles in more than one split={leaked}, "
            f"articles with more than one row={duplicated}"
        )

    def split_ratio():
        total = sum(split_counts.values())
        target = {
            "train": 100.0 * TRAIN_BUCKETS / SPLIT_BUCKETS,
            "validation": 100.0 * VALIDATION_BUCKETS / SPLIT_BUCKETS,
            "test": 100.0
            * (SPLIT_BUCKETS - TRAIN_BUCKETS - VALIDATION_BUCKETS)
            / SPLIT_BUCKETS,
        }
        actual = {
            split: 100.0 * split_counts.get(split, 0) / total if total else 0.0
            for split in target
        }
        ok = total > 0 and all(
            abs(actual[split] - target[split]) <= SPLIT_TOLERANCE_PP
            for split in target
        )
        shown = ", ".join(
            f"{split}={actual[split]:.2f}%" for split in target
        )
        return ok, f"{shown} (target 80/10/10, tolerance +/-{SPLIT_TOLERANCE_PP}pp)"

    check("ML training = ML-ready", matches_ml_ready)
    check("ML training split matches article_id hash", split_matches_hash)
    check("ML training article in exactly one split", one_split_per_article)
    check("ML training split ratio ~80/10/10", split_ratio)


def check_monitoring(s3):

    def latest_monitoring():
        bucket, prefix = split_s3(MONITORING_PATH)
        objects = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            objects.extend(
                o for o in page.get("Contents", []) if o["Key"].endswith(".json")
            )
        if not objects:
            return False, f"no monitoring JSON under {MONITORING_PATH}"
        latest = max(objects, key=lambda o: o["LastModified"])
        record_json = json.loads(
            s3.get_object(Bucket=bucket, Key=latest["Key"])["Body"].read()
        )
        ok = "records_ingested" in record_json
        return ok, (
            f"s3://{bucket}/{latest['Key']} "
            f"(recorded_at={record_json.get('recorded_at')}, "
            f"records_ingested={record_json.get('records_ingested')})"
        )

    check("Latest monitoring JSON exists", latest_monitoring)


def check_mlflow():

    def latest_run():
        import mlflow

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return False, f"experiment '{EXPERIMENT_NAME}' not found"
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.records_ingested >= 0",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            return False, f"no pipeline runs in '{EXPERIMENT_NAME}'"
        run = runs.iloc[0]
        ok = run["status"] == "FINISHED"
        return ok, (
            f"run_id={run['run_id']}, status={run['status']}, "
            f"started={run['start_time']}, "
            f"records_ingested={int(run['metrics.records_ingested'])}"
        )

    check("Latest MLflow pipeline run exists", latest_run)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    print("=" * 70)
    print("PIPELINE VALIDATION (read-only)")
    print("=" * 70)

    s3 = aws_client("s3")

    infrastructure_ok = check_infrastructure(s3)

    if infrastructure_ok:

        from .pipeline import create_spark

        spark = create_spark()
        spark.sparkContext.setLogLevel("ERROR")

        try:
            layers = load_layers(spark)
            check_consistency(spark, s3, layers)
            check_article_ids(layers)
            check_feature_ranges(layers)
            check_training_layout(layers)
        finally:
            spark.stop()

        check_monitoring(s3)

    check_mlflow()

    passed = sum(1 for _, ok in results if ok)
    failed = [name for name, ok in results if not ok]

    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {len(failed)} failed, {len(results)} checks")
    for name in failed:
        print(f"  FAILED: {name}")
    print("RESULT: " + ("PASS" if not failed else "FAIL"))
    print("=" * 70)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
