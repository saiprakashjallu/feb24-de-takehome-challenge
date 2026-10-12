"""
Data-quality load test: publish ~100,000 records with a heavy, known mix of
problems to an ISOLATED stream and bucket (see dqtest.env).

Default mix ("heavy errors"):
    80,000 valid articles
    12,000 invalid      (2,000 per validation rule x 6)
     5,000 exact duplicates (later copy of a valid article, same article_id)
     2,500 near-duplicates  (one word changed, new article_id, arrive later)
       500 malformed        (250 not JSON, 250 JSON but not an object)
   -------
   100,000 records

Expected pipeline result:
    malformed 500 -> dead-letter
    ingested 99,500 | invalid 12,000 | exact duplicates 5,000
    quarantined 17,000 | valid 82,500 | semantic duplicates 2,500
    Gold / ML-ready / ML training +80,000

Near-duplicates are published after everything else, so the original always
arrives first and stays canonical.

Run from processor/ with the test environment loaded:
    set -a; source dqtest.env; set +a
    .venv/bin/python publish_load_test.py --setup     # create stream + bucket
    .venv/bin/python publish_load_test.py --dry-run   # DQ preview, no writes
    .venv/bin/python publish_load_test.py             # publish
    .venv/bin/python publish_load_test.py --scale 0.25   # 25,000-record version
"""

import argparse
import json
import random
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

import boto3

from publish_mixed_batch import VOCABULARY, make_invalid, valid_article
from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_ENDPOINT_URL,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    KINESIS_STREAM,
    S3_BUCKET,
)


VALID = 80_000
INVALID_PER_RULE = 2_000
EXACT_DUPLICATES = 5_000
NEAR_DUPLICATES = 2_500
MALFORMED = 500

RULES = [
    "missing_article_id",
    "missing_source_id",
    "missing_title",
    "missing_body_text",
    "invalid_published_at",
    "body_text_too_short",
]

MAIN_STREAM, MAIN_BUCKET = "MyStream", "my-bucket"


def client(service):
    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def guard_isolated():
    if KINESIS_STREAM == MAIN_STREAM or S3_BUCKET == MAIN_BUCKET:
        raise SystemExit(
            f"Refusing to run against the main environment "
            f"({KINESIS_STREAM}, {S3_BUCKET}). "
            f"Load dqtest.env first: set -a; source dqtest.env; set +a"
        )


def near_duplicate(rng, original, run_id, number):
    words = original["body_text"].split()
    index = rng.randrange(len(words))
    words[index] = rng.choice(
        [word for word in VOCABULARY if word != words[index].strip(".").lower()]
    )
    article_id = f"dqtest-load-{run_id}-near-{number:06d}"
    copy = dict(original)
    copy.update(
        {
            "article_id": article_id,
            "body_text": " ".join(words),
            "url": f"https://news.example.com/articles/{article_id}",
            "published_at": (
                datetime.fromisoformat(original["published_at"])
                + timedelta(hours=2)
            ).isoformat(),
        }
    )
    return copy


def malformed_payload(rng, run_id, number):
    if number % 2:
        return f"dqtest-load-{run_id}-{number} {{not json".encode("utf-8")
    return json.dumps(
        rng.choice([["dqtest", run_id, number], "just a string", 12345])
    ).encode("utf-8")


def build(run_id, seed=7):
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    valid = []
    for number in range(1, VALID + 1):
        article = valid_article(rng, run_id, number, now)
        article["article_id"] = f"dqtest-load-{run_id}-{number:06d}"
        article["url"] = f"https://news.example.com/articles/{article['article_id']}"
        valid.append(article)

    # Disjoint originals for exact and near duplicates keep the counts exact.
    originals = rng.sample(valid, EXACT_DUPLICATES + NEAR_DUPLICATES)
    exact_sources = originals[:EXACT_DUPLICATES]
    near_sources = originals[EXACT_DUPLICATES:]

    main = [("valid", article) for article in valid]

    for original in exact_sources:
        duplicate = dict(original)
        duplicate["published_at"] = (
            datetime.fromisoformat(original["published_at"])
            + timedelta(hours=1)
        ).isoformat()
        main.append(("exact_duplicate", duplicate))

    number = 1
    for rule in RULES:
        for _ in range(INVALID_PER_RULE):
            record = make_invalid(rng, run_id, number, now, rule)
            if "article_id" in record:
                record["article_id"] = f"dqtest-load-{run_id}-bad-{number:06d}"
            main.append((rule, record))
            number += 1

    for number in range(1, MALFORMED + 1):
        main.append(("malformed", malformed_payload(rng, run_id, number)))

    rng.shuffle(main)

    tail = [
        ("near_duplicate", near_duplicate(rng, original, run_id, number))
        for number, original in enumerate(near_sources, start=1)
    ]
    rng.shuffle(tail)

    return main + tail


def print_plan(records):
    kinds = Counter(kind for kind, _ in records)
    invalid = sum(kinds[rule] for rule in RULES)
    print(f"Target: stream={KINESIS_STREAM} bucket={S3_BUCKET}")
    print(f"Records: {len(records):,}")
    print(f"  valid             {kinds['valid']:>7,}")
    print(f"  invalid           {invalid:>7,}  ({INVALID_PER_RULE:,} per rule x {len(RULES)})")
    print(f"  exact duplicates  {kinds['exact_duplicate']:>7,}")
    print(f"  near-duplicates   {kinds['near_duplicate']:>7,}")
    print(f"  malformed         {kinds['malformed']:>7,}")
    print("Expected pipeline result:")
    print(
        f"  malformed {kinds['malformed']:,} -> dead-letter | "
        f"ingested {len(records) - kinds['malformed']:,}"
    )
    print(
        f"  invalid {invalid:,} | exact duplicates {kinds['exact_duplicate']:,} | "
        f"quarantined {invalid + kinds['exact_duplicate']:,}"
    )
    print(
        f"  valid {kinds['valid'] + kinds['near_duplicate']:,} | "
        f"semantic duplicates {kinds['near_duplicate']:,} | "
        f"Gold +{kinds['valid']:,}"
    )


def setup():
    kinesis, s3 = client("kinesis"), client("s3")
    try:
        kinesis.create_stream(StreamName=KINESIS_STREAM, ShardCount=1)
        print(f"Creating stream {KINESIS_STREAM} ...")
    except kinesis.exceptions.ResourceInUseException:
        print(f"Stream {KINESIS_STREAM} already exists")
    for _ in range(30):
        status = kinesis.describe_stream(StreamName=KINESIS_STREAM)[
            "StreamDescription"
        ]["StreamStatus"]
        if status == "ACTIVE":
            break
        time.sleep(1)
    print(f"Stream {KINESIS_STREAM}: {status}")
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if S3_BUCKET in buckets:
        print(f"Bucket {S3_BUCKET} already exists")
    else:
        s3.create_bucket(Bucket=S3_BUCKET)
        print(f"Bucket {S3_BUCKET} created")


def dry_run(records):
    from src.data_quality import (
        print_quality_summary,
        split_valid_and_quarantine,
        validate_and_clean_articles,
    )
    from src.pipeline import articles_to_dataframe, create_spark

    articles = [record for kind, record in records if kind != "malformed"]
    spark = create_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        cleaned = validate_and_clean_articles(
            articles_to_dataframe(spark, articles)
        )
        print_quality_summary(cleaned)
        valid_df, quarantine_df = split_valid_and_quarantine(cleaned)
        print(f"DRY RUN valid rows:      {valid_df.count():,}")
        print(f"DRY RUN quarantine rows: {quarantine_df.count():,}")
    finally:
        spark.stop()


def publish(records):
    kinesis = client("kinesis")
    started = time.time()
    for start in range(0, len(records), 500):
        chunk = records[start:start + 500]
        entries = []
        for _, record in chunk:
            if isinstance(record, bytes):
                entries.append({"Data": record, "PartitionKey": str(uuid.uuid4())})
            else:
                entries.append(
                    {
                        "Data": json.dumps(record).encode("utf-8"),
                        "PartitionKey": record.get("article_id") or str(uuid.uuid4()),
                    }
                )
        for attempt in range(5):
            response = kinesis.put_records(StreamName=KINESIS_STREAM, Records=entries)
            if not response.get("FailedRecordCount"):
                break
            entries = [
                entry
                for entry, result in zip(entries, response["Records"])
                if "ErrorCode" in result
            ]
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Records still failing in chunk at {start}")
        done = start + len(chunk)
        if done % 10_000 == 0 or done == len(records):
            print(f"Published {done:,} / {len(records):,} ({time.time() - started:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every count, e.g. 0.25 for a 25,000-record batch",
    )
    args = parser.parse_args()

    global VALID, INVALID_PER_RULE, EXACT_DUPLICATES, NEAR_DUPLICATES, MALFORMED
    VALID = round(VALID * args.scale)
    INVALID_PER_RULE = round(INVALID_PER_RULE * args.scale)
    EXACT_DUPLICATES = round(EXACT_DUPLICATES * args.scale)
    NEAR_DUPLICATES = round(NEAR_DUPLICATES * args.scale)
    MALFORMED = round(MALFORMED * args.scale)

    guard_isolated()

    if args.setup:
        setup()
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    records = build(run_id)

    print("=" * 70)
    print(f"DQ LOAD TEST (run {run_id})")
    print("=" * 70)
    print_plan(records)
    print("=" * 70)

    if args.dry_run:
        dry_run(records)
    else:
        publish(records)


if __name__ == "__main__":
    main()
