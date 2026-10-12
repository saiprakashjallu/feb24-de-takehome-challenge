"""
Publish one realistic mixed batch to Kinesis so the data-quality stage has
real failures to catch:

    4,030 valid articles
      100 invalid articles (spread over the 6 validation rules)
       80 exact duplicates (a later copy of 80 of the valid articles)
    -----
    4,210 records  ->  expected: 4,030 valid, 100 invalid,
                                 80 exact duplicates, 180 quarantined

Every record uses source_name "DQ Test Source" and an article_id starting
with "dqtest-mixed-" (except the records that deliberately have no
article_id), so the test data can be filtered out of every layer.

Article text is generated from a fixed vocabulary with a seeded RNG (no
Faker dependency), so the batch is reproducible and the valid articles are
not near-duplicates of each other.

Run from processor/:
    .venv/bin/python publish_mixed_batch.py --dry-run   # no Kinesis/S3 writes
    .venv/bin/python publish_mixed_batch.py             # publish to Kinesis
"""

import argparse
import json
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

import boto3

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_ENDPOINT_URL,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    KINESIS_STREAM,
)


VALID_COUNT = 4030
EXACT_DUPLICATE_COUNT = 80

# 100 invalid records, one data_quality_issue each.
INVALID_PLAN = {
    "missing_article_id": 17,
    "missing_source_id": 17,
    "missing_title": 17,
    "missing_body_text": 17,
    "invalid_published_at": 16,
    "body_text_too_short": 16,
}

SOURCES = [
    "Tech Daily",
    "Global News",
    "Business Journal",
    "World News",
    "Sports Network",
]

VOCABULARY = """
city council budget transit plan vote residents district hospital school
teacher student market price energy power grid storm rain flood river bridge
road traffic police court judge law policy election candidate campaign voter
company profit revenue worker union strike factory farm crop harvest export
import trade tariff bank loan interest inflation housing rent tenant landlord
museum artist festival music film theater library science research study
scientist data report survey health doctor nurse patient clinic vaccine virus
climate carbon emission solar wind battery vehicle electric airport flight
port ship cargo border security military defense minister president senator
mayor governor agency office program funding grant tax spending deficit debt
growth recession jobs unemployment wage salary pension retirement insurance
technology software startup investor network internet privacy cyber attack
team coach player season match tournament league championship stadium fans
water supply drought wildfire forest park wildlife ocean coast beach tourism
hotel restaurant food safety recall product consumer retail store online
""".split()

VERBS = """
announced approved rejected proposed reported warned confirmed delayed
expanded reduced increased launched reviewed criticized welcomed funded
""".split()


def sentence(rng):
    words = rng.sample(VOCABULARY, rng.randint(7, 13))
    words.insert(rng.randint(1, len(words) - 1), rng.choice(VERBS))
    return " ".join(words).capitalize() + "."


def body(rng):
    # About 130-150 words, similar to the normal publisher's articles.
    sentences = []
    while sum(len(s.split()) for s in sentences) < rng.randint(125, 150):
        sentences.append(sentence(rng))
    return " ".join(sentences)


def valid_article(rng, run_id, number, now):
    article_id = f"dqtest-mixed-{run_id}-{number:05d}"
    return {
        "article_id": article_id,
        "source_id": f"dqtest-source-{rng.randint(1, 5)}",
        "source_name": "DQ Test Source",
        "title": sentence(rng).rstrip("."),
        "author": f"Test Author {rng.randint(1, 50)}",
        "body_text": body(rng),
        "url": f"https://news.example.com/articles/{article_id}",
        "published_at": (
            now - timedelta(minutes=rng.randint(60, 60 * 24 * 30))
        ).isoformat(),
        "language": "en",
    }


def make_invalid(rng, run_id, number, now, issue):
    record = valid_article(rng, run_id, number, now)
    record["article_id"] = f"dqtest-mixed-{run_id}-bad-{number:05d}"

    if issue == "missing_article_id":
        record.pop("article_id")
    elif issue == "missing_source_id":
        record["source_id"] = ""
    elif issue == "missing_title":
        record["title"] = None
    elif issue == "missing_body_text":
        record["body_text"] = ""
    elif issue == "invalid_published_at":
        record["published_at"] = rng.choice(
            ["not-a-date", "yesterday", "2026-13-45", "31/02/2026"]
        )
    elif issue == "body_text_too_short":
        record["body_text"] = rng.choice(["Short.", "Too short", "N/A"])

    return record


def build_batch(run_id, seed=42):
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    records = []

    valid = [
        valid_article(rng, run_id, number, now)
        for number in range(1, VALID_COUNT + 1)
    ]
    records += [("valid", record) for record in valid]

    for original in rng.sample(valid, EXACT_DUPLICATE_COUNT):
        duplicate = dict(original)
        duplicate["published_at"] = (
            datetime.fromisoformat(original["published_at"])
            + timedelta(hours=1)
        ).isoformat()
        records.append(("duplicate_article_id", duplicate))

    number = 1
    for issue, count in INVALID_PLAN.items():
        for _ in range(count):
            records.append(
                (issue, make_invalid(rng, run_id, number, now, issue))
            )
            number += 1

    rng.shuffle(records)
    return records


def print_plan(records):
    kinds = Counter(kind for kind, _ in records)
    print(f"Batch size: {len(records)}")
    print(f"  valid:                {kinds['valid']}")
    print(f"  exact duplicates:     {kinds['duplicate_article_id']}")
    print(f"  invalid:              {sum(INVALID_PLAN.values())}")
    for issue in INVALID_PLAN:
        print(f"    {issue:<22}{kinds[issue]}")
    print(
        "Expected DQ result: "
        f"{kinds['valid']} valid, {sum(INVALID_PLAN.values())} invalid, "
        f"{kinds['duplicate_article_id']} exact duplicates, "
        f"{sum(INVALID_PLAN.values()) + kinds['duplicate_article_id']} quarantined"
    )


def dry_run(records):
    """Run the real data-quality code on the batch in local Spark."""
    from src.data_quality import (
        print_quality_summary,
        split_valid_and_quarantine,
        validate_and_clean_articles,
    )
    from src.pipeline import articles_to_dataframe, create_spark

    spark = create_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = articles_to_dataframe(spark, [record for _, record in records])
        cleaned = validate_and_clean_articles(df)
        print_quality_summary(cleaned)
        valid_df, quarantine_df = split_valid_and_quarantine(cleaned)
        print(f"DRY RUN valid rows:      {valid_df.count()}")
        print(f"DRY RUN quarantine rows: {quarantine_df.count()}")
    finally:
        spark.stop()


def publish(records):
    kinesis = boto3.client(
        "kinesis",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    for start in range(0, len(records), 500):
        chunk = records[start:start + 500]
        response = kinesis.put_records(
            StreamName=KINESIS_STREAM,
            Records=[
                {
                    "Data": json.dumps(record).encode("utf-8"),
                    "PartitionKey": record.get("article_id") or str(uuid.uuid4()),
                }
                for _, record in chunk
            ],
        )
        if response.get("FailedRecordCount"):
            raise RuntimeError(
                f"{response['FailedRecordCount']} records failed to publish "
                f"in chunk starting at {start}"
            )
        print(f"Published {start + len(chunk)} / {len(records)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    records = build_batch(run_id)

    print("=" * 70)
    print(f"MIXED DQ BATCH (run {run_id})")
    print("=" * 70)
    print_plan(records)
    print("=" * 70)

    if args.dry_run:
        dry_run(records)
    else:
        publish(records)


if __name__ == "__main__":
    main()
