"""
Publish a small, labelled batch of test records to Kinesis to prove the
data-quality, quarantine, deduplication and dead-letter paths work end to end.

Every record uses source_name "DQ Test Source" and an article_id starting
with "dqtest-" (except the one that deliberately has no article_id), so the
rows are easy to find in every layer afterwards.

Run from processor/:
    .venv/bin/python publish_dq_test_records.py
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import boto3

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_ENDPOINT_URL,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    KINESIS_STREAM,
)


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
NOW = datetime.now(timezone.utc)

BODY_A = (
    "City officials approved a new transit plan on Tuesday that will add "
    "three bus rapid transit lines across the northern districts. The plan "
    "was debated for more than six hours before the council voted to fund "
    "the first phase. Supporters said the lines would cut average commute "
    "times by almost twenty minutes for residents who currently rely on two "
    "or more transfers. Critics argued that the budget did not include "
    "enough money for maintenance and that construction would disrupt local "
    "businesses for at least two years. The mayor said the city would "
    "publish a detailed construction schedule next month and hold public "
    "meetings in every affected neighborhood. Transit advocates welcomed the "
    "decision but warned that ridership targets depend on frequent service "
    "and reliable funding in future budgets."
)

# Same article with one word changed: a near-duplicate for MinHash.
BODY_A_NEAR_DUPLICATE = BODY_A.replace("Tuesday", "Wednesday")

BODY_B = (
    "A regional hospital network reported that emergency room wait times "
    "fell by fifteen percent after it introduced a new triage system in the "
    "spring. Nurses now use a shared dashboard to track patient arrivals and "
    "available beds in real time. Hospital leaders said the change also "
    "reduced the number of patients who left before being seen by a doctor. "
    "Staff unions said the improvement came with heavier workloads during "
    "peak hours and asked for additional hiring. The network plans to expand "
    "the system to three smaller clinics by the end of the year and will "
    "publish quarterly results so the public can follow its progress."
)


def article(article_id, body, **overrides):
    record = {
        "article_id": article_id,
        "source_id": "dqtest-source",
        "source_name": "DQ Test Source",
        "title": "DQ test article",
        "author": "Test Author",
        "body_text": body,
        "url": f"https://news.example.com/articles/{article_id}",
        "published_at": (NOW - timedelta(hours=2)).isoformat(),
        "language": "en",
    }
    record.update(overrides)
    return record


def test_records():
    valid_a = f"dqtest-{RUN_ID}-valid-a"
    valid_b = f"dqtest-{RUN_ID}-valid-b"

    return [
        (
            "VALID: normal article A",
            "kept: Silver, Gold, ML-ready",
            article(valid_a, BODY_A),
        ),
        (
            "VALID: normal article B",
            "kept: Silver, Gold, ML-ready",
            article(valid_b, BODY_B),
        ),
        (
            "SEMANTIC DUPLICATE of A (one word changed, new id, arrives later)",
            "Silver with is_duplicate=true (A stays canonical), NOT in Gold",
            article(f"dqtest-{RUN_ID}-near-dup-a", BODY_A_NEAR_DUPLICATE),
        ),
        (
            "EXACT DUPLICATE of B (same article_id, published later)",
            "quarantined: duplicate_article_id",
            article(
                valid_b,
                BODY_B,
                published_at=(NOW - timedelta(hours=1)).isoformat(),
            ),
        ),
        (
            "INVALID: article_id missing",
            "quarantined: missing_article_id",
            {
                key: value
                for key, value in article("unused", BODY_B).items()
                if key != "article_id"
            },
        ),
        (
            "INVALID: source_id empty",
            "quarantined: missing_source_id",
            article(f"dqtest-{RUN_ID}-no-source", BODY_B, source_id=""),
        ),
        (
            "INVALID: title missing",
            "quarantined: missing_title",
            article(f"dqtest-{RUN_ID}-no-title", BODY_B, title=None),
        ),
        (
            "INVALID: body_text empty",
            "quarantined: missing_body_text",
            article(f"dqtest-{RUN_ID}-empty-body", ""),
        ),
        (
            "INVALID: published_at not a date",
            "quarantined: invalid_published_at",
            article(
                f"dqtest-{RUN_ID}-bad-date", BODY_B, published_at="not-a-date"
            ),
        ),
        (
            "INVALID: body_text shorter than 10 characters",
            "quarantined: body_text_too_short",
            article(f"dqtest-{RUN_ID}-short-body", "Too short"),
        ),
        (
            "MALFORMED: payload is not JSON",
            "dead-letter: quarantine/malformed/, never in Bronze",
            f"dqtest-{RUN_ID} this is {{not valid json".encode("utf-8"),
        ),
        (
            "MALFORMED: payload is a JSON array, not an object",
            "dead-letter: quarantine/malformed/, never in Bronze",
            json.dumps(["dqtest", RUN_ID, "array"]).encode("utf-8"),
        ),
    ]


def main():

    kinesis = boto3.client(
        "kinesis",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    records = test_records()

    print("=" * 78)
    print(f"PUBLISHING {len(records)} DQ TEST RECORDS to {KINESIS_STREAM} (run {RUN_ID})")
    print("=" * 78)

    for number, (case, expected, payload) in enumerate(records, start=1):
        if isinstance(payload, bytes):
            data, article_id = payload, None
        else:
            data, article_id = json.dumps(payload).encode("utf-8"), payload.get("article_id")
        response = kinesis.put_record(
            StreamName=KINESIS_STREAM,
            Data=data,
            PartitionKey=article_id or str(uuid.uuid4()),
        )
        print(f"{number:>2}. {case}")
        print(f"    article_id={article_id!r}  seq=...{response['SequenceNumber'][-8:]}")
        print(f"    expected -> {expected}")

    print("=" * 78)
    print("Expected pipeline result for this batch:")
    print("  malformed 2 -> dead-letter (not counted as ingested)")
    print("  ingested 10 | invalid 6 | exact duplicates 1 | quarantined 7")
    print("  valid 3 -> Silver +3 | semantic duplicates 1 | Gold/ML-ready +2")
    print("  canonical of the A pair = valid-a (first to arrive)")
    print("=" * 78)


if __name__ == "__main__":
    main()
