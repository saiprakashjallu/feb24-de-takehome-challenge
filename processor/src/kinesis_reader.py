import base64
import json
import time
from datetime import datetime, timezone

import boto3

try:
    from .config import (
        AWS_ENDPOINT_URL,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_REGION,
        KINESIS_STREAM,
        MALFORMED_PATH,
    )
except ImportError:
    from config import (
        AWS_ENDPOINT_URL,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_REGION,
        KINESIS_STREAM,
        MALFORMED_PATH,
    )


def create_kinesis_client():
    return boto3.client(
        "kinesis",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def get_shards(client):
    response = client.describe_stream(
        StreamName=KINESIS_STREAM
    )

    return [
        shard["ShardId"]
        for shard in response["StreamDescription"]["Shards"]
    ]


def get_starting_iterator(
    client,
    shard_id,
    checkpoint=None,
):
    """
    Determine where to start reading a shard.

    If a checkpoint exists, start AFTER the
    previously processed sequence number.

    Otherwise start from TRIM_HORIZON.
    """

    if checkpoint:

        print(
            f"Checkpoint found for {shard_id}: "
            f"{checkpoint}"
        )

        response = client.get_shard_iterator(
            StreamName=KINESIS_STREAM,
            ShardId=shard_id,
            ShardIteratorType="AFTER_SEQUENCE_NUMBER",
            StartingSequenceNumber=checkpoint,
        )

    else:

        print(
            f"No checkpoint found for {shard_id}. "
            f"Starting from TRIM_HORIZON."
        )

        response = client.get_shard_iterator(
            StreamName=KINESIS_STREAM,
            ShardId=shard_id,
            ShardIteratorType="TRIM_HORIZON",
        )

    return response["ShardIterator"]


def read_records(
    client,
    shard_id,
    checkpoint=None,
):
    """
    Read records from a Kinesis shard.

    Returns:
        records,
        latest_sequence_number,
        malformed_records
    """

    iterator = get_starting_iterator(
        client,
        shard_id,
        checkpoint,
    )

    records = []

    malformed_records = []

    latest_sequence_number = checkpoint

    empty_polls = 0

    while empty_polls < 5:

        response = client.get_records(
            ShardIterator=iterator,
            Limit=100,
        )

        kinesis_records = response.get(
            "Records",
            [],
        )

        if not kinesis_records:

            empty_polls += 1

            time.sleep(1)

        else:

            empty_polls = 0

            for record in kinesis_records:

                try:

                    article = json.loads(
                        record["Data"].decode(
                            "utf-8"
                        )
                    )

                    if not isinstance(article, dict):
                        raise ValueError(
                            "record is valid JSON but not "
                            f"an object ({type(article).__name__})"
                        )

                    source_timestamp = record.get(
                        "ApproximateArrivalTimestamp"
                    )

                    if source_timestamp:

                        source_timestamp = (
                            source_timestamp
                            .astimezone(timezone.utc)
                            .isoformat()
                        )

                    ingest_timestamp = (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    )

                    sequence_number = str(
                        record["SequenceNumber"]
                    )

                    enriched_record = {
                        **article,

                        "_ingest_timestamp":
                            ingest_timestamp,

                        "_source_shard_id":
                            shard_id,

                        "_source_sequence_number":
                            sequence_number,

                        "_source_timestamp":
                            source_timestamp,

                        "_record_key":
                            record.get(
                                "PartitionKey"
                            ),

                        "_pipeline_version":
                            "1.0.0",
                    }

                    records.append(
                        enriched_record
                    )

                    latest_sequence_number = (
                        sequence_number
                    )

                except Exception as exc:

                    # Keep the record in the dead-letter area
                    # instead of dropping it, and move the
                    # checkpoint past it so it is not re-read
                    # forever.

                    malformed = malformed_record(
                        record,
                        shard_id,
                        exc,
                    )

                    malformed_records.append(
                        malformed
                    )

                    latest_sequence_number = (
                        malformed["sequence_number"]
                    )

                    print(
                        "Malformed Kinesis record sent to "
                        f"dead-letter: {shard_id} "
                        f"{malformed['sequence_number']} "
                        f"({malformed['error']})"
                    )

        iterator = response[
            "NextShardIterator"
        ]

    return (
        records,
        latest_sequence_number,
        malformed_records,
    )


def malformed_record(record, shard_id, exc):
    """
    Describe a record that could not be parsed as a JSON object.
    The raw bytes are kept (base64) so nothing is lost.
    """

    data = record.get("Data", b"")

    arrival = record.get("ApproximateArrivalTimestamp")

    return {
        "shard_id": shard_id,
        "sequence_number": str(record.get("SequenceNumber")),
        "partition_key": record.get("PartitionKey"),
        "arrival_timestamp": (
            arrival.astimezone(timezone.utc).isoformat()
            if arrival
            else None
        ),
        "error": f"{type(exc).__name__}: {exc}",
        "data_preview": data[:500].decode("utf-8", errors="replace"),
        "data_base64": base64.b64encode(data).decode("ascii"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def save_malformed_records(malformed_records):
    """
    Write each malformed record to
    MALFORMED_PATH/shard=<id>/seq=<sequence>.json.

    Keys are derived from the Kinesis sequence number, so a
    replayed batch overwrites the same objects instead of
    duplicating them.
    """

    if not malformed_records:
        return 0

    bucket, _, prefix = (
        MALFORMED_PATH
        .replace("s3://", "", 1)
        .partition("/")
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    for malformed in malformed_records:

        key = (
            f"{prefix.rstrip('/')}/shard={malformed['shard_id']}/"
            f"seq={malformed['sequence_number']}.json"
        )

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(malformed, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    print(
        f"{len(malformed_records)} malformed record(s) written to: "
        f"{MALFORMED_PATH}"
    )

    return len(malformed_records)


def read_all_articles(
    checkpoints=None,
):
    """
    Read all Kinesis shards incrementally.

    Args:
        checkpoints:
            Dictionary mapping shard IDs to
            previously processed sequence numbers.

    Returns:
        articles,
        updated_checkpoints,
        malformed_records
    """

    if checkpoints is None:
        checkpoints = {}

    client = create_kinesis_client()

    shards = get_shards(client)

    articles = []

    malformed_records = []

    updated_checkpoints = {}

    for shard_id in shards:

        checkpoint = checkpoints.get(
            shard_id
        )

        (
            shard_records,
            latest_sequence,
            shard_malformed,
        ) = read_records(
            client,
            shard_id,
            checkpoint,
        )

        articles.extend(
            shard_records
        )

        malformed_records.extend(
            shard_malformed
        )

        if latest_sequence:
            updated_checkpoints[
                shard_id
            ] = latest_sequence

    return (
        articles,
        updated_checkpoints,
        malformed_records,
    )