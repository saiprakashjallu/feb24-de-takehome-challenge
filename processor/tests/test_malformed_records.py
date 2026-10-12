import base64
import os
import sys
from datetime import datetime, timezone

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.kinesis_reader import malformed_record


def test_malformed_record_keeps_raw_bytes_and_reason():
    raw = b"this is {not json"
    record = {
        "Data": raw,
        "SequenceNumber": "4967903",
        "PartitionKey": "pk-1",
        "ApproximateArrivalTimestamp": datetime(
            2026, 10, 12, 13, 0, tzinfo=timezone.utc
        ),
    }

    malformed = malformed_record(
        record, "shardId-000000000000", ValueError("bad json")
    )

    assert malformed["shard_id"] == "shardId-000000000000"
    assert malformed["sequence_number"] == "4967903"
    assert malformed["partition_key"] == "pk-1"
    assert malformed["error"] == "ValueError: bad json"
    assert malformed["data_preview"] == "this is {not json"
    assert base64.b64decode(malformed["data_base64"]) == raw
    assert malformed["arrival_timestamp"] == "2026-10-12T13:00:00+00:00"
