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
        MONITORING_PATH,
    )
except ImportError:
    from config import (
        AWS_ENDPOINT_URL,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_REGION,
        MONITORING_PATH,
    )


def create_s3_client():

    return boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


def parse_s3_path(path):

    path = path.replace(
        "s3://",
        "",
        1,
    )

    bucket, key = path.split(
        "/",
        1,
    )

    return bucket, key


def record_pipeline_metrics(
    metrics,
):
    """
    Store one monitoring record for each
    pipeline execution.
    """

    client = create_s3_client()

    bucket, prefix = parse_s3_path(
        MONITORING_PATH
    )

    run_timestamp = datetime.now(
        timezone.utc
    )

    date_partition = (
        run_timestamp.strftime("%Y-%m-%d")
    )

    file_timestamp = (
        run_timestamp.strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    key = (
        prefix.rstrip("/")
        + f"/pipeline_date={date_partition}"
        + f"/run_{file_timestamp}.json"
    )

    monitoring_record = {
        "recorded_at": run_timestamp.isoformat(),
        **metrics,
    }

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(
            monitoring_record,
            indent=2,
        ).encode("utf-8"),
        ContentType="application/json",
    )

    print(
        f"Monitoring metrics written to: "
        f"s3://{bucket}/{key}"
    )