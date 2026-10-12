import json

import boto3

try:
    from .config import (
        AWS_ENDPOINT_URL,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_REGION,
        CHECKPOINT_PATH,
    )
except ImportError:
    from config import (
        AWS_ENDPOINT_URL,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_REGION,
        CHECKPOINT_PATH,
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


def checkpoint_key():

    return (
        CHECKPOINT_PATH
        .replace(
            "s3://",
            "",
            1,
        )
        + "kinesis_checkpoint.json"
    )


def load_checkpoints():

    client = create_s3_client()

    bucket, prefix = parse_s3_path(
        CHECKPOINT_PATH
    )

    key = prefix.rstrip(
        "/"
    ) + "/kinesis_checkpoint.json"

    try:

        response = client.get_object(
            Bucket=bucket,
            Key=key,
        )

        body = response[
            "Body"
        ].read().decode(
            "utf-8"
        )

        checkpoints = json.loads(
            body
        )

        print(
            f"Loaded checkpoints: "
            f"{checkpoints}"
        )

        return checkpoints

    except client.exceptions.NoSuchKey:

        print(
            "No checkpoint file found. "
            "Starting fresh."
        )

        return {}

    except Exception as exc:

        error_message = str(exc)

        if (
            "NoSuchKey" in error_message
            or "404" in error_message
        ):

            print(
                "No checkpoint file found. "
                "Starting fresh."
            )

            return {}

        raise


def save_checkpoints(
    checkpoints,
):

    client = create_s3_client()

    bucket, prefix = parse_s3_path(
        CHECKPOINT_PATH
    )

    key = prefix.rstrip(
        "/"
    ) + "/kinesis_checkpoint.json"

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(
            checkpoints,
            indent=2,
        ).encode(
            "utf-8"
        ),
        ContentType="application/json",
    )

    print(
        f"Checkpoints saved to: "
        f"s3://{bucket}/{key}"
    )