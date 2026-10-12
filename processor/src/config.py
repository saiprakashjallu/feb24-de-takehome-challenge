import os


AWS_ENDPOINT_URL = os.getenv(
    "AWS_ENDPOINT_URL",
    "http://localhost:4566",
)

AWS_ACCESS_KEY_ID = os.getenv(
    "AWS_ACCESS_KEY_ID",
    "test",
)

AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY",
    "test",
)

AWS_REGION = os.getenv(
    "AWS_DEFAULT_REGION",
    "us-east-1",
)

KINESIS_STREAM = os.getenv(
    "KINESIS_STREAM_NAME",
    "MyStream",
)

S3_BUCKET = os.getenv(
    "S3_BUCKET_NAME",
    "my-bucket",
)

BRONZE_PATH = os.getenv(
    "BRONZE_PATH",
    "s3://my-bucket/bronze/articles/",
)

ML_READY_PATH = os.getenv(
    "ML_READY_PATH",
    "s3://my-bucket/ml-ready/",
)

MONITORING_PATH = os.getenv(
    "MONITORING_PATH",
    "s3://my-bucket/monitoring/",
)

ML_TRAINING_PATH = os.getenv(
    "ML_TRAINING_PATH",
    "s3://my-bucket/ml/training/articles/",
)

MALFORMED_PATH = os.getenv(
    "MALFORMED_PATH",
    "s3://my-bucket/quarantine/malformed/",
)

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    "s3://my-bucket/checkpoints/kinesis/",
)