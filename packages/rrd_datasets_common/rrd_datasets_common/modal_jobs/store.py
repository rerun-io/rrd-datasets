"""
The storage backend behind the Modal jobs: every `STORAGE_BACKEND` branch lives here.

The worker and launcher call these functions without checking which backend is active. `s3`
uploads to AWS S3 — an OIDC-assumed role on the worker, ambient credentials on the launcher.
`hf` uploads to an HF Storage Bucket through its S3-compatible gateway, one static-key client
on both sides.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from rrd_datasets_common.modal_jobs.aws import assume_aws_role, local_s3_client
from rrd_datasets_common.modal_jobs.hf_bucket import HF_GATEWAY_REGION, HF_TRANSFER_CONFIG, gateway_client
from rrd_datasets_common.modal_jobs.hf_bucket import check_bucket as check_hf_bucket
from rrd_datasets_common.modal_jobs.image import hf_s3_secret
from rrd_datasets_common.storage import HF_BUCKET, HF_NAMESPACE, S3_BUCKET, S3_REGION, STORAGE_BACKEND

if TYPE_CHECKING:
    import modal
    from mypy_boto3_s3.client import S3Client

# The role must trust Modal's OIDC issuer and allow read/write on the output prefix (s3 backend
# only). Set `AWS_ROLE_ARN`, or replace the placeholder.
AWS_ROLE_ARN = os.getenv("AWS_ROLE_ARN", "arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>")


def worker_client() -> S3Client:
    """The upload client for a Modal worker."""
    if STORAGE_BACKEND == "hf":
        return gateway_client(HF_NAMESPACE)
    return assume_aws_role(AWS_ROLE_ARN, S3_REGION).client("s3", region_name=S3_REGION)


def launcher_client() -> S3Client:
    """The listing client for the local launcher."""
    if STORAGE_BACKEND == "hf":
        return gateway_client(HF_NAMESPACE)
    return local_s3_client(S3_REGION)


def extra_secrets() -> list[modal.Secret]:
    """Backend secrets to bind to the run, beyond the HuggingFace token."""
    return [hf_s3_secret()] if STORAGE_BACKEND == "hf" else []


def region_pin() -> str:
    """Where workers must run: beside the S3 bucket, or beside the gateway."""
    return S3_REGION if STORAGE_BACKEND == "s3" else HF_GATEWAY_REGION


def check_bucket() -> None:
    """
    Stop the launch when the target bucket does not exist, before any worker is spawned.

    Only the hf backend checks: launcher AWS credentials are optional on s3, so a failed probe
    there proves nothing — a missing S3 bucket surfaces in the workers.
    """
    if STORAGE_BACKEND == "hf":
        check_hf_bucket(HF_NAMESPACE, HF_BUCKET)


def worker_env() -> dict[str, str]:
    """Backend config baked into the worker image, so a worker resolves the same backend and bucket."""
    return {
        "STORAGE_BACKEND": STORAGE_BACKEND,
        "S3_BUCKET": S3_BUCKET,
        "S3_REGION": S3_REGION,
        "AWS_ROLE_ARN": AWS_ROLE_ARN,
        "HF_NAMESPACE": HF_NAMESPACE,
        "HF_BUCKET": HF_BUCKET,
    }


def upload_file(s3: S3Client, path: str, uri: str) -> None:
    """Upload `path` to `uri` with the backend's transfer settings."""
    bucket, key = s3_parts(uri)
    s3.upload_file(path, bucket, key, Config=HF_TRANSFER_CONFIG if STORAGE_BACKEND == "hf" else None)


def s3_parts(uri: str) -> tuple[str, str]:
    """Split an `s3://bucket/key` URI into `(bucket, key)`."""
    without_scheme = uri.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def s3_exists(s3: S3Client, uri: str) -> bool:
    """Whether an object exists at `uri`. A denied head counts as absent, since it proves nothing."""
    bucket, key = s3_parts(uri)
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("403", "404", "NoSuchKey"):
            return False
        raise
    return True


def s3_existing_keys(s3: S3Client, uri_prefix: str) -> set[str]:
    """
    Every non-empty object key under `uri_prefix`, from one paginated listing.

    Pages hold 1000 keys, so this costs what the bucket already holds rather than what you ask about.
    Zero-byte objects are left out, so a half-finished upload does not read as done.
    """
    bucket, key_prefix = s3_parts(uri_prefix)
    keys: set[str] = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key_prefix):
        keys.update(obj["Key"] for obj in page.get("Contents", ()) if obj["Size"] > 0)
    return keys
