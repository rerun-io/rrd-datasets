"""
HF Storage Bucket access for Modal jobs: a boto3 client on the S3-compatible gateway.

Namespaces are arguments rather than constants, so this file carries nothing private or specific
to one dataset. The HFAK key pair rides in `RCLONE_CONFIG_HF_*` variables, so the same exports
also configure an `rclone` remote for syncing the bucket locally.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

# The HFAK key pair: generated once from a fine-grained HF token scoped to the bucket.
HF_S3_KEYS = ("RCLONE_CONFIG_HF_ACCESS_KEY_ID", "RCLONE_CONFIG_HF_SECRET_ACCESS_KEY")

# The gateway accepts only us-east-1 signatures and is served from us-east-1, so workers pin
# there too.
HF_GATEWAY_REGION = "us-east-1"

# The gateway is unreliable assembling multipart uploads from many parts (`CompleteMultipartUpload`
# returns `InternalError`). A 2 GiB cutoff ships whole RRDs as one PUT; anything larger goes up in
# 2 GiB parts, one at a time, so a worker buffers at most one part.
GIB = 1024**3
HF_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=2 * GIB,
    multipart_chunksize=2 * GIB,
    max_concurrency=1,
)


def missing_hf_s3_keys() -> list[str]:
    """The HFAK variables not set in the caller's environment."""
    return [key for key in HF_S3_KEYS if not os.environ.get(key)]


def gateway_client(namespace: str) -> S3Client:
    """
    An S3 client on the HF bucket gateway for `namespace`, authenticated with the HFAK key pair.

    The gateway serves one endpoint per namespace and speaks plain S3, with two deviations: no
    virtual-host bucket names (path-style addressing only), and no `aws-chunked` uploads
    (checksum trailers only where a call requires them).
    """
    return boto3.client(
        "s3",
        endpoint_url=f"https://s3.hf.co/{namespace}",
        region_name=HF_GATEWAY_REGION,
        aws_access_key_id=os.environ["RCLONE_CONFIG_HF_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RCLONE_CONFIG_HF_SECRET_ACCESS_KEY"],
        config=Config(
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def check_bucket(namespace: str, bucket: str) -> None:
    """
    Stop the launch unless `bucket` exists in `namespace`.

    A worker uploads only after building its layers, so a missing bucket would burn a container's
    build time per episode before failing. One listing on the launcher catches it first.
    """
    try:
        existing = [entry["Name"] for entry in gateway_client(namespace).list_buckets()["Buckets"]]
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Cannot list buckets in {namespace}: {exc}") from exc

    if bucket in existing:
        return

    known = ", ".join(existing) or "none"
    if bucket == namespace:
        raise SystemExit(
            f"HF_BUCKET={bucket!r} repeats the namespace: the gateway endpoint already carries it, "
            f"so HF_BUCKET names the bucket bare. Buckets in {namespace}: {known}."
        )
    raise SystemExit(
        f"No bucket {namespace}/{bucket}. Buckets in {namespace}: {known}. "
        f"Create it with `hf buckets create {namespace}/{bucket} --private`."
    )
