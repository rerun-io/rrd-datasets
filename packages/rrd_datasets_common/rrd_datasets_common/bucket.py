"""
Bucket access on the active storage backend: the local client, object checks, and uploads.

Callers do not check which backend is active. `s3` uses the caller's own AWS credentials, and `hf`
goes through the HF Storage Bucket's S3-compatible gateway with a static key pair.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from rrd_datasets_common.hf_bucket import HF_TRANSFER_CONFIG, gateway_client, missing_hf_s3_keys
from rrd_datasets_common.hf_bucket import check_bucket as check_hf_bucket
from rrd_datasets_common.storage import HF_BUCKET, HF_NAMESPACE, S3_REGION, STORAGE_BACKEND

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


def launcher_client() -> S3Client:
    """The client for the local machine, used by the Modal launchers and the upload tasks."""
    if STORAGE_BACKEND == "hf":
        return gateway_client(HF_NAMESPACE)
    return boto3.client("s3", region_name=S3_REGION)


def check_bucket() -> None:
    """
    Stop when the target bucket does not exist, before any work starts.

    Only the hf backend checks: local AWS credentials are optional on s3, so a failed probe
    there proves nothing — a missing S3 bucket surfaces in the workers.
    """
    if STORAGE_BACKEND == "hf":
        check_hf_bucket(HF_NAMESPACE, HF_BUCKET)


def upload_file(s3: S3Client, path: str, uri: str) -> None:
    """Upload `path` to `uri` with the backend's transfer settings."""
    bucket, key = s3_parts(uri)
    s3.upload_file(path, bucket, key, Config=HF_TRANSFER_CONFIG if STORAGE_BACKEND == "hf" else None)


def upload_files(uploads: Mapping[Path, str], missing_hint: str = "") -> None:
    """Upload each local file in `uploads` to its `s3://` URI, after checking the files, credentials, and bucket."""
    if missing := [str(path) for path in uploads if not path.exists()]:
        hint = f" — {missing_hint}" if missing_hint else ""
        raise FileNotFoundError(f"{', '.join(missing)} not found{hint}.")
    if bad_uris := [uri for uri in uploads.values() if not uri.startswith("s3://")]:
        raise ValueError(f"Expected s3:// URIs, got {', '.join(bad_uris)}")

    if STORAGE_BACKEND == "hf" and (missing_keys := missing_hf_s3_keys()):
        raise SystemExit(
            f"Missing HF S3 credentials: {', '.join(missing_keys)} — generate them from an HF access token."
        )
    check_bucket()

    client = launcher_client()
    for path, uri in uploads.items():
        try:
            upload_file(client, str(path), uri)
        except NoCredentialsError:
            raise SystemExit(
                f"No credentials found for the upload to {uri}. Assume a profile/role with write "
                f"access to the bucket, then retry."
            ) from None
        print(f"Uploaded {path.name}: {uri}")


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
