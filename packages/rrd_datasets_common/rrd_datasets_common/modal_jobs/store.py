"""
The worker side of the storage backend: worker credentials, run secrets, region, and image config.

Workers call these functions without checking which backend is active. On `s3` a worker assumes an
AWS role through Modal's OIDC identity. On `hf` it uses the same gateway client as the local machine.
Bucket access shared with local tasks is in `rrd_datasets_common.bucket`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rrd_datasets_common.hf_bucket import HF_GATEWAY_REGION, gateway_client
from rrd_datasets_common.modal_jobs.aws import assume_aws_role
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


def extra_secrets() -> list[modal.Secret]:
    """Backend secrets to bind to the run, beyond the HuggingFace token."""
    return [hf_s3_secret()] if STORAGE_BACKEND == "hf" else []


def region_pin() -> str:
    """Where workers must run: beside the S3 bucket, or beside the gateway."""
    return S3_REGION if STORAGE_BACKEND == "s3" else HF_GATEWAY_REGION


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
