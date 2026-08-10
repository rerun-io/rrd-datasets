"""
AWS access for Modal jobs: credentials from Modal's OIDC identity.

Role ARNs are arguments rather than constants, so this file carries nothing private or specific
to one dataset.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


def assume_aws_role(role_arn: str, region: str, session_name: str = "ModalOIDCSession") -> boto3.Session:
    """Assume an AWS role using Modal's OIDC token. Returns an auto-refreshing boto3 `Session`."""
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session

    def refresh() -> dict[str, str]:
        response = boto3.client("sts", region_name=region).assume_role_with_web_identity(
            RoleArn=role_arn,
            RoleSessionName=session_name,
            WebIdentityToken=os.environ["MODAL_IDENTITY_TOKEN"],
        )
        creds = response["Credentials"]
        return {
            "access_key": creds["AccessKeyId"],
            "secret_key": creds["SecretAccessKey"],
            "token": creds["SessionToken"],
            "expiry_time": creds["Expiration"].isoformat(),
        }

    botocore_session = get_session()
    botocore_session._credentials = RefreshableCredentials.create_from_metadata(  # type: ignore[attr-defined]
        metadata=refresh(),
        refresh_using=refresh,
        method="sts-assume-role-with-web-identity",
    )
    return boto3.Session(botocore_session=botocore_session, region_name=region)


def local_s3_client(region: str) -> S3Client:
    """An S3 client on the caller's own credentials, for use outside a Modal worker."""
    return boto3.client("s3", region_name=region)
