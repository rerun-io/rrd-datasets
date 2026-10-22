"""
Upload local files, such as default blueprints and model assets, to the dataset's bucket.

Needs the `cloud` extra: the storage clients and bucket checks are in `modal_jobs.store`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from botocore.exceptions import NoCredentialsError

from rrd_datasets_common.modal_jobs.hf_bucket import missing_hf_s3_keys
from rrd_datasets_common.modal_jobs.store import check_bucket, launcher_client, upload_file
from rrd_datasets_common.storage import STORAGE_BACKEND


def upload_files(uploads: Mapping[Path, str], missing_hint: str = "") -> None:
    """Upload each local file in `uploads` to its `s3://` URI on the active storage backend."""
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
