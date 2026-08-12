"""
Upload the ABC-130k blueprint (`.rbl`) to the storage bucket.

`pixi run -e abc blueprint` writes the committed default `blueprints/abc-130k/default.rbl`; this
uploads it to the bucket, under the dataset's `blueprints/` prefix.

Run:  pixi run -e abc blueprint         # (re)generate the default blueprint
      pixi run -e abc upload-blueprint  # upload it
"""

from __future__ import annotations

import argparse
from pathlib import Path

from abc_130k.blueprint import BLUEPRINT_PATH
from abc_130k.storage import BLUEPRINT_PREFIX

DEFAULT_BLUEPRINT_URI = f"{BLUEPRINT_PREFIX}{BLUEPRINT_PATH.name}"


def upload_blueprint(blueprint: Path, s3_uri: str) -> None:
    """Upload the local `.rbl` at `blueprint` to `s3_uri` on the active storage backend."""
    from botocore.exceptions import NoCredentialsError

    from rrd_datasets_common.modal_jobs.store import launcher_client, upload_file

    if blueprint.suffix != ".rbl":
        raise ValueError(f"Expected an .rbl blueprint file, got {blueprint}")
    if not blueprint.exists():
        raise FileNotFoundError(f"{blueprint} not found — run `pixi run -e abc blueprint` first.")
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {s3_uri}")

    try:
        upload_file(launcher_client(), str(blueprint), s3_uri)
    except NoCredentialsError:
        raise SystemExit(
            f"No credentials found for the upload to {s3_uri}. Assume a profile/role with write "
            f"access to the bucket, then retry."
        ) from None
    print(f"Uploaded blueprint: {s3_uri}")


def main() -> None:
    """Upload the default blueprint (or a given `.rbl`) to the bucket."""
    parser = argparse.ArgumentParser(
        description="Upload the ABC-130k blueprint (.rbl) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("blueprint", nargs="?", type=Path, default=BLUEPRINT_PATH, help="Path to the .rbl to upload.")
    parser.add_argument("--s3-uri", default=DEFAULT_BLUEPRINT_URI, help="Destination S3 URI for the blueprint.")
    args = parser.parse_args()

    upload_blueprint(args.blueprint, args.s3_uri)


if __name__ == "__main__":
    main()
