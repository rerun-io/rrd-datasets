"""
Upload the LIBERO blueprint (`.rbl`) to the storage bucket.

Run:  pixi run -e libero blueprint         # generate the default blueprint
      pixi run -e libero upload-blueprint  # upload it
"""

from __future__ import annotations

import argparse
from pathlib import Path

from libero.blueprint import BLUEPRINT_PATH
from libero.storage import BLUEPRINT_URI


def upload_blueprint(blueprint: Path, uri: str) -> None:
    """Upload the local `.rbl` at `blueprint` to `uri` on the active storage backend."""
    from rrd_datasets_common.modal_jobs.hf_bucket import missing_hf_s3_keys
    from rrd_datasets_common.modal_jobs.store import check_bucket, launcher_client, upload_file
    from rrd_datasets_common.storage import STORAGE_BACKEND

    if blueprint.suffix != ".rbl":
        raise ValueError(f"Expected an .rbl blueprint file, got {blueprint}")
    if not blueprint.exists():
        raise FileNotFoundError(f"{blueprint} not found — run `pixi run -e libero blueprint` first.")
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri}")

    if STORAGE_BACKEND == "hf" and (missing := missing_hf_s3_keys()):
        raise SystemExit(f"Missing HF S3 credentials: {', '.join(missing)} — generate them from an HF access token.")
    check_bucket()
    upload_file(launcher_client(), str(blueprint), uri)
    print(f"Uploaded blueprint: {uri}")


def main() -> None:
    """Upload the default blueprint (or a given `.rbl`) to the bucket."""
    parser = argparse.ArgumentParser(
        description="Upload the LIBERO blueprint (.rbl) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("blueprint", nargs="?", type=Path, default=BLUEPRINT_PATH, help="Path to the .rbl to upload.")
    parser.add_argument("--uri", default=BLUEPRINT_URI, help="Destination URI for the blueprint (s3:// shape).")
    args = parser.parse_args()

    upload_blueprint(args.blueprint, args.uri)


if __name__ == "__main__":
    main()
