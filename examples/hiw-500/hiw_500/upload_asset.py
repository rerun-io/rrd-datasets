"""
Upload the shared HIW-500 model rrd to the storage bucket.

The Modal workers build the per-episode layers only; the shared model is written locally and
uploaded from here, so a bucket carries the same `assets/` file the local catalog registers.

Run:  pixi run -e hiw convert-urdf   # write the shared model rrd
      pixi run -e hiw upload-asset   # upload it
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hiw_500.base_layer import RRD_ROOT
from hiw_500.storage import ASSET_PREFIX
from hiw_500.urdf_layer import model_rrd_path

MODEL_PATH = model_rrd_path(RRD_ROOT)
MODEL_ASSET_URI = f"{ASSET_PREFIX}{MODEL_PATH.name}"


def upload_asset(asset: Path, uri: str) -> None:
    """Upload the local `.rrd` at `asset` to `uri` on the active storage backend."""
    from rrd_datasets_common.modal_jobs.hf_bucket import missing_hf_s3_keys
    from rrd_datasets_common.modal_jobs.store import check_bucket, launcher_client, upload_file
    from rrd_datasets_common.storage import STORAGE_BACKEND

    if asset.suffix != ".rrd":
        raise ValueError(f"Expected an .rrd recording, got {asset}")
    if not asset.exists():
        raise FileNotFoundError(f"{asset} not found — run `pixi run -e hiw convert-urdf` first.")
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri}")

    if STORAGE_BACKEND == "hf" and (missing := missing_hf_s3_keys()):
        raise SystemExit(f"Missing HF S3 credentials: {', '.join(missing)} — generate them from an HF access token.")
    check_bucket()
    upload_file(launcher_client(), str(asset), uri)
    print(f"Uploaded asset: {uri}")


def main() -> None:
    """Upload the shared model rrd (or a given `.rrd`) to the dataset's asset prefix."""
    parser = argparse.ArgumentParser(
        description="Upload the shared HIW-500 model rrd (.rrd) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("asset", nargs="?", type=Path, default=MODEL_PATH, help="Path to the .rrd asset to upload.")
    parser.add_argument("--uri", default=MODEL_ASSET_URI, help="Destination URI for the asset (s3:// shape).")
    args = parser.parse_args()

    upload_asset(args.asset, args.uri)


if __name__ == "__main__":
    main()
