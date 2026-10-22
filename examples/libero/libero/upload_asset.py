"""
Upload the shared LIBERO model rrd to the storage bucket.

The Modal workers build the per-demo layers only; the shared model is written locally and uploaded
from here, so a bucket carries the same `assets/` file the local catalog registers.

Run:  pixi run -e libero convert-urdf   # write the shared model rrd
      pixi run -e libero upload-asset   # upload it
"""

from __future__ import annotations

import argparse
from pathlib import Path

from libero.base_layer import RRD_ROOT
from libero.storage import ASSET_PREFIX
from libero.urdf_layer import model_rrd_path

MODEL_PATH = model_rrd_path(RRD_ROOT)
MODEL_ASSET_URI = f"{ASSET_PREFIX}{MODEL_PATH.name}"


def main() -> None:
    """Upload the shared model rrd (or a given `.rrd`) to the dataset's asset prefix."""
    parser = argparse.ArgumentParser(
        description="Upload the shared LIBERO model rrd (.rrd) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("asset", nargs="?", type=Path, default=MODEL_PATH, help="Path to the .rrd asset to upload.")
    parser.add_argument("--uri", default=MODEL_ASSET_URI, help="Destination URI for the asset (s3:// shape).")
    args = parser.parse_args()

    from rrd_datasets_common.bucket import upload_files

    upload_files({args.asset: args.uri}, missing_hint="run `pixi run -e libero convert-urdf` first")


if __name__ == "__main__":
    main()
