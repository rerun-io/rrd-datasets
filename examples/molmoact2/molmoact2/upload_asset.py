"""
Upload the two MolmoAct2 arm model rrds to the storage bucket.

The arm models are written locally and uploaded to the storage bucket.

Run:  pixi run -e molmo convert-urdf   # write both arm model rrds
      pixi run -e molmo upload-asset   # upload them
"""

from __future__ import annotations

import argparse

from molmoact2.base_layer import RRD_ROOT
from molmoact2.storage import ASSET_PREFIX
from molmoact2.urdf_layer import ARMS, model_rrd_path
from rrd_datasets_common.bucket import upload_files


def asset_uri(arm: str) -> str:
    """The bucket URI of one arm's model, matching its local path under `assets/`."""
    return f"{ASSET_PREFIX}{model_rrd_path(RRD_ROOT, arm).name}"


def main() -> None:
    """Upload both arm model rrds, or the arms named on the command line."""
    parser = argparse.ArgumentParser(
        description="Upload the MolmoAct2 arm model rrds to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("arms", nargs="*", choices=ARMS, help="The arms to upload (default: both).")
    args = parser.parse_args()

    uploads = {model_rrd_path(RRD_ROOT, arm): asset_uri(arm) for arm in args.arms or ARMS}
    upload_files(uploads, missing_hint="run `pixi run -e molmo convert-urdf` first")


if __name__ == "__main__":
    main()
