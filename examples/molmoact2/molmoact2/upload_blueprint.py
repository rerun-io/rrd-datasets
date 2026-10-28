"""
Upload the MolmoAct2 blueprint (`.rbl`) to the storage bucket.

Run:  pixi run -e molmo blueprint         # generate the default blueprint
      pixi run -e molmo upload-blueprint  # upload it
"""

from __future__ import annotations

import argparse
from pathlib import Path

from molmoact2.blueprint import BLUEPRINT_PATH
from molmoact2.storage import BLUEPRINT_URI
from rrd_datasets_common.bucket import upload_files


def main() -> None:
    """Upload the default blueprint (or a given `.rbl`) to the bucket."""
    parser = argparse.ArgumentParser(
        description="Upload the MolmoAct2 blueprint (.rbl) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("blueprint", nargs="?", type=Path, default=BLUEPRINT_PATH, help="Path to the .rbl to upload.")
    parser.add_argument("--uri", default=BLUEPRINT_URI, help="Destination URI for the blueprint (s3:// shape).")
    args = parser.parse_args()

    upload_files({args.blueprint: args.uri}, missing_hint="run `pixi run -e molmo blueprint` first")


if __name__ == "__main__":
    main()
