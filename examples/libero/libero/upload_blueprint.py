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


def main() -> None:
    """Upload the default blueprint (or a given `.rbl`) to the bucket."""
    parser = argparse.ArgumentParser(
        description="Upload the LIBERO blueprint (.rbl) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("blueprint", nargs="?", type=Path, default=BLUEPRINT_PATH, help="Path to the .rbl to upload.")
    parser.add_argument("--uri", default=BLUEPRINT_URI, help="Destination URI for the blueprint (s3:// shape).")
    args = parser.parse_args()

    from rrd_datasets_common.bucket import upload_files

    upload_files({args.blueprint: args.uri}, missing_hint="run `pixi run -e libero blueprint` first")


if __name__ == "__main__":
    main()
