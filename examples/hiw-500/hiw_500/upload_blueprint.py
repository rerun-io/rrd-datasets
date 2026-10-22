"""
Upload the HIW-500 blueprint (`.rbl`) to the storage bucket.

This uploads `blueprints/hiw-500/default.rbl` to the dataset's HF bucket at `blueprints/default.rbl`.

Run:  pixi run -e hiw blueprint         # (re)generate the default blueprint
      pixi run -e hiw upload-blueprint  # upload it
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hiw_500.blueprint import BLUEPRINT_PATH
from hiw_500.storage import BLUEPRINT_URI


def main() -> None:
    """Upload the default blueprint (or a given `.rbl`) to the bucket."""
    parser = argparse.ArgumentParser(
        description="Upload the HIW-500 blueprint (.rbl) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("blueprint", nargs="?", type=Path, default=BLUEPRINT_PATH, help="Path to the .rbl to upload.")
    parser.add_argument("--uri", default=BLUEPRINT_URI, help="Destination URI for the blueprint (s3:// shape).")
    args = parser.parse_args()

    from rrd_datasets_common.bucket import upload_files

    upload_files({args.blueprint: args.uri}, missing_hint="run `pixi run -e hiw blueprint` first")


if __name__ == "__main__":
    main()
