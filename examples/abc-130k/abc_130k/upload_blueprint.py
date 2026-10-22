"""
Upload the ABC-130k blueprint (`.rbl`) to the storage bucket.

`pixi run -e abc blueprint` writes the default `blueprints/abc-130k/default.rbl`; this
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


def main() -> None:
    """Upload the default blueprint (or a given `.rbl`) to the bucket."""
    parser = argparse.ArgumentParser(
        description="Upload the ABC-130k blueprint (.rbl) to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("blueprint", nargs="?", type=Path, default=BLUEPRINT_PATH, help="Path to the .rbl to upload.")
    parser.add_argument("--uri", default=DEFAULT_BLUEPRINT_URI, help="Destination URI for the blueprint (s3:// shape).")
    args = parser.parse_args()

    from rrd_datasets_common.bucket import upload_files

    upload_files({args.blueprint: args.uri}, missing_hint="run `pixi run -e abc blueprint` first")


if __name__ == "__main__":
    main()
