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


def asset_uri(arm: str) -> str:
    """The bucket URI of one arm's model, matching its local path under `assets/`."""
    return f"{ASSET_PREFIX}{model_rrd_path(RRD_ROOT, arm).name}"


def upload_assets(arms: list[str]) -> None:
    """Upload each arm's local model rrd to the dataset's asset prefix on the active storage backend."""
    from rrd_datasets_common.modal_jobs.hf_bucket import missing_hf_s3_keys
    from rrd_datasets_common.modal_jobs.store import check_bucket, launcher_client, upload_file
    from rrd_datasets_common.storage import STORAGE_BACKEND

    models = {arm: model_rrd_path(RRD_ROOT, arm) for arm in arms}
    missing_models = [str(path) for path in models.values() if not path.exists()]
    if missing_models:
        raise FileNotFoundError(f"{', '.join(missing_models)} not found — run `pixi run -e molmo convert-urdf` first.")

    if STORAGE_BACKEND == "hf" and (missing := missing_hf_s3_keys()):
        raise SystemExit(f"Missing HF S3 credentials: {', '.join(missing)} — generate them from an HF access token.")
    check_bucket()
    client = launcher_client()
    for arm, path in models.items():
        upload_file(client, str(path), asset_uri(arm))
        print(f"Uploaded asset: {asset_uri(arm)}")


def main() -> None:
    """Upload both arm model rrds, or the arms named on the command line."""
    parser = argparse.ArgumentParser(
        description="Upload the MolmoAct2 arm model rrds to the storage bucket.",
        epilog="Prerequisites and examples: see the README.",
    )
    parser.add_argument("arms", nargs="*", choices=ARMS, help="The arms to upload (default: both).")
    args = parser.parse_args()

    upload_assets(args.arms or ARMS)


if __name__ == "__main__":
    main()
