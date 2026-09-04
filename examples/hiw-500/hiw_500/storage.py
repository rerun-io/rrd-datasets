"""
The HIW-500 bucket layout.

Everything the dataset owns sits under one prefix, split by kind:

    <prefix>base/<recording_id>.rrd           # base layer
    <prefix><layer>/<recording_id>.rrd        # and the other derived layers
    <prefix>assets/urdf-model.rrd             # the shared robot model, once for the dataset
    <prefix>blueprints/default.rbl            # the default viewer layout

The backend and bucket come from `rrd_datasets_common.storage` (`STORAGE_BACKEND`,
`S3_BUCKET` / `HF_BUCKET`, …); only the layout under the prefix is this dataset's own.
The `hiw` pixi environments default the backend to `hf` and the bucket to `hiw-500`.
"""

from __future__ import annotations

from rrd_datasets_common.storage import dataset_prefix

DATASET_PREFIX = dataset_prefix("hiw-500")
BLUEPRINT_URI = f"{DATASET_PREFIX}blueprints/default.rbl"
ASSET_PREFIX = f"{DATASET_PREFIX}assets/"
