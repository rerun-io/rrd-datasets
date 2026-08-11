"""
The ABC-130k bucket layout.

Everything the dataset owns sits under one prefix, split by kind:

    <prefix>base/<recording_id>.rrd     # base-layer recordings
    <prefix>blueprints/default.rbl      # the default viewer layout

The backend and bucket come from `rrd_datasets_common.storage` (`STORAGE_BACKEND`,
`S3_BUCKET` / `HF_BUCKET`, …); only the layout under the prefix is this dataset's own.
"""

from __future__ import annotations

from rrd_datasets_common.storage import dataset_prefix

DATASET_PREFIX = dataset_prefix("abc-130k")
RRD_PREFIX = f"{DATASET_PREFIX}base/"
BLUEPRINT_PREFIX = f"{DATASET_PREFIX}blueprints/"
