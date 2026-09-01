"""
The LIBERO bucket layout.

Everything the dataset owns sits under one prefix, split by kind:

    <prefix><layer>/<suite>/<task>__<demo>.rrd    # one rrd per layer per demo
    <prefix>blueprints/default.rbl                # the default viewer layout

The backend and bucket come from `rrd_datasets_common.storage`. This module only owns the layout
under the prefix. The `libero` pixi environments default to the `hf` backend and the `libero` bucket.
"""

from __future__ import annotations

from rrd_datasets_common.storage import dataset_prefix

DATASET_PREFIX = dataset_prefix("libero")
BLUEPRINT_URI = f"{DATASET_PREFIX}blueprints/default.rbl"
