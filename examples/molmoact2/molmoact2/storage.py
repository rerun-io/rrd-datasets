"""
The MolmoAct2 bucket layout.

Everything the dataset owns sits under one prefix, split by kind:

    <prefix><layer>/episode_NNNNN.rrd             # one rrd per layer per episode
    <prefix>assets/urdf-model-{left,right}.rrd    # one arm model per arm, once for the dataset
    <prefix>blueprints/default.rbl                # the default viewer layout

The backend and bucket come from `rrd_datasets_common.storage`. This module only owns the layout
under the prefix. The `molmo` pixi environments default to the `hf` backend and the `molmoact2` bucket.
"""

from __future__ import annotations

from rrd_datasets_common.storage import dataset_prefix

DATASET_PREFIX = dataset_prefix("molmoact2")
BLUEPRINT_URI = f"{DATASET_PREFIX}blueprints/default.rbl"
ASSET_PREFIX = f"{DATASET_PREFIX}assets/"
