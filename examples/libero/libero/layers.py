"""
The layers every demo is converted into, in build order.

Each layer writes `<layer>/<recording_id>.rrd` (`rrd_datasets_common.paths.layer_relpath`) under
the local rrd root and the bucket prefix alike, so a layer directory syncs between the two without
renaming and `register` reads either side as is.
"""

from __future__ import annotations

LAYERS = ("base", "properties", "urdf", "cameras")
