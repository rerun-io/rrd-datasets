"""
The layers every episode is converted into, in build order.

Each layer writes `<layer>/<recording_id>.rrd` under the rrd root, so `register` finds every layer
of an episode by its recording id.
"""

from __future__ import annotations

LAYERS = ("base", "urdf", "cameras")
