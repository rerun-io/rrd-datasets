"""
Build every layer (base, derived archetypes, URDF, odom, cameras, IR, properties) for every episode in one command.

Each layer module owns its conversion and runs on its own (`convert-base` / `convert-derived-archetypes` /
`convert-urdf` / `convert-odom` / `convert-cameras` / `convert-ir` / `convert-properties`); this
module runs them in order. The URDF model is parsed once, written as the dataset's shared model
rrd, and reused across episodes. Episodes without a head stereo calibration skip the cameras
layer, without wrist IR streams the ir layer, and without `/wbc_lerobot` the derived archetypes
layer.

Run:  pixi run -e hiw convert            # all episodes under data/HIW-500/
      pixi run -e hiw convert <ep.mcap>  # a single episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.urdf import UrdfTree

from hiw_500 import (
    base_layer,
    camera_layer,
    derived_archetypes_layer,
    ir_layer,
    odom_layer,
    properties_layer,
    urdf_layer,
)
from hiw_500.base_layer import DATASET_ROOT, RRD_ROOT, Episode, discover_episodes
from hiw_500.layers import LAYERS


def convert_episode(ep: Episode, urdf: UrdfTree) -> list[str]:
    """Write one episode's layers in `LAYERS` order; returns those written (a layer skips when its input is absent)."""
    results = {
        "base": base_layer.convert_episode(ep, RRD_ROOT),
        "derived_archetypes": derived_archetypes_layer.convert_episode(ep, RRD_ROOT),
        "urdf": urdf_layer.convert_episode(urdf, ep, RRD_ROOT),
        "odom": odom_layer.convert_episode(ep, RRD_ROOT),
        "cameras": camera_layer.convert_episode(ep, RRD_ROOT),
        "ir": ir_layer.convert_episode(ep, RRD_ROOT),
        "properties": properties_layer.convert_episode(ep, RRD_ROOT),
    }
    return [layer for layer in LAYERS if results[layer] is not None]


def main() -> None:
    """Convert a single episode mcap (positional) or every episode under `DATASET_ROOT`, every layer."""
    parser = argparse.ArgumentParser(description="Convert HIW-500 episode MCAPs into Rerun RRDs (every layer).")
    parser.add_argument(
        "mcap", nargs="?", type=Path, help="A single episode mcap (default: every episode under data/HIW-500/)."
    )
    args = parser.parse_args()

    episodes = [base_layer.episode_from_mcap(args.mcap)] if args.mcap is not None else discover_episodes(DATASET_ROOT)
    if not episodes:
        print(f"No episodes found under {DATASET_ROOT}")
        print("-> download some first: 'pixi run -e hiw download' (see README).")
        return
    urdf = urdf_layer.load_urdf()
    urdf_layer.convert_model(urdf, RRD_ROOT)

    print(f"Converting {len(episodes)} episode(s) -> {RRD_ROOT}/<layer>/ ({' + '.join(LAYERS)})")
    for ep in episodes:
        written = convert_episode(ep, urdf)
        skipped = [layer for layer in LAYERS if layer not in written]
        note = f" ({', '.join(skipped)} skipped — inputs absent)" if skipped else ""
        print(f"  {ep.recording_id}: {len(written)} layers written{note}")


if __name__ == "__main__":
    main()
