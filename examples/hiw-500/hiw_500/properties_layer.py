"""
Build a properties *layer*: per-episode metadata that surfaces as catalog columns.

Recording properties (logged under `/__properties/<name>`) attach to the segment when registered,
so they become filterable/sortable columns in the catalog. The values come from `info.json`, the
calibration sidecars beside the episode, the dataset path, and a constant — never the mcap.
Written as a separate `.rrd` per episode sharing the base `recording_id`.

Run:  pixi run -e hiw convert-properties            # all episodes under data/HIW-500/
      pixi run -e hiw convert-properties <ep.mcap>  # a single episode
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rerun as rr

from hiw_500.base_layer import (
    APPLICATION_ID,
    DATASET_ROOT,
    RRD_ROOT,
    Episode,
    discover_episodes,
    episode_from_mcap,
    has_ir,
)
from rrd_datasets_common.paths import layer_relpath

ROBOT = "unitree_g1"


def _task_group(ep: Episode) -> str:
    """Top-level dataset folder for the episode (e.g. `Move-The-Pillow-To-The-Sofa-From-Floor`)."""
    try:
        return ep.mcap.resolve().relative_to(DATASET_ROOT.resolve()).parts[0]
    except ValueError:
        return ep.recording_id.split("__")[0]


def convert_episode(ep: Episode, rrd_root: Path) -> Path:
    info = ep.info
    out_path = rrd_root / layer_relpath("properties", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rr.RecordingStream(APPLICATION_ID, recording_id=ep.recording_id) as rec:
        rec.save(str(out_path))

        def prop(name: str, value: object, dtype: type | None = None) -> None:
            col: np.ndarray | list[object] = np.array([value], dtype=dtype) if dtype is not None else [value]
            rec.send_property(name, rr.AnyValues(**{name: col}))  # type: ignore[arg-type]

        prop("task", info.task)
        prop("task_group", _task_group(ep))
        prop("duration_sec", info.duration_sec, np.float64)
        prop("num_subtasks", len(info.subtasks), np.int64)
        prop("subtask_labels", ", ".join(s.task for s in info.subtasks))
        prop("scene", info.scene, np.int64)  # -1 when the episode's info.json names no scene
        prop("has_ir", has_ir(ep))
        prop("robot", ROBOT)
    return out_path


def main(argv: list[str]) -> None:
    episodes = [episode_from_mcap(Path(argv[1]))] if len(argv) > 1 else discover_episodes(DATASET_ROOT)
    print(f"Building properties layer for {len(episodes)} episode(s) -> {RRD_ROOT / 'properties'}/")
    for ep in episodes:
        out = convert_episode(ep, RRD_ROOT)
        print(f"  {ep.recording_id}: {out}")


if __name__ == "__main__":
    main(sys.argv)
