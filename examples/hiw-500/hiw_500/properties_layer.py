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
from rerun.experimental import Chunk, LazyChunkStream, OptimizationProfile

from hiw_500.base_layer import (
    APPLICATION_ID,
    DATASET_ROOT,
    PROPERTY,
    RRD_ROOT,
    Episode,
    discover_episodes,
    episode_from_mcap,
    has_ir,
)
from rrd_datasets_common.paths import layer_relpath

ROBOT = "unitree_g1"


def properties_chunk(ep: Episode) -> Chunk:
    """
    Every `info.json` field as a recording property, plus what the sidecars and the dataset add.

    The subtask boundaries are the exception: they are temporal, so the base logs them on
    `/task/subtask`, and only their count and labels appear here. One property holds every field,
    so the catalog columns read `property:episode:<name>`.
    """
    info = ep.info
    return Chunk.from_property(
        PROPERTY,
        rr.AnyValues(
            episode_name=[info.episode_name],
            task=[info.task],
            scene=np.array([info.scene], dtype=np.int64),  # -1 when info.json names no scene
            start_timestamp_ns=np.array([info.start_timestamp_ns], dtype=np.int64),
            end_timestamp_ns=np.array([info.end_timestamp_ns], dtype=np.int64),
            duration_sec=np.array([info.duration_sec], dtype=np.float64),
            num_subtasks=np.array([len(info.subtasks)], dtype=np.int64),
            subtask_labels=[", ".join(s.task for s in info.subtasks)],
            has_ir=[has_ir(ep)],
            robot=[ROBOT],
        ),
    )


def convert_episode(ep: Episode, rrd_root: Path) -> Path:
    out_path = rrd_root / layer_relpath("properties", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LazyChunkStream.from_iter([properties_chunk(ep)]).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main(argv: list[str]) -> None:
    episodes = [episode_from_mcap(Path(argv[1]))] if len(argv) > 1 else discover_episodes(DATASET_ROOT)
    print(f"Building properties layer for {len(episodes)} episode(s) -> {RRD_ROOT / 'properties'}/")
    for ep in episodes:
        out = convert_episode(ep, RRD_ROOT)
        print(f"  {ep.recording_id}: {out}")


if __name__ == "__main__":
    main(sys.argv)
