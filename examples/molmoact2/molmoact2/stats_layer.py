"""
Build the *stats* layer: the source's per-episode statistics.

Everything comes from `meta/episodes`, so building this layer needs no data or video file. The
statistics of the state, the action and each camera are static `FeatureStats` entities under
`/stats/<feature>`. Four recording properties also hold each arm's joint and gripper spread, so
the catalog can filter episodes by how much each arm moves.

Run:  pixi run -e molmo convert-stats              # every episode with a base RRD under rrds/molmoact2/base/
      pixi run -e molmo convert-stats <in.rrd>     # one episode
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile

from molmoact2.base_layer import APPLICATION_ID, RRD_ROOT
from molmoact2.episode_index import LOCAL_DIR, episode_file_sets, episode_of
from rrd_datasets_common.paths import layer_relpath, resolve_input_path

LAYER = "stats"
INPUT_LAYER = "base"

STATS_ARCHETYPE = "FeatureStats"
STATS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99", "count")

# One property holding the four spreads, so the catalog columns read `property:stats:<name>`.
PROPERTY = "stats"

ARMS = ("left", "right")


@dataclass(frozen=True)
class EpisodeStats:
    """The `meta/episodes` values the stats layer needs, read once per run."""

    features: list[str]  # the state, the action, then each camera, as the source names them
    names: dict[str, list[str]]  # the element names of the state and the action, from `info.json`
    # One row per episode, in episode order: every statistic of every feature.
    episodes: pa.Table
    spreads: dict[str, np.ndarray]  # one value per episode for each property


def read_episode_stats(dataset_root: Path) -> EpisodeStats:
    """Read the per-episode statistics under `dataset_root/meta`."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    cameras = [file_set.video_key for file_set in episode_file_sets(dataset_root) if file_set.video_key]
    features = ["observation.state", "action", *cameras]
    columns = ["episode_index"] + [f"stats/{feature}/{stat}" for feature in features for stat in STATS]
    episodes = pa.concat_tables(
        pq.read_table(path, columns=columns) for path in sorted(dataset_root.glob("meta/episodes/*/*.parquet"))
    ).sort_by("episode_index")
    # Row i must hold episode i, since the layer looks up an episode by position.
    if not np.array_equal(episodes["episode_index"].to_numpy(), np.arange(episodes.num_rows)):
        raise ValueError(f"meta/episodes in {dataset_root} does not hold episodes 0..{episodes.num_rows - 1}")
    names = {feature: list(info["features"][feature]["names"]) for feature in ("observation.state", "action")}
    return EpisodeStats(features, names, episodes, arm_spreads(episodes, names["action"]))


def arm_spreads(episodes: pa.Table, names: list[str]) -> dict[str, np.ndarray]:
    """How far each arm's joints and gripper move in each episode, as `q99 - q01` of the command."""
    shape = (episodes.num_rows, len(names))
    # The command, not the measured state: at the start of some episodes the measured arm
    # settles toward the command, and the state's spread would count that as motion.
    q01 = episodes["stats/action/q01"].combine_chunks().flatten().to_numpy().reshape(shape)
    q99 = episodes["stats/action/q99"].combine_chunks().flatten().to_numpy().reshape(shape)
    # Unlike `max - min`, the 1st to 99th percentile range leaves out single-frame glitches.
    spread = q99 - q01
    joints = {arm: [i for i, name in enumerate(names) if name.startswith(f"{arm}_joint_")] for arm in ARMS}
    return {
        # The largest spread of the arm's six joints, in radians.
        **{f"{arm}_joint_spread": spread[:, joints[arm]].max(axis=1) for arm in ARMS},
        # The gripper's spread, in the source's range from 0 (closed) to 1 (open).
        **{f"{arm}_gripper_spread": spread[:, names.index(f"{arm}_gripper.pos")] for arm in ARMS},
    }


def single_row(values: pa.ListArray) -> pa.ListArray:
    """A one-row statistic with its nested lists flattened to one value per element."""
    # Camera statistics are stored per channel as [3][1][1]; the state and action ones are flat.
    flat = values.flatten()
    while pa.types.is_list(flat.type):
        flat = flat.flatten()
    return pa.ListArray.from_arrays(pa.array([0, len(flat)], pa.int32()), flat)


def feature_chunk(stats: EpisodeStats, feature: str, episode: int) -> Chunk:
    """The source's statistics of one feature in one episode, as a static entity."""
    row = stats.episodes.slice(episode, 1)
    components: dict[str, pa.Array | list[list[str]]] = {
        stat: single_row(row[f"stats/{feature}/{stat}"].combine_chunks()) for stat in STATS
    }
    if feature in stats.names:
        components["names"] = [stats.names[feature]]
    return Chunk.from_columns(
        f"/stats/{feature}",
        indexes=[],
        columns=rr.DynamicArchetype.columns(archetype=STATS_ARCHETYPE, components=components),
    )


def properties_chunk(stats: EpisodeStats, episode: int) -> Chunk:
    """The episode's arm spreads, as recording properties."""
    values = rr.AnyValues()
    for name, spreads in stats.spreads.items():
        values = values.with_component_from_data(name, spreads[episode : episode + 1])
    return Chunk.from_property(PROPERTY, values)


def write_layer(recording_id: str, stats: EpisodeStats, out_dir: Path) -> Path:
    """Write one episode's stats layer under `out_dir`; returns the written path."""
    episode = episode_of(recording_id)
    out_path = out_dir / layer_relpath(LAYER, recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [*(feature_chunk(stats, feature, episode) for feature in stats.features), properties_chunk(stats, episode)]
    LazyChunkStream.from_iter(chunks).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=recording_id
    )
    return out_path


def discover_inputs(path: Path) -> list[Path]:
    """The input RRDs at `path`: the file itself, or a directory's top level."""
    return [path] if path.is_file() else sorted(path.glob("*.rrd"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the stats layer of every episode that has a base RRD.")
    parser.add_argument(
        "input", nargs="?", type=Path, default=RRD_ROOT / INPUT_LAYER, help="A base RRD, or a directory of them."
    )
    parser.add_argument("--data-dir", type=Path, default=LOCAL_DIR, help="The downloaded dataset root.")
    parser.add_argument("--out-dir", type=Path, default=RRD_ROOT, help="Root the stats/ folder is written under.")
    args = parser.parse_args()

    inputs = discover_inputs(resolve_input_path(args.input))
    if not inputs:
        raise SystemExit(f"No .rrd at {args.input} — run `pixi run -e molmo convert-{INPUT_LAYER}` first.")
    stats = read_episode_stats(args.data_dir)
    print(f"Building {LAYER} layer for {len(inputs)} recording(s) -> {args.out_dir / LAYER}/")
    for input_rrd in inputs:
        # Every layer file is named after its recording id.
        out = write_layer(input_rrd.stem, stats, args.out_dir)
        print(f"  {out.stem}: {out.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
