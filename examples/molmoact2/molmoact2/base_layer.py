"""
Convert MolmoAct2 episodes into per-episode Rerun RRDs: the base layer.

Each episode is written as `LeRobotReader` emits it, with two additions: a `timestamp` timeline
next to `frame_index`, and recording properties holding the episode's metadata, its source files,
whether its streams disagree on the number of frames, and which cameras' video windows disagree
with the keyframes of their file.
One RRD per episode, with the recording id `episode_NNNNN`.

Run:  pixi run -e molmo convert-base                       # every downloaded episode
      pixi run -e molmo convert-base --from 800 --to 809   # downloaded episodes 800 to 809
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rerun as rr
from rerun.chunk import Chunk, ChunkStore, LazyChunkStream, OptimizationProfile
from rerun.experimental import LeRobotReader, Mp4Reader

from molmoact2.episode_index import (
    HF_REVISION,
    LOCAL_DIR,
    EpisodeFileSet,
    WorkItem,
    discover_local_groups,
    episode_file_sets,
    recording_id,
)
from rrd_datasets_common.paths import dataset_rrd_dir, layer_relpath

LAYER = "base"
RRD_ROOT = dataset_rrd_dir("molmoact2")
APPLICATION_ID = "molmoact2"

# One property holding every field, so the catalog columns read `property:episode:<name>`.
PROPERTY = "episode"

# The components that hold one row per frame: the state and action scalars, and the video samples.
FRAME_COMPONENTS = ["Scalars:scalars", "VideoStream:sample"]

# The source names each camera `observation.images.<camera>`; properties list cameras by `<camera>`.
CAMERA_KEY_PREFIX = "observation.images."

# `LeRobotReader` keeps `frame_index` as the only timeline and drops the source's `timestamp`
# column. The timeline added back carries the source's float32 seconds as whole nanoseconds.
TIMESTAMP_TIMELINE = "timestamp"
TIMESTAMP_FIELD_METADATA = {
    b"rerun:kind": b"index",
    b"rerun:index_name": TIMESTAMP_TIMELINE.encode(),
    b"rerun:is_sorted": b"true",
}


@dataclass(frozen=True)
class SourceMeta:
    """The `meta/` values the base layer needs, read once per run."""

    robot_type: str
    fps: int
    root: Path  # the dataset root the source files are under
    file_sets: list[EpisodeFileSet]  # the data parquets, then one per camera
    # One row per episode, in episode order: episode_index, length, task, instruction, each file
    # set's chunk_index and file_index, and each camera's from_timestamp.
    episodes: pa.Table


def read_source_meta(dataset_root: Path) -> SourceMeta:
    """Read the dataset-wide and per-episode metadata under `dataset_root/meta`."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    file_sets = episode_file_sets(dataset_root)
    columns = ["episode_index", "length", "tasks"]
    columns += [f"{file_set.prefix}/{part}" for file_set in file_sets for part in ("chunk_index", "file_index")]
    columns += [f"{file_set.prefix}/from_timestamp" for file_set in file_sets if file_set.video_key]
    episodes = pa.concat_tables(
        pq.read_table(path, columns=columns) for path in sorted(dataset_root.glob("meta/episodes/*/*.parquet"))
    ).sort_by("episode_index")
    # Row i must hold episode i, since properties look an episode up by position.
    if not np.array_equal(episodes["episode_index"].to_numpy(), np.arange(episodes.num_rows)):
        raise ValueError(f"meta/episodes in {dataset_root} does not hold episodes 0..{episodes.num_rows - 1}")
    task_count = pc.list_value_length(episodes["tasks"]).to_numpy()
    if not (task_count == 1).all():
        odd = np.flatnonzero(task_count != 1)
        raise ValueError(f"{len(odd)} episodes do not have exactly one task, e.g. episode {odd[0]}")
    episodes = episodes.drop_columns("tasks").append_column("task", pc.list_flatten(episodes["tasks"]))
    annotated = pq.read_table(
        dataset_root / "meta" / "tasks_annotated.parquet", columns=["episode_index", "task"]
    ).sort_by("episode_index")
    if not np.array_equal(annotated["episode_index"].to_numpy(), episodes["episode_index"].to_numpy()):
        raise ValueError(f"tasks_annotated.parquet in {dataset_root} does not list every episode once")
    episodes = episodes.append_column("instruction", annotated["task"])
    return SourceMeta(str(info["robot_type"]), int(info["fps"]), dataset_root, file_sets, episodes)


def episode_file(meta: SourceMeta, file_set: EpisodeFileSet, episode: int) -> str:
    """The repo path of the file in `file_set` that holds the episode."""
    return file_set.path(
        int(meta.episodes[f"{file_set.prefix}/chunk_index"][episode].as_py()),
        int(meta.episodes[f"{file_set.prefix}/file_index"][episode].as_py()),
    )


def source_files(meta: SourceMeta, episode: int) -> list[str]:
    """The repo paths of the episode's data parquet and of each camera's video file."""
    return [episode_file(meta, file_set, episode) for file_set in meta.file_sets]


# A video file holds many consecutive episodes, and the check for each episode reads the
# same keyframe list. Caching the file's keyframes ensures that each file is read only once.
@lru_cache(maxsize=8)
def keyframe_frames(video: Path, fps: int) -> np.ndarray:
    """The frame numbers of a video file's keyframes, in order."""
    stream = Mp4Reader(video, timeline_name="video").stream()
    chunks = stream.filter(components=["VideoStream:is_keyframe"]).to_chunks()
    nanos = np.concatenate([chunk.to_record_batch().column("video").cast(pa.int64()).to_numpy() for chunk in chunks])
    return np.sort(np.rint(nanos * fps / 1e9).astype(np.int64))


def window_misaligned(keyframes: np.ndarray, start: int) -> bool:
    """Whether a video window starting at frame `start` disagrees with the keyframes of its file."""
    # A GOP (group of pictures) is a keyframe and the delta frames that follow it. LeRobot encodes
    # each episode as a separate video segment, and a segment is supposed to start with a new GOP.
    # With GOPs longer than one frame (for MolmoAct2, GOP=2), a window that matches its segment
    # starts on a keyframe followed by a delta frame. Two cases break this. If the window starts on
    # a delta frame, it starts in the middle of a GOP. If its first two frames are both keyframes,
    # its first frame is a one-frame GOP that belongs to the segment before it.
    first_is_keyframe, second_is_keyframe = np.isin([start, start + 1], keyframes)
    return bool(not first_is_keyframe or second_is_keyframe)


def keyframe_misaligned_cameras(meta: SourceMeta, episode: int) -> list[str]:
    """The cameras whose video window for the episode disagrees with the keyframes of its file."""
    cameras = []
    for file_set in meta.file_sets:
        if not file_set.video_key:
            continue
        start = round(float(meta.episodes[f"{file_set.prefix}/from_timestamp"][episode].as_py()) * meta.fps)
        keyframes = keyframe_frames(meta.root / episode_file(meta, file_set, episode), meta.fps)
        if window_misaligned(keyframes, start):
            cameras.append(file_set.video_key.removeprefix(CAMERA_KEY_PREFIX))
    return cameras


def num_frames_mismatched(store: ChunkStore, length: int) -> bool:
    """Whether the state, the action, or any camera holds a number of frames other than `length`."""
    rows: dict[str, int] = {}
    for chunk in store.stream().filter(components=FRAME_COMPONENTS, is_static=False):
        rows[chunk.entity_path] = rows.get(chunk.entity_path, 0) + chunk.num_rows
    return any(count != length for count in rows.values())


def properties_chunk(meta: SourceMeta, episode: int, frames_mismatched: bool, misaligned_cameras: list[str]) -> Chunk:
    """The episode's recording properties, typed the same in every episode."""
    row = meta.episodes.slice(episode, 1)
    # The explicit list type keeps an empty list typed like a filled one.
    string_list = pa.list_(pa.string())
    return Chunk.from_property(
        PROPERTY,
        rr.AnyValues(
            episode_index=np.array([episode], dtype=np.int64),
            task=row["task"].to_pylist(),
            instruction=row["instruction"].to_pylist(),
            length=row["length"].to_numpy(),
            robot_type=[meta.robot_type],
            source_revision=[HF_REVISION],
            source_files=pa.array([source_files(meta, episode)], type=string_list),
            num_frames_mismatched=np.array([frames_mismatched], dtype=np.bool_),
            keyframe_misaligned_cameras=pa.array([misaligned_cameras], type=string_list),
        ),
    )


def timestamp_nanos(frame_index: np.ndarray, fps: int) -> np.ndarray:
    """
    The source `timestamp` of each frame, in nanoseconds.

    The source stores `float32(frame_index / fps)` seconds, so the same value is computed here and
    rounded to whole nanoseconds; `float32(nanos / 1e9)` gives the stored value back exactly.
    """
    seconds = (frame_index / fps).astype(np.float32)
    return np.round(seconds.astype(np.float64) * 1e9).astype(np.int64)


def add_timestamp_timeline(chunk: Chunk, fps: int) -> Chunk:
    """Add the `timestamp` timeline to a temporal chunk, leaving its components untouched."""
    if chunk.is_static:
        return chunk
    batch = chunk.to_record_batch()
    # Index columns must come before the data columns.
    position = batch.schema.get_field_index("frame_index") + 1
    field = pa.field(TIMESTAMP_TIMELINE, pa.duration("ns"), nullable=False, metadata=TIMESTAMP_FIELD_METADATA)
    nanos = pa.array(timestamp_nanos(batch.column("frame_index").to_numpy(), fps), type=pa.duration("ns"))
    (with_timestamp,) = Chunk.from_record_batch(batch.add_column(position, field, nanos))
    return with_timestamp


def episode_stream(reader: LeRobotReader, meta: SourceMeta, episode: int) -> LazyChunkStream:
    """One episode as `LeRobotReader` emits it, plus the `timestamp` timeline."""
    return reader.stream(episode).map(partial(add_timestamp_timeline, fps=meta.fps))


def convert_episode(episode: int, reader: LeRobotReader, meta: SourceMeta, rrd_root: Path) -> Path:
    """Write one episode's base layer; returns the written path."""
    recording = recording_id(episode)
    out_path = rrd_root / layer_relpath(LAYER, recording)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Collected once: the frame counts come from the read episode, before the properties are built.
    # The object-store profile compacts the reader's small chunks into the sizes catalog reads expect.
    store = episode_stream(reader, meta, episode).collect(optimize=OptimizationProfile.OBJECT_STORE)
    length = int(meta.episodes["length"][episode].as_py())
    properties = properties_chunk(
        meta, episode, num_frames_mismatched(store, length), keyframe_misaligned_cameras(meta, episode)
    )
    LazyChunkStream.merge(store.stream(), LazyChunkStream.from_iter([properties])).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=recording
    )
    return out_path


def convert_group(group: WorkItem, reader: LeRobotReader, meta: SourceMeta, rrd_root: Path) -> list[Path]:
    """Write the base layer of every episode in a file group; returns the written paths."""
    return [convert_episode(episode, reader, meta, rrd_root) for episode in group.episodes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded MolmoAct2 episodes into base-layer RRDs.")
    parser.add_argument("--from", dest="first", type=int, default=0, help="First episode to convert (default: 0).")
    parser.add_argument("--to", dest="last", type=int, help="Last episode to convert, included (default: the last).")
    parser.add_argument("--data-dir", type=Path, default=LOCAL_DIR, help="The downloaded dataset root.")
    parser.add_argument("--out-dir", type=Path, default=RRD_ROOT, help="Root the base/ folder is written under.")
    args = parser.parse_args()

    downloaded = [episode for group in discover_local_groups(args.data_dir) for episode in group.episodes]
    meta = read_source_meta(args.data_dir)
    last = args.last if args.last is not None else meta.episodes.num_rows - 1
    episodes = [episode for episode in downloaded if args.first <= episode <= last]
    if not episodes:
        raise SystemExit(f"No downloaded episode in {args.first}–{last} — run `pixi run -e molmo download` first.")
    reader = LeRobotReader(args.data_dir)
    print(f"Building {LAYER} layer for {len(episodes)} episode(s) -> {args.out_dir / LAYER}/")
    skipped = last - args.first + 1 - len(episodes)
    if skipped:
        print(f"  {skipped} episode(s) in {args.first}–{last} are not downloaded and are skipped.")
    for episode in episodes:
        out_path = convert_episode(episode, reader, meta, args.out_dir)
        print(f"  {out_path.stem}: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
