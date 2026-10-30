"""Round trip of the base layer: every source data column and the properties come back from the written RRDs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from conftest import EPISODE_TASKS, INSTRUCTIONS, JOINT_NAMES, LENGTHS, TASKS, frame_columns
from rerun.chunk import ChunkStore, LazyStore, RrdReader
from rerun.experimental import LeRobotReader

from molmoact2.base_layer import (
    PROPERTY,
    convert_group,
    episode_file,
    keyframe_misaligned_cameras,
    read_source_meta,
    timestamp_nanos,
    window_misaligned,
)
from molmoact2.episode_index import LOCAL_DIR, discover_local_groups, groups_from_meta

SAMPLE_GROUP = 31673  # a single-episode file group from `SAMPLES`
CAMERAS = ("top", "left", "right")

# In top `file-823`, the windows of 12054–12061 start on a delta frame. In left `file-588`, 12051's
# window starts on a video segment one frame long, so its first two frames are keyframes.
MISALIGNED_CAMERAS = {12051: ["left"], 12052: [], 12053: [], **{episode: ["top"] for episode in range(12054, 12062)}}


def temporal_rows(
    store: ChunkStore | LazyStore, entity: str, component: str
) -> tuple[np.ndarray, np.ndarray, pa.Array]:
    """The `frame_index`, `timestamp` nanoseconds, and component values of an entity, in frame order."""
    stream = store.stream().filter(content=entity, components=[component], is_static=False)
    batches = [chunk.to_record_batch() for chunk in stream.to_chunks()]
    frames = np.concatenate([batch.column("frame_index").to_numpy() for batch in batches])
    nanos = np.concatenate([batch.column("timestamp").cast(pa.int64()).to_numpy() for batch in batches])
    values = pa.concat_arrays([batch.column(component) for batch in batches])
    order = np.argsort(frames, kind="stable")
    return frames[order], nanos[order], values.take(pa.array(order))


def properties(store: ChunkStore | LazyStore) -> pa.RecordBatch:
    (chunk,) = store.stream().filter(content=f"/__properties/{PROPERTY}").to_chunks()
    return chunk.to_record_batch()


def convert_all(dataset_root: Path, out_dir: Path) -> list[LazyStore]:
    """Convert every file group of `dataset_root`; returns one store per episode, in episode order."""
    reader = LeRobotReader(dataset_root)
    meta = read_source_meta(dataset_root)
    paths = [path for group in groups_from_meta(dataset_root) for path in convert_group(group, reader, meta, out_dir)]
    return [RrdReader(str(path)).store() for path in paths]


def test_round_trip_rebuilds_data_and_properties(dataset_root: Path, tmp_path: Path) -> None:
    stores = convert_all(dataset_root, tmp_path / "rrds")
    schemas = {tuple((field.name, field.type) for field in properties(store).schema) for store in stores}
    assert len(schemas) == 1, "property types differ between episodes"

    task_names = pq.read_table(dataset_root / "meta" / "tasks.parquet")["__index_level_0__"]
    rebuilt: dict[str, list[np.ndarray]] = {name: [] for name in frame_columns()}
    first_index = 0
    for episode, store in enumerate(stores):
        batch = properties(store)
        props = batch.to_pydict()
        assert props["episode_index"] == [[episode]]
        assert props["length"] == [[LENGTHS[episode]]]
        # The fixture's tasks are out of order, so a lookup by the wrong row shows up here.
        assert props["task"] == [[TASKS[EPISODE_TASKS[episode]]]]
        assert props["instruction"] == [[INSTRUCTIONS[episode]]]
        assert props["robot_type"] == [["bi_yam_follower"]]
        assert props["source_files"] == [[["data/chunk-000/file-000.parquet"]]]
        assert props["num_frames_mismatched"] == [[False]]
        # Every fixture episode has an empty camera list, which must still be typed like a filled list.
        assert batch.schema.field("keyframe_misaligned_cameras").type == batch.schema.field("source_files").type

        for name in ("action", "observation.state"):
            frames, nanos, values = temporal_rows(store, f"/{name}", "Scalars:scalars")
            rebuilt[name].append(pc.list_flatten(values).to_numpy().reshape(len(frames), len(JOINT_NAMES)))
        _, _, texts = temporal_rows(store, "/task", "TextDocument:text")
        rebuilt["task_index"].append(pc.index_in(pc.list_flatten(texts), value_set=task_names).to_numpy())
        rebuilt["timestamp"].append((nanos / 1e9).astype(np.float32))
        rebuilt["frame_index"].append(frames)
        rebuilt["episode_index"].append(np.full(len(frames), episode))
        rebuilt["index"].append(first_index + frames)
        first_index += LENGTHS[episode]

    for name, source in frame_columns().items():
        np.testing.assert_array_equal(np.concatenate(rebuilt[name]).astype(source.dtype), source, err_msg=name)


@pytest.mark.skipif(not (LOCAL_DIR / "meta" / "info.json").exists(), reason="needs `pixi run -e molmo download`")
def test_downloaded_episode_has_one_video_sample_per_frame(tmp_path: Path) -> None:
    groups = [group for group in discover_local_groups(LOCAL_DIR) if group.episodes[0] == SAMPLE_GROUP]
    if not groups:
        pytest.skip(f"file group {SAMPLE_GROUP} is not downloaded")
    (group,) = groups
    meta = read_source_meta(LOCAL_DIR)
    (path,) = convert_group(group, LeRobotReader(LOCAL_DIR), meta, tmp_path)
    store = RrdReader(str(path)).store()
    props = properties(store).to_pydict()
    assert props["num_frames_mismatched"] == [[False]]
    # The data parquet, then one video file per camera; the conversion just read all of them.
    ((files,),) = props["source_files"]
    assert files[0].startswith("data/")
    assert len(files) == 1 + len(CAMERAS)
    assert all((LOCAL_DIR / file).is_file() for file in files), files

    for camera in CAMERAS:
        frames, nanos, _ = temporal_rows(store, f"/observation.images.{camera}", "VideoStream:sample")
        np.testing.assert_array_equal(frames, np.arange(props["length"][0][0]), err_msg=camera)
        np.testing.assert_array_equal(nanos, timestamp_nanos(frames, meta.fps), err_msg=camera)
        # `is_keyframe` is only written on keyframe rows.
        keyframes, _, _ = temporal_rows(store, f"/observation.images.{camera}", "VideoStream:is_keyframe")
        assert keyframes[0] == 0, f"{camera} does not start on a keyframe"


@pytest.mark.parametrize(
    ("keyframes", "start", "misaligned"),
    [
        ([0, 2, 4, 5, 7], 5, False),  # starts right after an odd-length episode's last keyframe
        ([0, 2, 4, 6], 3, True),  # starts on a delta frame
        ([0, 2, 3, 4, 6], 3, True),  # starts on a segment one frame long
    ],
)
def test_window_misaligned(keyframes: list[int], start: int, misaligned: bool) -> None:
    assert window_misaligned(np.array(keyframes), start) == misaligned


@pytest.mark.skipif(not (LOCAL_DIR / "meta" / "info.json").exists(), reason="needs `pixi run -e molmo download`")
def test_downloaded_videos_flag_keyframe_misaligned_cameras() -> None:
    meta = read_source_meta(LOCAL_DIR)
    videos = {
        LOCAL_DIR / episode_file(meta, file_set, episode)
        for episode in MISALIGNED_CAMERAS
        for file_set in meta.file_sets
        if file_set.video_key
    }
    if not all(video.is_file() for video in videos):
        pytest.skip("the video files of episodes 12051–12061 are not downloaded")
    found = {episode: keyframe_misaligned_cameras(meta, episode) for episode in MISALIGNED_CAMERAS}
    assert found == MISALIGNED_CAMERAS
