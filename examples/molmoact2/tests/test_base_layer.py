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

from molmoact2.base_layer import PROPERTY, convert_group, read_source_meta, timestamp_nanos
from molmoact2.episode_index import LOCAL_DIR, discover_local_groups, groups_from_meta

SAMPLE_GROUP = 31673  # a single-episode file group from `SAMPLES`
CAMERAS = ("top", "left", "right")


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
        props = properties(store).to_pydict()
        assert props["episode_index"] == [[episode]]
        assert props["length"] == [[LENGTHS[episode]]]
        # The fixture's tasks are out of order, so a lookup by the wrong row shows up here.
        assert props["task"] == [[TASKS[EPISODE_TASKS[episode]]]]
        assert props["instruction"] == [[INSTRUCTIONS[episode]]]
        assert props["robot_type"] == [["bi_yam_follower"]]
        assert props["has_frame_mismatch"] == [[False]]

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
    assert props["has_frame_mismatch"] == [[False]]

    for camera in CAMERAS:
        frames, nanos, _ = temporal_rows(store, f"/observation.images.{camera}", "VideoStream:sample")
        np.testing.assert_array_equal(frames, np.arange(props["length"][0][0]), err_msg=camera)
        np.testing.assert_array_equal(nanos, timestamp_nanos(frames, meta.fps), err_msg=camera)
        # `is_keyframe` is only written on keyframe rows.
        keyframes, _, _ = temporal_rows(store, f"/observation.images.{camera}", "VideoStream:is_keyframe")
        assert keyframes[0] == 0, f"{camera} does not start on a keyframe"
