"""Tests for the file groups and the recording ids."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from molmoact2.episode_index import WorkItem, discover_local_groups, groups_from_meta, recording_id

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
TOP, WRIST = "observation.images.top", "observation.images.left"

# A fake dataset of six episodes. Each list gives the file index of every episode in one file set.
# Episodes 0–3 form one group only through their neighbors, since no file holds all four:
# 0 and 1 share data file 0, 1 and 2 share top file 0, and 2 and 3 share data file 1.
# This chain is the part of the grouping most likely to break.
# Episodes 4 and 5 start new files in every file set, so they form a second group.
DATA_FILES = [0, 0, 1, 1, 2, 2]
TOP_FILES = [0, 0, 0, 1, 2, 2]
WRIST_FILES = [0, 0, 1, 1, 2, 2]
TASKS = ["fold", "fold", "fold", "fold", "pack", "pack"]


def write_meta(root: Path, top_files: list[int] = TOP_FILES) -> None:
    """Write the fake dataset's LeRobot v3 `meta/` folder under `root`."""
    meta = root / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    info = {
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "observation.state": {"dtype": "float32"},
            TOP: {"dtype": "video"},
            WRIST: {"dtype": "video"},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    zeros = [0] * len(TASKS)
    episodes = pa.table({
        "episode_index": pa.array(range(len(TASKS)), pa.int64()),
        "tasks": pa.array([[task] for task in TASKS], pa.list_(pa.string())),
        "data/chunk_index": pa.array(zeros, pa.int64()),
        "data/file_index": pa.array(DATA_FILES, pa.int64()),
        f"videos/{TOP}/chunk_index": pa.array(zeros, pa.int64()),
        f"videos/{TOP}/file_index": pa.array(top_files, pa.int64()),
        f"videos/{WRIST}/chunk_index": pa.array(zeros, pa.int64()),
        f"videos/{WRIST}/file_index": pa.array(WRIST_FILES, pa.int64()),
    })
    pq.write_table(episodes, meta / "episodes" / "chunk-000" / "file-000.parquet")


def touch(root: Path, paths: tuple[str, ...]) -> None:
    for relpath in paths:
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


# The two groups `groups_from_meta` should build from the fake dataset.
FIRST_GROUP = WorkItem(
    episodes=(0, 1, 2, 3),
    tasks=("fold",),
    files=(
        "data/chunk-000/file-000.parquet",
        "data/chunk-000/file-001.parquet",
        f"videos/{WRIST}/chunk-000/file-000.mp4",
        f"videos/{WRIST}/chunk-000/file-001.mp4",
        f"videos/{TOP}/chunk-000/file-000.mp4",
        f"videos/{TOP}/chunk-000/file-001.mp4",
    ),
)
SECOND_GROUP = WorkItem(
    episodes=(4, 5),
    tasks=("pack",),
    files=(
        "data/chunk-000/file-002.parquet",
        f"videos/{WRIST}/chunk-000/file-002.mp4",
        f"videos/{TOP}/chunk-000/file-002.mp4",
    ),
)


def test_recording_id_pads_to_five_digits() -> None:
    assert recording_id(7) == "episode_00007"
    assert recording_id(32245) == "episode_32245"


def test_groups_split_where_no_file_is_shared(tmp_path: Path) -> None:
    write_meta(tmp_path)
    assert groups_from_meta(tmp_path) == [FIRST_GROUP, SECOND_GROUP]


def test_file_with_non_consecutive_episodes_raises(tmp_path: Path) -> None:
    write_meta(tmp_path, top_files=[0, 1, 0, 1, 2, 2])
    with pytest.raises(ValueError, match="consecutive"):
        groups_from_meta(tmp_path)


def test_local_discovery_keeps_complete_groups(tmp_path: Path) -> None:
    write_meta(tmp_path)
    touch(tmp_path, SECOND_GROUP.files)
    touch(tmp_path, FIRST_GROUP.files[:2])
    assert discover_local_groups(tmp_path) == [SECOND_GROUP]
