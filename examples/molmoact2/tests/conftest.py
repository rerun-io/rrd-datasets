"""A miniature LeRobot v3 dataset shaped like MolmoAct2, synthesized so no sample file is committed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

FPS = 30
JOINT_NAMES = [f"{side}_{joint}.pos" for side in ("left", "right") for joint in ("joint_0", "gripper")]
TASKS = ["fold", "pack"]
EPISODE_TASKS = [1, 0, 1]  # task index of each episode
LENGTHS = [3, 5, 4]
INSTRUCTIONS = ["Pack the box.", "Fold the cloth.", "Pack the cloth into the box."]


def frame_columns() -> dict[str, np.ndarray]:
    """Every data-row column of the fixture, in row order, as the source stores it."""
    frame = np.concatenate([np.arange(length) for length in LENGTHS]).astype(np.int64)
    episode = np.repeat(np.arange(len(LENGTHS)), LENGTHS).astype(np.int64)
    values = np.random.default_rng(0).standard_normal((len(frame), 2 * len(JOINT_NAMES))).astype(np.float32)
    return {
        "action": values[:, : len(JOINT_NAMES)],
        "observation.state": values[:, len(JOINT_NAMES) :],
        "timestamp": (frame / FPS).astype(np.float32),
        "frame_index": frame,
        "episode_index": episode,
        "index": np.arange(len(frame), dtype=np.int64),
        "task_index": np.array(EPISODE_TASKS, dtype=np.int64)[episode],
    }


def write_dataset(root: Path) -> None:
    """Write the fixture's `meta/` and its single data file under `root`, without video."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    vector = {"dtype": "float32", "shape": [len(JOINT_NAMES)], "names": JOINT_NAMES}
    scalar = {"shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "bi_yam_follower",
        "total_episodes": len(LENGTHS),
        "total_frames": sum(LENGTHS),
        "total_tasks": len(TASKS),
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(LENGTHS)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": vector,
            "observation.state": vector,
            "timestamp": {"dtype": "float32", **scalar},
            "frame_index": {"dtype": "int64", **scalar},
            "episode_index": {"dtype": "int64", **scalar},
            "index": {"dtype": "int64", **scalar},
            "task_index": {"dtype": "int64", **scalar},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    # The source keeps the task string as the pandas index of tasks.parquet.
    tasks = pa.table({"task_index": pa.array(range(len(TASKS)), pa.int64()), "__index_level_0__": TASKS})
    pq.write_table(tasks, root / "meta" / "tasks.parquet")
    annotated = pa.table({"task": INSTRUCTIONS, "episode_index": pa.array(range(len(LENGTHS)), pa.int64())})
    pq.write_table(annotated, root / "meta" / "tasks_annotated.parquet")
    ends = np.cumsum(LENGTHS)
    episodes = pa.table({
        "episode_index": pa.array(range(len(LENGTHS)), pa.int64()),
        "tasks": pa.array([[TASKS[task]] for task in EPISODE_TASKS], pa.list_(pa.string())),
        "length": pa.array(LENGTHS, pa.int64()),
        "data/chunk_index": pa.array([0] * len(LENGTHS), pa.int64()),
        "data/file_index": pa.array([0] * len(LENGTHS), pa.int64()),
        "dataset_from_index": pa.array(ends - LENGTHS, pa.int64()),
        "dataset_to_index": pa.array(ends, pa.int64()),
    })
    pq.write_table(episodes, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    columns = frame_columns()
    width = len(JOINT_NAMES)
    data = pa.table({
        name: pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), width) if values.ndim == 2 else values
        for name, values in columns.items()
    })
    pq.write_table(data, root / "data" / "chunk-000" / "file-000.parquet")


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "MolmoAct2"
    write_dataset(root)
    return root
