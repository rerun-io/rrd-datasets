"""The properties layer's one chunk: every `info.json` field but the subtask boundaries, plus the sidecar facts."""

from __future__ import annotations

import json
from pathlib import Path

from rerun.experimental import Chunk

from hiw_500.base_layer import PROPERTY_PATH, Episode, EpisodeInfo
from hiw_500.properties_layer import ROBOT, properties_chunk

INFO = {
    "episode_name": "episode_2026-02-24_14-31-06",
    "task": "move the pillow to the sofa from floor",
    "start_timestamp_ns": 1771914673276999936,
    "end_timestamp_ns": 1771914696836000000,
    "duration_ns": 23559000064,
    "duration_sec": 23.559000064,
    "subtasks": [
        {"task": "move to bed", "timestamp_ns": 1771914678316999936},
        {"task": "pick pillow", "timestamp_ns": 1771914683217999872},
    ],
    "scene": 1,
}


def _values(chunk: Chunk) -> dict[str, object]:
    """Component name to its one value; `subtask_labels` keeps its whole row as the one list-valued property."""
    batch = chunk.to_record_batch()
    rows = {name: batch.column(name).to_pylist()[0] for name in batch.schema.names if name != "rerun.controls.RowId"}
    return {name: row if name == "subtask_labels" else row[0] for name, row in rows.items()}


def _episode(root: Path, info: dict[str, object]) -> Episode:
    """An episode directory holding only an `info.json`; the mcap itself is never opened."""
    path = root / "info.json"
    path.write_text(json.dumps(info))
    return Episode(
        mcap=root / "episode_0001.mcap",
        info=EpisodeInfo.from_json(path),
        recording_id="Move-The-Pillow-To-The-Sofa-From-Floor__episode_2026-02-24_14-31-06__episode_0001",
        head_calib=None,
    )


def test_every_info_field_is_a_property_except_the_boundaries_and_duration_ns(tmp_path: Path) -> None:
    chunk = properties_chunk(_episode(tmp_path, INFO))
    assert chunk.is_static
    assert chunk.entity_path == PROPERTY_PATH
    assert _values(chunk) == {
        "episode_name": "episode_2026-02-24_14-31-06",
        "task": "move the pillow to the sofa from floor",
        "scene": 1,
        "start_timestamp_ns": 1771914673276999936,
        "end_timestamp_ns": 1771914696836000000,
        "duration_sec": 23.559000064,
        "num_subtasks": 2,
        "subtask_labels": ["move to bed", "pick pillow"],
        "has_ir": False,
        "robot": ROBOT,
    }


def test_a_missing_info_json_still_fills_every_column(tmp_path: Path) -> None:
    """The columns exist on every segment, so an absent sidecar reads as the defaults rather than a gap."""
    episode = Episode(mcap=tmp_path / "episode_0001.mcap", info=EpisodeInfo(), recording_id="T__s__e", head_calib=None)
    values = _values(properties_chunk(episode))
    assert values["scene"] == -1
    assert values["start_timestamp_ns"] == values["end_timestamp_ns"] == 0
    assert values["num_subtasks"] == 0
    assert values["subtask_labels"] == []
