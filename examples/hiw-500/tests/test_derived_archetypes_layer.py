"""
The derived archetypes layer parses `/wbc_lerobot` once and leaves the source text to the base layer.

Runs on a synthetic MCAP with two JSON messages, and on one without the topic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcap_ros2.writer import Writer

from hiw_500.base_layer import episode_from_mcap
from hiw_500.derived_archetypes_layer import EE_NAMES, MSG_WBC, WBC_TOPIC, convert_episode, derived_stream

STRING_SCHEMA = "string data\n"


def _payload(offset: float) -> dict[str, Any]:
    return {
        "ee_state": [offset + i for i in range(12)],
        "ee_action": [offset + 100 + i for i in range(12)],
        "gripper_controls": {"left_trigger": 0.5, "left_squeeze": 0.0, "right_trigger": 1.0, "right_squeeze": 0.25},
        "pivot": [offset] * 7,
    }


def _episode(root: Path, topic: str, payloads: list[dict[str, Any]]) -> Path:
    """An episode directory whose MCAP carries `payloads` as `std_msgs/String` JSON on `topic`."""
    episode = root / "Task" / "episode_2026-01-01_00-00-00" / "episode_0001"
    episode.mkdir(parents=True)
    mcap = episode / "episode_0001.mcap"
    with open(mcap, "wb") as file:
        writer = Writer(file)
        schema = writer.register_msgdef("std_msgs/msg/String", STRING_SCHEMA)
        for sequence, payload in enumerate(payloads):
            time_ns = 10**9 * (sequence + 1)
            writer.write_message(topic, schema, {"data": json.dumps(payload)}, time_ns, time_ns, sequence)
        writer.finish()
    return mcap


def test_the_json_becomes_one_struct_and_four_markers(tmp_path: Path) -> None:
    payloads = [_payload(0.0), _payload(1000.0)]
    store = derived_stream(_episode(tmp_path, WBC_TOPIC, payloads)).collect()

    entities = {chunk.entity_path for chunk in store.stream()}
    assert entities == {
        WBC_TOPIC,
        *(f"/lerobot/{kind}/{arm}" for kind in ("ee_state", "ee_action") for arm in ("left", "right")),
    }

    table = store.reader(index="message_publish_time", contents=[WBC_TOPIC]).to_arrow_table()
    assert not [name for name in table.schema.names if name.endswith(("TextDocument:text", "McapSchema:data"))]
    structs = [row[0] for row in table.column(f"{WBC_TOPIC}:{MSG_WBC}").to_pylist()]
    assert [struct["ee_state"] for struct in structs] == [payload["ee_state"] for payload in payloads]
    assert structs[0]["gripper_controls"]["right_squeeze"] == 0.25
    assert table.column(f"{WBC_TOPIC}:ee_names").to_pylist() == [EE_NAMES] * 2

    right = store.reader(index="message_publish_time", contents=["/lerobot/ee_action/right"]).to_arrow_table()
    translations = [row[0] for row in right.column("/lerobot/ee_action/right:Transform3D:translation").to_pylist()]
    assert translations == [payload["ee_action"][6:9] for payload in payloads]


def test_an_episode_without_the_topic_skips_the_layer(tmp_path: Path) -> None:
    mcap = _episode(tmp_path, "/annotation", [{"task": "sweep"}])
    assert convert_episode(episode_from_mcap(mcap), tmp_path / "rrds") is None
