"""
The census end to end: a channel the decoders cannot read is flagged, and its bytes are kept anyway.

No episode in the dataset sample triggers this path, so the test writes an MCAP whose one channel
carries payloads too short for its schema, converts it, and reads the layer back.
"""

from __future__ import annotations

from pathlib import Path

from mcap.writer import Writer
from rerun.experimental import RrdReader

from hiw_500.base_layer import MCAP_PROPERTY, PROPERTY_PATH, convert_episode, episode_from_mcap

TOPIC = "/stamped/broken"
MESSAGES = 5


def _broken_episode(root: Path) -> Path:
    """An episode directory holding one MCAP; every message is two bytes under a twelve-byte schema."""
    episode = root / "Task" / "episode_2026-01-01_00-00-00" / "episode_0001"
    episode.mkdir(parents=True)
    mcap = episode / "episode_0001.mcap"
    with open(mcap, "wb") as file:
        writer = Writer(file)
        writer.start(profile="ros2", library="synthetic")
        schema = writer.register_schema("test/msg/Broken", "ros2msg", b"int32 a\nfloat64 b\n")
        channel = writer.register_channel(TOPIC, "cdr", schema)
        for sequence in range(MESSAGES):
            time_ns = 10**9 * (sequence + 1)
            writer.add_message(channel, log_time=time_ns, publish_time=time_ns, data=b"\x00\x01", sequence=sequence)
        writer.finish()
    return mcap


def test_an_undecodable_channel_is_flagged_and_kept_raw(tmp_path: Path) -> None:
    rrd = convert_episode(episode_from_mcap(_broken_episode(tmp_path / "data")), tmp_path / "rrds")
    reader = RrdReader(str(rrd))
    static: dict[str, dict[str, object]] = {}
    raw_rows = 0
    for chunk in reader.stream(store=reader.recordings()[0]):
        batch = chunk.to_record_batch()
        if chunk.is_static:
            static.setdefault(chunk.entity_path, {}).update({
                name: batch.column(name).to_pylist()[0] for name in batch.schema.names if name != "rerun.controls.RowId"
            })
        elif chunk.entity_path == TOPIC and "McapMessage:data" in batch.schema.names:
            raw_rows += chunk.num_rows

    assert static[PROPERTY_PATH]["has_undecodable"] == [True]
    assert static[PROPERTY_PATH]["undecodable_topics"] == [TOPIC]
    assert raw_rows == MESSAGES
    assert static[TOPIC]["McapSchema:name"] == ["test/msg/Broken"]  # the schema and channel rows stay beside the bytes
    assert static[f"/__properties/{MCAP_PROPERTY}"]["library"] == ["synthetic"]
