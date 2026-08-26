"""
Every record of the MCAP comes out of the converted layers.

The base, IR and derived archetypes layers of the smallest cached episode are built once and held
against the MCAP summary: every channel with all its messages, every schema and channel row, the
camera fields, the head original, the file records. Skipped when no episode is downloaded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from rerun.experimental import McapChannelInfo, McapReader, RrdReader

from hiw_500 import base_layer, derived_archetypes_layer, ir_layer
from hiw_500.base_layer import CAMERA_FORMAT, CAMERA_HEADER, HEAD_TOPIC, MCAP_PROPERTY, episode_from_mcap

BOOKKEEPING = {f"McapSchema:{name}" for name in ("data", "encoding", "id", "name")} | {
    f"McapChannel:{name}" for name in ("id", "message_encoding", "metadata", "topic")
}


@dataclass
class Inventory:
    """What the layers hold: temporal rows per entity and component, static components per entity."""

    channels: tuple[McapChannelInfo, ...]
    rows: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    static: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def decoded(self, entity: str) -> int:
        """Messages decoded on an entity: the widest of its chunk families, each one row per message."""
        return max(self.rows.get(entity, {}).values(), default=0)


@pytest.fixture(scope="module")
def inventory(cached_episode: Path, tmp_path_factory: pytest.TempPathFactory) -> Inventory:
    root = tmp_path_factory.mktemp("rrds")
    episode = episode_from_mcap(cached_episode)
    layers = [
        base_layer.convert_episode(episode, root),
        ir_layer.convert_episode(episode, root),
        derived_archetypes_layer.convert_episode(episode, root),
    ]
    inventory = Inventory(channels=McapReader(str(cached_episode)).info().channels)
    for rrd in filter(None, layers):
        reader = RrdReader(str(rrd))
        for chunk in reader.stream(store=reader.recordings()[0]):
            batch = chunk.to_record_batch()
            for name in batch.schema.names:
                if name == "rerun.controls.RowId" or name in chunk.timeline_names:
                    continue
                if chunk.is_static:
                    inventory.static[chunk.entity_path].add(name)
                else:
                    inventory.rows[chunk.entity_path][name] += chunk.num_rows
        rrd.unlink()
    return inventory


def test_every_channel_keeps_all_its_messages(inventory: Inventory) -> None:
    short = {
        channel.topic: (inventory.decoded(channel.topic), channel.message_count)
        for channel in inventory.channels
        if inventory.decoded(channel.topic) != channel.message_count
    }
    assert short == {}


def test_every_topic_keeps_its_schema_and_channel_rows(inventory: Inventory) -> None:
    missing = {
        channel.topic: sorted(BOOKKEEPING - inventory.static[channel.topic])
        for channel in inventory.channels
        if not BOOKKEEPING <= inventory.static[channel.topic]
    }
    assert missing == {}


def test_cameras_keep_every_field_per_row(inventory: Inventory) -> None:
    cameras = [channel for channel in inventory.channels if channel.topic.startswith("/camera/")]
    assert cameras
    for channel in cameras:
        rows = inventory.rows[channel.topic]
        assert rows["EncodedImage:blob"] == channel.message_count, channel.topic
        assert rows[CAMERA_HEADER] == rows[CAMERA_FORMAT] == channel.message_count, channel.topic


def test_the_head_original_stays_beside_its_halves(inventory: Inventory) -> None:
    assert "EncodedImage:blob" in inventory.rows[HEAD_TOPIC]
    assert all("EncodedImage:blob" in inventory.rows[f"/camera/head/{side}"] for side in ("left", "right"))


def test_the_file_records_are_kept(inventory: Inventory) -> None:
    assert "rosbag2" in inventory.static["/__mcap_metadata"]
    assert any(name.startswith("McapStatistics:") for name in inventory.static["/__mcap_properties"])
    assert {"profile", "library", "compression"} <= inventory.static[f"/__properties/{MCAP_PROPERTY}"]
