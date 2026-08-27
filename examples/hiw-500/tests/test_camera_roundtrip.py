"""
Round-trip test of the camera messages to ensure the original message can be reconstructed: every `sensor_msgs/CompressedImage` field comes back out of the layers.

Each message is rebuilt from the layer's columns, re-encoded as CDR and compared with the recorded
bytes — on a synthetic MCAP whose `format` and `frame_id` are not the dataset's constants, and on a
cached episode when one is present.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from mcap_ros2.writer import Writer
from PIL import Image
from rerun.experimental import ChunkStore, LazyChunkStream, McapReader

from hiw_500.base_layer import (
    CAMERA_FORMAT,
    CAMERA_HEADER,
    IR_TOPICS,
    RGB_CAMERA_TOPICS,
    base_stream,
    camera_fields_stream,
)
from hiw_500.ir_layer import ir_stream

CAMERA_SCHEMA = "sensor_msgs/msg/CompressedImage"
# The dataset's schema without its comments: the same three definitions rosbag2 writes.
SCHEMA_TEXT = """\
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
WRIST_TOPIC = "/camera/left_wrist/image/compressed"


@dataclass(frozen=True)
class CameraMessage:
    """One `CompressedImage` as recorded: its two MCAP times and its four fields."""

    log_time_ns: int
    publish_time_ns: int
    sec: int
    nanosec: int
    frame_id: str
    format: str
    data: bytes


def write_camera_mcap(path: Path, topic: str, schema_text: str, messages: list[CameraMessage]) -> None:
    with open(path, "wb") as file:
        writer = Writer(file)
        schema = writer.register_msgdef(CAMERA_SCHEMA, schema_text)
        for sequence, msg in enumerate(messages):
            payload = {
                "header": {"stamp": {"sec": msg.sec, "nanosec": msg.nanosec}, "frame_id": msg.frame_id},
                "format": msg.format,
                "data": msg.data,
            }
            writer.write_message(topic, schema, payload, msg.log_time_ns, msg.publish_time_ns, sequence)
        writer.finish()


def _blobs(column: pa.ChunkedArray) -> list[bytes]:
    """The bytes of a one-instance-per-row blob column, sliced out of the flat buffer."""
    blobs = column.combine_chunks().flatten()
    offsets = blobs.offsets.to_numpy()
    values: np.ndarray = blobs.values.to_numpy(zero_copy_only=False)
    return [values[offsets[i] : offsets[i + 1]].tobytes() for i in range(len(blobs))]


def _nanos(column: pa.ChunkedArray) -> list[int]:
    return [int(value) for value in column.cast(pa.int64()).to_pylist()]


def raw_messages(path: Path, topic: str) -> list[tuple[int, int, bytes]]:
    """Every message of `topic` as `(log_time, publish_time, CDR bytes)`, in log-time order."""
    out: list[tuple[int, int, bytes]] = []
    stream = McapReader(str(path), decoders=["raw"], include_topic_regex=[f"^{re.escape(topic)}$"]).stream()
    for chunk in stream:
        if chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        table = pa.Table.from_batches([batch])
        out.extend(
            zip(
                _nanos(table.column("message_log_time")),
                _nanos(table.column("message_publish_time")),
                _blobs(table.column("McapMessage:data")),
            )
        )
    return sorted(out)


def rebuild(store: ChunkStore, topic: str) -> tuple[list[CameraMessage], str]:
    """The topic's messages rebuilt from the layer's columns, plus the `.msg` text the layer kept for them."""
    table = store.reader(index="message_log_time", contents=[topic]).to_arrow_table()
    headers = [row[0] for row in table.column(f"{topic}:{CAMERA_HEADER}").to_pylist()]
    formats = [row[0] for row in table.column(f"{topic}:{CAMERA_FORMAT}").to_pylist()]
    stamps = _nanos(table.column("ros2_timestamp"))
    messages = [
        CameraMessage(log, pub, header["stamp"]["sec"], header["stamp"]["nanosec"], header["frame_id"], fmt, data)
        for log, pub, header, fmt, data in zip(
            _nanos(table.column("message_log_time")),
            _nanos(table.column("message_publish_time")),
            headers,
            formats,
            _blobs(table.column(f"{topic}:EncodedImage:blob")),
        )
    ]
    assert stamps == [msg.sec * 1_000_000_000 + msg.nanosec for msg in messages]
    schema_text = _blobs(table.column(f"{topic}:McapSchema:data"))[0].decode()
    return sorted(messages, key=lambda msg: (msg.log_time_ns, msg.publish_time_ns)), schema_text


def assert_same_messages(rebuilt: list[tuple[int, int, bytes]], recorded: list[tuple[int, int, bytes]]) -> None:
    """
    Same times and same CDR payload, up to the recorder's trailing alignment pad.

    rmw_fastrtps pads a serialized message to a 4-byte boundary; the pad follows the last field and
    carries no data, so a re-encode without it is the same message.
    """
    assert len(rebuilt) == len(recorded)
    for (log, pub, payload), (recorded_log, recorded_pub, recorded_payload) in zip(rebuilt, recorded):
        assert (log, pub) == (recorded_log, recorded_pub)
        pad = recorded_payload[len(payload) :]
        assert recorded_payload[: len(payload)] == payload
        assert len(pad) < 4 and not pad.strip(b"\0"), pad


def _jpeg(shade: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (shade, shade, shade)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_a_camera_message_round_trips_byte_for_byte(tmp_path: Path) -> None:
    """A `format` and `frame_id` the dataset never uses survive: the header column, not a convention, carries them."""
    messages = [
        CameraMessage(
            10_000_000_007 + i * 10**9,
            10_000_000_009 + i * 10**9,
            10 + i,
            5 + i,
            "left_wrist_optical",
            "bgr8; jpeg compressed bgr8",
            _jpeg(20 + i),
        )
        for i in range(3)
    ]
    source = tmp_path / "source.mcap"
    write_camera_mcap(source, WRIST_TOPIC, SCHEMA_TEXT, messages)

    store = LazyChunkStream.merge(base_stream(source), camera_fields_stream(source, RGB_CAMERA_TOPICS)).collect()
    rebuilt, schema_text = rebuild(store, WRIST_TOPIC)
    assert rebuilt == messages
    assert schema_text == SCHEMA_TEXT

    frames = (
        store.reader(index="message_log_time", contents=[WRIST_TOPIC])
        .to_arrow_table()
        .column(f"{WRIST_TOPIC}:CoordinateFrame:frame")
    )
    assert (
        frames.to_pylist() == [["left_wrist_optical_image_plane"]] * 3
    )  # the decoder's rewrite; the header column keeps the original

    again = tmp_path / "rebuilt.mcap"
    write_camera_mcap(again, WRIST_TOPIC, schema_text, rebuilt)
    assert_same_messages(raw_messages(again, WRIST_TOPIC), raw_messages(source, WRIST_TOPIC))


def test_the_cached_episode_cameras_round_trip_byte_for_byte(cached_episode: Path, tmp_path: Path) -> None:
    """Against a rosbag2-written file: the re-encoded CDR of every camera message equals the recorded bytes."""
    mcap = cached_episode
    topics = [
        channel.topic for channel in McapReader(str(mcap)).info().channels if channel.topic.startswith("/camera/")
    ]
    is_ir = re.compile(IR_TOPICS[0]).match
    stores = {
        False: LazyChunkStream.merge(base_stream(mcap), camera_fields_stream(mcap, RGB_CAMERA_TOPICS)).collect(),
        True: ir_stream(mcap).collect() if any(is_ir(topic) for topic in topics) else None,
    }
    for index, topic in enumerate(topics):
        store = stores[bool(is_ir(topic))]
        assert store is not None
        rebuilt, schema_text = rebuild(store, topic)
        again = tmp_path / f"{index}.mcap"
        write_camera_mcap(again, topic, schema_text, rebuilt)
        assert_same_messages(raw_messages(again, topic), raw_messages(mcap, topic))
