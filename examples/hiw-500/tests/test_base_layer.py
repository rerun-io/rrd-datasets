"""
Tests for the base layer's calibration passthrough, joint arrays, and channel census — no MCAP, no network.

Calibration files are archived verbatim, so both halves of that promise are guarded here: the file
text survives byte for byte, and the per-serial wrist files keep one entity path across rigs. The
census is what makes silently dropped messages visible, so it runs against synthetic chunks shaped
exactly as `McapReader` emits them — a statistics table on `/__mcap_properties`, a static channel
id per raw topic, and decoded rows on the counting entity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.experimental import Chunk

from hiw_500.base_layer import (
    CENSUS_PROXIES,
    CHANNEL_ID,
    G1_JOINT_NAMES,
    STAT_CHANNEL_COUNTS,
    Episode,
    EpisodeInfo,
    _motors,
    calibration_chunks,
    census_chunk,
    has_ir,
    joint_names_chunks,
    undecodable_topics,
)

HEAD_YAML = """\
# head stereo pair
baseline: 60.530000000000001
image_size: [640, 480]
"""
WRIST_JSON = '{"color": {"intrinsics": {"fx": 435.96734619140625}}}\n'


def _episode(root: Path, sidecars: dict[str, str]) -> Episode:
    """An episode directory holding only the given sidecars; the mcap itself is never opened."""
    for rel, text in sidecars.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Episode(
        mcap=root / "episode_0001.mcap",
        info=EpisodeInfo(),
        recording_id="Sweep-Floor__episode_2026-05-19_14-35-04__episode_0001",
        head_calib=None,
    )


def _cell(chunk: Chunk, column: str) -> object:
    """The one value in a one-row chunk's component column."""
    (row,) = chunk.to_record_batch().column(column).to_pylist()
    (value,) = row
    return value


def _by_entity(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {chunk.entity_path: chunk for chunk in chunks}


def test_calibration_files_are_archived_verbatim(tmp_path: Path) -> None:
    episode = _episode(
        tmp_path,
        {
            "calibration/params/head_camera_params.yaml": HEAD_YAML,
            "calibration/notes.txt": "rig A\n",
        },
    )
    chunks = _by_entity(calibration_chunks(episode))
    assert set(chunks) == {"/calibration/params/head_camera_params", "/calibration/notes"}

    head = chunks["/calibration/params/head_camera_params"]
    assert head.is_static
    assert _cell(head, "TextDocument:text") == HEAD_YAML
    assert _cell(head, "TextDocument:media_type") == "application/x-yaml"
    assert _cell(head, "path") == "calibration/params/head_camera_params.yaml"
    assert _cell(chunks["/calibration/notes"], "TextDocument:media_type") == "text/plain"


def test_wrist_calibrations_get_one_entity_path_across_rigs(tmp_path: Path) -> None:
    """Serials differ per rig, so they ride in a component while the entity path stays put."""
    episode = _episode(
        tmp_path,
        {
            "calibration/params/camera_409122273272.json": WRIST_JSON,
            "calibration/params/camera_323622270214.json": WRIST_JSON,
        },
    )
    chunks = _by_entity(calibration_chunks(episode))
    assert set(chunks) == {"/calibration/params/wrist_camera1", "/calibration/params/wrist_camera2"}

    first = chunks["/calibration/params/wrist_camera1"]
    assert _cell(first, "id") == "323622270214"  # numbered by sorted filename, never a left/right guess
    assert _cell(first, "path") == "calibration/params/camera_323622270214.json"
    assert _cell(first, "TextDocument:media_type") == "application/json"
    assert _cell(chunks["/calibration/params/wrist_camera2"], "id") == "409122273272"


def test_an_episode_without_calibration_contributes_no_chunks(tmp_path: Path) -> None:
    assert calibration_chunks(_episode(tmp_path, {"info.json": "{}"})) == []


def test_an_info_json_without_a_scene_reads_as_minus_one(tmp_path: Path) -> None:
    """`scene` is a catalog column, so the absent case needs a stable value rather than a crash."""
    path = tmp_path / "info.json"
    path.write_text('{"task": "sweep floor", "duration_sec": 16.6}')
    assert EpisodeInfo.from_json(path).scene == -1
    assert EpisodeInfo.from_json(tmp_path / "absent.json").scene == -1


def test_ir_is_inferred_from_the_wrist_calibrations(tmp_path: Path) -> None:
    """The properties layer answers this without the mcap, so the sidecars have to carry it."""
    with_ir = _episode(tmp_path / "with", {"calibration/params/camera_409122273272.json": WRIST_JSON})
    without = _episode(tmp_path / "without", {"calibration/params/head_camera_params.yaml": HEAD_YAML})
    assert has_ir(with_ir)
    assert not has_ir(without)
    assert not has_ir(_episode(tmp_path / "bare", {"info.json": "{}"}))


def _motor_messages(rows: int) -> pa.Array:
    """Lowstate-shaped rows as the reader emits them (a length-1 message list per row); q of motor i is 100r + i."""
    return pa.array([
        [{"data": {"motor_state": [{"q": 100.0 * r + i, "tau_est": 2000.0 + i} for i in range(35)]}}]
        for r in range(rows)
    ])


def test_the_motor_selector_yields_29_joints_in_motor_order() -> None:
    """The motor arrays are 35-wide with indices 29-34 unused; the joint arrays must stop at 29."""
    values = _motors("motor_state", "q").execute_per_row(_motor_messages(2))
    assert values is not None
    assert values.to_pylist() == [[100.0 * r + i for i in range(29)] for r in range(2)]
    taus = _motors("motor_state", "tau_est").execute_per_row(_motor_messages(1))
    assert taus is not None
    assert taus.to_pylist() == [[2000.0 + i for i in range(29)]]


def test_joint_names_ride_statically_beside_the_arrays() -> None:
    """Series i of every joint array is `joint_names[i]` — the mapping the arrays are read by."""
    chunks = _by_entity(joint_names_chunks())
    assert set(chunks) == {"/state/joint", "/cmd/joint"}
    for chunk in chunks.values():
        assert chunk.is_static
        (row,) = chunk.to_record_batch().column("joint_names").to_pylist()
        assert row == G1_JOINT_NAMES


# `McapStatistics:channel_message_counts` as the reader emits it: one instance per row holding the
# whole channel_id -> message_count table.
COUNTS_TYPE = pa.list_(pa.struct([("channel_id", pa.uint16()), ("message_count", pa.uint64())]))

LOWSTATE = "/stamped/lowstate"
DEX1_STATE = "/stamped/dex1/left/state"


def _column(descriptor: str, values: pa.Array) -> rr.ComponentColumn:
    return rr.ComponentColumn(descriptor, rr.AnyBatchValue(descriptor, values))


def _statistics_chunk(counts: dict[int, int]) -> Chunk:
    """The MCAP summary's own per-channel message counts."""
    table = [{"channel_id": channel_id, "message_count": count} for channel_id, count in counts.items()]
    return Chunk.from_columns(
        "/__mcap_properties",
        indexes=[],
        columns=[_column(STAT_CHANNEL_COUNTS, pa.array([table], type=COUNTS_TYPE))],
    )


def _channel_chunk(topic: str, channel_id: int) -> Chunk:
    """A raw topic entity's static channel id — the join key into the statistics table."""
    return Chunk.from_columns(
        topic,
        indexes=[],
        columns=[_column(CHANNEL_ID, pa.array([channel_id], type=pa.uint16()))],
    )


def _decoded_chunk(entity: str, rows: int) -> Chunk:
    """`rows` decoded messages on the entity the census counts a topic by."""
    return Chunk.from_columns(
        entity,
        indexes=[rr.TimeColumn("message_publish_time", timestamp=np.arange(rows).astype("datetime64[ns]"))],
        columns=rr.Scalars.columns(scalars=np.zeros(rows)),
    )


def test_an_episode_that_decoded_everything_flags_nothing() -> None:
    chunks = [
        _statistics_chunk({1: 5, 2: 2}),
        _channel_chunk(LOWSTATE, 1),
        _channel_chunk("/annotation", 2),
        _decoded_chunk(CENSUS_PROXIES[LOWSTATE], 3),  # a topic's rows may arrive in several chunks
        _decoded_chunk(CENSUS_PROXIES[LOWSTATE], 2),
        _decoded_chunk("/annotation", 2),  # a passthrough topic counts rows on its own entity
    ]
    assert undecodable_topics(chunks) == []


def test_topics_short_of_their_channel_count_are_flagged() -> None:
    chunks = [
        _statistics_chunk({1: 5, 2: 2, 3: 5126}),
        _channel_chunk(LOWSTATE, 1),
        _channel_chunk("/annotation", 2),
        _channel_chunk(DEX1_STATE, 3),
        _decoded_chunk(CENSUS_PROXIES[LOWSTATE], 5),
        _decoded_chunk("/annotation", 1),
    ]
    # dex1 decoded none of its 5126 messages (the Feb 2026 payload mismatch); /annotation lost one.
    assert undecodable_topics(chunks) == ["/annotation", DEX1_STATE]


def test_the_census_verdict_is_stamped_on_episode() -> None:
    flagged = census_chunk([DEX1_STATE])
    assert flagged.entity_path == "/episode"
    assert flagged.is_static
    assert _cell(flagged, "has_undecodable") is True
    assert _cell(flagged, "undecodable_topics") == DEX1_STATE

    clean = census_chunk([])
    assert _cell(clean, "has_undecodable") is False
    assert _cell(clean, "undecodable_topics") == ""
