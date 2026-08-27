"""
Tests for the base layer's calibration passthrough, series labels, and channel census — no MCAP, no network.

Calibration files are archived verbatim, so both halves of that promise are guarded here: the file
text survives byte for byte, and the per-serial wrist files keep one entity path across rigs. The
census is what makes silently dropped messages visible, so it runs against the summary's counts and
synthetic chunks shaped as `McapReader` emits them: decoded rows on the topic entity, one chunk
family per component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import yaml
from rerun.experimental import Chunk

from hiw_500.base_layer import (
    CALIBRATION_ARCHETYPE,
    G1_JOINT_NAMES,
    MCAP_PROPERTY,
    PROPERTY_PATH,
    Episode,
    EpisodeInfo,
    _calibration_leaves,
    calibration_chunks,
    calibration_components,
    census_chunk,
    has_ir,
    mcap_chunk,
    names_chunks,
    undecodable_topics,
)

HEAD_YAML = """\
# head stereo pair
baseline: 60.530000000000001
image_size: [640, 480]
"""


def _wrist_json(serial: str) -> str:
    """A wrist sidecar carrying its own serial, which is where the serial now comes from."""
    return json.dumps({"color": {"intrinsics": {"fx": 435.96734619140625}}, "serial_number": serial})


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


def _list_cell(chunk: Chunk, column: str) -> list[object]:
    """The one list in a one-row chunk's list-valued component column."""
    rows: list[list[object]] = chunk.to_record_batch().column(column).to_pylist()
    (row,) = rows
    return row


def _by_entity(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {chunk.entity_path: chunk for chunk in chunks}


def test_calibration_values_land_as_named_components(tmp_path: Path) -> None:
    """Each value gets its own component, named by its path in the file."""
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
    assert _cell(head, "CalibrationFile:baseline") == 60.530000000000001
    assert _list_cell(head, "CalibrationFile:image_size") == [640, 480]
    assert _cell(head, "CalibrationFile:path") == "calibration/params/head_camera_params.yaml"

    # A suffix with no loader keeps its text rather than being dropped.
    assert _cell(chunks["/calibration/notes"], "CalibrationFile:text") == "rig A\n"


def test_wrist_calibrations_get_one_entity_path_across_rigs(tmp_path: Path) -> None:
    """Serials differ per rig, so they ride in a component while the entity path stays put."""
    episode = _episode(
        tmp_path,
        {
            "calibration/params/camera_409122273272.json": _wrist_json("409122273272"),
            "calibration/params/camera_323622270214.json": _wrist_json("323622270214"),
        },
    )
    chunks = _by_entity(calibration_chunks(episode))
    assert set(chunks) == {"/calibration/params/wrist_camera1", "/calibration/params/wrist_camera2"}

    first = chunks["/calibration/params/wrist_camera1"]
    # Numbered by sorted filename, never a left/right guess; the serial rides in the file's own field.
    assert _cell(first, "CalibrationFile:serial_number") == "323622270214"
    assert _cell(first, "CalibrationFile:path") == "calibration/params/camera_323622270214.json"
    assert _cell(first, "CalibrationFile:color.intrinsics.fx") == 435.96734619140625
    second = chunks["/calibration/params/wrist_camera2"]
    assert _cell(second, "CalibrationFile:serial_number") == "409122273272"


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
    with_ir = _episode(tmp_path / "with", {"calibration/params/camera_409122273272.json": _wrist_json("409122273272")})
    without = _episode(tmp_path / "without", {"calibration/params/head_camera_params.yaml": HEAD_YAML})
    assert has_ir(with_ir)
    assert not has_ir(without)
    assert not has_ir(_episode(tmp_path / "bare", {"info.json": "{}"}))


def test_joint_names_ride_statically_beside_the_structs() -> None:
    """Element i of the motor arrays is `joint_names[i]`."""
    chunks = _by_entity(names_chunks())
    assert set(chunks) == {"/stamped/lowstate", "/stamped/lowcmd"}
    for chunk in chunks.values():
        assert chunk.is_static
        assert _list_cell(chunk, "joint_names") == G1_JOINT_NAMES


LOWSTATE = "/stamped/lowstate"
DEX1_STATE = "/stamped/dex1/left/state"


def _decoded_chunk(entity: str, rows: int) -> Chunk:
    """`rows` decoded messages on a topic entity."""
    return Chunk.from_columns(
        entity,
        indexes=[rr.TimeColumn("message_publish_time", timestamp=np.arange(rows).astype("datetime64[ns]"))],
        columns=rr.Scalars.columns(scalars=np.zeros(rows)),
    )


def _derived_chunk(entity: str, rows: int) -> Chunk:
    """A second chunk family on the entity, as a lens or a second reader adds: one row per message again."""
    return Chunk.from_columns(
        entity,
        indexes=[rr.TimeColumn("message_publish_time", timestamp=np.arange(rows).astype("datetime64[ns]"))],
        columns=rr.TextDocument.columns(text=["x"] * rows),
    )


def test_an_episode_that_decoded_everything_flags_nothing() -> None:
    expected = {LOWSTATE: 5, "/annotation": 2}
    chunks = [
        _decoded_chunk(LOWSTATE, 3),  # a topic's rows may arrive in several chunks
        _decoded_chunk(LOWSTATE, 2),
        _decoded_chunk("/annotation", 2),
    ]
    assert undecodable_topics(expected, chunks) == []


def test_topics_short_of_their_channel_count_are_flagged() -> None:
    expected = {LOWSTATE: 5, "/annotation": 2, DEX1_STATE: 5126}
    chunks = [_decoded_chunk(LOWSTATE, 5), _decoded_chunk("/annotation", 1)]
    # dex1 decoded none of its 5126 messages (the Feb 2026 payload mismatch); /annotation lost one.
    assert undecodable_topics(expected, chunks) == ["/annotation", DEX1_STATE]


def test_a_second_component_on_the_entity_is_not_more_messages() -> None:
    expected = {LOWSTATE: 4}
    chunks = [_decoded_chunk(LOWSTATE, 2), _derived_chunk(LOWSTATE, 2)]
    assert undecodable_topics(expected, chunks) == [LOWSTATE]


def test_the_census_verdict_is_an_episode_property() -> None:
    flagged = census_chunk([DEX1_STATE])
    assert flagged.entity_path == PROPERTY_PATH
    assert flagged.is_static
    assert _cell(flagged, "has_undecodable") is True
    assert _list_cell(flagged, "undecodable_topics") == [DEX1_STATE]

    clean = census_chunk([])
    assert _cell(clean, "has_undecodable") is False
    assert _list_cell(clean, "undecodable_topics") == []


def test_the_mcap_header_is_an_episode_property() -> None:
    chunk = mcap_chunk("ros2", "libmcap 0.8.0", [""])
    assert chunk.entity_path == f"/__properties/{MCAP_PROPERTY}"
    assert chunk.is_static
    assert _cell(chunk, "profile") == "ros2"
    assert _cell(chunk, "library") == "libmcap 0.8.0"
    assert _list_cell(chunk, "compression") == ["none"]


# A sidecar covering every leaf kind the vendors use: nested dicts, a matrix, flat lists of floats
# and ints, an empty list, and the three scalar types.
ROUND_TRIP_YAML = """\
camera_matrix_left:
- [316.297, 0.0, 331.734]
- [0.0, 316.542, 228.072]
- [0.0, 0.0, 1.0]
dist_coeffs_left: [0.13826, -0.254975, -0.0167411]
image_size: [640, 480]
baseline: 59.7295
rms_error: 3.34749
success: true
notes: rig A
"""
ROUND_TRIP_JSON = json.dumps({
    "color": {"intrinsics": {"fx": 435.96734619140625, "width": 640, "model": "brown_conrady", "coeffs": []}},
    "ir1": {"extrinsics_to_color": {"rotation": [1.0, 0.0, 0.0], "translation": [9.87e-06, 1e-05, 1e-05]}},
    "serial_number": "409122273272",
})


def _rebuild(leaves: dict[str, Any]) -> dict[str, Any]:
    """The inverse of the dotted naming: every `a.b.c` name back into nested dicts."""
    root: dict[str, Any] = {}
    for name, value in leaves.items():
        node = root
        *parents, last = name.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return root


def test_a_calibration_sidecar_survives_the_round_trip(tmp_path: Path) -> None:
    """Names rebuild the source exactly, and each leaf reaches its component whole, as a one-row batch."""
    for name, text in (("head.yaml", ROUND_TRIP_YAML), ("camera_409122273272.json", ROUND_TRIP_JSON)):
        path = tmp_path / name
        path.write_text(text)
        source = yaml.safe_load(text) if name.endswith(".yaml") else json.loads(text)

        leaves = _calibration_leaves(source)
        assert _rebuild(leaves) == source

        components = calibration_components(path, Path("calibration/params") / name)
        for key, value in leaves.items():
            assert components[key].to_pylist() == [value], key
        assert components["path"].to_pylist() == [f"calibration/params/{name}"]


def test_every_component_hangs_off_the_calibration_archetype(tmp_path: Path) -> None:
    """One archetype for both vendor schemas, whatever the rig."""
    path = tmp_path / "head.yaml"
    path.write_text(ROUND_TRIP_YAML)
    components = calibration_components(path, Path("head.yaml"))
    chunk = Chunk.from_columns(
        "/calibration/params/head",
        indexes=[],
        columns=rr.DynamicArchetype.columns(archetype=CALIBRATION_ARCHETYPE, components=components),
    )
    named = [f.name for f in chunk.to_record_batch().schema if f.name.startswith(CALIBRATION_ARCHETYPE)]
    assert len(named) == len(components)
    assert f"{CALIBRATION_ARCHETYPE}:camera_matrix_left" in named


def test_a_matrix_keeps_its_shape_without_a_shape_component(tmp_path: Path) -> None:
    """A nested list is self-describing; a flattened one would need a shape sibling."""
    path = tmp_path / "head.yaml"
    path.write_text(ROUND_TRIP_YAML)
    components = calibration_components(path, Path("head.yaml"))
    assert not [name for name in components if name.endswith(".shape")]
    (matrix,) = components["camera_matrix_left"].to_pylist()
    assert matrix == [[316.297, 0.0, 331.734], [0.0, 316.542, 228.072], [0.0, 0.0, 1.0]]


def test_an_empty_list_keeps_a_typed_element(tmp_path: Path) -> None:
    """Inference reads an empty list as `list<null>`, which then drops the values of other episodes."""
    path = tmp_path / "camera_1.json"
    path.write_text(ROUND_TRIP_JSON)
    components = calibration_components(path, Path("camera_1.json"))
    assert pa.types.is_floating(components["color.intrinsics.coeffs"].type.value_type)


def test_a_leaf_arrow_has_no_type_for_keeps_its_text(tmp_path: Path) -> None:
    """A missing value or a YAML date must not fail the whole episode."""
    path = tmp_path / "odd.yaml"
    path.write_text("baseline:\ncalibrated_on: 2026-08-20\ncoeffs: [null, 1.0]\n")
    components = calibration_components(path, Path("odd.yaml"))
    assert components["baseline"].to_pylist() == [None]
    assert components["calibrated_on"].to_pylist() == ["2026-08-20"]
    assert components["coeffs"].to_pylist() == [[None, 1.0]]


def test_a_sidecar_carrying_its_own_path_is_an_error(tmp_path: Path) -> None:
    """`path` holds the source filename, so overwriting a vendor field with it would lose data."""
    path = tmp_path / "camera_1.json"
    path.write_text(json.dumps({"path": "vendor/original.json"}))
    with pytest.raises(ValueError, match="path"):
        calibration_components(path, Path("camera_1.json"))
