"""Tests for the conversion: the pure transforms, and a synthesized-file round trip through both layers."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
from rerun.experimental import Hdf5Reader, RrdReader

from libero import properties_layer
from libero.base_layer import (
    convert_demo,
    demo_keys,
    discover_cameras,
    flip_vertical,
    to_scalars,
)

NUM_STEPS = 3
HEIGHT, WIDTH = 4, 2


def test_flip_reverses_rows_and_keeps_pixels_intact() -> None:
    image = np.arange(HEIGHT * WIDTH * 3, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    arr = pa.array([image.reshape(-1)], type=pa.list_(pa.uint8()))
    flipped = flip_vertical(HEIGHT, WIDTH)(arr)
    result = np.asarray(flipped.to_pylist()[0], dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(result, image[::-1])


def test_scalars_wrap_plain_and_list_columns_alike() -> None:
    assert to_scalars(pa.array([0, 1], type=pa.uint8())).to_pylist() == [[0.0], [1.0]]
    assert to_scalars(pa.array([[1.0, 2.0]], type=pa.list_(pa.float64()))).to_pylist() == [[1.0, 2.0]]


def _write_demo(data: h5py.Group, demo: str, rng: np.random.Generator) -> None:
    group = data.create_group(demo)
    obs = group.create_group("obs")
    obs["agentview_rgb"] = rng.integers(0, 255, size=(NUM_STEPS, HEIGHT, WIDTH, 3), dtype=np.uint8)
    obs["eye_in_hand_rgb"] = rng.integers(0, 255, size=(NUM_STEPS, HEIGHT, WIDTH, 3), dtype=np.uint8)
    obs["ee_pos"] = rng.random((NUM_STEPS, 3))
    obs["ee_ori"] = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])  # last magnitude > π
    obs["joint_states"] = rng.random((NUM_STEPS, 7))
    obs["gripper_states"] = rng.random((NUM_STEPS, 2))
    obs["ee_states"] = rng.random((NUM_STEPS, 6))  # dropped
    group["actions"] = rng.random((NUM_STEPS, 7))
    group["rewards"] = np.array([0, 0, 1], dtype=np.uint8)
    group["dones"] = np.array([0, 0, 1], dtype=np.uint8)
    group["states"] = rng.random((NUM_STEPS, 5))  # dropped
    group["robot_states"] = rng.random((NUM_STEPS, 9))  # dropped
    group.attrs["model_file"] = f"<mujoco model={demo!r}/>"
    group.attrs["init_state"] = group["states"][0]
    group.attrs["num_samples"] = NUM_STEPS


def _write_fixture(path: Path) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["env_name"] = "Libero_Test_Env"
        data.attrs["problem_info"] = json.dumps({"language_instruction": "turn on the stove"})
        data.attrs["macros_image_convention"] = "opengl"
        data.attrs["num_demos"] = 4
        for demo in ("demo_0", "demo_1", "demo_2", "demo_10"):
            _write_demo(data, demo, rng)


def _chunks_by_entity(rrd_path: Path) -> dict[str, list[pa.RecordBatch]]:
    reader = RrdReader(str(rrd_path))
    entries = reader.recordings()
    assert len(entries) == 1
    batches: dict[str, list[pa.RecordBatch]] = {}
    for chunk in reader.stream(store=entries[0]):
        batches.setdefault(chunk.entity_path, []).append(chunk.to_record_batch())
    return batches


def _column(batches: list[pa.RecordBatch], name: str) -> pa.Array:
    arrays = [batch.column(name) for batch in batches if name in batch.schema.names]
    assert arrays, name
    return pa.concat_arrays([
        array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array for array in arrays
    ])


def _rows(batches: list[pa.RecordBatch], name: str) -> list[object]:
    """A component column's row values — each row arrives as a one-instance batch and unwraps to its value."""
    return [row[0] for row in _column(batches, name).to_pylist()]


def test_demo_keys_sort_numerically(tmp_path: Path) -> None:
    fixture = tmp_path / "kitchen_demo.hdf5"
    _write_fixture(fixture)
    assert demo_keys(Hdf5Reader(fixture)) == ["demo_0", "demo_1", "demo_2", "demo_10"]


def test_cameras_are_discovered_by_shape(tmp_path: Path) -> None:
    fixture = tmp_path / "kitchen_demo.hdf5"
    _write_fixture(fixture)
    cameras = discover_cameras(Hdf5Reader(fixture), "demo_0")
    assert [(camera.name, camera.height, camera.width) for camera in cameras] == [
        ("agentview", HEIGHT, WIDTH),
        ("eye_in_hand", HEIGHT, WIDTH),
    ]


def test_base_layer_round_trip(tmp_path: Path) -> None:
    """The synthesized file converts into exactly the documented entities, values intact."""
    fixture = tmp_path / "kitchen_demo.hdf5"
    _write_fixture(fixture)
    reader = Hdf5Reader(fixture)
    out = convert_demo(reader, "suite/kitchen", "demo_0", tmp_path / "rrds")
    assert out == tmp_path / "rrds" / "base" / "suite" / "kitchen" / "demo_0.rrd"

    entries = RrdReader(str(out)).recordings()
    assert entries[0].recording_id == "suite/kitchen/demo_0"
    assert entries[0].application_id == "libero"

    batches = _chunks_by_entity(out)
    assert sorted(batches) == [
        "/action",
        "/camera/agentview",
        "/camera/eye_in_hand",
        "/done",
        "/replay/init_state",
        "/replay/model_file",
        "/reward",
        "/robot/ee_ori",
        "/robot/ee_pos",
        "/robot/gripper_states",
        "/robot/joint_states",
        "/task/instruction",
    ]

    with h5py.File(fixture) as file:
        source_images = np.asarray(file["data/demo_0/obs/agentview_rgb"])
        source_ori = np.asarray(file["data/demo_0/obs/ee_ori"])
        source_init = np.asarray(file["data/demo_0"].attrs["init_state"])

    buffer_column = _column(batches["/camera/agentview"], "Image:buffer")
    assert not buffer_column.type.value_type.value_field.nullable, "viewers reject nullable blob elements"
    buffers = _rows(batches["/camera/agentview"], "Image:buffer")
    converted = np.asarray(buffers, dtype=np.uint8).reshape(NUM_STEPS, HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(converted, source_images[:, ::-1])

    rotvecs = np.asarray(_rows(batches["/robot/ee_ori"], "Scalars:scalars"))
    np.testing.assert_array_equal(rotvecs, source_ori)
    assert _column(batches["/robot/ee_ori"], "SeriesLines:names").to_pylist() == [["rx", "ry", "rz"]]

    sim_time = _column(batches["/reward"], "sim_time")
    assert sim_time.cast(pa.int64()).to_pylist() == [250_000_000 + 50_000_000 * i for i in range(NUM_STEPS)]
    assert _rows(batches["/reward"], "Scalars:scalars") == [[0.0], [0.0], [1.0]]
    assert _rows(batches["/reward"], "SeriesLines:names") == ["reward"]
    assert _rows(batches["/done"], "SeriesLines:names") == ["done"]

    assert _rows(batches["/task/instruction"], "TextDocument:text") == ["turn on the stove"]
    assert _rows(batches["/replay/model_file"], "model_file") == ["<mujoco model='demo_0'/>"]
    np.testing.assert_allclose(np.asarray(_rows(batches["/replay/init_state"], "init_state")[0]), source_init)


def test_properties_layer_round_trip(tmp_path: Path) -> None:
    fixture = tmp_path / "KITCHEN_SCENE3_kitchen_demo.hdf5"
    _write_fixture(fixture)
    reader = Hdf5Reader(fixture)
    facts = properties_layer.task_facts(reader, "suite/KITCHEN_SCENE3_kitchen")
    assert facts.scene == "KITCHEN_SCENE3"
    assert facts.language == "turn on the stove"
    assert facts.source_file == "suite/KITCHEN_SCENE3_kitchen_demo.hdf5"

    out = properties_layer.convert_demo(reader, facts, "demo_0", tmp_path / "rrds")
    entries = RrdReader(str(out)).recordings()
    assert entries[0].recording_id == "suite/KITCHEN_SCENE3_kitchen/demo_0"

    batches = _chunks_by_entity(out)
    properties = next(batch for entity, batch in batches.items() if "episode" in entity)
    names = {name for batch in properties for name in batch.schema.names}
    assert {name.rsplit(":", 1)[-1] for name in names} >= {
        "suite",
        "task",
        "scene",
        "language",
        "env_name",
        "num_samples",
        "source_file",
    }
