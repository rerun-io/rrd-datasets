"""Tests for the conversion: the pure transforms, and a synthesized-file round trip through the layers."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pytest
from conftest import ENV_ARGS, HEIGHT, MODEL_FILE, NUM_STEPS, WIDTH, write_fixture
from rerun.experimental import Hdf5Reader, RrdReader

from libero import properties_layer
from libero.base_layer import convert_demo, demo_keys, discover_cameras, flip_vertical
from libero.episodes import LOCAL_DIR


def test_flip_reverses_rows_and_keeps_pixels_intact() -> None:
    image = np.arange(HEIGHT * WIDTH * 3, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    arr = pa.array([image.reshape(-1)], type=pa.list_(pa.uint8()))
    flipped = flip_vertical(HEIGHT, WIDTH)(arr)
    result = np.asarray(flipped.to_pylist()[0], dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(result, image[::-1])


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
    write_fixture(fixture)
    assert demo_keys(Hdf5Reader(fixture)) == ["demo_0", "demo_1", "demo_2", "demo_10"]


def test_cameras_are_discovered_by_shape(tmp_path: Path) -> None:
    fixture = tmp_path / "kitchen_demo.hdf5"
    write_fixture(fixture)
    cameras = discover_cameras(Hdf5Reader(fixture), "demo_0")
    assert [(camera.name, camera.height, camera.width) for camera in cameras] == [
        ("agentview", HEIGHT, WIDTH),
        ("eye_in_hand", HEIGHT, WIDTH),
    ]


def test_base_layer_round_trip(tmp_path: Path) -> None:
    """The synthesized file converts into the reflected entities, values and dtypes intact."""
    fixture = tmp_path / "kitchen_demo.hdf5"
    write_fixture(fixture)
    reader = Hdf5Reader(fixture)
    out = convert_demo(reader, "suite/kitchen", "demo_0", tmp_path / "rrds")
    assert out == tmp_path / "rrds" / "base" / "suite" / "kitchen__demo_0.rrd"

    entries = RrdReader(str(out)).recordings()
    assert entries[0].recording_id == "suite/kitchen__demo_0"
    assert entries[0].application_id == "libero"

    batches = _chunks_by_entity(out)
    assert sorted(batches) == [
        "/__hdf5_properties",
        "/camera/agentview",
        "/camera/eye_in_hand",
        "/demo",
        "/demo/__hdf5_properties",
        "/demo/obs",
        "/task/instruction",
    ]

    with h5py.File(fixture) as file:
        demo = file["data/demo_0"]
        source_images = np.asarray(demo["obs/agentview_rgb"])
        source_ori = np.asarray(demo["obs/ee_ori"])
        source_states = np.asarray(demo["states"])
        source_init = np.asarray(demo.attrs["init_state"])

    buffer_column = _column(batches["/camera/agentview"], "Image:buffer")
    assert not buffer_column.type.value_type.value_field.nullable, "viewers reject nullable blob elements"
    buffers = _rows(batches["/camera/agentview"], "Image:buffer")
    converted = np.asarray(buffers, dtype=np.uint8).reshape(NUM_STEPS, HEIGHT, WIDTH, 3)
    np.testing.assert_array_equal(converted, source_images[:, ::-1])

    obs_columns = {name for batch in batches["/demo/obs"] for name in batch.schema.names}
    assert {"ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states"} <= obs_columns
    assert not {"agentview_rgb", "eye_in_hand_rgb"} & obs_columns, "the bottom-up blobs leave with the flip"
    np.testing.assert_array_equal(np.asarray(_rows(batches["/demo/obs"], "ee_ori")), source_ori)

    root_columns = {name for batch in batches["/demo"] for name in batch.schema.names}
    assert {"actions", "dones", "rewards", "robot_states", "states"} <= root_columns
    np.testing.assert_array_equal(np.asarray(_rows(batches["/demo"], "states")), source_states)
    # The state vector's width is per scene; as a variable-length list every demo fits one catalog schema.
    assert pa.types.is_list(_column(batches["/demo"], "states").type.value_type)
    assert pa.types.is_fixed_size_list(_column(batches["/demo"], "actions").type.value_type), "other widths stay fixed"
    assert pa.types.is_uint8(_column(batches["/demo"], "rewards").type.value_type), "dtypes survive"
    assert _rows(batches["/demo"], "rewards") == [0, 0, 1]

    sim_time = _column(batches["/demo"], "sim_time")
    assert sim_time.cast(pa.int64()).to_pylist() == [250_000_000 + 50_000_000 * i for i in range(NUM_STEPS)]

    attrs = batches["/demo/__hdf5_properties"]
    assert _rows(attrs, "model_file") == [MODEL_FILE]
    assert _rows(attrs, "num_samples") == [NUM_STEPS]
    np.testing.assert_allclose(np.asarray(_rows(attrs, "init_state")[0]), source_init)
    assert pa.types.is_list(_column(attrs, "init_state").type.value_type)

    task_attrs = batches["/__hdf5_properties"]
    task_columns = {name for batch in task_attrs for name in batch.schema.names}
    assert task_columns >= {
        "bddl_file_name",
        "env_args",
        "env_name",
        "macros_image_convention",
        "num_demos",
        "problem_info",
        "tag",
        "total",
    }
    assert _rows(task_attrs, "total") == [4 * NUM_STEPS]
    assert _rows(task_attrs, "env_args") == [json.dumps(ENV_ARGS)], "the raw JSON stays beside its parsed form"
    assert _rows(task_attrs, "env_args:parsed") == [ENV_ARGS]
    assert _rows(task_attrs, "problem_info:parsed") == [
        {"problem_name": "libero_test", "language_instruction": "turn on the stove"}
    ]
    assert _rows(batches["/task/instruction"], "TextDocument:text") == ["turn on the stove"]


# Index and timeline columns ride in every chunk; they are not datasets.
_INDEX_COLUMNS = {"row_index", "sim_time"}


def _components(batches: list[pa.RecordBatch]) -> list[str]:
    """The component columns of an entity, in first-seen order."""
    names: dict[str, None] = {}
    for batch in batches:
        for name in batch.schema.names:
            if name not in _INDEX_COLUMNS and not name.startswith("rerun.controls"):
                names.setdefault(name)
    return list(names)


def _ordered(batches: list[pa.RecordBatch], name: str) -> pa.Array:
    """A column in `row_index` order, one value per row, gathered from the chunks that carry it."""
    parts = [(batch.column("row_index"), batch.column(name)) for batch in batches if name in batch.schema.names]
    assert parts, name
    index = np.concatenate([_plain(rows).to_numpy() for rows, _ in parts])
    values = pa.concat_arrays([_plain(column) for _, column in parts])
    _, first_seen = np.unique(index, return_index=True)
    return values.take(pa.array(first_seen))


def _plain(array: pa.Array | pa.ChunkedArray) -> pa.Array:
    return array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array


def _dataset(batches: list[pa.RecordBatch], name: str) -> np.ndarray:
    """A reflected dataset as the array the reader saw: `(N, K)` for list rows, `(N,)` for scalars."""
    instances = _ordered(batches, name).flatten()
    if pa.types.is_fixed_size_list(instances.type) or pa.types.is_list(instances.type):
        flat = np.asarray(instances.flatten().to_numpy(zero_copy_only=False))
        return flat.reshape(len(instances), -1)
    return np.asarray(instances.to_numpy(zero_copy_only=False))


def _camera_dataset(batches: list[pa.RecordBatch]) -> np.ndarray:
    """The source frames back from the upright `Image` buffers, shaped by the static `Image:format`."""
    image_format = _rows(batches, "Image:format")[0]
    assert isinstance(image_format, dict)
    frames = _ordered(batches, "Image:buffer").flatten()
    pixels = np.asarray(frames.flatten().to_numpy(zero_copy_only=False))
    upright = pixels.reshape(len(frames), image_format["height"], image_format["width"], 3)
    return upright[:, ::-1]


def _write_attrs(target: h5py.Group, batches: list[pa.RecordBatch]) -> None:
    """The reflected attributes back onto an HDF5 object; the parsed JSON copies are derived and stay out."""
    for name in _components(batches):
        if name.endswith(":parsed"):
            continue
        value = _rows(batches, name)[0]
        target.attrs[name] = np.asarray(value) if isinstance(value, list) else value


def _rebuild(rrd: Path, hdf5: Path, demo: str) -> None:
    """The inverse of `convert_demo`: the base RRD written back as a one-demo task file."""
    batches = _chunks_by_entity(rrd)
    with h5py.File(hdf5, "w") as file:
        data = file.create_group("data")
        _write_attrs(data, batches["/__hdf5_properties"])
        group = data.create_group(demo)
        _write_attrs(group, batches["/demo/__hdf5_properties"])
        for entity, chunks in batches.items():
            if entity.startswith("/demo") and not entity.endswith("__hdf5_properties"):
                target = group if entity == "/demo" else group.require_group(entity.removeprefix("/demo/"))
                for name in _components(chunks):
                    target[name] = _dataset(chunks, name)
            elif entity.startswith("/camera/"):
                # Cameras came out of `obs/<name>_rgb`; the entity keeps the name without the suffix.
                group.require_group("obs")[f"{entity.removeprefix('/camera/')}_rgb"] = _camera_dataset(chunks)


def _datasets(group: h5py.Group) -> dict[str, h5py.Dataset]:
    found: dict[str, h5py.Dataset] = {}
    for name, item in group.items():
        if isinstance(item, h5py.Dataset):
            found[name] = item
        elif isinstance(item, h5py.Group):
            found.update({f"{name}/{path}": dataset for path, dataset in _datasets(item).items()})
    return found


def _assert_same_attrs(expected: h5py.AttributeManager, actual: h5py.AttributeManager) -> None:
    assert sorted(actual) == sorted(expected)
    for key in expected:
        want, got = np.asarray(expected[key]), np.asarray(actual[key])
        assert got.dtype == want.dtype, key
        np.testing.assert_array_equal(got, want)


def _assert_same_demo(source: h5py.File, copy: h5py.File, demo: str) -> None:
    """The rebuilt file matches the source in tree, dtypes, shapes, values and attributes."""
    _assert_same_attrs(source["data"].attrs, copy["data"].attrs)
    _assert_same_attrs(source[f"data/{demo}"].attrs, copy[f"data/{demo}"].attrs)
    expected, actual = _datasets(source[f"data/{demo}"]), _datasets(copy[f"data/{demo}"])
    assert sorted(actual) == sorted(expected)
    for path, want in expected.items():
        got = actual[path]
        assert (got.dtype, got.shape) == (want.dtype, want.shape), path
        np.testing.assert_array_equal(got[()], want[()])


def _assert_sim_time_is_the_mujoco_clock(batches: list[pa.RecordBatch]) -> None:
    """The synthesized timeline agrees with `states[:, 0]`, MuJoCo's own clock, on every row."""
    sim_time = _ordered(batches, "sim_time").cast(pa.int64()).to_numpy() / 1e9
    np.testing.assert_allclose(sim_time, _dataset(batches, "states")[:, 0], atol=1e-6)


def test_the_base_layer_rebuilds_the_source_file(tmp_path: Path) -> None:
    fixture = tmp_path / "kitchen_demo.hdf5"
    write_fixture(fixture)
    out = convert_demo(Hdf5Reader(fixture), "suite/kitchen", "demo_0", tmp_path / "rrds")
    rebuilt = tmp_path / "rebuilt.hdf5"
    _rebuild(out, rebuilt, "demo_0")
    with h5py.File(fixture) as source, h5py.File(rebuilt) as copy:
        _assert_same_demo(source, copy, "demo_0")
    _assert_sim_time_is_the_mujoco_clock(_chunks_by_entity(out)["/demo"])


SAMPLE_TASK_FILE = LOCAL_DIR / "libero_goal" / "turn_on_the_stove_demo.hdf5"


@pytest.mark.skipif(not SAMPLE_TASK_FILE.exists(), reason="sample task file not downloaded")
def test_a_downloaded_demo_rebuilds(tmp_path: Path) -> None:
    out = convert_demo(Hdf5Reader(SAMPLE_TASK_FILE), "libero_goal/turn_on_the_stove", "demo_0", tmp_path / "rrds")
    rebuilt = tmp_path / "rebuilt.hdf5"
    _rebuild(out, rebuilt, "demo_0")
    with h5py.File(SAMPLE_TASK_FILE) as source, h5py.File(rebuilt) as copy:
        _assert_same_demo(source, copy, "demo_0")
    _assert_sim_time_is_the_mujoco_clock(_chunks_by_entity(out)["/demo"])


def test_properties_layer_round_trip(tmp_path: Path) -> None:
    fixture = tmp_path / "KITCHEN_SCENE3_kitchen_demo.hdf5"
    write_fixture(fixture)
    reader = Hdf5Reader(fixture)
    facts = properties_layer.task_facts(reader, "suite/KITCHEN_SCENE3_kitchen")
    assert facts.scene == "KITCHEN_SCENE3"
    assert facts.language == "turn on the stove"
    assert facts.source_file == "suite/KITCHEN_SCENE3_kitchen_demo.hdf5"

    out = properties_layer.convert_demo(reader, facts, "demo_0", tmp_path / "rrds")
    entries = RrdReader(str(out)).recordings()
    assert entries[0].recording_id == "suite/KITCHEN_SCENE3_kitchen__demo_0"

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


def test_scene_prefix_covers_multi_word_scenes(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.hdf5"
    write_fixture(fixture)
    reader = Hdf5Reader(fixture)

    def scene(task: str) -> str:
        return properties_layer.task_facts(reader, task).scene

    assert scene("libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup") == "LIVING_ROOM_SCENE1"
    assert scene("libero_10/KITCHEN_SCENE10_put_the_yellow_mug_on_the_plate") == "KITCHEN_SCENE10"
    assert scene("libero_goal/turn_on_the_stove") == ""
