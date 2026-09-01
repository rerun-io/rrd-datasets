"""Tests for the URDF layer: the joint mapping, the scene offset, and a synthesized-file round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from conftest import ARM_POSE, BASE_POS, MODEL_FILE, NUM_STEPS, write_fixture
from rerun.experimental import Hdf5Reader, RrdReader

from libero import urdf_layer
from libero.urdf_layer import (
    JOINT_NAMES_URDF,
    N_JOINTS,
    TRANSFORMS,
    WORLD_FRAME,
    WORLD_FROM_BASE,
    base_pose,
    convert_demo,
    load_urdf,
    read_joint_values,
    transform_batches,
)

FINGER_LIMIT = 0.04


def _obs(joint_states: list[float], gripper_states: list[float]) -> pa.Array:
    """One row shaped like the reader's `/obs` struct column."""
    return pa.StructArray.from_arrays(
        [
            pa.array([joint_states], type=pa.list_(pa.float64(), 7)),
            pa.array([gripper_states], type=pa.list_(pa.float64(), 2)),
        ],
        names=["joint_states", "gripper_states"],
    )


def test_the_mapping_names_every_moving_joint_once() -> None:
    """A URDF revision that adds or renames a moving joint must not silently leave it at rest."""
    moving = {joint.name for joint in load_urdf().joints() if joint.joint_type != "fixed"}
    assert set(JOINT_NAMES_URDF) == moving
    assert len(JOINT_NAMES_URDF) == len(moving)
    prismatic = [joint.child_link for joint in load_urdf().joints() if joint.joint_type == "prismatic"]
    assert prismatic == ["fer_leftfinger", "fer_rightfinger"]


def test_the_second_finger_is_negated_into_the_urdf_range() -> None:
    """Robosuite signs the fingers against each other; both URDF joints open positively."""
    arm, fingers = read_joint_values(_obs(list(ARM_POSE), [0.031, -0.028]))
    assert arm[0] == pytest.approx(ARM_POSE)
    assert fingers[0] == pytest.approx([0.031, 0.028])
    assert ((0.0 <= fingers) & (fingers <= FINGER_LIMIT)).all()


def _finger_translations(opening: float) -> dict[str, np.ndarray]:
    """The two finger links' positions in the hand frame, at a given opening."""
    urdf = load_urdf()
    batch = transform_batches(urdf, _obs(list(ARM_POSE), [opening, -opening]))
    assert len(batch) == 1
    assert len(batch[0]) == N_JOINTS
    return {
        entry["child_frame"].as_py(): np.asarray(entry["translation"].as_py())
        for entry in batch[0]
        if "finger" in entry["child_frame"].as_py()
    }


def test_the_fingers_open_symmetrically() -> None:
    """
    The fingers travel apart because `fer_finger_joint2` is turned π about z.

    Losing that rotation — a wrong sign, or an SDK composing prismatic motion differently — collapses
    the gap to zero.
    """
    opening = 0.02
    fingers = _finger_translations(opening)
    left, right = fingers["fer_leftfinger"], fingers["fer_rightfinger"]
    assert left[1] == pytest.approx(opening, abs=1e-6)
    assert right[1] == pytest.approx(-opening, abs=1e-6)
    np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=1e-6)


def test_base_pose_reads_the_scene_offset() -> None:
    translation, quaternion = base_pose(MODEL_FILE)
    assert translation == pytest.approx(BASE_POS)
    assert quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])  # xyzw identity


def test_a_model_file_without_the_arm_fails_loudly() -> None:
    """A silent fallback would put the arm at the world origin, which reads as a plausible pose."""
    with pytest.raises(ValueError, match="robot0_base"):
        base_pose("<mujoco><worldbody/></mujoco>")


def _transform_edges(rrd_path: Path) -> tuple[set[tuple[str, str]], int]:
    """Every `parent -> child` frame pair in the layer, and the temporal transform row count."""
    reader = RrdReader(str(rrd_path))
    (entry,) = reader.recordings()
    edges: set[tuple[str, str]] = set()
    temporal_rows = 0
    for chunk in reader.stream(store=entry):
        batch = chunk.to_record_batch()
        if "Transform3D:child_frame" not in batch.schema.names:
            continue
        parents = [value[0] for value in batch.column("Transform3D:parent_frame").to_pylist()]
        children = [value[0] for value in batch.column("Transform3D:child_frame").to_pylist()]
        edges.update(zip(parents, children))
        if chunk.entity_path == TRANSFORMS and not chunk.is_static:
            temporal_rows += batch.num_rows
    return edges, temporal_rows


def test_urdf_layer_round_trip(tmp_path: Path) -> None:
    fixture = tmp_path / "suite_task_demo.hdf5"
    write_fixture(fixture)
    reader = Hdf5Reader(fixture)
    out = convert_demo(load_urdf(), reader, "suite/task", "demo_0", tmp_path / "rrds")

    entries = RrdReader(str(out)).recordings()
    assert [entry.recording_id for entry in entries] == ["suite/task__demo_0"]

    batches: dict[str, list[pa.RecordBatch]] = {}
    for chunk in RrdReader(str(out)).stream(store=entries[0]):
        batches.setdefault(chunk.entity_path, []).append(chunk.to_record_batch())
    assert TRANSFORMS in batches
    assert any(path.startswith(f"/{urdf_layer.ENTITY_PREFIX}/") and "visual_geometries" in path for path in batches)
    (edge,) = batches[WORLD_FROM_BASE]
    np.testing.assert_allclose(edge.column("Transform3D:translation").to_pylist()[0][0], BASE_POS, atol=1e-6)

    # The layer rides the base layer's timelines, or nothing lines up in time.
    names = {name for batch in batches[TRANSFORMS] for name in batch.schema.names}
    assert {"row_index", "sim_time"} <= names

    edges, temporal_rows = _transform_edges(out)
    assert temporal_rows == NUM_STEPS * N_JOINTS

    # Exactly one root, and every frame reaches it — otherwise links collapse onto the origin.
    parent_of = {child: parent for parent, child in edges}
    frames = set(parent_of) | set(parent_of.values())
    roots = {frame for frame in frames if frame not in parent_of}
    assert roots == {WORLD_FRAME}
    for frame in frames:
        seen: set[str] = set()
        current = frame
        while current in parent_of and current not in seen:
            seen.add(current)
            current = parent_of[current]
        assert current == WORLD_FRAME, f"{frame} does not reach {WORLD_FRAME}"
