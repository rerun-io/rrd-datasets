"""
Tests for the URDF model, the joint mapping the FK layer feeds it, and the model/FK layer split.

The motor-to-joint mapping is not in the data: `JOINT_NAMES_URDF` asserts that motor index i is
URDF joint `<name>_joint`, and FK writes a confident wrong pose if that drifts — a reordered or
renamed joint in the model, or an SDK change in `UrdfTree`. These tests hold the model, the
mapping, and the FK output shape together; only the layer round trip needs an MCAP.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from rerun.experimental import RrdReader
from rerun.urdf import UrdfTree

from hiw_500.base_layer import N_JOINTS, episode_from_mcap
from hiw_500.odom_layer import ROOT_FRAME
from hiw_500.urdf_layer import (
    JOINT_NAMES_URDF,
    MODEL_RECORDING_ID,
    TRANSFORMS,
    convert_episode,
    convert_model,
    load_urdf,
)

IDENTITY_QUATERNION = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def urdf() -> UrdfTree:
    """The tree exactly as `urdf_layer.main` builds it."""
    return load_urdf()


def test_the_model_rrd_is_a_valid_asset(urdf: UrdfTree, tmp_path: Path) -> None:
    """The catalog rejects temporal chunks in an asset, so the model rrd must stay static only."""
    reader = RrdReader(str(convert_model(urdf, tmp_path)))
    (entry,) = reader.recordings()
    assert entry.recording_id == MODEL_RECORDING_ID
    chunks = list(reader.stream(store=entry))
    assert chunks
    assert all(chunk.is_static for chunk in chunks)
    assert any("visual_geometries" in chunk.entity_path for chunk in chunks)
    assert not any("collision_geometries" in chunk.entity_path for chunk in chunks)


def test_the_episode_layer_leaves_the_meshes_to_the_shared_model(
    urdf: UrdfTree, cached_episode: Path, tmp_path: Path
) -> None:
    """The 24 MB of G1 meshes ship once per dataset, so an episode's layer holds only its FK rows."""
    out = convert_episode(urdf, episode_from_mcap(cached_episode), tmp_path)
    reader = RrdReader(str(out))
    (entry,) = reader.recordings()
    assert {chunk.entity_path for chunk in reader.stream(store=entry)} == {TRANSFORMS}


def test_the_29_motors_are_the_urdf_revolute_joints_in_motor_order(urdf: UrdfTree) -> None:
    """Motor index i maps to revolute joint i; a reordered model would silently cross-wire the pose."""
    revolute = [joint.name for joint in urdf.joints() if joint.joint_type == "revolute"]
    assert len(revolute) == N_JOINTS
    assert revolute == JOINT_NAMES_URDF


def test_the_finger_joints_are_prismatic_and_outside_the_motor_mapping(urdf: UrdfTree) -> None:
    """The Dex1 fingers stay at rest, so they must never appear among the driven joints."""
    prismatic = {joint.name for joint in urdf.joints() if joint.joint_type == "prismatic"}
    assert len(prismatic) == 4
    assert not prismatic & set(JOINT_NAMES_URDF)


def test_the_tree_roots_at_the_frame_the_odom_layer_extends(urdf: UrdfTree) -> None:
    """The odom layer logs an `odom -> ROOT_FRAME` edge; a renamed root strands the robot at the origin."""
    assert urdf.root_link().name == ROOT_FRAME


def test_fk_at_rest_reproduces_each_joints_origin(urdf: UrdfTree) -> None:
    """Zero angles must return the model's rest pose — the check that catches unit or order drift."""
    names = pa.array([JOINT_NAMES_URDF])
    values = pa.array([[0.0] * N_JOINTS])
    (batch,) = urdf.compute_joint_transform_batches(names, values)
    entries = {entry["child_frame"]: entry for entry in batch.as_py()}
    assert len(entries) == N_JOINTS
    for joint in urdf.joints():
        if joint.joint_type != "revolute":
            continue
        entry = entries[joint.child_link]
        assert entry["parent_frame"] == joint.parent_link
        assert tuple(entry["translation"]) == pytest.approx(joint.origin_xyz)
        if joint.origin_rpy == (0.0, 0.0, 0.0):
            assert entry["quaternion"] == pytest.approx(IDENTITY_QUATERNION)
