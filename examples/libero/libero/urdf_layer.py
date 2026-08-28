"""
Build the URDF forward-kinematics layer for each LIBERO demo.

The vendored Franka `fer` model, posed by the joint columns of the base layer, written as its own
`.rrd` per demo under the base `recording_id`. Two lenses on one `Hdf5Reader` stream: the first
builds a `JointTransformBatch` per row, the second scatters each batch into per-joint `Transform3D`
rows. A static `world -> base` edge from the demo's MuJoCo XML stands the arm where the scene puts it.

Run:  pixi run -e libero convert-urdf              # every downloaded task file
      pixi run -e libero convert-urdf <task.hdf5>  # a single task file
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from rerun.experimental import (
    Chunk,
    DeriveLens,
    Hdf5Reader,
    LazyChunkStream,
    OptimizationProfile,
    Selector,
)
from rerun.urdf import UrdfTree

from libero.base_layer import APPLICATION_ID, RRD_ROOT, demo_keys, discover_cameras, task_files, with_sim_time
from libero.episodes import recording_id
from rrd_datasets_common.paths import layer_relpath

# pyarrow.compute ships incomplete stubs.
list_flatten = pc.list_flatten  # type: ignore[attr-defined]

URDF_PATH = Path(__file__).resolve().parents[1] / "urdf" / "fer" / "fer.urdf"
ENTITY_PREFIX = "urdf"
TRANSFORMS = f"/{ENTITY_PREFIX}/transforms"
WORLD_FROM_BASE = f"/{ENTITY_PREFIX}/world_from_base"
WORLD_FRAME = "world"

# The reader emits the demo's observations as one struct column.
OBS_ENTITY = "/obs"
OBS_STRUCT = "data"
BATCH = "rerun.urdf.JointTransformBatch"

# `obs/joint_states[i]` drives `fer_joint{i+1}`; the two fingers follow from `obs/gripper_states`.
ARM_JOINTS = [f"fer_joint{i + 1}" for i in range(7)]
FINGER_JOINTS = ["fer_finger_joint1", "fer_finger_joint2"]
JOINT_NAMES_URDF = ARM_JOINTS + FINGER_JOINTS
N_JOINTS = len(JOINT_NAMES_URDF)

# --------------------------------------------------------------------------------------
# prismatic finger workaround
# rerun-sdk 0.36.1 `compute_joint_transform_batches` slides a prismatic joint along its axis
# without the joint origin's rotation, so the finger transforms are built here. Delete this
# section once the upstream fix lands.
# --------------------------------------------------------------------------------------

# The SDK's batch entries have non-nullable elements; ours must match to concatenate with them.
_VEC3 = pa.list_(pa.field("item", pa.float32(), nullable=False), 3)
_QUAT = pa.list_(pa.field("item", pa.float32(), nullable=False), 4)


@dataclass(frozen=True)
class SlidingJoint:
    """A prismatic joint resolved to motion in its parent frame."""

    parent_frame: str
    child_frame: str
    origin: np.ndarray  # parent-frame translation at joint value 0
    slide: np.ndarray  # parent-frame displacement per unit of joint value
    quaternion: np.ndarray  # xyzw; a slide does not rotate the child


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw as a rotation matrix."""
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _rpy_quaternion(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw as an xyzw quaternion."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64) / 2.0
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def sliding_joints(urdf: UrdfTree) -> list[SlidingJoint]:
    """
    The finger joints, with the slide direction resolved through the joint origin's rotation.

    `fer_finger_joint2` is turned π about z so the fingers open apart; without that rotation the
    gap stays at zero.
    """
    joints = {joint.name: joint for joint in urdf.joints()}
    resolved = []
    for name in FINGER_JOINTS:
        joint = joints[name]
        rotation = _rpy_matrix(joint.origin_rpy)
        resolved.append(
            SlidingJoint(
                parent_frame=joint.parent_link,
                child_frame=joint.child_link,
                origin=np.asarray(joint.origin_xyz, dtype=np.float64),
                slide=rotation @ np.asarray(joint.axis, dtype=np.float64),
                quaternion=_rpy_quaternion(joint.origin_rpy),
            )
        )
    return resolved


def _sliding_entries(joints: list[SlidingJoint], values: np.ndarray, template: pa.DataType) -> pa.Array:
    """One batch entry per sliding joint per row, ordered row by row to match the arm entries."""
    rows = len(values)
    translations = np.stack(
        [joint.origin + np.outer(values[:, index], joint.slide) for index, joint in enumerate(joints)],
        axis=1,
    ).reshape(-1, 3)
    quaternions = np.tile(np.stack([joint.quaternion for joint in joints]), (rows, 1))
    return pa.StructArray.from_arrays(
        [
            pa.FixedSizeListArray.from_arrays(pa.array(translations.reshape(-1), type=pa.float32()), type=_VEC3),
            pa.FixedSizeListArray.from_arrays(pa.array(quaternions.reshape(-1), type=pa.float32()), type=_QUAT),
            pa.array([joint.parent_frame for joint in joints] * rows, type=pa.string()),
            pa.array([joint.child_frame for joint in joints] * rows, type=pa.string()),
        ],
        fields=list(template),
    )


# --------------------------------------------------------------------------------------
# forward kinematics
# --------------------------------------------------------------------------------------


def _obs_struct(msgs: pa.Array) -> pa.Array:
    """The observations arrive as a length-1 list per row; flatten to one struct per row."""
    return list_flatten(msgs) if pa.types.is_list(msgs.type) else msgs


def read_joint_values(msgs: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """The arm angles `(n, 7)` in radians and the finger openings `(n, 2)` in metres."""
    obs = _obs_struct(msgs)
    arm = np.asarray(obs.field("joint_states").to_numpy(zero_copy_only=False).tolist(), dtype=np.float64)
    grip = np.asarray(obs.field("gripper_states").to_numpy(zero_copy_only=False).tolist(), dtype=np.float64)
    # Both URDF finger joints are limited to [0, 0.04]; robosuite reports the second one negated.
    return arm, np.column_stack([grip[:, 0], -grip[:, 1]])


def transform_batches(urdf: UrdfTree, joints: list[SlidingJoint], msgs: pa.Array) -> pa.Array:
    """One `JointTransformBatch` per row: the arm solved by the SDK, the fingers built here."""
    arm_values, finger_values = read_joint_values(msgs)
    rows = len(arm_values)
    arm_offsets = pa.array(np.arange(rows + 1) * len(ARM_JOINTS), type=pa.int32())
    names = pa.ListArray.from_arrays(arm_offsets, pa.array(ARM_JOINTS * rows, type=pa.string()))
    values = pa.ListArray.from_arrays(arm_offsets, pa.array(arm_values.reshape(-1)))

    arm = urdf.compute_joint_transform_batches(names, values, clamp=False).flatten()
    fingers = _sliding_entries(joints, finger_values, arm.type)  # prismatic finger workaround

    # Interleave so each row holds its arm entries followed by its finger entries.
    arm_index = np.arange(rows)[:, None] * len(ARM_JOINTS) + np.arange(len(ARM_JOINTS))[None, :]
    finger_index = len(arm) + np.arange(rows)[:, None] * len(joints) + np.arange(len(joints))[None, :]
    order = np.concatenate([arm_index, finger_index], axis=1).reshape(-1)
    entries = pa.concat_arrays([arm, fingers]).take(pa.array(order))
    return pa.ListArray.from_arrays(pa.array(np.arange(rows + 1) * N_JOINTS, type=pa.int32()), entries)


# --------------------------------------------------------------------------------------
# scene placement
# --------------------------------------------------------------------------------------


def base_pose(model_file: str) -> tuple[list[float], list[float]]:
    """
    The scene's `world -> base` transform as `(translation, xyzw quaternion)`.

    Read from the `robot0_base` body of the demo's MuJoCo XML; the offset differs per scene.
    """
    body = ET.fromstring(model_file).find(".//body[@name='robot0_base']")
    if body is None:
        raise ValueError("model_file has no `robot0_base` body — cannot place the arm in the scene")
    translation = [float(value) for value in body.get("pos", "0 0 0").split()]
    w, x, y, z = (float(value) for value in body.get("quat", "1 0 0 0").split())  # MuJoCo stores wxyz
    return translation, [x, y, z, w]


def world_from_base_chunk(model_file: str) -> Chunk:
    """The static `world -> base` edge that stands the arm where the scene puts it."""
    translation, quaternion = base_pose(model_file)
    return Chunk.from_columns(
        WORLD_FROM_BASE,
        indexes=[],
        columns=rr.Transform3D.columns(
            translation=[translation],
            quaternion=[quaternion],
            parent_frame=[WORLD_FRAME],
            child_frame=["base"],
        ),
    )


def fk_stream(urdf: UrdfTree, joints: list[SlidingJoint], reader: Hdf5Reader, demo: str) -> LazyChunkStream:
    """Two-lens FK: observations -> JointTransformBatch -> scattered per-joint `Transform3D`."""
    # FK needs only the joint datasets; the camera frames are most of the file.
    cameras = [f"obs/{camera.source}" for camera in discover_cameras(reader, demo)]
    obs = reader.stream(root_group=f"/data/{demo}", ignore_datasets=cameras, use_structs=True)
    return (
        obs.lenses(
            DeriveLens(OBS_STRUCT, output_entity="/tmp/batches").to_component(
                BATCH,
                Selector(".").pipe(lambda msgs: transform_batches(urdf, joints, msgs)),
            ),
            content=OBS_ENTITY,
            output_mode="forward_all",
        )
        .lenses(
            DeriveLens(BATCH, output_entity=TRANSFORMS, scatter=True)
            .to_component(rr.Transform3D.descriptor_translation(), Selector("[].translation"))
            .to_component(rr.Transform3D.descriptor_quaternion(), Selector("[].quaternion"))
            .to_component(rr.Transform3D.descriptor_parent_frame(), Selector("[].parent_frame"))
            .to_component(rr.Transform3D.descriptor_child_frame(), Selector("[].child_frame")),
            content="/tmp/batches",
            output_mode="drop_unmatched",
        )
        .filter(content=TRANSFORMS)
        .flat_map(with_sim_time)
    )


# --------------------------------------------------------------------------------------
# demo driver
# --------------------------------------------------------------------------------------


def convert_demo(urdf: UrdfTree, reader: Hdf5Reader, task: str, demo: str, rrd_root: Path) -> Path:
    """Write one demo's URDF layer; returns the written path."""
    model_file = str(reader.attributes(f"/data/{demo}")["model_file"])
    rec_id = recording_id(task, demo)
    out_path = rrd_root / layer_relpath("urdf", rec_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = urdf.stream(include_joint_transforms=True)
    edge = LazyChunkStream.from_iter([world_from_base_chunk(model_file)])
    fk = fk_stream(urdf, sliding_joints(urdf), reader, demo)
    LazyChunkStream.merge(model, edge, fk).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=rec_id
    )
    return out_path


def load_urdf() -> UrdfTree:
    """The vendored `fer` model, rooted under `/urdf`."""
    return UrdfTree.from_file_path(
        str(URDF_PATH),
        entity_path_prefix=ENTITY_PREFIX,
        static_transform_entity_path=f"{ENTITY_PREFIX}/tf_static",
    )


def main(argv: list[str]) -> None:
    urdf = load_urdf()
    inputs = task_files(argv)
    print(f"Building URDF layer for {len(inputs)} task file(s) -> {RRD_ROOT / 'urdf'}/")
    for path, task in inputs:
        reader = Hdf5Reader(path)
        for demo in demo_keys(reader):
            out = convert_demo(urdf, reader, task, demo, RRD_ROOT)
            print(f"  {recording_id(task, demo)}: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv)
