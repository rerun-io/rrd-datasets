"""
Build the URDF forward-kinematics layer for each LIBERO demo.

The vendored Franka `fer` model is written as a shared model rrd (`convert_model`); each
demo's own `.rrd` carries only what varies. Two lenses on
one `Hdf5Reader` stream build the FK.

Run:  pixi run -e libero convert-urdf              # every downloaded task file
      pixi run -e libero convert-urdf <task.hdf5>  # a single task file
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
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

# Used in three places: the model rrd's file name, its recording id, and its asset id on the catalog.
MODEL_RECORDING_ID = "urdf-model"
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
# forward kinematics
# --------------------------------------------------------------------------------------


def _obs_struct(msgs: pa.Array) -> pa.Array:
    """The observations arrive as a length-1 list per row; flatten to one struct per row."""
    return list_flatten(msgs) if pa.types.is_list(msgs.type) else msgs


def read_joint_values(msgs: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """The arm angles `(n, 7)` in radians and the finger openings `(n, 2)` in metres."""
    obs = _obs_struct(msgs)
    arm = obs.field("joint_states").flatten().to_numpy().reshape(len(obs), -1)
    grip = obs.field("gripper_states").flatten().to_numpy().reshape(len(obs), -1)
    # Both URDF finger joints are limited to [0, 0.04]; robosuite reports the second one negated.
    return arm, np.column_stack([grip[:, 0], -grip[:, 1]])


def transform_batches(urdf: UrdfTree, msgs: pa.Array) -> pa.Array:
    """One `JointTransformBatch` per row, all nine joints solved by the SDK."""
    arm_values, finger_values = read_joint_values(msgs)
    joint_values = np.column_stack([arm_values, finger_values])
    rows = len(joint_values)
    offsets = pa.array(np.arange(rows + 1) * N_JOINTS, type=pa.int32())
    names = pa.ListArray.from_arrays(offsets, pa.array(JOINT_NAMES_URDF * rows, type=pa.string()))
    values = pa.ListArray.from_arrays(offsets, pa.array(joint_values.reshape(-1)))
    return urdf.compute_joint_transform_batches(names, values, clamp=False)


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


def fk_stream(urdf: UrdfTree, reader: Hdf5Reader, demo: str) -> LazyChunkStream:
    """Two-lens FK: observations -> JointTransformBatch -> scattered per-joint `Transform3D`."""
    # FK needs only the joint datasets; the camera frames are most of the file.
    cameras = [f"obs/{camera.source}" for camera in discover_cameras(reader, demo)]
    obs = reader.stream(root_group=f"/data/{demo}", ignore_datasets=cameras, use_structs=True)
    return (
        obs.lenses(
            DeriveLens(OBS_STRUCT, output_entity="/tmp/batches").to_component(
                BATCH,
                Selector(".").pipe(lambda msgs: transform_batches(urdf, msgs)),
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


def model_rrd_path(rrd_root: Path) -> Path:
    """Where the shared model rrd lives under a dataset's rrd root."""
    return rrd_root / "assets" / f"{MODEL_RECORDING_ID}.rrd"


def convert_model(urdf: UrdfTree, rrd_root: Path) -> Path:
    """Write the shared model rrd: the meshes and fixed transforms every demo's urdf layer poses."""
    out_path = model_rrd_path(rrd_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urdf.stream(include_joint_transforms=True).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=MODEL_RECORDING_ID
    )
    return out_path


def convert_demo(urdf: UrdfTree, reader: Hdf5Reader, task: str, demo: str, rrd_root: Path) -> Path:
    """Write one demo's URDF layer; returns the written path."""
    model_file = str(reader.attributes(f"/data/{demo}")["model_file"])
    rec_id = recording_id(task, demo)
    out_path = rrd_root / layer_relpath("urdf", rec_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    edge = LazyChunkStream.from_iter([world_from_base_chunk(model_file)])
    fk = fk_stream(urdf, reader, demo)
    LazyChunkStream.merge(edge, fk).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
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
    model = convert_model(urdf, RRD_ROOT)
    print(f"Shared model rrd: {model} ({model.stat().st_size / 1e6:.1f} MB)")
    print(f"Building URDF layer for {len(inputs)} task file(s) -> {RRD_ROOT / 'urdf'}/")
    for path, task in inputs:
        reader = Hdf5Reader(path)
        for demo in demo_keys(reader):
            out = convert_demo(urdf, reader, task, demo, RRD_ROOT)
            print(f"  {recording_id(task, demo)}: {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv)
