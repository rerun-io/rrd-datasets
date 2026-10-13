"""
Build the *urdf* layer: both YAM arms posed by forward kinematics.

Derived from the base layer's `/observation.state`.
The vendored model is written as the asset `assets/urdf-model.rrd` for the whole dataset.
Each episode's `.rrd` holds the joint transforms and the two edges that place the arms in one `world` frame.

Run:  pixi run -e molmo convert-urdf              # every base RRD under rrds/molmoact2/base/
      pixi run -e molmo convert-urdf <in.rrd>     # one recording
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.chunk import (
    Chunk,
    DeriveLens,
    LazyChunkStream,
    LazyStore,
    MutateLens,
    OptimizationProfile,
    RrdReader,
    Selector,
)
from rerun.urdf import UrdfTree

from molmoact2.base_layer import APPLICATION_ID, RRD_ROOT
from rrd_datasets_common.paths import layer_relpath

LAYER = "urdf"
INPUT_LAYER = "base"

URDF_PATH = Path(__file__).resolve().parents[1] / "urdf" / "yam" / "yam.urdf"

# Used in three places: the asset rrd's file name, its recording id, and its asset id on the catalog.
MODEL_RECORDING_ID = "urdf-model"

# Every mesh is painted this packed RGBA instead of the per-link colors the URDF declares.
MODEL_ALBEDO = 0xFFFFFFFF

ARMS = ["left", "right"]
WORLD_FRAME = "world"
BATCH = "rerun.urdf.JointTransformBatch"

STATE_ENTITY = "/observation.state"
STATE_COMPONENT = "Scalars:scalars"

# The 14 state values hold the left arm's six joints and gripper, then the right arm's.
# `{arm}_joint_{i}.pos` drives `joint{i+1}` in radians; `{arm}_gripper.pos` drives both fingers.
VALUES_PER_ARM = 7
ARM_JOINTS = [f"joint{index + 1}" for index in range(6)]
FINGER_JOINTS = ["joint7", "joint8"]
JOINT_NAMES = ARM_JOINTS + FINGER_JOINTS

# The gripper value runs from 0 (closed) to 1 (open), and both finger joints travel over
# `[-FINGER_TRAVEL, 0]` meters, with 0 closed.
FINGER_TRAVEL = 0.04695

# The MolmoAct2 dataset does not publish this number.
# Both arms face +x, and this spacing along y is an estimate based on visual observation.
BASE_SPACING = 0.50


def arm_entity(arm: str) -> str:
    """The entity every part of one arm is logged under, model and layer alike."""
    # `UrdfTree` takes its entity prefix as a single path part.
    return f"/urdf_{arm}"


def load_urdfs() -> dict[str, UrdfTree]:
    """The vendored YAM model once per arm, each with its own entity path and frame names."""
    return {
        arm: UrdfTree.from_file_path(
            str(URDF_PATH),
            entity_path_prefix=arm_entity(arm).lstrip("/"),
            frame_prefix=f"{arm}_",
            static_transform_entity_path=f"{arm_entity(arm)}/tf_static",
        )
        for arm in ARMS
    }


# --------------------------------------------------------------------------------------
# forward kinematics
# --------------------------------------------------------------------------------------


def joint_values(state: pa.Array, arm: str) -> np.ndarray:
    """One arm's eight URDF joint values per state row: six angles, then both fingers in metres."""
    offset = ARMS.index(arm) * VALUES_PER_ARM
    values = state.flatten().to_numpy().reshape(len(state), -1)[:, offset : offset + VALUES_PER_ARM]
    angles, gripper = values[:, : len(ARM_JOINTS)], values[:, len(ARM_JOINTS)]
    fingers = -FINGER_TRAVEL * gripper
    return np.column_stack([angles, fingers, fingers])


def transform_batches(urdf: UrdfTree, state: pa.Array, arm: str) -> pa.Array:
    """One `JointTransformBatch` per state row, for one arm's eight joints."""
    values = joint_values(state, arm)
    rows = len(values)
    offsets = pa.array(np.arange(rows + 1) * len(JOINT_NAMES), type=pa.int32())
    names = pa.ListArray.from_arrays(offsets, pa.array(JOINT_NAMES * rows, type=pa.string()))
    return urdf.compute_joint_transform_batches(
        names, pa.ListArray.from_arrays(offsets, pa.array(values.reshape(-1))), clamp=False
    )


def batch_chunk(chunk: Chunk, urdf: UrdfTree, arm: str) -> Chunk:
    """A state chunk's rows as one arm's `JointTransformBatch` rows, on the same timelines."""
    state = chunk.to_record_batch()
    batches = transform_batches(urdf, state.column(STATE_COMPONENT), arm)
    # One batch per row: wrapped in a single-element list, so the batch itself is the component value.
    one_per_row = pa.ListArray.from_arrays(pa.array(np.arange(len(batches) + 1), type=pa.int32()), batches)
    columns = {name: state.column(name) for name in chunk.timeline_names}
    columns[f"/tmp/{arm}_batches:{BATCH}"] = one_per_row
    (out,) = Chunk.from_record_batch(pa.RecordBatch.from_pydict(columns), index=chunk.timeline_names)
    return out


def arm_fk_stream(store: LazyStore, urdf: UrdfTree, arm: str) -> LazyChunkStream:
    """FK for one arm: state rows -> `JointTransformBatch` -> per-joint `Transform3D` rows."""
    transforms = f"{arm_entity(arm)}/transforms"
    return (
        store.stream()
        .filter(content=STATE_ENTITY, components=[STATE_COMPONENT], is_static=False)
        # A lens cannot build the batches: its selector receives every row's 14 values as one flat array.
        .map(partial(batch_chunk, urdf=urdf, arm=arm))
        .lenses(
            DeriveLens(BATCH, output_entity=transforms, scatter=True)
            .to_component(rr.Transform3D.descriptor_translation(), Selector("[].translation"))
            .to_component(rr.Transform3D.descriptor_quaternion(), Selector("[].quaternion"))
            .to_component(rr.Transform3D.descriptor_parent_frame(), Selector("[].parent_frame"))
            .to_component(rr.Transform3D.descriptor_child_frame(), Selector("[].child_frame")),
            output_mode="drop_unmatched",
        )
    )


def world_from_base_chunk(arm: str) -> Chunk:
    """The static `world -> {arm}_base` edge, with the left arm on +y."""
    side = 1.0 if arm == "left" else -1.0
    return Chunk.from_columns(
        f"{arm_entity(arm)}/world_from_base",
        indexes=[],
        columns=rr.Transform3D.columns(
            translation=[[0.0, side * BASE_SPACING / 2, 0.0]],
            parent_frame=[WORLD_FRAME],
            child_frame=[f"{arm}_base"],
        ),
    )


def urdf_stream(store: LazyStore, urdfs: dict[str, UrdfTree]) -> LazyChunkStream:
    """The derivation: `/observation.state` -> both arms' joint transforms, plus their world edges."""
    edges = LazyChunkStream.from_iter([world_from_base_chunk(arm) for arm in ARMS])
    return LazyChunkStream.merge(edges, *(arm_fk_stream(store, urdfs[arm], arm) for arm in ARMS))


# --------------------------------------------------------------------------------------
# episode driver
# --------------------------------------------------------------------------------------


def model_rrd_path(out_dir: Path) -> Path:
    """Where the dataset asset with the arm model is under an rrd root."""
    return out_dir / "assets" / f"{MODEL_RECORDING_ID}.rrd"


def albedo_lens() -> MutateLens:
    """Repaint every mesh `MODEL_ALBEDO`, so the shading alone shows the shape."""

    def repaint(albedo: pa.Array) -> pa.Array:
        return pa.array(np.full(len(albedo), MODEL_ALBEDO, dtype=np.uint32), type=albedo.type)

    return MutateLens("Asset3D:albedo_factor", Selector(".").pipe(repaint))


def write_asset(urdfs: dict[str, UrdfTree], out_dir: Path) -> Path:
    """Write both arms' meshes and rest-pose joint transforms once, under their own recording id."""
    out_path = model_rrd_path(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = LazyChunkStream.merge(*(urdfs[arm].stream(include_joint_transforms=True) for arm in ARMS))
    # The lens changes only the albedo; every other chunk passes through.
    model = model.lenses(albedo_lens(), output_mode="forward_unmatched")
    model.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=MODEL_RECORDING_ID
    )
    return out_path


def write_layer(input_rrd: Path, urdfs: dict[str, UrdfTree], out_dir: Path) -> Path:
    """Write one episode's urdf layer under `out_dir`; returns the written path."""
    reader = RrdReader(str(input_rrd))
    recordings = reader.recordings()
    if len(recordings) != 1:
        raise ValueError(f"{input_rrd}: expected one recording, found {len(recordings)}")
    recording_id = recordings[0].recording_id
    out_path = out_dir / layer_relpath(LAYER, recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    derived = urdf_stream(reader.store(), urdfs).collect(optimize=OptimizationProfile.OBJECT_STORE)
    derived.write_rrd(str(out_path), application_id=APPLICATION_ID, recording_id=recording_id)
    return out_path


def discover_inputs(path: Path) -> list[Path]:
    """The input RRDs at `path`: the file itself, or a directory's top level."""
    return [path] if path.is_file() else sorted(path.glob("*.rrd"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the urdf layer from base RRDs.")
    parser.add_argument(
        "input", nargs="?", type=Path, default=RRD_ROOT / INPUT_LAYER, help="An RRD, or a directory of them."
    )
    parser.add_argument("--out-dir", type=Path, default=RRD_ROOT, help="Root the urdf/ and assets/ folders go under.")
    args = parser.parse_args()

    urdfs = load_urdfs()
    model = write_asset(urdfs, args.out_dir)
    print(f"Model asset: {model} ({model.stat().st_size / 1e6:.1f} MB)")
    inputs = discover_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No .rrd at {args.input} — run `pixi run -e molmo convert-{INPUT_LAYER}` first.")
    print(f"Building {LAYER} layer for {len(inputs)} recording(s) -> {args.out_dir / LAYER}/")
    for input_rrd in inputs:
        out = write_layer(input_rrd, urdfs, args.out_dir)
        print(f"  {out.stem}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
