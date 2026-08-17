"""
Build a URDF forward-kinematics *layer* for each HIW-500 episode.

A derived layer: the animated Unitree G1 mesh, driven by the joint positions in
`/stamped/lowstate`. Written as a separate `.rrd` per episode sharing the base `recording_id`,
so base + URDF load as one logical recording.

Two lenses on one `McapReader` joint stream: the first `DeriveLens` calls
`compute_joint_transform_batches` per row, and a second `scatter` lens explodes each batch into
per-joint `Transform3D` rows at `/robot/transforms`.

The model is `g1_29dof_mode_15_with_dex1_1.urdf` (Dex1 variant). Its 29 revolute joints follow
the documented Unitree G1 motor order, so motor index i maps to `<name>_joint`. The 4 prismatic
Dex1 finger joints stay at rest.

Run:  pixi run -e hiw convert-urdf            # all episodes under data/HIW-500/
      pixi run -e hiw convert-urdf <ep.mcap>  # a single episode
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
    DeriveLens,
    LazyChunkStream,
    McapReader,
    OptimizationProfile,
    Selector,
)
from rerun.urdf import UrdfTree

from hiw_500.base_layer import (
    APPLICATION_ID,
    DATASET_ROOT,
    G1_JOINT_NAMES,
    MSG_LOWSTATE,
    N_JOINTS,
    RRD_ROOT,
    Episode,
    discover_episodes,
    episode_from_mcap,
)
from rrd_datasets_common.paths import layer_relpath

# pyarrow.compute ships incomplete stubs; alias once (used inside the FK lens pipe).
list_element = pc.list_element  # type: ignore[attr-defined]
list_flatten = pc.list_flatten  # type: ignore[attr-defined]

URDF_PATH = Path("urdf/g1/g1_29dof_mode_15_with_dex1_1.urdf")
ENTITY_PREFIX = "robot"
LOWSTATE_TOPIC = "/stamped/lowstate"
BATCH = "rerun.urdf.JointTransformBatch"

# Motor index -> URDF joint name (verified: identical order, "_joint" suffix).
JOINT_NAMES_URDF = [f"{name}_joint" for name in G1_JOINT_NAMES]

# Entity paths log under /robot/<robot-name>/...; the content glob only supports a trailing
# `**`, so the collision-drop pattern needs the literal robot-name segment.
ROBOT_NAME = ET.parse(URDF_PATH).getroot().attrib["name"]
COLLISION_GLOB = f"/{ENTITY_PREFIX}/{ROBOT_NAME}/collision_geometries/**"

_OFFSETS = pa.array(np.arange(0, 4096 * N_JOINTS, N_JOINTS, dtype=np.int32))  # reused, sliced per call


def _msg_struct(msgs: pa.Array) -> pa.Array:
    """The lowstate `:message` arrives as a length-1 list per row; flatten to one struct per row."""
    return list_flatten(msgs) if pa.types.is_list(msgs.type) else msgs


def read_values(msgs: pa.Array) -> pa.Array:
    """Per-timestamp list of the 29 joint positions, in URDF joint order."""
    motor = _msg_struct(msgs).field("data").field("motor_state")
    n = len(motor)
    q = np.stack(
        [list_element(motor, i).field("q").to_numpy(zero_copy_only=False) for i in range(N_JOINTS)],
        axis=1,
    ).reshape(-1)
    return pa.ListArray.from_arrays(_OFFSETS[: n + 1], pa.array(q.astype(np.float64)))


def read_names(msgs: pa.Array) -> pa.Array:
    """Per-timestamp list of the 29 URDF joint names, aligned to `read_values`."""
    n = len(_msg_struct(msgs))
    return pa.ListArray.from_arrays(_OFFSETS[: n + 1], pa.array(JOINT_NAMES_URDF * n, type=pa.string()))


def fk_stream(urdf: UrdfTree, path: Path) -> LazyChunkStream:
    """Two-lens FK: lowstate messages -> JointTransformBatch -> scattered per-joint Transform3D."""
    joints = McapReader(str(path), include_topic_regex=[f"^{LOWSTATE_TOPIC}$"]).stream()
    return (
        joints.lenses(
            DeriveLens(MSG_LOWSTATE, output_entity="/tmp/batches").to_component(
                BATCH,
                Selector(".").pipe(lambda m: urdf.compute_joint_transform_batches(read_names(m), read_values(m))),
            ),
            content=LOWSTATE_TOPIC,
            output_mode="forward_all",
        )
        .lenses(
            DeriveLens(BATCH, output_entity="/robot/transforms", scatter=True)
            .to_component(rr.Transform3D.descriptor_translation(), Selector("[].translation"))
            .to_component(rr.Transform3D.descriptor_quaternion(), Selector("[].quaternion"))
            .to_component(rr.Transform3D.descriptor_parent_frame(), Selector("[].parent_frame"))
            .to_component(rr.Transform3D.descriptor_child_frame(), Selector("[].child_frame")),
            content="/tmp/batches",
            output_mode="drop_unmatched",
        )
        .filter(content="/robot/transforms")
    )


def convert_episode(urdf: UrdfTree, ep: Episode, rrd_root: Path) -> Path:
    model = urdf.stream(include_joint_transforms=True).drop(content=COLLISION_GLOB)
    out_path = rrd_root / layer_relpath("urdf", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LazyChunkStream.merge(model, fk_stream(urdf, ep.mcap)).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main(argv: list[str]) -> None:
    urdf = UrdfTree.from_file_path(
        str(URDF_PATH),
        entity_path_prefix=ENTITY_PREFIX,
        static_transform_entity_path=f"{ENTITY_PREFIX}/tf_static",
    )
    episodes = [episode_from_mcap(Path(argv[1]))] if len(argv) > 1 else discover_episodes(DATASET_ROOT)
    print(f"Building URDF FK layer for {len(episodes)} episode(s) -> {RRD_ROOT / 'urdf'}/")
    for ep in episodes:
        out = convert_episode(urdf, ep, RRD_ROOT)
        print(f"  {ep.recording_id}: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv)
