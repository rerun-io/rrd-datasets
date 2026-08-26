"""
Build an odometry *layer* that connects the URDF robot to the world frame per episode.

The URDF FK layer roots the G1 at the `pelvis` frame, so on its own the robot animates in place
at the origin. This layer adds the missing edge: a time-varying `odom -> pelvis` `Transform3D`
derived from `/lf/odommodestate` (base position + orientation), under which the whole robot
tree moves through the scene.

One `McapReader` stream and a single `DeriveLens` turn the SportModeState message into that
edge (translation + reordered quaternion + constant frame names). Written as a separate `.rrd`
per episode sharing the base `recording_id`.

Run:  pixi run -e hiw convert-odom            # all episodes under data/HIW-500/
      pixi run -e hiw convert-odom <ep.mcap>  # a single episode
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from rerun.experimental import (
    DeriveLens,
    McapReader,
    OptimizationProfile,
    Selector,
)

from hiw_500.base_layer import (
    APPLICATION_ID,
    DATASET_ROOT,
    MSG_ODOM,
    RRD_ROOT,
    VEC3,
    Episode,
    const_str,
    discover_episodes,
    episode_from_mcap,
)
from rrd_datasets_common.paths import layer_relpath

ODOM_TOPIC = "/lf/odommodestate"
WORLD_FRAME = "odom"
ROOT_FRAME = "pelvis"  # the URDF root frame (verified via UrdfTree.root_link())
EDGE_ENTITY = "/odom/pelvis"
# Arrow type of a Transform3D quaternion.
QUAT = pa.list_(pa.float32(), 4)


def to_vec3(arr: pa.Array) -> pa.Array:
    """A list<float>[3] field -> Transform3D translation type."""
    return pa.FixedSizeListArray.from_arrays(pc.list_flatten(arr).cast(pa.float32()), 3).cast(VEC3)


def quat_wxyz_to_xyzw(arr: pa.Array) -> pa.Array:
    """Unitree quaternions are [w, x, y, z]; Rerun wants [x, y, z, w]."""
    wxyz: np.ndarray = pc.list_flatten(arr).to_numpy(zero_copy_only=False).reshape(-1, 4)
    xyzw = wxyz[:, [1, 2, 3, 0]].astype(np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(xyzw), 4).cast(QUAT)


def odom_edge_lens() -> DeriveLens:
    """SportModeState -> `odom -> pelvis` Transform3D edge."""
    return (
        DeriveLens(MSG_ODOM, output_entity=EDGE_ENTITY)
        .to_component(rr.Transform3D.descriptor_translation(), Selector(".position").pipe(to_vec3))
        .to_component(rr.Transform3D.descriptor_quaternion(), Selector(".imu_state.quaternion").pipe(quat_wxyz_to_xyzw))
        .to_component(rr.Transform3D.descriptor_parent_frame(), Selector(".position").pipe(const_str(WORLD_FRAME)))
        .to_component(rr.Transform3D.descriptor_child_frame(), Selector(".position").pipe(const_str(ROOT_FRAME)))
    )


def convert_episode(ep: Episode, rrd_root: Path) -> Path:
    stream = McapReader(str(ep.mcap), include_topic_regex=[f"^{ODOM_TOPIC}$"]).stream()
    stream = stream.lenses(odom_edge_lens(), content=ODOM_TOPIC, output_mode="drop_unmatched").filter(
        content=EDGE_ENTITY
    )
    out_path = rrd_root / layer_relpath("odom", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main(argv: list[str]) -> None:
    episodes = [episode_from_mcap(Path(argv[1]))] if len(argv) > 1 else discover_episodes(DATASET_ROOT)
    print(f"Building odometry layer for {len(episodes)} episode(s) -> {RRD_ROOT / 'odom'}/")
    for ep in episodes:
        out = convert_episode(ep, RRD_ROOT)
        print(f"  {ep.recording_id}: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv)
