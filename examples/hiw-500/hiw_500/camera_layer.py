"""
Build a camera *layer* that places the head camera in 3D per episode.

The head camera is an RGB stereo pair, mounted in the URDF at the fixed `d435_link` frame
(`d435_joint` off `torso_link`), so its mounting extrinsic is already in the URDF layer's
frame graph. The base layer splits the side-by-side stereo stream into `/camera/head/left` and
`/camera/head/right`; this layer gives each eye a `Transform3D` (the `d435_link -> *_optical`
rotation), a `Pinhole` (intrinsics), and a `CoordinateFrame` that binds the entity and its
image to the optical frame. The left eye sits at `d435_link`; the right is offset by the
stereo baseline.

Per-eye intrinsics and the baseline come from the episode's own head stereo calibration
(`calibration/params/head_camera_params.yaml`). Episodes without one — the older sessions — skip
this layer and keep their head images 2D. Unused from that calibration: distortion (`Pinhole`
has no distortion model), the rectification outputs, and the small left->right rotation `R`
(the right eye is placed by the baseline alone).

The two wrist cameras stay as 2D image views: they have no mounting extrinsic (absent from the
URDF, and the calibration files carry only intra-camera IR↔color extrinsics, not hand-eye).

Run:  pixi run -e hiw convert-cameras            # all episodes under data/HIW-500/
      pixi run -e hiw convert-cameras <ep.mcap>  # a single episode
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import yaml
from rerun.experimental import (
    Chunk,
    LazyChunkStream,
    OptimizationProfile,
)

from hiw_500.base_layer import APPLICATION_ID, DATASET_ROOT, RRD_ROOT, Episode, discover_episodes, episode_from_mcap
from rrd_datasets_common.paths import layer_relpath

D435_LINK_FRAME = "d435_link"  # the URDF fixed-joint camera frame (mount / parent)
# The head stream is a side-by-side stereo pair; the base layer split it into these two entities.
LEFT_ENTITY = "/camera/head/left"
RIGHT_ENTITY = "/camera/head/right"
LEFT_OPTICAL = "d435_left_optical"  # optical frame (z-forward, x-right, y-down)
RIGHT_OPTICAL = "d435_right_optical"


@dataclass
class HeadCalibration:
    """Per-eye pinhole intrinsics and the stereo baseline of the head camera pair."""

    k_left: np.ndarray
    k_right: np.ndarray
    width: int
    height: int
    baseline_m: float

    @classmethod
    def from_yaml(cls, path: Path) -> HeadCalibration:
        """Parse an episode's OpenCV stereo calibration; the yaml stores the baseline in millimetres."""
        calib = yaml.safe_load(path.read_text())
        width, height = (int(v) for v in calib["image_size"])
        return cls(
            k_left=np.array(calib["camera_matrix_left"], dtype=np.float64),
            k_right=np.array(calib["camera_matrix_right"], dtype=np.float64),
            width=width,
            height=height,
            baseline_m=float(calib["baseline"]) / 1000.0,
        )


def optical_quaternion() -> np.ndarray:
    """
    `d435_link` (ROS FLU body) -> optical (RDF) rotation, as an xyzw quaternion.

    Optical axes expressed in the link frame: x_opt=-y_link, y_opt=-z_link, z_opt=+x_link.
    """
    r = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # columns = optical axes
    t = r.trace()
    w = np.sqrt(max(0.0, 1.0 + t)) / 2.0
    x = (r[2, 1] - r[1, 2]) / (4.0 * w)
    y = (r[0, 2] - r[2, 0]) / (4.0 * w)
    z = (r[1, 0] - r[0, 1]) / (4.0 * w)
    return np.array([x, y, z, w])


def _head_camera_chunk(entity: str, optical_frame: str, k: np.ndarray, w: int, h: int, offset_y: float) -> Chunk:
    """
    A head-camera entity: extrinsic Transform3D + Pinhole + CoordinateFrame.

    The Transform3D gives the FLU->optical rotation plus the in-link translation (`offset_y` along
    link -y == optical +x, i.e. the stereo baseline for the right eye); the Pinhole gives the
    intrinsics. Both name `parent_frame=d435_link` (the URDF mount) and `child_frame=optical_frame`,
    so the `d435_link -> optical_frame` edge exists in the named frame graph. The CoordinateFrame
    then places *this entity* (and the head image the base layer logs on it) in `optical_frame` —
    without it the entity sits in its own implicit frame with no transform path to the scene root.
    """
    return Chunk.from_columns(
        entity,
        indexes=[],
        columns=[
            *rr.Transform3D.columns(
                translation=[[0.0, offset_y, 0.0]],
                quaternion=[optical_quaternion()],
                parent_frame=[D435_LINK_FRAME],
                child_frame=[optical_frame],
            ),
            *rr.Pinhole.columns(
                image_from_camera=[k],
                resolution=[[w, h]],
                parent_frame=[D435_LINK_FRAME],
                child_frame=[optical_frame],
            ),
            *rr.CoordinateFrame.columns(frame=[optical_frame]),
        ],
    )


def camera_chunks(calib: HeadCalibration) -> list[Chunk]:
    """
    Place both head stereo images in 3D: left at `d435_link`, right offset by the baseline.

    The left eye sits at the URDF `d435_link` origin; the right eye is offset by the stereo
    baseline along the optical +x (== link -y) axis.
    """
    return [
        _head_camera_chunk(LEFT_ENTITY, LEFT_OPTICAL, calib.k_left, calib.width, calib.height, offset_y=0.0),
        _head_camera_chunk(
            RIGHT_ENTITY, RIGHT_OPTICAL, calib.k_right, calib.width, calib.height, offset_y=-calib.baseline_m
        ),
    ]


def convert_episode(ep: Episode, rrd_root: Path) -> Path | None:
    """Write the episode's camera layer, or return `None` when it ships no head calibration."""
    if ep.head_calib is None:
        return None
    out_path = rrd_root / layer_relpath("cameras", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = camera_chunks(HeadCalibration.from_yaml(ep.head_calib))
    LazyChunkStream.from_iter(chunks).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main(argv: list[str]) -> None:
    episodes = [episode_from_mcap(Path(argv[1]))] if len(argv) > 1 else discover_episodes(DATASET_ROOT)
    print(f"Building camera layer for {len(episodes)} episode(s) -> {RRD_ROOT / 'cameras'}/")
    for ep in episodes:
        out = convert_episode(ep, rrd_root=RRD_ROOT)
        if out is None:
            print(f"  {ep.recording_id}: skipped — no head calibration")
        else:
            print(f"  {ep.recording_id}: {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv)
