"""
Build the *cameras* layer: the three cameras placed in the 3D scene, so their videos show there.

Derived from the base layer, which gives the recording id. The layer is static: it adds a
`Transform3D`, a `Pinhole` and a `CoordinateFrame` to each camera's video entity. The wrist cameras
hang off the gripper flanges that the urdf layer poses, so they move with the arms.

Run:  pixi run -e molmo convert-cameras              # every base RRD under rrds/molmoact2/base/
      pixi run -e molmo convert-cameras <in.rrd>     # one recording
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile, RrdReader

from molmoact2.base_layer import APPLICATION_ID, RRD_ROOT
from molmoact2.urdf_layer import WORLD_FRAME
from rrd_datasets_common.paths import layer_relpath, resolve_input_path

LAYER = "cameras"
INPUT_LAYER = "base"

WIDTH, HEIGHT = 640, 360

# No calibration is published. Every camera is taken as a RealSense D405, with the 87° horizontal
# field of view of its spec and the principal point at the image centre.
HORIZONTAL_FOV_DEG = 87.0

# i2rt's YAM station model mounts each wrist camera on the gripper flange, turned -25° about x
# (`{side}_camera_joint` in `yam_station_linear_4310_d405.urdf`, i2rt-robotics/i2rt at 120c3c8).
WRIST_TRANSLATION = [0.0, -0.070435, 0.077006]
WRIST_TURN_DEG = -25.0

# The top camera's pose is not published. These values make the posed arms match the top video of
# episodes 800, 7309 and 31673: ahead of the bases, slightly left, 72° below the horizon.
TOP_TRANSLATION = [0.09, 0.02, 0.73]
TOP_YAW_DEG, TOP_TILT_DEG, TOP_ROLL_DEG = 2.0, 72.0, 2.0


@dataclass(frozen=True)
class CameraMount:
    """Where one camera sits, relative to its parent frame."""

    camera: str  # the source camera name: `top`, `left` or `right`
    parent_frame: str
    translation: list[float]
    rotation: np.ndarray  # columns: the camera's right, down and forward axes in the parent frame


def turn_about_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ])


def look_rotation(yaw_deg: float, tilt_deg: float, roll_deg: float) -> np.ndarray:
    """A camera's axes in a z-up frame: heading `yaw` from +x, `tilt` below the horizon, `roll` about the view axis."""
    yaw, tilt, roll = (math.radians(value) for value in (yaw_deg, tilt_deg, roll_deg))
    forward = np.array([math.cos(tilt) * math.cos(yaw), math.cos(tilt) * math.sin(yaw), -math.sin(tilt)])
    level_right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    level_down = np.cross(forward, level_right)
    right = math.cos(roll) * level_right + math.sin(roll) * level_down
    return np.column_stack([right, np.cross(forward, right), forward])


MOUNTS = [
    CameraMount("top", WORLD_FRAME, TOP_TRANSLATION, look_rotation(TOP_YAW_DEG, TOP_TILT_DEG, TOP_ROLL_DEG)),
    *(CameraMount(arm, f"{arm}_gripper", WRIST_TRANSLATION, turn_about_x(WRIST_TURN_DEG)) for arm in ("left", "right")),
]


def image_from_camera() -> np.ndarray:
    focal = (WIDTH / 2) / math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2)
    return np.array([[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]])


def camera_chunk(mount: CameraMount) -> Chunk:
    """
    The camera entity's static pose, intrinsics and frame binding.

    The `Transform3D` and the `Pinhole` both name the `parent -> camera` edge, and the
    `CoordinateFrame` puts the entity, video included, in that frame.
    """
    frame = f"{mount.camera}_camera"
    return Chunk.from_columns(
        f"/observation.images.{mount.camera}",
        indexes=[],
        columns=[
            *rr.Transform3D.columns(
                translation=[mount.translation],
                mat3x3=[mount.rotation],
                parent_frame=[mount.parent_frame],
                child_frame=[frame],
            ),
            *rr.Pinhole.columns(
                image_from_camera=[image_from_camera()],
                resolution=[[WIDTH, HEIGHT]],
                camera_xyz=[rr.ViewCoordinates.RDF],
                parent_frame=[mount.parent_frame],
                child_frame=[frame],
            ),
            *rr.CoordinateFrame.columns(frame=[frame]),
        ],
    )


def write_layer(input_rrd: Path, out_dir: Path) -> Path:
    """Write one episode's cameras layer under `out_dir`; returns the written path."""
    recordings = RrdReader(str(input_rrd)).recordings()
    if len(recordings) != 1:
        raise ValueError(f"{input_rrd}: expected one recording, found {len(recordings)}")
    recording_id = recordings[0].recording_id
    out_path = out_dir / layer_relpath(LAYER, recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = LazyChunkStream.from_iter([camera_chunk(mount) for mount in MOUNTS])
    chunks.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=recording_id
    )
    return out_path


def discover_inputs(path: Path) -> list[Path]:
    """The input RRDs at `path`: the file itself, or a directory's top level."""
    return [path] if path.is_file() else sorted(path.glob("*.rrd"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cameras layer from base RRDs.")
    parser.add_argument(
        "input", nargs="?", type=Path, default=RRD_ROOT / INPUT_LAYER, help="An RRD, or a directory of them."
    )
    parser.add_argument("--out-dir", type=Path, default=RRD_ROOT, help="Root the cameras/ folder is written under.")
    args = parser.parse_args()

    inputs = discover_inputs(resolve_input_path(args.input))
    if not inputs:
        raise SystemExit(f"No .rrd at {args.input} — run `pixi run -e molmo convert-{INPUT_LAYER}` first.")
    print(f"Building {LAYER} layer for {len(inputs)} recording(s) -> {args.out_dir / LAYER}/")
    for input_rrd in inputs:
        out = write_layer(input_rrd, args.out_dir)
        print(f"  {out.stem}: {out.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
