"""
Build a camera *layer* that places the two recorded cameras in 3D per demo.

The demo's MuJoCo XML (`model_file`) gives each camera a pose and a vertical field of view. This
layer turns them into a static `Transform3D`, `Pinhole` and `CoordinateFrame` on the base layer's
`/camera/<name>` image entities: `agentview` hangs off the world frame, `eye_in_hand` off the URDF
hand frame, so it moves with the arm.

Run:  pixi run -e libero convert-cameras              # every downloaded task file
      pixi run -e libero convert-cameras <task.hdf5>  # a single task file
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.experimental import Chunk, Hdf5Reader, LazyChunkStream, OptimizationProfile

from libero.base_layer import APPLICATION_ID, RRD_ROOT, Camera, demo_keys, discover_cameras, task_files
from libero.episodes import recording_id
from libero.urdf_layer import WORLD_FRAME
from rrd_datasets_common.paths import layer_relpath

CAMERA_ENTITY_PREFIX = "/camera"

# MuJoCo's default `fovy`, for a `<camera>` element that gives none.
DEFAULT_FOVY_DEG = 45.0

# MuJoCo cameras look along -z with +y up.
CAMERA_XYZ = rr.ViewCoordinates.RUB

# MuJoCo body -> URDF frame, for body-mounted cameras. robosuite's hand body and the URDF's
# `fer_hand` coincide: 0.1065 m along link 7, twisted -45°.
BODY_FRAMES = {"robot0_right_hand": "fer_hand"}


@dataclass(frozen=True)
class CameraMount:
    """One recorded camera: where it sits, on which frame, and how wide it sees."""

    name: str  # entity leaf, e.g. "eye_in_hand"
    parent_frame: str
    translation: list[float]
    quaternion: list[float]  # xyzw
    fovy_deg: float


def _find_camera(root: ET.Element, name: str) -> tuple[ET.Element, str]:
    """The `<camera>` element for a recorded image and the frame it is fixed to."""
    names = {name, f"robot0_{name}"}
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("model_file has no worldbody")
    for element in worldbody.findall("camera"):
        if element.get("name") in names:
            return element, WORLD_FRAME
    for body in worldbody.iter("body"):
        for element in body.findall("camera"):
            if element.get("name") in names:
                frame = BODY_FRAMES.get(body.get("name", ""))
                if frame is None:
                    raise ValueError(f"camera {name!r} rides body {body.get('name')!r}, which has no URDF frame")
                return element, frame
    raise ValueError(f"model_file has no camera named {name!r} or robot0_{name!r}")


def camera_mounts(model_file: str, cameras: list[Camera]) -> list[CameraMount]:
    """The recorded cameras' mounts, read from the demo's MuJoCo XML."""
    root = ET.fromstring(model_file)
    mounts = []
    for camera in cameras:
        element, parent_frame = _find_camera(root, camera.name)
        w, x, y, z = (float(value) for value in element.get("quat", "1 0 0 0").split())  # MuJoCo stores wxyz
        mounts.append(
            CameraMount(
                name=camera.name,
                parent_frame=parent_frame,
                translation=[float(value) for value in element.get("pos", "0 0 0").split()],
                quaternion=[x, y, z, w],
                fovy_deg=float(element.get("fovy", DEFAULT_FOVY_DEG)),
            )
        )
    return mounts


def image_from_camera(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """The pinhole matrix robosuite derives from MuJoCo's vertical field of view: square pixels, centred principal point."""
    focal = 0.5 * height / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])


def camera_chunk(mount: CameraMount, camera: Camera) -> Chunk:
    """
    The camera entity's static pose, intrinsics and frame binding.

    The `Transform3D` and the `Pinhole` both name the `parent -> camera` edge; the `CoordinateFrame`
    puts the entity, image included, in that frame.
    """
    return Chunk.from_columns(
        f"{CAMERA_ENTITY_PREFIX}/{mount.name}",
        indexes=[],
        columns=[
            *rr.Transform3D.columns(
                translation=[mount.translation],
                quaternion=[mount.quaternion],
                parent_frame=[mount.parent_frame],
                child_frame=[mount.name],
            ),
            *rr.Pinhole.columns(
                image_from_camera=[image_from_camera(mount.fovy_deg, camera.width, camera.height)],
                resolution=[[camera.width, camera.height]],
                camera_xyz=[CAMERA_XYZ],
                parent_frame=[mount.parent_frame],
                child_frame=[mount.name],
            ),
            *rr.CoordinateFrame.columns(frame=[mount.name]),
        ],
    )


def convert_demo(reader: Hdf5Reader, task: str, demo: str, rrd_root: Path) -> Path:
    """Write one demo's camera layer; returns the written path."""
    cameras = discover_cameras(reader, demo)
    mounts = camera_mounts(str(reader.attributes(f"/data/{demo}")["model_file"]), cameras)
    rec_id = recording_id(task, demo)
    out_path = rrd_root / layer_relpath("cameras", rec_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = [camera_chunk(mount, camera) for mount, camera in zip(mounts, cameras)]
    LazyChunkStream.from_iter(chunks).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=rec_id
    )
    return out_path


def main(argv: list[str]) -> None:
    inputs = task_files(argv)
    print(f"Building camera layer for {len(inputs)} task file(s) -> {RRD_ROOT / 'cameras'}/")
    for path, task in inputs:
        reader = Hdf5Reader(path)
        for demo in demo_keys(reader):
            out = convert_demo(reader, task, demo, RRD_ROOT)
            print(f"  {recording_id(task, demo)}: {out}")


if __name__ == "__main__":
    main(sys.argv)
