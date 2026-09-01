"""Tests for the camera layer: the intrinsics, the mounts read from the MuJoCo XML, and a synthesized-file round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import AGENTVIEW_POS, EYE_IN_HAND_POS, HEIGHT, MODEL_FILE, WIDTH, write_fixture
from rerun.experimental import Hdf5Reader, RrdReader

from libero.base_layer import Camera
from libero.camera_layer import camera_mounts, convert_demo, image_from_camera
from libero.urdf_layer import WORLD_FRAME

CAMERAS = [Camera("agentview", "agentview_rgb", HEIGHT, WIDTH), Camera("eye_in_hand", "eye_in_hand_rgb", HEIGHT, WIDTH)]


def test_the_focal_length_follows_the_vertical_field_of_view() -> None:
    """Robosuite's convention: `f = H / (2 tan(fovy / 2))`, square pixels, principal point at the centre."""
    k = image_from_camera(90.0, 128, 128)
    assert k[0, 0] == pytest.approx(64.0) and k[1, 1] == pytest.approx(64.0)
    assert k[0, 2] == pytest.approx(64.0) and k[1, 2] == pytest.approx(64.0)
    assert image_from_camera(45.0, 128, 128)[0, 0] == pytest.approx(154.5, abs=0.1)
    assert image_from_camera(75.0, 128, 128)[0, 0] == pytest.approx(83.4, abs=0.1)


def test_mounts_come_from_the_model_file() -> None:
    agentview, eye_in_hand = camera_mounts(MODEL_FILE, CAMERAS)

    assert agentview.parent_frame == WORLD_FRAME
    assert agentview.translation == pytest.approx(AGENTVIEW_POS)
    assert agentview.fovy_deg == 45.0  # MuJoCo's default, the XML gives none

    assert eye_in_hand.parent_frame == "fer_hand"
    assert eye_in_hand.translation == pytest.approx(EYE_IN_HAND_POS)
    assert eye_in_hand.fovy_deg == 75.0
    # MuJoCo writes wxyz; Rerun takes xyzw.
    assert eye_in_hand.quaternion == pytest.approx([0.707107, 0.707107, 0.0, 0.0])


def test_a_camera_on_an_unmapped_body_fails_loudly() -> None:
    """A camera on a body with no URDF frame would otherwise float at the origin."""
    xml = '<mujoco><worldbody><body name="robot0_base"><camera name="robot0_eye_in_hand"/></body></worldbody></mujoco>'
    with pytest.raises(ValueError, match="robot0_base"):
        camera_mounts(xml, CAMERAS[1:])


def test_a_missing_camera_fails_loudly() -> None:
    with pytest.raises(ValueError, match="agentview"):
        camera_mounts("<mujoco><worldbody/></mujoco>", CAMERAS[:1])


def test_camera_layer_round_trip(tmp_path: Path) -> None:
    fixture = tmp_path / "suite_task_demo.hdf5"
    write_fixture(fixture)
    out = convert_demo(Hdf5Reader(fixture), "suite/task", "demo_0", tmp_path / "rrds")

    reader = RrdReader(str(out))
    (entry,) = reader.recordings()
    assert entry.recording_id == "suite/task__demo_0"

    edges: dict[str, tuple[str, str]] = {}
    components: dict[str, set[str]] = {}
    for chunk in reader.stream(store=entry):
        batch = chunk.to_record_batch()
        assert chunk.is_static
        components.setdefault(chunk.entity_path, set()).update(batch.schema.names)
        if "Transform3D:parent_frame" in batch.schema.names:
            edges[chunk.entity_path] = (
                batch.column("Transform3D:parent_frame").to_pylist()[0][0],
                batch.column("Transform3D:child_frame").to_pylist()[0][0],
            )

    assert edges == {
        "/camera/agentview": (WORLD_FRAME, "agentview"),
        "/camera/eye_in_hand": ("fer_hand", "eye_in_hand"),
    }
    for entity in ("/camera/agentview", "/camera/eye_in_hand"):
        assert {
            "Pinhole:image_from_camera",
            "Pinhole:resolution",
            "Pinhole:camera_xyz",
            "CoordinateFrame:frame",
        } <= components[entity]

    # The eye-in-hand camera is the one with the explicit, wider field of view.
    for chunk in reader.stream(store=entry):
        batch = chunk.to_record_batch()
        if chunk.entity_path == "/camera/eye_in_hand" and "Pinhole:image_from_camera" in batch.schema.names:
            k = np.asarray(batch.column("Pinhole:image_from_camera").to_pylist()[0][0]).reshape(3, 3)
            assert k[0, 0] == pytest.approx(image_from_camera(75.0, WIDTH, HEIGHT)[0, 0])
