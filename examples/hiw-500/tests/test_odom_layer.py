"""The odometry layer's `start` frame: the robot's initial pose, flattened to the floor and reduced to its heading."""

from __future__ import annotations

import math

from hiw_500.odom_layer import START_ENTITY, START_FRAME, WORLD_FRAME, start_frame_chunk, yaw_only


def _quaternion_xyzw(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll / 2),
        math.sin(roll / 2),
        math.cos(pitch / 2),
        math.sin(pitch / 2),
        math.cos(yaw / 2),
        math.sin(yaw / 2),
    )
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def test_yaw_only_keeps_the_heading_and_drops_roll_and_pitch() -> None:
    heading = 0.8
    x, y, z, w = yaw_only(_quaternion_xyzw(roll=0.2, pitch=-0.3, yaw=heading))
    assert (x, y) == (0.0, 0.0)
    assert math.isclose(2 * math.atan2(z, w), heading, abs_tol=1e-9)


def test_the_start_frame_sits_on_the_floor_under_the_first_pose() -> None:
    chunk = start_frame_chunk([-3.5, -0.25, 0.7], _quaternion_xyzw(0.0, 0.0, 0.5))
    assert chunk.entity_path == START_ENTITY
    assert chunk.is_static
    batch = chunk.to_record_batch()
    assert batch.column("Transform3D:translation").to_pylist() == [
        [[-3.5, -0.25, 0.0]]
    ]  # float32 column, so values exactly representable
    assert batch.column("Transform3D:parent_frame").to_pylist() == [[WORLD_FRAME]]
    assert batch.column("Transform3D:child_frame").to_pylist() == [[START_FRAME]]
