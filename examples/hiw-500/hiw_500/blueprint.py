"""
Build the default Rerun blueprint for HIW-500 episodes and save it as an `.rbl`.

The blueprint decides how an episode is shown. The `register` task installs this as the dataset's default
so every segment opens with a sensible humanoid-manipulation layout: a 3D scene, the four camera
streams, the task log, and the joint / end-effector signal plots.

The 3D scene must target the `odom` frame — the robot pose lives in a named frame graph
(`odom -> pelvis -> ...`), so a view left at the root frame renders empty ("No transform path").

Run:  pixi run -e hiw blueprint   # regenerates blueprints/hiw-500/default.rbl
"""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from hiw_500.base_layer import APPLICATION_ID
from rrd_datasets_common.paths import default_blueprint_path

# Regenerating overwrites the committed default, so the change shows up in
# `git status` and ships with the next commit.
BLUEPRINT_PATH = default_blueprint_path("hiw-500")

LEFT_WRIST = "/camera/left_wrist/image/compressed"
RIGHT_WRIST = "/camera/right_wrist/image/compressed"

# The IR layer's streams: a stereo infrared pair per wrist. Episodes recorded before the IR
# cameras reached the rig have no such entities, and the tab simply comes up empty for them.
WRIST_IR = [
    ("/camera/left_wrist/ir1/compressed", "L IR1"),
    ("/camera/left_wrist/ir2/compressed", "L IR2"),
    ("/camera/right_wrist/ir1/compressed", "R IR1"),
    ("/camera/right_wrist/ir2/compressed", "R IR2"),
]


def _wrist_view(origin: str, name: str) -> rrb.Spatial2DView:
    # The decoder leaves CoordinateFrame:frame="" on the wrist cams, which keeps the image from
    # showing in a 2D view; override it to the entity's own implicit frame so it renders.
    # TODO(michael): remove this override once we have gripper cam extrinsics (then the wrist
    # cams get a real frame and can be placed in 3D like the head cam).
    return rrb.Spatial2DView(
        origin=origin,
        name=name,
        overrides={origin: rr.CoordinateFrame(frame="tf#")},  # type: ignore[dict-item]
    )


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                # Left: 3D scene with the subtask state timeline beneath it.
                rrb.Vertical(
                    # 3D scene in the odom frame: robot mesh + FK, base, EE positions, left head cam.
                    rrb.Spatial3DView(
                        origin="/",
                        name="Scene",
                        contents=[
                            "+ /robot/**",
                            "+ /odom/**",
                            "+ /state/base",
                            # Only the left head eye in 3D — the right overlaps it distractingly (it
                            # stays in its own 2D pane). TODO(michael): include the wrist cams here
                            # too once we have gripper cam extrinsics; for now they have no real frame.
                            "+ /camera/head/left/**",
                            "+ /lerobot/ee_state/**",
                            "+ /lerobot/ee_action/**",
                        ],
                        spatial_information=rrb.SpatialInformation(target_frame="odom"),
                        eye_controls=rrb.archetypes.EyeControls3D(
                            position=[-2.2, -3.2, 0.7], look_target=[-1.8, -1.9, 0.6]
                        ),
                        # Pull the head image plane close (25 cm) and make it semi-transparent so it
                        # doesn't occlude the robot/scene behind it. Scoped to the 3D view only.
                        overrides={
                            "/camera/head/left": [
                                rr.Pinhole.from_fields(image_plane_distance=0.25),
                                rr.EncodedImage.from_fields(opacity=0.5),
                            ]
                        },
                    ),
                    rrb.StateTimelineView(origin="/task/subtask", name="Subtasks"),
                    row_shares=[6, 1],
                ),
                # Right: the head pair above the wrists, whose colour and infrared views
                # share one slot as tabs — the same cameras, two modalities.
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="/camera/head/left", name="Head L"),
                        rrb.Spatial2DView(origin="/camera/head/right", name="Head R"),
                        name="Head",
                    ),
                    rrb.Tabs(
                        rrb.Horizontal(
                            _wrist_view(LEFT_WRIST, "Wrist L"),
                            _wrist_view(RIGHT_WRIST, "Wrist R"),
                            name="RGB",
                        ),
                        rrb.Grid(
                            *(_wrist_view(origin, name) for origin, name in WRIST_IR),
                            grid_columns=2,
                            name="IR",
                        ),
                        name="Wrists",
                    ),
                    name="Cameras",
                ),
                column_shares=[3, 2],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/state/joint", name="Joints"),
                rrb.TimeSeriesView(
                    origin="/lerobot",
                    name="End-effector",
                    contents=["+ /lerobot/ee_state/**", "+ /lerobot/ee_action/**", "+ /lerobot/gripper/**"],
                ),
                name="Signals",
            ),
            row_shares=[7, 3],
        ),
        rrb.TimePanel(
            timeline="message_publish_time",
            play_state=rrb.components.PlayState.Following,
            state=rrb.components.PanelState.Collapsed,
        ),
        collapse_panels=True,
    )


def main() -> None:
    BLUEPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_blueprint().save(APPLICATION_ID, str(BLUEPRINT_PATH))
    print(f"Wrote blueprint -> {BLUEPRINT_PATH}")


if __name__ == "__main__":
    main()
