"""
Build the default Rerun blueprint for HIW-500 episodes and save it as an `.rbl`.

The blueprint decides how an episode is shown: a 3D scene, the camera streams, the subtask lane and
the signal plots. The plots read the raw message structs through component mappings, so no scalar
is materialised in any layer. The `register` task installs the result as the dataset's default.

Run:  pixi run -e hiw blueprint   # regenerates blueprints/hiw-500/default.rbl
"""

from __future__ import annotations

from collections.abc import Sequence

import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint.datatypes import ComponentSourceKind, VisualizerComponentMapping
from rerun.blueprint.visualizers import Visualizer

from hiw_500.base_layer import (
    APPLICATION_ID,
    G1_JOINT_NAMES,
    MOTOR_SLOTS,
    MSG_LOWSTATE,
    MSG_MOTOR_CMD,
    MSG_MOTOR_STATE,
    N_JOINTS,
)
from hiw_500.derived_archetypes_layer import EE_NAMES, MSG_WBC, WBC_TOPIC
from hiw_500.odom_layer import START_FRAME
from rrd_datasets_common.paths import default_blueprint_path

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


LOWSTATE_TOPIC = "/stamped/lowstate"


def _series(
    component: str, selector: str, names: str | Sequence[str], visible: Sequence[bool] | None = None
) -> Visualizer:
    """
    The series a selector reads out of a message struct on the view's entity.

    A `SeriesLines` visualizer takes its `Scalars` input from `selector` applied to `component`: a
    scalar path is one series, a `[]` path one series per array element, named and shown in order.
    """
    return rr.SeriesLines(names=names, visible_series=visible).visualizer(
        mappings=[
            VisualizerComponentMapping(
                target="Scalars:scalars",
                source_kind=ComponentSourceKind.SourceComponent,
                source_component=component,
                selector=selector,
            )
        ]
    )


def joint_series() -> list[Visualizer]:
    """
    The angles of the 29 real motors in the 35-slot `motor_state` array, named by joint; the unused slots stay hidden.

    One `[].q` mapping serves every series. A mapping per index copies every `MotorState` field of every
    row for each series on each frame, which halves the viewer's frame rate.
    Only the angle: `dq` and `tau_est` sit in the same struct for a mapping or a drag onto the view when wanted.
    """
    visible = [True] * N_JOINTS + [False] * (MOTOR_SLOTS - N_JOINTS)
    return [_series(MSG_LOWSTATE, ".data.motor_state[].q", G1_JOINT_NAMES, visible)]


def ee_series() -> list[Visualizer]:
    """`<kind>/<arm>/<field>` for the two width-12 end-effector arrays of the wbc struct."""
    return [
        _series(MSG_WBC, f".ee_{kind}[{index}]", f"{kind}/{name}")
        for kind in ("state", "action")
        for index, name in enumerate(EE_NAMES)
    ]


def gripper_control_series() -> list[Visualizer]:
    """The four teleop gripper inputs of the wbc struct."""
    return [
        _series(MSG_WBC, f".gripper_controls.{control}", control)
        for control in ("left_trigger", "left_squeeze", "right_trigger", "right_squeeze")
    ]


def dex1_series() -> dict[str, list[Visualizer]]:
    """The measured and commanded jaw angle of each dex1 gripper, on its own topic entity."""
    return {
        **{
            f"/stamped/dex1/{side}/state": [_series(MSG_MOTOR_STATE, ".data.states[0].q", f"state/{side}/q")]
            for side in ("left", "right")
        },
        **{
            f"/stamped/dex1/{side}/cmd": [_series(MSG_MOTOR_CMD, ".data.cmds[0].q", f"cmd/{side}/q")]
            for side in ("left", "right")
        },
    }


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
                    # 3D scene in the episode's `start` frame: robot mesh + FK, EE positions, left head cam.
                    # `start` is fixed in `odom` at the robot's initial pose (odom layer), so one eye frames
                    # every episode alike while the robot moves through a still world. The frame graph
                    # (`odom -> pelvis -> …`) has to connect: left at the root frame the view renders empty
                    # ("No transform path").
                    rrb.Spatial3DView(
                        origin="/",
                        name="Scene",
                        contents=[
                            "+ /robot/**",
                            "+ /odom/**",
                            # Only the left head eye in 3D — the right overlaps it distractingly (it
                            # stays in its own 2D pane). TODO(michael): include the wrist cams here
                            # too once we have gripper cam extrinsics; for now they have no real frame.
                            "+ /camera/head/left/**",
                            "+ /lerobot/ee_state/**",
                            "+ /lerobot/ee_action/**",
                        ],
                        spatial_information=rrb.SpatialInformation(target_frame=START_FRAME),
                        # 1.9 m off the robot's rear-left quarter, looking at its hip (0.7 m up).
                        eye_controls=rrb.archetypes.EyeControls3D(
                            position=[-1.2, -1.38, 1.12], look_target=[0.0, 0.0, 0.6]
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
                    row_shares=[4, 1],
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
            # Every plot maps its series out of a struct; the instructions on an entity replace the
            # viewer's defaults, so voltage, temperature and the other fields stay off these axes.
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin=LOWSTATE_TOPIC,
                    name="Joints (q)",
                    overrides={LOWSTATE_TOPIC: joint_series()},  # type: ignore[dict-item]
                ),
                # End-effector poses and gripper controls plot on unrelated scales, so they get
                # a view each rather than one axis that flattens both.
                rrb.TimeSeriesView(
                    origin=WBC_TOPIC,
                    name="End-effector",
                    overrides={WBC_TOPIC: ee_series()},  # type: ignore[dict-item]
                ),
                rrb.Tabs(
                    # What the gripper actually did: the dex1 jaw, measured against commanded.
                    rrb.TimeSeriesView(
                        origin="/stamped/dex1",
                        name="Gripper(Dex1)",
                        overrides=dex1_series(),  # type: ignore[arg-type]
                    ),
                    # The teleop inputs behind it, whose 0-10 range would flatten the jaw angle.
                    rrb.TimeSeriesView(
                        origin=WBC_TOPIC,
                        name="Gripper(LeRobot)",
                        overrides={WBC_TOPIC: gripper_control_series()},  # type: ignore[dict-item]
                    ),
                    name="Gripper",
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
