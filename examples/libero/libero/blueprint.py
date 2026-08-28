"""
Build the default Rerun blueprint for LIBERO demos and save it as an `.rbl`.

The blueprint decides how a demo is shown. Registered as the dataset's default, every segment
opens with the same layout: the task instruction and the posed arm on the left, the two camera
panes on the right, and the joint / gripper / action / end-effector plots along the bottom. The
plots read straight off the reflected columns.

Run:  pixi run -e libero blueprint   # regenerates blueprints/libero/default.rbl
"""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from libero import camera_layer, urdf_layer
from libero.base_layer import APPLICATION_ID, DEMO_ENTITY, OBS_ENTITY
from rrd_datasets_common.paths import default_blueprint_path

BLUEPRINT_PATH = default_blueprint_path("libero")

# Legend labels — the arrays otherwise fall back to bare index labels 0…6.
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]  # Panda arm joints, in motor order
GRIPPER_NAMES = ["finger1", "finger2"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]  # OSC_POSE deltas + gripper
EE_POS_NAMES = ["x", "y", "z"]
EE_ORI_NAMES = ["rx", "ry", "rz"]


def series(component: str, names: list[str]) -> rr.Visualizer:
    """
    A line-series visualizer reading a reflected array column.

    The base layer derives no `Scalars`, so the mapping binds `Scalars:scalars` to `component`;
    `[]` spreads the array into one series per element, labelled by `names`.
    """
    return rr.SeriesLines(names=names).visualizer(
        mappings=[
            rrb.datatypes.VisualizerComponentMapping(
                target="Scalars:scalars",
                source_kind=rrb.datatypes.ComponentSourceKind.SourceComponent,
                source_component=component,
                selector="[]",
            )
        ]
    )


URDF_ORIGIN = f"/{urdf_layer.ENTITY_PREFIX}"
CAMERA_ORIGIN = camera_layer.CAMERA_ENTITY_PREFIX


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            # Top: the instruction strip and the 3D arm on the left, the two camera streams
            # stacked on the right.
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/task/instruction", name="Instruction"),
                    # The eye sits behind the agent camera and off to its side, so the pane shows that
                    # camera's frustum and image plane next to the arm.
                    rrb.Spatial3DView(
                        origin="/",
                        contents=[f"+ {URDF_ORIGIN}/fer/**", f"+ {CAMERA_ORIGIN}/**"],
                        name="Robot",
                        eye_controls=rrb.archetypes.EyeControls3D(position=[1.0, 0.5, 1.8]),
                        overrides={
                            f"{CAMERA_ORIGIN}/agentview": [
                                rr.Pinhole.from_fields(image_plane_distance=0.4),
                                rr.Image.from_fields(opacity=0.6),
                            ],
                            f"{CAMERA_ORIGIN}/eye_in_hand": [
                                rr.Pinhole.from_fields(image_plane_distance=0.12),
                                rr.Image.from_fields(opacity=0.6),
                            ],
                        },
                    ),
                    row_shares=[1, 6],
                ),
                # The camera frames are square, so the column width sets their on-screen height.
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/camera/agentview", name="Agent view"),
                    rrb.Spatial2DView(origin="/camera/eye_in_hand", name="Eye in hand"),
                ),
                column_shares=[5, 2],
            ),
            # Bottom: one plot per signal group — their value ranges differ too much to share axes.
            # Each view lists exactly the entity it plots, so the other columns on that entity stay out.
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin=OBS_ENTITY,
                    contents=f"+ {OBS_ENTITY}",
                    name="Joints",
                    overrides={OBS_ENTITY: [series("joint_states", JOINT_NAMES)]},
                ),
                rrb.TimeSeriesView(
                    origin=OBS_ENTITY,
                    contents=f"+ {OBS_ENTITY}",
                    name="Gripper",
                    overrides={OBS_ENTITY: [series("gripper_states", GRIPPER_NAMES)]},
                ),
                rrb.TimeSeriesView(
                    origin=DEMO_ENTITY,
                    contents=f"+ {DEMO_ENTITY}",
                    name="Action",
                    overrides={DEMO_ENTITY: [series("actions", ACTION_NAMES)]},
                ),
                # Position (m) and the rotation vector (rad) share one pane.
                rrb.TimeSeriesView(
                    origin=OBS_ENTITY,
                    contents=f"+ {OBS_ENTITY}",
                    name="End effector",
                    overrides={OBS_ENTITY: [series("ee_pos", EE_POS_NAMES), series("ee_ori", EE_ORI_NAMES)]},
                ),
                name="Signals",
            ),
            row_shares=[8, 3],
        ),
        rrb.TimePanel(
            timeline="sim_time",
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
