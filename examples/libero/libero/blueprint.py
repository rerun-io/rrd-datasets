"""
Build the default Rerun blueprint for LIBERO demos and save it as an `.rbl`.

The blueprint decides how a demo is shown. Registered as the dataset's default, every segment
opens with the same layout: the two camera panes with the task instruction above them, and the
joint / gripper / action / end-effector plots.

Run:  pixi run -e libero blueprint   # regenerates blueprints/libero/default.rbl
"""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from libero.base_layer import APPLICATION_ID
from rrd_datasets_common.paths import default_blueprint_path

BLUEPRINT_PATH = default_blueprint_path("libero")

# Legend labels — the arrays otherwise fall back to bare index labels 0…6.
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]  # Panda arm joints, in motor order
GRIPPER_NAMES = ["finger1", "finger2"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]  # OSC_POSE deltas + gripper


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            # Left: the task instruction, with the two camera streams beneath it.
            rrb.Vertical(
                rrb.TextDocumentView(origin="/task/instruction", name="Instruction"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="/camera/agentview", name="Agent view"),
                    rrb.Spatial2DView(origin="/camera/eye_in_hand", name="Eye in hand"),
                    name="Cameras",
                ),
                row_shares=[1, 6],
            ),
            # Right: one plot per signal group — their value ranges differ too much to share axes.
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/robot/joint_states",
                    name="Joints",
                    defaults=[rr.SeriesLines(names=JOINT_NAMES)],
                ),
                rrb.TimeSeriesView(
                    origin="/robot/gripper_states",
                    name="Gripper",
                    defaults=[rr.SeriesLines(names=GRIPPER_NAMES)],
                ),
                rrb.TimeSeriesView(
                    origin="/action",
                    name="Action",
                    defaults=[rr.SeriesLines(names=ACTION_NAMES)],
                ),
                # Position (m) and the rotation vector (rad) share one pane; their labels ride
                # statically in the data.
                rrb.TimeSeriesView(
                    origin="/",
                    name="End effector",
                    contents=["+ /robot/ee_pos/**", "+ /robot/ee_ori/**"],
                ),
                name="Signals",
            ),
            column_shares=[3, 2],
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
