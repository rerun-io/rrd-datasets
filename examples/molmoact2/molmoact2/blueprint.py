"""
Build the default Rerun blueprint for MolmoAct2 episodes and save it as an `.rbl`.

Registered as the dataset's default, every episode opens with the same layout: the instruction
and the three cameras on top, and one state plot and one action plot per arm below.

Run:  pixi run -e molmo blueprint   # regenerates blueprints/molmoact2/default.rbl
"""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from molmoact2.base_layer import APPLICATION_ID, PROPERTY, TIMESTAMP_TIMELINE
from rrd_datasets_common.paths import default_blueprint_path

BLUEPRINT_PATH = default_blueprint_path("molmoact2")

ACTION_ENTITY = "/action"
STATE_ENTITY = "/observation.state"
# The source camera names, with the view titles: `left` and `right` are the wrist cameras.
CAMERAS = {"top": "Top", "left": "Left wrist", "right": "Right wrist"}
PROPERTIES_ENTITY = f"/__properties/{PROPERTY}"

# The 14 state and action values hold the left arm's 6 joints and gripper, then the right arm's,
# named as in the source's `info.json`.
ARMS = ["left", "right"]
JOINTS = [*(f"joint_{index}" for index in range(6)), "gripper"]
SERIES_NAMES = [f"{arm}_{joint}.pos" for arm in ARMS for joint in JOINTS]

# One color per joint, the same in all four plots, so a joint's state and action match.
JOINT_COLORS = [0x3987E5FF, 0xD95926FF, 0x199E70FF, 0xC98500FF, 0xD55181FF, 0x008300FF, 0x9085E9FF]

# Without a fixed range, a plot shows only a window around the time cursor, which cuts off the
# start of long episodes.
WHOLE_EPISODE = rr.TimeRange(start=rr.TimeRangeBoundary.infinite(), end=rr.TimeRangeBoundary.infinite())


def arm_lines(arm: str) -> rr.SeriesLines:
    """
    Show only one arm's 7 series of the 14.

    The legend lists hidden series too. The other arm's 7 share one name, so the legend shows them
    as a single entry that turns that arm back on.
    """
    visible = [name.startswith(f"{arm}_") for name in SERIES_NAMES]
    other_arm = next(other for other in ARMS if other != arm)
    names = [name if shown else f"{other_arm} arm" for name, shown in zip(SERIES_NAMES, visible)]
    return rr.SeriesLines.from_fields(colors=JOINT_COLORS * len(ARMS), visible_series=visible, names=names)


def arm_plot(entity: str, arm: str, title: str) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        origin=entity,
        contents=f"+ {entity}",
        name=f"{arm.capitalize()} {title}",
        overrides={entity: [arm_lines(arm)]},
        axis_x=rrb.archetypes.TimeAxis(view_range=WHOLE_EPISODE),
    )


def instruction_view() -> rrb.TextDocumentView:
    """The episode's free-text instruction, read from the recording properties."""
    return rrb.TextDocumentView(
        origin=PROPERTIES_ENTITY,
        contents=f"+ {PROPERTIES_ENTITY}",
        name="Instruction",
        overrides={
            PROPERTIES_ENTITY: [
                rr.TextDocument.from_fields().visualizer(
                    mappings=[
                        rrb.encodings.VisualizerComponentMapping(
                            target="TextDocument:text",
                            source_kind=rrb.encodings.ComponentSourceKind.SourceComponent,
                            source_component="instruction",
                        )
                    ]
                )
            ]
        },
    )


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            instruction_view(),
            rrb.Horizontal(
                *(
                    rrb.Spatial2DView(origin=f"/observation.images.{camera}", name=title)
                    for camera, title in CAMERAS.items()
                ),
                name="Cameras",
            ),
            rrb.Grid(
                *(arm_plot(STATE_ENTITY, arm, "state") for arm in ARMS),
                *(arm_plot(ACTION_ENTITY, arm, "action") for arm in ARMS),
                grid_columns=2,
                name="Signals",
            ),
            row_shares=[4, 22, 34],
        ),
        rrb.TimePanel(
            timeline=TIMESTAMP_TIMELINE,
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
