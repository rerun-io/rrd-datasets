"""
Default Rerun blueprint for the ABC-130k base-layer recordings.

Writes the default `blueprints/abc-130k/default.rbl` bound to the shared application id
`abc_130k`, so it can be applied ad hoc (`rerun <rrds> blueprints/abc-130k/default.rbl`) or
registered as a catalog dataset's default blueprint once that layer exists.
There is no 3D view: the base layer carries no transforms or geometry that is useful for 3D view.

Run:  pixi run -e abc blueprint
"""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from abc_130k.convert import APPLICATION_ID, ARM_JOINT_NAMES, Kind, Side
from rrd_datasets_common.paths import default_blueprint_path

BLUEPRINT_PATH = default_blueprint_path("abc-130k")


def _camera_view(origin: str, name: str) -> rrb.Spatial2DView:
    """
    A 2D view for one `VideoStream` camera entity.

    The MCAP decoder tags each camera with a `CoordinateFrame` that has no materialized transform
    tree in the base layer, which keeps the video from showing in a 2D view. Override it to the
    entity's own implicit frame (`tf#`) so the stream renders. The `Pinhole` calibration lives on the
    sibling `<origin>-info` entity and is not needed to display the video.
    """
    return rrb.Spatial2DView(
        origin=origin,
        name=name,
        overrides={origin: rr.CoordinateFrame(frame="tf#")},  # type: ignore[dict-item]
    )


def _signal_contents(signal: str) -> list[str]:
    """
    `+ include` rules picking one signal (`q`/`dq`/`tau`) for a plot.

    Wildcards match whole subtrees, not leaf names, so we list each entity: the arm array (6 joints) and
    the gripper, per side. Action carries only the commanded position, so `dq`/`tau` are state-only. The
    converter normalizes the gripper onto its own entity, so each plot is just arm + gripper per side.
    """
    kinds = (Kind.STATE, Kind.ACTION) if signal == "q" else (Kind.STATE,)
    paths = [f"/{side}/arm/{kind}/{signal}" for side in Side for kind in kinds]
    paths += [f"/{side}/gripper/{kind}/{signal}" for side in Side for kind in kinds]
    return [f"+ {p}" for p in paths]


def _series_name(side: Side, kind: Kind, leaf: str) -> str:
    """
    One legend label: side initial (`L`/`R`), `cmd` for action, then the DoF name.

    Arrays otherwise fall back to bare index labels, so several in one plot collide; the side and `cmd`
    prefixes keep every series distinct. Shared by the arm joints and the gripper.
    """
    return f"{side[0].upper()}{' cmd' if kind == Kind.ACTION else ''} {leaf}"


def _arm_overrides(signal: str) -> dict[str, rr.SeriesLines]:
    """
    Legend labels for the arm arrays -- the 6 joints, on every plot.

    The converter normalizes the arm to joints-only (the gripper is split onto its own entity), so the
    arrays are always 6 wide and never carry a gripper series here.
    """
    kinds = (Kind.STATE, Kind.ACTION) if signal == "q" else (Kind.STATE,)
    return {
        f"/{side}/arm/{kind}/{signal}": rr.SeriesLines(names=[_series_name(side, kind, n) for n in ARM_JOINT_NAMES])
        for side in Side
        for kind in kinds
    }


def _gripper_overrides(signal: str) -> dict[str, rr.SeriesLines]:
    """
    Legend label for the gripper entity, matching the joint label scheme (`L gripper`, `R cmd gripper`).

    The converter routes the gripper's dq/tau onto this entity too, so it's labeled on every plot.
    """
    kinds = (Kind.STATE, Kind.ACTION) if signal == "q" else (Kind.STATE,)
    return {
        f"/{side}/gripper/{kind}/{signal}": rr.SeriesLines(names=[_series_name(side, kind, "gripper")])
        for side in Side
        for kind in kinds
    }


def _signal_view(signal: str, name: str) -> rrb.TimeSeriesView:
    """One signal plot: labeled arm + gripper series, legend bottom-left."""
    overrides = {**_arm_overrides(signal), **_gripper_overrides(signal)}
    return rrb.TimeSeriesView(
        origin="/",
        name=name,
        contents=_signal_contents(signal),
        overrides=overrides,  # type: ignore[arg-type]
        plot_legend=rrb.PlotLegend(rrb.components.Corner2D.LeftBottom),
    )


def build_blueprint() -> rrb.Blueprint:
    """The default ABC-130k layout: cameras + instruction/subtasks over the signal plots."""
    return rrb.Blueprint(
        rrb.Vertical(
            # Camera row: the top cameras share one slot via tabs (mono `/top-camera`, then ZED-X
            # stereo `/top-{left,right}-camera` side by side), followed by the two wrists; a view
            # whose origin is absent in a recording renders empty.
            rrb.Horizontal(
                rrb.Tabs(
                    _camera_view("/top-camera", "Top"),
                    rrb.Horizontal(
                        _camera_view("/top-left-camera", "Top L"),
                        _camera_view("/top-right-camera", "Top R"),
                        name="Top Stereo",
                    ),
                    name="Top",
                ),
                _camera_view("/left-wrist-camera", "Wrist L"),
                _camera_view("/right-wrist-camera", "Wrist R"),
                name="Cameras",
            ),
            # Middle: the whole-episode instruction and the subtask state timeline share one slot via tabs.
            rrb.Tabs(
                rrb.TextDocumentView(origin="/instruction", name="Instruction"),
                rrb.StateTimelineView(origin="/task/subtask", name="Subtasks"),
            ),
            # Bottom: signals split by kind so clashing value ranges get their own axis.
            rrb.Horizontal(
                _signal_view("q", "Positions (q)"),
                _signal_view("dq", "Velocities (dq)"),
                _signal_view("tau", "Torques (tau)"),
                name="Signals",
            ),
            row_shares=[7, 3, 6],
        ),
        rrb.TimePanel(
            timeline="message_publish_time",
            play_state=rrb.components.PlayState.Following,
            state=rrb.components.PanelState.Collapsed,
        ),
        collapse_panels=True,
    )


def main() -> None:
    """Write the blueprint to the committed default path."""
    BLUEPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_blueprint().save(APPLICATION_ID, str(BLUEPRINT_PATH))
    print(f"Wrote blueprint -> {BLUEPRINT_PATH}")


if __name__ == "__main__":
    main()
