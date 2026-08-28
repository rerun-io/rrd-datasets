"""
Tests that the blueprint builds under the pinned SDK and reads back with every declared view.

The blueprint is a generated artifact (`pixi run -e libero blueprint`), and nothing else exercises
the `rrb` constructors between SDK bumps. Saved blueprints differ byte for byte on every save, so
the tests compare structure — the store identity, the tally of view classes, and the component
mappings that feed the plots — never bytes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rerun.experimental import RrdReader

from libero.base_layer import APPLICATION_ID
from libero.blueprint import build_blueprint

# One entry per view in `build_blueprint`: the two camera panes, the instruction text, and the
# joint / gripper / action / end-effector plots.
EXPECTED_VIEWS = {"2D": 2, "TextDocument": 1, "TimeSeries": 4}

# The reflected columns each plot reads; the end-effector pane reads two.
EXPECTED_SOURCES = {"joint_states", "gripper_states", "actions", "ee_pos", "ee_ori"}


def _read_back(rbl: Path) -> tuple[Counter[str], list[dict[str, object]]]:
    """The view-class tally and the visualizer mappings of a saved blueprint, read back through the SDK."""
    reader = RrdReader(str(rbl))
    (entry,) = reader.blueprints()
    assert entry.kind == "blueprint"
    assert entry.application_id == APPLICATION_ID
    classes: Counter[str] = Counter()
    mappings: list[dict[str, object]] = []
    for chunk in reader.stream(store=entry):
        batch = chunk.to_record_batch()
        if "ViewBlueprint:class_identifier" in batch.schema.names:
            for (identifier,) in batch.column("ViewBlueprint:class_identifier").to_pylist():
                classes[identifier] += 1
        if "VisualizerInstruction:component_map" in batch.schema.names:
            for row in batch.column("VisualizerInstruction:component_map").to_pylist():
                mappings.extend(row)
    return classes, mappings


def test_a_fresh_save_holds_every_declared_view(tmp_path: Path) -> None:
    path = tmp_path / "default.rbl"
    build_blueprint().save(APPLICATION_ID, str(path))
    classes, _ = _read_back(path)
    assert classes == EXPECTED_VIEWS


def test_every_plot_maps_scalars_onto_a_reflected_column(tmp_path: Path) -> None:
    """The plots read the base layer's array columns through component mappings; a bump that changes how those serialize shows here."""
    path = tmp_path / "default.rbl"
    build_blueprint().save(APPLICATION_ID, str(path))
    _, mappings = _read_back(path)
    assert {mapping["source_component"] for mapping in mappings} == EXPECTED_SOURCES
    assert {mapping["target"] for mapping in mappings} == {"Scalars:scalars"}
    assert {mapping["selector"] for mapping in mappings} == {"[]"}
