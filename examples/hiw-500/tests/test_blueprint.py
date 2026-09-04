"""
Tests that the blueprint builds under the pinned SDK and reads back with every declared view.

The blueprint is a generated artifact (`pixi run -e hiw blueprint`), and nothing else exercises
the `rrb` constructors between SDK bumps. Saved blueprints differ byte for byte on every save, so
the test compares structure — the store identity and the tally of view classes — never bytes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rerun.experimental import RrdReader

from hiw_500.base_layer import APPLICATION_ID, INSTRUCTION_ENTITY, SUBTASK_ENTITY
from hiw_500.blueprint import INSTRUCTION_COLOR, build_blueprint

# One entry per view in `build_blueprint`: the 3D scene; head pair, wrist RGB pair, and four wrist
# IR panes; the joint, end-effector, and two gripper plots; the annotation timeline.
EXPECTED_VIEWS = {"3D": 1, "2D": 8, "TimeSeries": 4, "StateTimeline": 1}
# One `SeriesLines` instruction per selector: the joint array (all 29 angles from one `[]` mapping),
# 12 EE fields x 2 arrays, 4 dex1 jaw angles, 4 gripper controls.
EXPECTED_INSTRUCTIONS = 1 + 12 * 2 + 4 + 4


def _tally(rbl: Path) -> tuple[Counter[str], int]:
    """The view-class tally and the `SeriesLines` instruction count of a saved blueprint, read back through the SDK."""
    reader = RrdReader(str(rbl))
    (entry,) = reader.blueprints()
    assert entry.kind == "blueprint"
    assert entry.application_id == APPLICATION_ID
    classes: Counter[str] = Counter()
    instructions = 0
    for chunk in reader.stream(store=entry):
        batch = chunk.to_record_batch()
        if "ViewBlueprint:class_identifier" in batch.schema.names:
            for (identifier,) in batch.column("ViewBlueprint:class_identifier").to_pylist():
                classes[identifier] += 1
        if "VisualizerInstruction:visualizer_type" in batch.schema.names:
            instructions += sum(
                row == ["SeriesLines"] for row in batch.column("VisualizerInstruction:visualizer_type").to_pylist()
            )
    return classes, instructions


def test_a_fresh_save_holds_every_declared_view(tmp_path: Path) -> None:
    path = tmp_path / "default.rbl"
    build_blueprint().save(APPLICATION_ID, str(path))
    assert _tally(path) == (EXPECTED_VIEWS, EXPECTED_INSTRUCTIONS)


def test_the_annotation_view_stacks_the_instruction_over_the_subtasks(tmp_path: Path) -> None:
    """Both lanes in one view, and the instruction held to its flat grey — lanes stack in path order."""
    path = tmp_path / "default.rbl"
    build_blueprint().save(APPLICATION_ID, str(path))
    reader = RrdReader(str(path))
    (entry,) = reader.blueprints()
    queries: list[list[str]] = []
    colors: list[list[int]] = []
    for chunk in reader.stream(store=entry):
        batch = chunk.to_record_batch()
        if "ViewContents:query" in batch.schema.names:
            queries += batch.column("ViewContents:query").to_pylist()
        if "StateConfiguration:colors" in batch.schema.names:
            colors += batch.column("StateConfiguration:colors").to_pylist()
    assert [f"+ {INSTRUCTION_ENTITY}", f"+ {SUBTASK_ENTITY}"] in queries
    assert colors == [[INSTRUCTION_COLOR]]
