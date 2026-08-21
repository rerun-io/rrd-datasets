"""
Tests that the blueprint builds under the pinned SDK and reads back with every declared view.

The blueprint is a generated artifact (`pixi run -e abc blueprint`), and nothing else exercises
the `rrb` constructors between SDK bumps. Saved blueprints differ byte for byte on every save, so
the test compares structure — the store identity and the tally of view classes — never bytes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rerun.experimental import RrdReader

from abc_130k.blueprint import build_blueprint
from abc_130k.convert import APPLICATION_ID

# One entry per view in `build_blueprint`: the five camera panes (top mono, top stereo pair, both
# wrists); the instruction document and the subtask timeline; the q, dq, and tau plots.
EXPECTED_VIEWS = {"2D": 5, "TextDocument": 1, "StateTimeline": 1, "TimeSeries": 3}


def _view_classes(rbl: Path) -> Counter[str]:
    """The view-class tally of a saved blueprint, read back through the SDK."""
    reader = RrdReader(str(rbl))
    (entry,) = reader.blueprints()
    assert entry.kind == "blueprint"
    assert entry.application_id == APPLICATION_ID
    classes: Counter[str] = Counter()
    for chunk in reader.stream(store=entry):
        batch = chunk.to_record_batch()
        if "ViewBlueprint:class_identifier" in batch.schema.names:
            for (identifier,) in batch.column("ViewBlueprint:class_identifier").to_pylist():
                classes[identifier] += 1
    return classes


def test_a_fresh_save_holds_every_declared_view(tmp_path: Path) -> None:
    path = tmp_path / "default.rbl"
    build_blueprint().save(APPLICATION_ID, str(path))
    assert _view_classes(path) == EXPECTED_VIEWS
