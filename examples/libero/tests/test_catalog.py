"""Tests for the catalog registration: how demos are found under the per-layer directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from libero.catalog import demo_ids, register_demos
from rrd_datasets_common.paths import layer_relpath


def _touch(root: Path, *relative: str) -> None:
    for rel in relative:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).touch()


def test_demo_ids_carry_the_suite_directory(tmp_path: Path) -> None:
    """The id is `<suite>/<stem>`, so a demo's other layers resolve through `layer_relpath`."""
    _touch(
        tmp_path,
        "base/libero_goal/turn_on_the_stove__demo_0.rrd",
        "base/libero_goal/turn_on_the_stove__demo_1.rrd",
        "base/libero_10/KITCHEN_SCENE3_turn_on_the_stove__demo_0.rrd",
        "urdf/libero_goal/turn_on_the_stove__demo_0.rrd",
    )
    ids = demo_ids(tmp_path)
    assert ids == [
        "libero_10/KITCHEN_SCENE3_turn_on_the_stove__demo_0",
        "libero_goal/turn_on_the_stove__demo_0",
        "libero_goal/turn_on_the_stove__demo_1",
    ]
    assert (tmp_path / layer_relpath("urdf", ids[1])).exists()
    assert not (tmp_path / layer_relpath("urdf", ids[2])).exists()


def test_an_empty_rrd_dir_fails_before_touching_the_catalog(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="base"):
        register_demos("rerun+http://127.0.0.1:1", "libero", tmp_path, tmp_path / "default.rbl")
