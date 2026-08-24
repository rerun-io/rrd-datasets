"""Tests for the task file discovery and the id derivations."""

from __future__ import annotations

from pathlib import Path

from libero.download import SAMPLES
from libero.episodes import WorkItem, discover_local_task_files, recording_id, task_files_from_files, task_id

LISTING = {
    ".gitattributes",
    "README.md",
    "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    "libero_90/KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5",
    "libero_goal/turn_on_the_stove_demo.hdf5",
}


def test_task_id_strips_the_suffix() -> None:
    assert task_id("libero_goal/turn_on_the_stove_demo.hdf5") == "libero_goal/turn_on_the_stove"


def test_recording_id_nests_demo_under_task() -> None:
    assert recording_id("libero_goal/turn_on_the_stove", "demo_0") == "libero_goal/turn_on_the_stove/demo_0"


def test_task_files_from_files_keeps_only_task_files_sorted() -> None:
    items = task_files_from_files(LISTING)
    assert [item.path for item in items] == [
        "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        "libero_90/KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5",
        "libero_goal/turn_on_the_stove_demo.hdf5",
    ]
    assert items[2] == WorkItem("libero_goal/turn_on_the_stove_demo.hdf5", "libero_goal/turn_on_the_stove")


def test_task_files_from_files_filters_by_substring() -> None:
    items = task_files_from_files(LISTING, "libero_90/")
    assert [item.task_id for item in items] == ["libero_90/KITCHEN_SCENE3_turn_on_the_stove"]


def test_local_discovery_matches_the_repo_layout(tmp_path: Path) -> None:
    for relpath in ["libero_goal/turn_on_the_stove_demo.hdf5", "libero_goal/notes.txt"]:
        target = tmp_path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    assert discover_local_task_files(tmp_path) == [
        WorkItem("libero_goal/turn_on_the_stove_demo.hdf5", "libero_goal/turn_on_the_stove")
    ]


def test_samples_cover_every_suite_once() -> None:
    assert sorted(sample.split("/")[0] for sample in SAMPLES) == [
        "libero_10",
        "libero_90",
        "libero_goal",
        "libero_object",
        "libero_spatial",
    ]
    assert all(sample.endswith("_demo.hdf5") for sample in SAMPLES)
