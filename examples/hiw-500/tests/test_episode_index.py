"""
Tests for HIW-500 episode discovery: synthetic file lists and empty local files, no network.

Every way of naming an episode — the Modal launcher's HuggingFace paths, a local scan, a single
mcap on the command line — must produce the same recording id, since `catalog.py` keys segments on
it. A mismatch scatters one episode's layers across several segments, and ids that drop the task
and session collide (every `episode_0001` becomes one segment).
"""

from __future__ import annotations

from pathlib import Path

from hiw_500.base_layer import discover_episodes, episode_from_mcap, recording_id_for
from hiw_500.episode_index import episodes_from_files, recording_id

TASK = "Move-The-Pillow-To-The-Sofa-From-Floor"
SESSION = "episode_2026-02-24_10-05-40"

FILES = {
    f"{TASK}/{SESSION}/episode_0001/episode_0001.mcap",
    f"{TASK}/{SESSION}/episode_0001/info.json",
    f"{TASK}/{SESSION}/episode_0002/episode_0002.mcap",
    f"{TASK}/{SESSION}/episode_0002/info.json",
    f"{TASK}/{SESSION}/episode_0002/calibration/params/head_camera_params.yaml",
    f"{TASK}/{SESSION}/episode_0002/calibration/params/camera_323622270214.json",
    f"{TASK}/{SESSION}/episode_0002/calibration/params/camera_409122273599.json",
    "Sweep-Floor/episode_2026-03-01_09-00-00/episode_0001/episode_0001.mcap",
    "Sweep-Floor/episode_2026-03-01_09-00-00/episode_0001/info.json",
    "README.md",
}


def test_recording_id_joins_the_directory_names() -> None:
    assert recording_id(f"{TASK}/{SESSION}/episode_0001/episode_0001.mcap") == f"{TASK}__{SESSION}__episode_0001"


def test_every_episode_is_found_with_its_sidecar() -> None:
    items = episodes_from_files(FILES)
    assert [item.recording_id for item in items] == [
        f"{TASK}__{SESSION}__episode_0001",
        f"{TASK}__{SESSION}__episode_0002",
        "Sweep-Floor__episode_2026-03-01_09-00-00__episode_0001",
    ]
    assert items[0].mcap == f"{TASK}/{SESSION}/episode_0001/episode_0001.mcap"
    assert items[0].info == f"{TASK}/{SESSION}/episode_0001/info.json"
    assert items[0].head_calib == ""
    assert items[0].wrist_calibs == []
    assert items[0].has_ir is False
    assert items[1].head_calib == f"{TASK}/{SESSION}/episode_0002/calibration/params/head_camera_params.yaml"
    assert items[1].wrist_calibs == [
        f"{TASK}/{SESSION}/episode_0002/calibration/params/camera_323622270214.json",
        f"{TASK}/{SESSION}/episode_0002/calibration/params/camera_409122273599.json",
    ]
    assert items[1].has_ir is True


def test_ordering_is_task_then_session_then_episode() -> None:
    files = {
        "B-Task/episode_2026-01-02_00-00-00/episode_0001/episode_0001.mcap",
        "A-Task/episode_2026-01-09_00-00-00/episode_0001/episode_0001.mcap",
        "A-Task/episode_2026-01-02_00-00-00/episode_0002/episode_0002.mcap",
        "A-Task/episode_2026-01-02_00-00-00/episode_0001/episode_0001.mcap",
    }
    assert [item.recording_id for item in episodes_from_files(files)] == [
        "A-Task__episode_2026-01-02_00-00-00__episode_0001",
        "A-Task__episode_2026-01-02_00-00-00__episode_0002",
        "A-Task__episode_2026-01-09_00-00-00__episode_0001",
        "B-Task__episode_2026-01-02_00-00-00__episode_0001",
    ]


def test_an_episode_without_a_sidecar_is_still_converted() -> None:
    files = {f"{TASK}/{SESSION}/episode_0003/episode_0003.mcap"}
    (item,) = episodes_from_files(files)
    assert item.info == ""


def test_non_episode_files_are_ignored() -> None:
    files = {
        f"{TASK}/{SESSION}/episode_0001/camera_409122273599.json",
        f"{TASK}/{SESSION}/episode_0001/notes.mcap",  # not an episode_*.mcap
        "episode_loose.mcap",  # no task/session directories above it
        "README.md",
    }
    assert episodes_from_files(files) == []


def _local_dataset(root: Path) -> list[Path]:
    """An empty stand-in for the dataset tree; only the paths matter to the id."""
    mcaps = []
    for rel in sorted(f for f in FILES if f.endswith(".mcap")):
        mcap = root / rel
        mcap.parent.mkdir(parents=True, exist_ok=True)
        mcap.touch()
        mcaps.append(mcap)
    return mcaps


def test_a_local_mcap_gets_the_same_id_as_its_hub_path(tmp_path: Path) -> None:
    for mcap in _local_dataset(tmp_path):
        assert recording_id_for(mcap) == recording_id(str(mcap.relative_to(tmp_path)))


def test_naming_one_mcap_gives_the_id_a_scan_would(tmp_path: Path) -> None:
    """The single-episode CLI path (`pixi run -e hiw convert <ep.mcap>`) must not name the file differently."""
    mcaps = _local_dataset(tmp_path)
    scanned = {ep.mcap: ep.recording_id for ep in discover_episodes(tmp_path)}
    assert scanned == {mcap: episode_from_mcap(mcap).recording_id for mcap in mcaps}
    assert scanned[mcaps[0]] == f"{TASK}__{SESSION}__episode_0001"


def test_task_filter_matches_a_substring_of_the_path() -> None:
    assert [item.recording_id for item in episodes_from_files(FILES, "Sweep-Floor")] == [
        "Sweep-Floor__episode_2026-03-01_09-00-00__episode_0001"
    ]
    assert episodes_from_files(FILES, "No-Such-Task") == []
