"""
Turn the HF repo's file list into the episodes to convert, each with its recording id.

A HIW-500 episode is a directory — `<task>/<session>/<episode_NNNN>/` — holding the episode's MCAP
and its sidecars (`info.json`, and on newer sessions a head stereo calibration). Recording ids join the
three directory names, so they match what `base_layer.discover_episodes` produces locally and
what `catalog.py` keys segments on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rrd_datasets_common.hf_repo import hf_file_index

HF_REPO_ID = "BitRobot/HIW-500"

# `.cache/` is gitignored. Delete this file to force a re-listing.
CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "hf_files.json.gz"

_INFO_NAME = "info.json"
_HEAD_CALIB_RELPATH = "calibration/params/head_camera_params.yaml"


@dataclass
class WorkItem:
    """One episode to convert: where it lives in the repo, and the recording id derived from it."""

    mcap: str  # e.g. "Clean-Up-The-Room/episode_2026-02-25_10-16-03/episode_0001/episode_0001.mcap"
    info: str  # the sidecar beside it, or "" when the episode ships none
    head_calib: str  # the head stereo calibration under calibration/params/, or "" when absent
    wrist_calibs: list[str]  # the per-serial wrist calibrations under calibration/params/, sorted
    has_ir: bool  # whether the episode records the wrist IR streams (they ship with wrist calibrations)
    recording_id: str  # "<task>__<session>__<episode_NNNN>", matching the local converter


def recording_id(mcap: str) -> str:
    """Derive `<task>__<session>__<episode_NNNN>` from a repo-relative MCAP path."""
    return "__".join(mcap.split("/")[:-1])


def episodes_from_files(files: set[str], task_filter: str = "") -> list[WorkItem]:
    """
    Every episode in a repo listing whose path contains `task_filter`.

    Sorting the paths orders the episodes by task, then session, then episode number.
    """
    # Wrist IR streams and the per-serial wrist calibrations come and go together (they appeared
    # on the rig at the same time), so the calibration sidecars answer "does this episode record
    # IR?" without opening the mcap.
    wrist_calibs: dict[str, list[str]] = {}
    for path in files:
        if "/calibration/params/camera_" in path and path.endswith(".json"):
            wrist_calibs.setdefault(path.split("/calibration/", 1)[0], []).append(path)
    items = []
    for path in sorted(files):
        directory, _, name = path.rpartition("/")
        if not directory or not name.startswith("episode_") or not name.endswith(".mcap"):
            continue
        if task_filter and task_filter not in path:
            continue
        info = f"{directory}/{_INFO_NAME}"
        head_calib = f"{directory}/{_HEAD_CALIB_RELPATH}"
        calibs = sorted(wrist_calibs.get(directory, []))
        items.append(
            WorkItem(
                path,
                info if info in files else "",
                head_calib if head_calib in files else "",
                calibs,
                bool(calibs),
                recording_id(path),
            )
        )
    return items


def discover_episodes(repo_id: str = HF_REPO_ID, task_filter: str = "") -> list[WorkItem]:
    """Every matching episode in `repo_id`, from the cached file listing."""
    return episodes_from_files(hf_file_index(repo_id, CACHE_PATH), task_filter)
