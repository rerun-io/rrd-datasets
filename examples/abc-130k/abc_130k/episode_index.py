"""
Turn the HF repo's file list into the episodes to convert, each with its recording id.

An ABC-130k episode is a directory — `data/<split>/<task>/episode_<uuid>/` — holding `episode.mcap`
and optionally `annotation.mcap`. Recording ids are `<task>__<uuid>`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rrd_datasets_common.hf_repo import hf_file_index

HF_REPO_ID = "XDOF/ABC-130k"

# Pinned so a re-upload cannot change what the converter reads. A full sha, since branches and
# tags move. Bump it deliberately to pick up newly published episodes.
HF_REVISION = "29136bc9b9e38d320b00ffcddbbe4cd0e3278c58"

# `.cache/` is gitignored. Delete this file to force a re-listing.
CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "hf_files.json.gz"

_EPISODE_SUFFIX = "/episode.mcap"
_ANNOTATION_SUFFIX = "/annotation.mcap"


@dataclass
class WorkItem:
    """One episode to convert: where it lives in the repo, and the recording id derived from it."""

    episode_dir: str  # e.g. "data/train/fold_and_stack_the_t_shirts/episode_<uuid>"
    recording_id: str  # "<task>__<uuid>", matching the local converter
    has_annotation: bool  # settled here, so a worker makes no Hub call to find out


def recording_id(episode_dir: str) -> str:
    """Derive `<task>__<uuid>` from a `.../<task>/episode_<uuid>` directory."""
    parts = episode_dir.rstrip("/").split("/")
    return f"{parts[-2]}__{parts[-1].removeprefix('episode_')}"


def episodes_from_files(files: set[str], task_filter: str = "") -> list[WorkItem]:
    """
    Every episode in a repo listing whose directory contains `task_filter`.

    Sorting the paths orders the episodes by split, then task, then uuid.
    """
    annotated = {path[: -len(_ANNOTATION_SUFFIX)] for path in files if path.endswith(_ANNOTATION_SUFFIX)}
    items = []
    for path in sorted(files):
        if not path.endswith(_EPISODE_SUFFIX):
            continue
        episode_dir = path[: -len(_EPISODE_SUFFIX)]
        if task_filter and task_filter not in episode_dir:
            continue
        items.append(WorkItem(episode_dir, recording_id(episode_dir), episode_dir in annotated))
    return items


def discover_episodes(repo_id: str = HF_REPO_ID, task_filter: str = "") -> list[WorkItem]:
    """Every matching episode in `repo_id`, from the cached file listing."""
    return episodes_from_files(hf_file_index(repo_id, CACHE_PATH, HF_REVISION), task_filter)
