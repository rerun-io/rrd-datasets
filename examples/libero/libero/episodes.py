"""
Turn the LIBERO repo's file list into the task files to convert, each with its task id.

A LIBERO task is one HDF5 file — `<suite>/<task>_demo.hdf5` — holding ~50 teleoperated demos.
One demo becomes one recording, so ids nest: `task_id = <suite>/<task>` names the file and
`recording_id = <task_id>/<demo_key>` names one demo, with the demo keys (`demo_0`, …) read
from inside the file. Task ids derive from the file path alone, so the HF listing (Modal
fan-out) and a local `data/LIBERO` scan (local conversion) key the same recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rrd_datasets_common.hf_repo import hf_file_index
from rrd_datasets_common.paths import dataset_data_dir

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"

# The source revision every run reads: a full commit sha, so listings and downloads cannot shift
# under a re-upload. Bump it deliberately to pick up dataset changes.
HF_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"  # 2025-05-18

# `.cache/` is gitignored. Delete this file to force a re-listing.
CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "hf_files.json.gz"

LOCAL_DIR = dataset_data_dir("LIBERO")

_TASK_FILE_SUFFIX = "_demo.hdf5"


@dataclass
class WorkItem:
    """One task file to convert: where it lives in the repo, and the task id derived from it."""

    path: str  # e.g. "libero_goal/turn_on_the_stove_demo.hdf5"
    task_id: str  # "libero_goal/turn_on_the_stove"


def task_id(path: str) -> str:
    """Derive `<suite>/<task>` from a repo-relative task file path."""
    return path.removesuffix(_TASK_FILE_SUFFIX)


def recording_id(task: str, demo: str) -> str:
    """The id of one demo's recordings: `<suite>/<task>/<demo_key>`."""
    return f"{task}/{demo}"


def task_files_from_files(files: set[str], path_filter: str = "") -> list[WorkItem]:
    """
    Every task file in a repo listing whose path contains `path_filter`.

    Sorting the paths orders the files by suite, then task name.
    """
    return [
        WorkItem(path, task_id(path))
        for path in sorted(files)
        if path.endswith(_TASK_FILE_SUFFIX) and path_filter in path
    ]


def discover_task_files(repo_id: str = HF_REPO_ID, path_filter: str = "") -> list[WorkItem]:
    """Every matching task file in `repo_id`, from the cached file listing."""
    return task_files_from_files(hf_file_index(repo_id, CACHE_PATH, HF_REVISION), path_filter)


def discover_local_task_files(data_dir: Path | None = None) -> list[WorkItem]:
    """Every downloaded task file under `data_dir`, keyed like the repo listing."""
    root = data_dir if data_dir is not None else LOCAL_DIR
    files = {file.relative_to(root).as_posix() for file in root.glob(f"*/*{_TASK_FILE_SUFFIX}")}
    if not files:
        raise RuntimeError(f"No task files under {root} — run `pixi run -e libero download` first.")
    return task_files_from_files(files)
