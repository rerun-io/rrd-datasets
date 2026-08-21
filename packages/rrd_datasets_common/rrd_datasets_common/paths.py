"""
The shared filesystem layout of the workspace.

One `data/` root holds the raw datasets, one `rrds/` root holds the converted
recordings, and each example generates its default blueprint under
`blueprints/<example>/`. Examples resolve every such path through this module
instead of hard-coding locations relative to their own directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """
    The workspace root.

    Pixi tasks get it from `$PIXI_PROJECT_ROOT`, which pixi sets to the manifest
    directory regardless of the task's `cwd`. Outside pixi activation (a notebook
    kernel started from the env's interpreter, `python -m` in a plain shell), the
    nearest ancestor holding `pixi.toml` is the root — the workspace has exactly
    one manifest. When neither applies (a Modal worker), fall back to the working
    directory; callers there pass explicit paths anyway.
    """
    env_root = os.environ.get("PIXI_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pixi.toml").exists():
            return candidate
    return Path.cwd()


def dataset_data_dir(dataset: str) -> Path:
    """The download root of a raw dataset, e.g. `data/HIW-500`."""
    return repo_root() / "data" / dataset


def dataset_rrd_dir(example: str) -> Path:
    """The output root of an example's converted recordings, e.g. `rrds/hiw-500`."""
    return repo_root() / "rrds" / example


def default_blueprint_path(example: str) -> Path:
    """The generated default blueprint of an example."""
    return repo_root() / "blueprints" / example / "default.rbl"


def resolve_input_path(path: Path) -> Path:
    """
    A user-typed input path, tried as given first, then against the workspace root.

    Pixi tasks run in their example directory while the documented commands write
    paths repo-root-relative; accepting both bases keeps either invocation working.
    A path that exists nowhere comes back untouched, for the caller's error message.
    """
    if path.is_absolute() or path.exists():
        return path
    rooted = repo_root() / path
    return rooted if rooted.exists() else path


def layer_relpath(layer: str, recording_id: str) -> str:
    """
    The relative path of one layer recording, e.g. `base/<recording_id>.rrd`.

    The same relative path applies under the local rrd root (`dataset_rrd_dir`) and
    under the dataset's bucket prefix, so a layer directory syncs between the two
    without renaming and the catalog registers either side as is.
    """
    return f"{layer}/{recording_id}.rrd"
