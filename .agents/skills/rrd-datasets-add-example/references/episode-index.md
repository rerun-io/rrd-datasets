# Episode index template

`[pkg]/episode_index.py` holds the list of what to convert.
It reads the source's file listing once and turns it into one work item per [unit].
The Modal launcher starts one worker per work item.
The recording ids minted here become the file stem of every layer and the id of each catalog segment.
`references/data-sources.md` covers the listing itself and says what to pin, what to cache, and which calls to avoid.

The work items come from two places: the pinned source listing on Modal, and a local scan of `data/`.
Both assume two things about this module:

1. The recording id is minted from the file path only, without opening it.
2. The work items are built from a set of paths passed in, not from a listing the builder fetches itself.
   Both sources then feed the same builder, so a [unit] gets the same work item either way.

The template below uses the Hugging Face listing driver.
Another source swaps the constants at the top and the `hf_file_index` call for its own equivalents.
`WorkItem`, the id derivation, and the item builder do not change.

---

```python
"""
Turn the [source]'s file list into the [unit]s to convert, each with its recording id.

A [Dataset Name] [unit] is [what one is, and which files it owns].
Recording ids [how they are built], so they match what local discovery produces
and what `catalog.py` keys segments on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rrd_datasets_common.hf_repo import hf_file_index

HF_REPO_ID = "[org/repo]"

# Pinned so a re-upload cannot change what the converter reads. A full sha, since branches and
# tags move. Bump it deliberately to pick up newly published [unit]s.
HF_REVISION = "[full commit sha]"

# `.cache/` is gitignored. Delete this file to force a re-listing.
CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "hf_files.json.gz"


@dataclass
class WorkItem:
    """One [unit] to convert: its path in the source, and the recording id derived from it."""

    [main]: str  # e.g. "[a source-relative path]"
    [sidecar]: str  # the sidecar beside it, or "" when the [unit] ships none
    [flag]: bool  # whether the [unit] has [stream], judged by which sidecar files are next to it
    recording_id: str  # "[id format]", matching the local converter


def recording_id([main]: str) -> str:
    """Derive `[id format]` from a source-relative [main] path."""


def [unit]s_from_files(files: set[str], task_filter: str = "") -> list[WorkItem]:
    """
    Every [unit] in a source listing whose path contains `task_filter`.

    Sorting the paths orders the [unit]s by [what that order means].
    """


def discover_[unit]s(repo_id: str = HF_REPO_ID, task_filter: str = "") -> list[WorkItem]:
    """Every matching [unit] in `repo_id`, from the cached file listing."""
    return [unit]s_from_files(hf_file_index(repo_id, CACHE_PATH, HF_REVISION), task_filter)
```

`task_filter` is the one option the index offers, and the Modal launcher's `--task-filter` flag uses it.

## When the unit is not one episode

Where the source stores many recordings in one file, the index keys the file and a second function names the recordings inside it.
libero does this, so its `WorkItem` has a task id and `recording_id(task, demo)` builds the id of one demo.

## Local discovery

Where the example converts from downloaded files, add a `discover_local_*` that scans `dataset_data_dir("[Dataset Name]")` and passes the paths it finds to `[unit]s_from_files`.
It raises with the `pixi run -e [env] download` line when the directory is empty.
