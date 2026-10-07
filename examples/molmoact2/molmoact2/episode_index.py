"""
Build the file groups to convert from the MolmoAct2 episode metadata.

LeRobot v3 stores consecutive episodes together in shared parquet and mp4 files, and each camera
starts a new file at a different episode. A file group is a run of consecutive episodes that shares
no file with any other group, so no two workers download the same file. `meta/episodes` is the only
place that says which files each episode uses.

Recording ids come from `episode_index` alone, so they match what local discovery produces and what
`catalog.py` keys segments on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from rrd_datasets_common.hf_repo import hf_file_index
from rrd_datasets_common.paths import dataset_data_dir

HF_REPO_ID = "allenai/MolmoAct2-BimanualYAM-Dataset"

# Pinned so a re-upload cannot change what the converter reads. A full sha, since branches and
# tags move. Bump it deliberately to pick up newly published episodes.
HF_REVISION = "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c"

# `.cache/` is gitignored. Delete this file to force a re-listing.
CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache" / "hf_files.json.gz"

LOCAL_DIR = dataset_data_dir("MolmoAct2")

_META_PREFIX = "meta/"


@dataclass(frozen=True)
class EpisodeFileSet:
    """Files for one episode-associated data source, such as data parquets or one camera."""

    prefix: str  # the column prefix in `meta/episodes`, e.g. "videos/observation.images.top"
    path_template: str  # `data_path` or `video_path` from `info.json`
    video_key: str  # the camera feature, or "" for the data parquets

    def path(self, chunk_index: int, file_index: int) -> str:
        """The repo-relative path of one file in the set."""
        return self.path_template.format(chunk_index=chunk_index, file_index=file_index, video_key=self.video_key)


@dataclass(frozen=True)
class WorkItem:
    """One file group to convert: its episodes, their tasks, and the files they use."""

    episodes: tuple[int, ...]  # episode indices, ascending and consecutive
    tasks: tuple[str, ...]  # the distinct tasks of those episodes, sorted
    files: tuple[str, ...]  # repo-relative data and video files, sorted

    @property
    def group_id(self) -> str:
        """The group's name in logs, e.g. `episodes_00016-00069`."""
        return f"episodes_{self.episodes[0]:05d}-{self.episodes[-1]:05d}"

    @property
    def recording_ids(self) -> list[str]:
        """The recording id of every episode in the group."""
        return [recording_id(episode) for episode in self.episodes]


def recording_id(episode_index: int) -> str:
    """Derive `episode_NNNNN` from an episode index, zero-padded so the ids sort in episode order."""
    return f"episode_{episode_index:05d}"


def groups_from_meta(dataset_root: Path, task_filter: str = "") -> list[WorkItem]:
    """Every file group under `dataset_root` with a task containing `task_filter`."""
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    video_keys = sorted(name for name, feature in info["features"].items() if feature["dtype"] == "video")
    file_sets = [EpisodeFileSet("data", info["data_path"], "")]
    file_sets += [EpisodeFileSet(f"videos/{key}", info["video_path"], key) for key in video_keys]
    columns = ["episode_index", "tasks"]
    columns += [f"{file_set.prefix}/{part}" for file_set in file_sets for part in ("chunk_index", "file_index")]
    episodes = pa.concat_tables(
        pq.read_table(path, columns=columns) for path in sorted(dataset_root.glob("meta/episodes/*/*.parquet"))
    ).sort_by("episode_index")

    # Chunk and file index packed into one int64, which sorts in the order LeRobot writes the files.
    file_keys = []
    for file_set in file_sets:
        chunk = episodes[f"{file_set.prefix}/chunk_index"].to_numpy().astype(np.int64)
        file = episodes[f"{file_set.prefix}/file_index"].to_numpy().astype(np.int64)
        key = (chunk << 32) | file
        # The boundaries below compare neighboring episodes only, which is correct while every file
        # holds consecutive episodes.
        if np.any(np.diff(key) < 0):
            raise ValueError(f"{file_set.prefix} files do not hold consecutive episodes in {dataset_root}")
        file_keys.append(key)

    shares_with_previous = np.zeros(episodes.num_rows, dtype=bool)
    for key in file_keys:
        shares_with_previous[1:] |= key[1:] == key[:-1]
    starts = np.flatnonzero(~shares_with_previous)
    stops = [*starts[1:], episodes.num_rows]

    episode_index = episodes["episode_index"].to_numpy()
    groups = []
    for start, stop in zip(starts, stops, strict=True):
        tasks = tuple(sorted(pc.unique(pc.list_flatten(episodes["tasks"].slice(start, stop - start))).to_pylist()))
        if task_filter and not any(task_filter in task for task in tasks):
            continue
        files = set()
        for file_set, key in zip(file_sets, file_keys, strict=True):
            for file_key in np.unique(key[start:stop]):
                chunk_index, file_index = divmod(int(file_key), 1 << 32)
                files.add(file_set.path(chunk_index, file_index))
        groups.append(WorkItem(tuple(episode_index[start:stop].tolist()), tasks, tuple(sorted(files))))
    return groups


def fetch_meta(files: set[str], local_dir: Path = LOCAL_DIR) -> None:
    """Download every `meta/` file of the pinned revision into `local_dir`, skipping files already there."""
    for path in sorted(path for path in files if path.startswith(_META_PREFIX)):
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=HF_REVISION,
            filename=path,
            local_dir=local_dir,
        )


def discover_groups(task_filter: str = "", local_dir: Path = LOCAL_DIR) -> list[WorkItem]:
    """Every matching file group in the pinned repo, after fetching its `meta/` into `local_dir`."""
    files = hf_file_index(HF_REPO_ID, CACHE_PATH, HF_REVISION)
    fetch_meta(files, local_dir)
    groups = groups_from_meta(local_dir, task_filter)
    missing = sorted({path for group in groups for path in group.files} - files)
    if missing:
        raise RuntimeError(f"{len(missing)} files named in meta/episodes are not in {HF_REPO_ID}, e.g. {missing[0]}")
    return groups


def discover_local_groups(data_dir: Path = LOCAL_DIR) -> list[WorkItem]:
    """Every file group whose files are all downloaded under `data_dir`."""
    groups = []
    if (data_dir / "meta" / "info.json").exists():
        groups = [
            group for group in groups_from_meta(data_dir) if all((data_dir / path).exists() for path in group.files)
        ]
    if not groups:
        raise RuntimeError(f"No complete file group under {data_dir} — run `pixi run -e molmo download` first.")
    return groups
