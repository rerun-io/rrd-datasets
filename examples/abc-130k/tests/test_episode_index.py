"""Tests for the episode index and its cached repo listing: synthetic tree, no network."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from abc_130k import episode_index
from rrd_datasets_common import hf_repo

REPO_ID = "XDOF/ABC-130k"
SHA = "a" * 40
OTHER_SHA = "b" * 40

# Files the real repo carries alongside the episodes; discovery must ignore all of them.
NON_EPISODE_FILES = {
    ".gitattributes",
    "README.md",
    "docs/YAM_DATA_FORMAT.md",
    "images/yam_data_example.mp4",
    "meta/train_report.txt",
    "meta/val_report.txt",
}

Expected = list[tuple[str, str, bool]]


def _tree(counts: dict[str, int]) -> tuple[set[str], Expected]:
    """A fake repo file list, plus the `(episode_dir, recording_id, has_annotation)` it should yield."""
    files = set(NON_EPISODE_FILES)
    expected: Expected = []
    for split_task, count in counts.items():
        for i in range(count):
            uuid = f"{i:08d}-4d85-4251-bb57-e1abbc88e527"
            episode_dir = f"data/{split_task}/episode_{uuid}"
            files.add(f"{episode_dir}/episode.mcap")
            annotated = i % 3 == 0
            if annotated:
                files.add(f"{episode_dir}/annotation.mcap")
            expected.append((episode_dir, f"{split_task.split('/')[1]}__{uuid}", annotated))
    return files, sorted(expected)


class FakeApi:
    """Stands in for `HfApi`, counting both calls so a cache hit can be shown to make neither."""

    def __init__(self, files: set[str], sha: str | None) -> None:
        self._files = files
        self._sha = sha
        self.listings = 0
        self.lookups = 0

    def repo_info(self, *args: Any, **kwargs: Any) -> Any:
        """The configured revision, or a connection failure when it is `None`."""
        self.lookups += 1
        if self._sha is None:
            raise ConnectionError("offline")
        return SimpleNamespace(sha=self._sha)

    def list_repo_files(self, *args: Any, **kwargs: Any) -> set[str]:
        """The fake file list, counting the call."""
        self.listings += 1
        return self._files


def _discover(
    files: set[str], sha: str | None, cache: Path, task_filter: str = "", revision: str | None = SHA
) -> tuple[Expected, int]:
    """Run `discover_episodes` against `FakeApi`, returning its result and the listings it made."""
    api = FakeApi(files, sha)
    with (
        patch.object(episode_index, "CACHE_PATH", cache),
        patch.object(episode_index, "HF_REVISION", revision),
        patch("huggingface_hub.HfApi", lambda: api),
    ):
        items = episode_index.discover_episodes(REPO_ID, task_filter)
    return [(item.episode_dir, item.recording_id, item.has_annotation) for item in items], api.listings


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "hf_files.json.gz"


def test_first_run_lists_then_reuses_the_cache(cache: Path) -> None:
    files, expected = _tree({"train/taskA": 40, "train/taskB": 7, "val/taskA": 5})

    got, listings = _discover(files, SHA, cache)
    assert got == expected
    assert listings == 1

    got, listings = _discover(files, SHA, cache)
    assert got == expected
    assert listings == 0


def test_task_filter_matches_a_substring_of_the_path(cache: Path) -> None:
    files, expected = _tree({"train/taskA": 12, "train/taskB": 12, "val/taskA": 4})
    _discover(files, SHA, cache)

    for task_filter in ("taskB", "val/", "taskA", "nothing"):
        got, _ = _discover(files, SHA, cache, task_filter)
        assert got == [item for item in expected if task_filter in item[0]]


def test_changed_revision_relists(cache: Path) -> None:
    files, expected = _tree({"train/taskA": 20})
    _discover(files, SHA, cache)

    got, listings = _discover(files, OTHER_SHA, cache, revision=OTHER_SHA)
    assert got == expected
    assert listings == 1
    assert json.loads(gzip.decompress(cache.read_bytes()))["sha"] == OTHER_SHA


def test_corrupt_cache_is_a_miss(cache: Path) -> None:
    files, expected = _tree({"train/taskA": 20})
    cache.write_bytes(b"not gzip at all")

    got, listings = _discover(files, SHA, cache)
    assert got == expected
    assert listings == 1


def test_older_cache_version_is_a_miss(cache: Path) -> None:
    cache.write_bytes(gzip.compress(json.dumps({"version": 0, "sha": SHA, "files": []}).encode()))
    assert hf_repo._read_cache(cache) is None


def test_unreadable_revision_falls_back_to_the_cache(cache: Path) -> None:
    """Without a pinned sha there is nothing to check the cache against, so it is used as-is."""
    files, expected = _tree({"train/taskA": 20})
    _discover(files, SHA, cache, revision=None)

    got, listings = _discover(files, None, cache, revision=None)
    assert got == expected
    assert listings == 0


def test_unreadable_revision_without_a_cache_exits(cache: Path) -> None:
    files, _ = _tree({"train/taskA": 20})
    with pytest.raises(SystemExit):
        _discover(files, None, cache, revision=None)


def test_a_pinned_sha_checks_the_cache_without_the_hub(cache: Path) -> None:
    """A full sha already says what it pins, so a cache hit costs no api call even offline."""
    files, _ = _tree({"train/taskA": 20})
    api = FakeApi(files, None)

    with patch("huggingface_hub.HfApi", lambda: api):
        assert hf_repo.hf_file_index(REPO_ID, cache, SHA) == files
        assert hf_repo.hf_file_index(REPO_ID, cache, SHA) == files

    assert api.listings == 1
    assert api.lookups == 0


def test_a_pinned_sha_refuses_another_revisions_cache(cache: Path) -> None:
    """The bug this guards: offline, a stale listing must not pass as the pinned revision."""
    files, _ = _tree({"train/taskA": 20})
    api = FakeApi(files, None)

    with patch("huggingface_hub.HfApi", lambda: api):
        hf_repo.hf_file_index(REPO_ID, cache, SHA)
        hf_repo.hf_file_index(REPO_ID, cache, OTHER_SHA)

    assert api.listings == 2
    assert json.loads(gzip.decompress(cache.read_bytes()))["sha"] == OTHER_SHA
