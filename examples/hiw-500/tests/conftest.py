"""Shared fixtures: the downloaded episode the integration tests run against."""

from __future__ import annotations

from pathlib import Path

import pytest

from hiw_500.base_layer import DATASET_ROOT


def cached_episodes() -> list[Path]:
    """Every downloaded episode MCAP, smallest first."""
    if not DATASET_ROOT.is_dir():
        return []
    return sorted(DATASET_ROOT.rglob("episode_*.mcap"), key=lambda path: path.stat().st_size)


@pytest.fixture(scope="session")
def cached_episode() -> Path:
    """The smallest downloaded episode; tests that need one skip when nothing is downloaded."""
    episodes = cached_episodes()
    if not episodes:
        pytest.skip("no HIW-500 episode under data/")
    return episodes[0]
