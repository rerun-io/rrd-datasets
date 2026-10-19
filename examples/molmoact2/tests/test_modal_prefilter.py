"""Tests for the Modal launcher's bucket prefilter: synthetic, no dataset and no bucket access."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from botocore.exceptions import NoCredentialsError

from molmoact2.episode_index import WorkItem, recording_id
from molmoact2.layers import LAYERS
from rrd_datasets_common.modal_jobs.hf_bucket import HF_S3_KEYS
from rrd_datasets_common.paths import layer_relpath

# Importing the launcher builds Modal objects, which read the HF token and the HF S3 credentials.
# The values are never used.
os.environ.setdefault("HF_TOKEN", "unit-test-token")
for _key in HF_S3_KEYS:
    os.environ.setdefault(_key, "unit-test-key")

DATASET_PREFIX = "s3://test-bucket/molmoact2/"


def _launcher() -> ModuleType:
    """The launcher module, imported late so `$HF_TOKEN` is set first."""
    from molmoact2.modal_jobs import convert_groups

    return convert_groups


def _items(count: int, episodes: int = 4) -> list[WorkItem]:
    """`count` file groups of `episodes` consecutive episodes each."""
    return [
        WorkItem(tuple(range(index * episodes, (index + 1) * episodes)), ("Fold the cloth",), ())
        for index in range(count)
    ]


def test_drop_converted_keeps_every_group_with_a_missing_layer() -> None:
    launcher = _launcher()
    layers = ["base", "cameras"]
    complete, partial, empty = _items(3)
    done = {f"molmoact2/{layer_relpath(layer, rec_id)}" for layer in layers for rec_id in complete.recording_ids}
    # The second group has everything but one episode's cameras layer.
    done |= {f"molmoact2/{layer_relpath(layer, rec_id)}" for layer in layers for rec_id in partial.recording_ids}
    done.discard(f"molmoact2/{layer_relpath('cameras', recording_id(6))}")

    with (
        patch.object(launcher, "s3_existing_keys", return_value=done),
        patch.object(launcher, "launcher_client"),
        patch.object(launcher, "DATASET_PREFIX", DATASET_PREFIX),
    ):
        assert launcher.drop_converted([complete, partial, empty], layers) == [partial, empty]


def test_drop_converted_returns_none_when_the_bucket_is_unreadable() -> None:
    """The launcher then starts every worker, and each one checks the bucket itself."""
    launcher = _launcher()
    with (
        patch.object(launcher, "s3_existing_keys", side_effect=NoCredentialsError()),
        patch.object(launcher, "launcher_client"),
    ):
        assert launcher.drop_converted(_items(1), ["base"]) is None


def test_a_synced_bucket_registers_without_renaming(tmp_path: Path) -> None:
    """The bucket keys are the local relative paths, so a copy of the bucket registers as is."""
    from molmoact2 import catalog

    launcher = _launcher()
    (item,) = _items(1)
    with patch.object(launcher, "DATASET_PREFIX", DATASET_PREFIX):
        for rec_id in item.recording_ids:
            for layer in LAYERS:
                path = tmp_path / launcher.layer_dest(layer, rec_id).removeprefix(DATASET_PREFIX)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")

    assert catalog.recording_ids(tmp_path) == item.recording_ids
