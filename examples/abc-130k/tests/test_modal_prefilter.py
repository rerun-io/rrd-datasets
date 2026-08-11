"""Tests for the Modal launcher's S3 prefilter: synthetic, no dataset and no AWS access."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from abc_130k.episode_index import WorkItem
from rrd_datasets_common.modal_jobs.store import s3_existing_keys

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

# Importing the launcher builds Modal objects and reads `$HF_TOKEN`; its value is never used here.
os.environ.setdefault("HF_TOKEN", "unit-test-token")

PREFIX = "s3://test-bucket/abc-130k/base/"


class FakeS3:
    """Stands in for a boto3 S3 client, serving canned `list_objects_v2` pages."""

    def __init__(self, *pages: list[tuple[str, int]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> FakeS3:
        """Serve as its own paginator."""
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs: Any) -> Any:
        """Yield the canned pages, recording the arguments it was called with."""
        self.calls.append(kwargs)
        return ({"Contents": [{"Key": key, "Size": size} for key, size in page]} for page in self.pages)


def _launcher() -> Any:
    """The launcher module, imported late so `$HF_TOKEN` is set first."""
    from abc_130k.modal_jobs import convert_episodes

    return convert_episodes


def _items(count: int) -> list[WorkItem]:
    return [WorkItem(f"data/train/taskA/episode_{i}", f"taskA__{i}", False) for i in range(count)]


def test_existing_keys_spans_pages_and_skips_empty_objects() -> None:
    s3 = FakeS3(
        [("abc-130k/base/taskA__1.rrd", 100), ("abc-130k/base/taskA__2.rrd", 0)],
        [("abc-130k/base/taskB__3.rrd", 5000)],
        [],
    )
    assert s3_existing_keys(cast("S3Client", s3), PREFIX) == {
        "abc-130k/base/taskA__1.rrd",
        "abc-130k/base/taskB__3.rrd",
    }
    assert s3.calls == [{"Bucket": "test-bucket", "Prefix": "abc-130k/base/"}]


def test_existing_keys_of_empty_prefix() -> None:
    assert s3_existing_keys(cast("S3Client", FakeS3([])), PREFIX) == set()


def test_drop_converted_keeps_only_the_missing_episodes() -> None:
    launcher = _launcher()
    done = {f"abc-130k/base/taskA__{i}.rrd" for i in range(25)}
    with (
        patch.object(launcher, "s3_existing_keys", return_value=done),
        patch.object(launcher, "launcher_client"),
        patch.object(launcher, "RRD_PREFIX", PREFIX),
    ):
        todo = launcher._drop_converted(_items(30))

    assert todo is not None
    assert [item.recording_id for item in todo] == [f"taskA__{i}" for i in range(25, 30)]
    # `main` slices after this filter, so a small --limit still yields unconverted episodes.
    assert [item.recording_id for item in todo[:3]] == ["taskA__25", "taskA__26", "taskA__27"]


@pytest.mark.parametrize(
    "failure",
    [
        NoCredentialsError(),
        ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "ListObjectsV2"),
    ],
    ids=["no-credentials", "access-denied"],
)
def test_drop_converted_returns_none_when_the_bucket_is_unreadable(failure: Exception) -> None:
    launcher = _launcher()
    with (
        patch.object(launcher, "s3_existing_keys", side_effect=failure),
        patch.object(launcher, "launcher_client"),
        patch.object(launcher, "RRD_PREFIX", PREFIX),
    ):
        assert launcher._drop_converted(_items(30)) is None
