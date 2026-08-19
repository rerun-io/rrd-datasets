"""Tests for the Modal launcher's layer selection and S3 prefilter: synthetic, no dataset and no AWS access."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from hiw_500.episode_index import WorkItem
from hiw_500.layers import LAYERS
from rrd_datasets_common.modal_jobs.hf_bucket import HF_S3_KEYS
from rrd_datasets_common.modal_jobs.store import s3_existing_keys
from rrd_datasets_common.paths import layer_relpath

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

# Importing the launcher builds Modal objects, which read the launcher-side credentials: the
# HuggingFace token, plus the HF S3 credentials on the default `hf` backend. No value here is ever used.
os.environ.setdefault("HF_TOKEN", "unit-test-token")
for _key in HF_S3_KEYS:
    os.environ.setdefault(_key, "unit-test-key")

BUCKET_URI = "s3://test-bucket/"
DATASET_PREFIX = f"{BUCKET_URI}hiw-500/"


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
    from hiw_500.modal_jobs import convert_episodes

    return convert_episodes


def _items(
    count: int,
    head_calib: str = "TaskA/session/episode/calibration/params/head_camera_params.yaml",
    has_ir: bool = True,
) -> list[WorkItem]:
    wrist_calibs = ["TaskA/session/episode/calibration/params/camera_000000000001.json"] if has_ir else []
    return [
        WorkItem(
            f"TaskA/session/episode_{i:04d}/episode_{i:04d}.mcap",
            "TaskA/session/episode/info.json",
            head_calib,
            wrist_calibs,
            has_ir,
            f"TaskA__{i}",
        )
        for i in range(count)
    ]


def _done(layers: list[str], recording_ids: list[str]) -> set[str]:
    """The keys a bucket would already hold for these episodes' layers, under the dataset prefix."""
    return {f"hiw-500/{layer_relpath(layer, rid)}" for layer in layers for rid in recording_ids}


def _serve(done: set[str]) -> Callable[[Any, str], set[str]]:
    """A `s3_existing_keys` stand-in that answers from `done`."""
    return lambda _s3, _uri_prefix: done


@contextmanager
def _patched(launcher: Any, keys: Any) -> Iterator[None]:
    """Patch the launcher's S3 seam and point it at the test bucket."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(launcher, "s3_existing_keys", side_effect=keys))
        stack.enter_context(patch.object(launcher, "launcher_client"))
        stack.enter_context(patch.object(launcher, "DATASET_PREFIX", DATASET_PREFIX))
        yield


# --------------------------------------------------------------------------------------
# layer selection
# --------------------------------------------------------------------------------------


def test_all_selects_every_layer_and_an_empty_flag_means_the_same() -> None:
    launcher = _launcher()
    assert launcher.parse_layers("all") == list(launcher.LAYERS)
    assert launcher.parse_layers("") == list(launcher.LAYERS)


def test_a_subset_comes_back_in_layer_order_however_it_is_typed() -> None:
    launcher = _launcher()
    assert launcher.parse_layers("urdf,base") == ["base", "urdf"]
    assert launcher.parse_layers(" properties , base ") == ["base", "properties"]


def test_an_unknown_layer_is_rejected() -> None:
    launcher = _launcher()
    with pytest.raises(SystemExit, match="urfd"):
        launcher.parse_layers("base,urfd")


def test_only_the_layers_that_read_them_pull_inputs() -> None:
    launcher = _launcher()
    assert not launcher.NEEDS_MCAP.intersection(["cameras", "properties"])
    assert not launcher.NEEDS_INFO.intersection(["cameras", "urdf", "odom", "ir"])
    assert launcher.NEEDS_MCAP.intersection(["base"])
    assert launcher.NEEDS_MCAP.intersection(["ir"])
    # base embeds every calibration file verbatim, cameras reads only the head calibration.
    assert launcher.NEEDS_HEAD_CALIB == {"base", "cameras"}
    assert launcher.NEEDS_WRIST_CALIBS == {"base", "properties"}


def test_expected_layers_drops_only_what_the_episode_cannot_produce() -> None:
    launcher = _launcher()
    (full,) = _items(1)
    (without_calib,) = _items(1, head_calib="")
    (without_ir,) = _items(1, has_ir=False)
    assert launcher.expected_layers(full, list(LAYERS)) == list(LAYERS)
    assert launcher.expected_layers(without_calib, list(LAYERS)) == [layer for layer in LAYERS if layer != "cameras"]
    assert launcher.expected_layers(without_ir, list(LAYERS)) == [layer for layer in LAYERS if layer != "ir"]


def test_the_s3_object_keys_are_the_local_relative_paths() -> None:
    """`register` reads a synced bucket only if the object keys match what the local flow writes."""
    launcher = _launcher()
    with patch.object(launcher, "DATASET_PREFIX", DATASET_PREFIX):
        assert launcher.layer_dest("base", "TaskA__1") == f"{DATASET_PREFIX}base/TaskA__1.rrd"
        assert launcher.layer_dest("urdf", "TaskA__1") == f"{DATASET_PREFIX}urdf/TaskA__1.rrd"
        assert launcher.layer_dest("properties", "TaskA__1") == f"{DATASET_PREFIX}properties/TaskA__1.rrd"


def test_each_kind_of_file_owns_a_prefix() -> None:
    """
    The layer directories and the blueprint must not collide.

    Syncing the layer directories is the documented way to pull the recordings down, so anything
    that is not a recording has to sit outside every `<prefix><layer>/` directory.
    """
    from hiw_500 import storage

    assert storage.BLUEPRINT_URI.startswith(storage.DATASET_PREFIX)
    blueprint_dir = storage.BLUEPRINT_URI.removeprefix(storage.DATASET_PREFIX).split("/")[0]
    assert blueprint_dir not in LAYERS


def test_the_job_and_the_catalog_agree_on_every_layer_name() -> None:
    """One shared tuple, so an S3 object can never sit in a directory `register` would not look in."""
    from hiw_500 import catalog

    launcher = _launcher()
    assert catalog.LAYERS is LAYERS
    assert launcher.LAYERS is LAYERS


def test_a_synced_bucket_registers_without_renaming(tmp_path: Path) -> None:
    """The README tells people to sync the bucket into `rrds/hiw-500/` and register — the paths have to line up."""
    from hiw_500 import catalog

    launcher = _launcher()
    with patch.object(launcher, "DATASET_PREFIX", DATASET_PREFIX):
        for rid in ("TaskA__session__episode_0001", "TaskA__session__episode_0002"):
            for layer in launcher.LAYERS:
                key = launcher.layer_dest(layer, rid).removeprefix(DATASET_PREFIX)
                path = tmp_path / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")

    # `register` starts from the base RRDs and finds each layer by its relative path.
    assert [path.name for path in catalog._base_rrds(tmp_path)] == [
        "TaskA__session__episode_0001.rrd",
        "TaskA__session__episode_0002.rrd",
    ]
    for base in catalog._base_rrds(tmp_path):
        rid = base.name.removesuffix(".rrd")
        for layer in LAYERS:
            assert (tmp_path / layer_relpath(layer, rid)).exists()


# --------------------------------------------------------------------------------------
# prefilter
# --------------------------------------------------------------------------------------


def test_existing_keys_spans_pages_and_skips_empty_objects() -> None:
    s3 = FakeS3(
        [("hiw-500/base/TaskA__1.rrd", 100), ("hiw-500/base/TaskA__2.rrd", 0)],
        [("hiw-500/urdf/TaskB__3.rrd", 5000)],
        [],
    )
    assert s3_existing_keys(cast("S3Client", s3), DATASET_PREFIX) == {
        "hiw-500/base/TaskA__1.rrd",
        "hiw-500/urdf/TaskB__3.rrd",
    }
    assert s3.calls == [{"Bucket": "test-bucket", "Prefix": "hiw-500/"}]


def test_existing_keys_of_empty_prefix() -> None:
    assert s3_existing_keys(cast("S3Client", FakeS3([])), DATASET_PREFIX) == set()


def test_one_listing_answers_for_every_layer() -> None:
    """All layer directories share the dataset prefix, so the launcher must not pay for a listing per layer."""
    launcher = _launcher()
    calls: list[str] = []

    def record(_s3: Any, uri_prefix: str) -> set[str]:
        calls.append(uri_prefix)
        return set()

    with _patched(launcher, record):
        launcher._drop_converted(_items(1), ["base", "urdf", "odom", "cameras", "properties"])

    assert calls == [DATASET_PREFIX]


def test_drop_converted_keeps_only_the_missing_episodes() -> None:
    launcher = _launcher()
    layers = ["base", "urdf"]
    done = _done(layers, [f"TaskA__{i}" for i in range(25)])
    with _patched(launcher, _serve(done)):
        todo = launcher._drop_converted(_items(30), layers)

    assert todo is not None
    assert [item.recording_id for item in todo] == [f"TaskA__{i}" for i in range(25, 30)]
    # `main` slices after this filter, so a small --limit still yields unconverted episodes.
    assert [item.recording_id for item in todo[:3]] == ["TaskA__25", "TaskA__26", "TaskA__27"]


def test_an_episode_missing_one_layer_is_rebuilt() -> None:
    launcher = _launcher()
    layers = ["base", "urdf", "odom"]
    done = _done(layers, ["TaskA__0", "TaskA__1"])
    # TaskA__1 uploaded its base and urdf but never its odom layer.
    done.discard("hiw-500/odom/TaskA__1.rrd")
    with _patched(launcher, _serve(done)):
        todo = launcher._drop_converted(_items(2), layers)

    assert todo is not None
    assert [item.recording_id for item in todo] == ["TaskA__1"]


def test_a_layer_outside_the_selection_does_not_hold_an_episode_back() -> None:
    launcher = _launcher()
    done = _done(["base"], ["TaskA__0"])
    with _patched(launcher, _serve(done)):
        assert launcher._drop_converted(_items(1), ["base"]) == []


def test_an_episode_without_head_calibration_does_not_wait_for_a_cameras_layer() -> None:
    """A cameras layer that can never exist must not keep respawning the episode's worker."""
    launcher = _launcher()
    done = _done(["base"], ["TaskA__0"])
    with _patched(launcher, _serve(done)):
        assert launcher._drop_converted(_items(1, head_calib=""), ["base", "cameras"]) == []


def test_an_episode_without_ir_streams_does_not_wait_for_an_ir_layer() -> None:
    launcher = _launcher()
    done = _done(["base"], ["TaskA__0"])
    with _patched(launcher, _serve(done)):
        assert launcher._drop_converted(_items(1, has_ir=False), ["base", "ir"]) == []


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
    with _patched(launcher, failure):
        assert launcher._drop_converted(_items(30), ["base"]) is None
