"""Tests for the Modal launcher's layer selection and bucket prefilter: synthetic, no dataset and no bucket access."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from libero.episodes import WorkItem
from libero.layers import LAYERS
from rrd_datasets_common.modal_jobs.hf_bucket import HF_S3_KEYS
from rrd_datasets_common.paths import layer_relpath

# Importing the launcher builds Modal objects, which read the HF token and the HF S3 credentials.
# The values are never used.
os.environ.setdefault("HF_TOKEN", "unit-test-token")
for _key in HF_S3_KEYS:
    os.environ.setdefault(_key, "unit-test-key")

BUCKET_URI = "s3://test-bucket/"
DATASET_PREFIX = f"{BUCKET_URI}libero/"


def _launcher() -> Any:
    """The launcher module, imported late so `$HF_TOKEN` is set first."""
    from libero.modal_jobs import convert_tasks

    return convert_tasks


def _items(count: int) -> list[WorkItem]:
    return [WorkItem(f"suite_a/task_{i}_demo.hdf5", f"suite_a/task_{i}") for i in range(count)]


def _done(layers: list[str], task_ids: list[str], demos: int = 50) -> set[str]:
    """The keys a bucket would already hold for these task files' demos, under the dataset prefix."""
    return {
        f"libero/{layer_relpath(layer, f'{tid}__demo_{index}')}"
        for layer in layers
        for tid in task_ids
        for index in range(demos)
    }


def _serve(done: set[str]) -> Callable[[Any, str], set[str]]:
    """An `s3_existing_keys` stand-in that answers from `done`."""
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
    assert launcher.parse_layers(" cameras , properties ") == ["properties", "cameras"]


def test_an_unknown_layer_is_rejected() -> None:
    launcher = _launcher()
    with pytest.raises(SystemExit, match="urfd"):
        launcher.parse_layers("base,urfd")


def test_the_s3_object_keys_are_the_local_relative_paths() -> None:
    """`register` reads a synced bucket only if the object keys match what the local flow writes."""
    launcher = _launcher()
    with patch.object(launcher, "DATASET_PREFIX", DATASET_PREFIX):
        rec_id = "libero_goal/turn_on_the_stove__demo_0"
        assert launcher.layer_dest("base", rec_id) == f"{DATASET_PREFIX}base/{rec_id}.rrd"
        assert launcher.layer_dest("urdf", rec_id) == f"{DATASET_PREFIX}urdf/{rec_id}.rrd"
        assert launcher.layer_dest("properties", rec_id) == f"{DATASET_PREFIX}properties/{rec_id}.rrd"


def test_each_kind_of_file_owns_a_prefix() -> None:
    """
    The layer directories, the blueprint and the shared asset must not collide.

    Syncing the layer directories is the documented way to pull the recordings, so nothing else
    may live inside them.
    """
    from libero import storage

    assert storage.BLUEPRINT_URI.startswith(storage.DATASET_PREFIX)
    blueprint_dir = storage.BLUEPRINT_URI.removeprefix(storage.DATASET_PREFIX).split("/")[0]
    assert blueprint_dir not in LAYERS

    assert storage.ASSET_PREFIX.startswith(storage.DATASET_PREFIX)
    asset_dir = storage.ASSET_PREFIX.removeprefix(storage.DATASET_PREFIX).rstrip("/")
    assert asset_dir not in LAYERS
    assert asset_dir != blueprint_dir


def test_the_uploaded_asset_lands_where_register_looks_for_it(tmp_path: Path) -> None:
    """`upload-asset` and `register` have to agree on the shared model's place under the prefix."""
    from libero import storage, upload_asset
    from libero.urdf_layer import model_rrd_path

    key = upload_asset.MODEL_ASSET_URI.removeprefix(storage.DATASET_PREFIX)
    assert (tmp_path / key) == model_rrd_path(tmp_path)


def test_the_job_and_the_catalog_agree_on_every_layer_name() -> None:
    """One shared tuple, so an S3 object can never sit in a directory `register` would not look in."""
    from libero import catalog

    launcher = _launcher()
    assert catalog.LAYERS is LAYERS
    assert launcher.LAYERS is LAYERS


def test_a_task_file_expects_one_recording_per_demo() -> None:
    launcher = _launcher()
    (item,) = _items(1)
    ids = launcher.expected_recording_ids(item)
    assert len(ids) == launcher.DEMOS_PER_TASK
    assert ids[0] == "suite_a/task_0__demo_0"
    assert ids[-1] == f"suite_a/task_0__demo_{launcher.DEMOS_PER_TASK - 1}"


def test_a_synced_bucket_registers_without_renaming(tmp_path: Path) -> None:
    """The README tells people to sync the bucket into `rrds/libero/` and register — the paths have to line up."""
    from libero import catalog

    launcher = _launcher()
    rec_ids = ["libero_goal/turn_on_the_stove__demo_0", "libero_goal/turn_on_the_stove__demo_1"]
    with patch.object(launcher, "DATASET_PREFIX", DATASET_PREFIX):
        for rec_id in rec_ids:
            for layer in launcher.LAYERS:
                key = launcher.layer_dest(layer, rec_id).removeprefix(DATASET_PREFIX)
                path = tmp_path / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")

    # `register` starts from the base RRDs and finds each layer by its relative path.
    assert catalog.demo_ids(tmp_path) == rec_ids
    for rec_id in rec_ids:
        for layer in LAYERS:
            assert (tmp_path / layer_relpath(layer, rec_id)).exists()


# --------------------------------------------------------------------------------------
# prefilter
# --------------------------------------------------------------------------------------


def test_one_listing_answers_for_every_layer() -> None:
    """All layer directories share the dataset prefix, so the launcher must not pay for a listing per layer."""
    launcher = _launcher()
    calls: list[str] = []

    def record(_s3: Any, uri_prefix: str) -> set[str]:
        calls.append(uri_prefix)
        return set()

    with _patched(launcher, record):
        launcher._drop_converted(_items(1), list(LAYERS))

    assert calls == [DATASET_PREFIX]


def test_drop_converted_keeps_only_the_missing_task_files() -> None:
    launcher = _launcher()
    layers = ["base", "urdf"]
    done = _done(layers, [f"suite_a/task_{i}" for i in range(25)])
    with _patched(launcher, _serve(done)):
        todo = launcher._drop_converted(_items(30), layers)

    assert todo is not None
    assert [item.task_id for item in todo] == [f"suite_a/task_{i}" for i in range(25, 30)]
    # `main` slices after this filter, so a small --limit still yields unconverted task files.
    assert [item.task_id for item in todo[:3]] == ["suite_a/task_25", "suite_a/task_26", "suite_a/task_27"]


def test_a_task_file_missing_one_demo_layer_is_rebuilt() -> None:
    launcher = _launcher()
    layers = ["base", "cameras"]
    done = _done(layers, ["suite_a/task_0", "suite_a/task_1"])
    # task_1 uploaded everything but one demo's cameras layer.
    done.discard("libero/cameras/suite_a/task_1__demo_31.rrd")
    with _patched(launcher, _serve(done)):
        todo = launcher._drop_converted(_items(2), layers)

    assert todo is not None
    assert [item.task_id for item in todo] == ["suite_a/task_1"]


def test_a_layer_outside_the_selection_does_not_hold_a_task_file_back() -> None:
    launcher = _launcher()
    done = _done(["base"], ["suite_a/task_0"])
    with _patched(launcher, _serve(done)):
        assert launcher._drop_converted(_items(1), ["base"]) == []


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
