"""Tests for `STORAGE_BACKEND` selection: storage config, the store selector, and the gateway client. No network."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

HF_S3_ENV = {
    "HF_BUCKET_ACCESS_KEY_ID": "HFAKunittest",
    "HF_BUCKET_SECRET_ACCESS_KEY": "unit-test-secret",
}


@pytest.fixture(autouse=True)
def restore_modules() -> Iterator[None]:
    """Re-resolve the backend modules after each test, so reloads here never leak into other tests."""
    yield
    from rrd_datasets_common import storage
    from rrd_datasets_common.modal_jobs import store

    importlib.reload(storage)
    importlib.reload(store)


def _reload(monkeypatch: pytest.MonkeyPatch, **env: str) -> tuple[ModuleType, ModuleType]:
    """`(storage, store)` re-imported under `env`, since both resolve the backend at import time."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from rrd_datasets_common import storage
    from rrd_datasets_common.modal_jobs import store

    importlib.reload(storage)
    return storage, importlib.reload(store)


# --------------------------------------------------------------------------------------
# backend resolution
# --------------------------------------------------------------------------------------


def test_the_default_backend_is_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    storage, store = _reload(monkeypatch, S3_BUCKET="aws-bucket")
    assert storage.STORAGE_BACKEND == "s3"
    assert storage.BUCKET == "aws-bucket"
    assert store.region_pin() == storage.S3_REGION
    assert store.extra_secrets() == []


def test_the_s3_backend_nests_each_dataset_under_its_own_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """An AWS bucket may be shared, so every dataset keeps its own prefix."""
    storage, _ = _reload(monkeypatch, S3_BUCKET="aws-bucket")
    assert storage.dataset_prefix("demo-data") == "s3://aws-bucket/demo-data/"


def test_the_hf_backend_swaps_the_bucket_but_keeps_the_uri_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hf bucket is dedicated to one dataset, so the layout sits at its root — no dataset segment."""
    storage, _ = _reload(monkeypatch, STORAGE_BACKEND="hf", HF_BUCKET="hf-bucket", S3_BUCKET="aws-bucket")
    assert storage.BUCKET == "hf-bucket"
    assert storage.dataset_prefix("demo-data") == "s3://hf-bucket/"


def test_an_unknown_backend_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="lamp"):
        _reload(monkeypatch, STORAGE_BACKEND="lamp")


# --------------------------------------------------------------------------------------
# the hf branch of the selector
# --------------------------------------------------------------------------------------


def test_hf_clients_point_at_the_gateway_on_both_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    _, store = _reload(monkeypatch, STORAGE_BACKEND="hf", HF_NAMESPACE="acme", **HF_S3_ENV)
    for client in (store.worker_client(), store.launcher_client()):
        assert client.meta.endpoint_url == "https://s3.hf.co/acme"


def test_the_hf_backend_pins_workers_beside_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    _, store = _reload(monkeypatch, STORAGE_BACKEND="hf")
    assert store.region_pin() == "us-east-1"


def test_missing_hfak_keys_abort_the_launch_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _, store = _reload(monkeypatch, STORAGE_BACKEND="hf")
    for key in HF_S3_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit, match="HF_BUCKET_ACCESS_KEY_ID"):
        store.extra_secrets()


def test_present_hfak_keys_become_one_run_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _, store = _reload(monkeypatch, STORAGE_BACKEND="hf", **HF_S3_ENV)
    assert len(store.extra_secrets()) == 1


# --------------------------------------------------------------------------------------
# uploads
# --------------------------------------------------------------------------------------


class FakeUploader:
    """Stands in for a boto3 client, recording its one `upload_file` call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def upload_file(self, path: str, bucket: str, key: str, Config: object = None) -> None:  # noqa: N803
        """Record the call instead of uploading."""
        self.calls.append({"path": path, "bucket": bucket, "key": key, "config": Config})


def test_hf_uploads_send_whole_rrds_as_one_put(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway is unreliable assembling many multipart parts, so anything under 2 GiB skips multipart."""
    _, store = _reload(monkeypatch, STORAGE_BACKEND="hf")
    uploader = FakeUploader()
    store.upload_file(uploader, "/tmp/x.rrd", "s3://bkt/rrds/x.rrd")

    (call,) = uploader.calls
    assert (call["bucket"], call["key"]) == ("bkt", "rrds/x.rrd")
    config = call["config"]
    assert config is not None
    assert config.multipart_threshold == 2 * 1024**3  # type: ignore[attr-defined]
    assert config.multipart_chunksize == 2 * 1024**3  # type: ignore[attr-defined]


def test_s3_uploads_keep_the_boto3_transfer_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _, store = _reload(monkeypatch, STORAGE_BACKEND="s3")
    uploader = FakeUploader()
    store.upload_file(uploader, "/tmp/x.rrd", "s3://bkt/rrds/x.rrd")

    (call,) = uploader.calls
    assert call["config"] is None


# --------------------------------------------------------------------------------------
# bucket existence check
# --------------------------------------------------------------------------------------


class FakeGateway:
    """Stands in for a gateway client, serving a canned `list_buckets` answer."""

    def __init__(self, buckets: list[str] | Exception) -> None:
        self.buckets = buckets

    def list_buckets(self) -> dict[str, list[dict[str, str]]]:
        """The canned listing, or the canned failure."""
        if isinstance(self.buckets, Exception):
            raise self.buckets
        return {"Buckets": [{"Name": name} for name in self.buckets]}


def _gateway_check(monkeypatch: pytest.MonkeyPatch, buckets: list[str] | Exception) -> None:
    """Run `check_bucket("acme", "demo-data")` against a gateway holding `buckets`."""
    from rrd_datasets_common.modal_jobs import hf_bucket

    monkeypatch.setattr(hf_bucket, "gateway_client", lambda _namespace: FakeGateway(buckets))
    hf_bucket.check_bucket("acme", "demo-data")


def test_check_bucket_accepts_an_existing_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    _gateway_check(monkeypatch, ["other", "demo-data"])


def test_check_bucket_names_the_create_command_for_a_missing_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="hf buckets create acme/demo-data"):
        _gateway_check(monkeypatch, ["other"])


def test_check_bucket_catches_the_namespace_repeated_as_the_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    from rrd_datasets_common.modal_jobs import hf_bucket

    monkeypatch.setattr(hf_bucket, "gateway_client", lambda _namespace: FakeGateway(["demo-data"]))
    with pytest.raises(SystemExit, match="names the bucket bare"):
        hf_bucket.check_bucket("acme", "acme")


def test_check_bucket_reports_an_unlistable_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    from botocore.exceptions import ClientError

    denied = ClientError({"Error": {"Code": "403", "Message": "denied"}}, "ListBuckets")
    with pytest.raises(SystemExit, match="Cannot list buckets in acme"):
        _gateway_check(monkeypatch, denied)


def _no_gateway(namespace: str) -> FakeGateway:
    raise AssertionError("the s3 backend must not touch the gateway")


def test_check_bucket_is_a_no_op_on_the_s3_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """No probe on s3: the launcher's AWS credentials are optional, so a failure would prove nothing."""
    from rrd_datasets_common.modal_jobs import hf_bucket

    _, store = _reload(monkeypatch, STORAGE_BACKEND="s3")
    monkeypatch.setattr(hf_bucket, "gateway_client", _no_gateway)
    store.check_bucket()


# --------------------------------------------------------------------------------------
# gateway client config
# --------------------------------------------------------------------------------------


def test_the_gateway_client_respects_the_gateway_deviations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path-style addressing and no unasked-for checksums — the two ways the gateway is not plain S3."""
    for key, value in HF_S3_ENV.items():
        monkeypatch.setenv(key, value)
    from rrd_datasets_common.modal_jobs.hf_bucket import gateway_client

    client = gateway_client("acme")
    assert client.meta.region_name == "us-east-1"  # the gateway rejects signatures for any other region
    config = client.meta.config
    # botocore Config keeps its options as dynamic attributes the stubs do not declare.
    assert config.s3 == {"addressing_style": "path"}  # type: ignore[attr-defined]
    assert config.request_checksum_calculation == "when_required"  # type: ignore[attr-defined]
    assert config.response_checksum_validation == "when_required"  # type: ignore[attr-defined]
