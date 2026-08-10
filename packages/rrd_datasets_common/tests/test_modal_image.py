"""Tests for the worker image's dependency list: a missing dep only shows up on a running worker."""

from __future__ import annotations

from rrd_datasets_common.modal_jobs.image import _split_requirements

REQUIREMENTS = [
    "rerun-sdk[datafusion]>=0.34,<0.35",
    "numpy>=1.26",
    "huggingface_hub>=1.21.0,<2",
    "modal>=1,<2",
    "boto3>=1.34",
    "rerun-notebook>=0.34,<0.35",
]


def test_split_requirements_holds_rerun_back_and_drops_modal() -> None:
    other, rerun = _split_requirements(REQUIREMENTS)
    assert other == ["numpy>=1.26", "huggingface_hub>=1.21.0,<2", "boto3>=1.34"]
    assert rerun == ["rerun-sdk[datafusion]>=0.34,<0.35", "rerun-notebook>=0.34,<0.35"]


def test_split_requirements_of_nothing() -> None:
    assert _split_requirements([]) == ([], [])
