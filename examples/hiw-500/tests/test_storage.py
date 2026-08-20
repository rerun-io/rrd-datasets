"""Tests for this example's storage defaults, which the `hiw` pixi environments declare."""

from __future__ import annotations

import tomllib

from rrd_datasets_common.paths import repo_root

DEFAULTS = {
    "STORAGE_BACKEND": "${STORAGE_BACKEND:-hf}",
    "HF_BUCKET": "${HF_BUCKET:-hiw-500}",
}


def test_the_example_declares_its_storage_defaults() -> None:
    """
    Converted recordings go to a HuggingFace bucket named after the dataset.

    The `${VAR:-default}` form is what leaves a caller's own `STORAGE_BACKEND=s3` intact: pixi's
    activation wins over the surrounding shell, so a literal value here would silently override it.
    """
    manifest = tomllib.loads((repo_root() / "pixi.toml").read_text())
    declared = manifest["feature"]["hiw-500"]["activation"]["env"]
    assert {key: declared.get(key) for key in DEFAULTS} == DEFAULTS
