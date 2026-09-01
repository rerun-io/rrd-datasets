"""Tests for this example's storage defaults, which the `libero` pixi environments declare."""

from __future__ import annotations

import tomllib

from rrd_datasets_common.paths import repo_root

DEFAULTS = {
    "STORAGE_BACKEND": "${STORAGE_BACKEND:-hf}",
    "HF_BUCKET": "${HF_BUCKET:-libero}",
}


def test_storage_defaults() -> None:
    """
    Test that the `libero` environments declare the storage defaults in `${VAR:-default}` form.

    A literal value would silently override a caller's own setting, since pixi's activation wins
    over the surrounding shell.
    """
    manifest = tomllib.loads((repo_root() / "pixi.toml").read_text())
    declared = manifest["feature"]["libero"]["activation"]["env"]
    assert {key: declared.get(key) for key in DEFAULTS} == DEFAULTS
