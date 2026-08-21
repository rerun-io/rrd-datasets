"""
Tests that the repo pins exactly one rerun version, in every file that pins it.

The pin lives in several files at once: pixi.toml holds the local environments' version, and each
example's pyproject.toml holds what its Modal image installs. A version bump edits all of them by
hand, so the half-bumped state — local environment on one version, Modal workers on another —
writes diverging `.rrd` output with no error anywhere. These tests hold every pin to one exact
version and that version to the installed SDK.
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

from rrd_datasets_common.paths import repo_root

# An exact rerun pin as a PEP 508 dependency string: name, optional extras, `==version`.
EXACT_PIN = re.compile(r"^(?P<name>rerun-[a-z0-9-]+)(\[[^]]*\])?==(?P<version>.+)$")


def _pixi_pins(manifest: Path) -> dict[str, str]:
    """Every rerun package pinned in any pixi feature's pypi-dependencies, keyed by feature and name."""
    features = tomllib.loads(manifest.read_text())["feature"]
    pins: dict[str, str] = {}
    for feature_name, feature in features.items():
        for name, spec in feature.get("pypi-dependencies", {}).items():
            if name.startswith("rerun-") and isinstance(spec, str):
                assert spec.startswith("=="), f"feature {feature_name} pins {name} inexactly: {spec}"
                pins[f"pixi.toml:{feature_name}:{name}"] = spec.removeprefix("==")
    return pins


def _pyproject_pins(pyproject: Path) -> dict[str, str]:
    """Every rerun dependency of one example, keyed by example and name; each must be an exact pin."""
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    pins: dict[str, str] = {}
    for dependency in dependencies:
        if dependency.startswith("rerun-"):
            match = EXACT_PIN.match(dependency)
            assert match, f"{pyproject} pins rerun inexactly: {dependency}"
            pins[f"{pyproject.parent.name}:{match['name']}"] = match["version"]
    return pins


def _all_pins() -> dict[str, str]:
    root = repo_root()
    pins = _pixi_pins(root / "pixi.toml")
    assert pins, "pixi.toml pins no rerun package"
    for pyproject in sorted(root.glob("examples/*/pyproject.toml")):
        example_pins = _pyproject_pins(pyproject)
        assert example_pins, f"{pyproject} pins no rerun package"
        pins |= example_pins
    return pins


def test_every_rerun_pin_names_the_same_version() -> None:
    """A bump edits several files by hand; a missed one splits local and Modal output."""
    pins = _all_pins()
    # pixi.toml plus at least two examples — the equality must never pass on an empty scan.
    assert len(pins) >= 3, pins
    assert len(set(pins.values())) == 1, pins


def test_the_installed_sdk_is_the_pinned_version() -> None:
    """Catches a bump that edited the manifests but skipped `pixi install`."""
    (pinned,) = set(_all_pins().values())
    assert version("rerun-sdk") == pinned
