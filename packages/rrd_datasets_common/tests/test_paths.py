from __future__ import annotations

from pathlib import Path

import pytest

from rrd_datasets_common.paths import (
    dataset_data_dir,
    dataset_rrd_dir,
    default_blueprint_path,
    repo_root,
    resolve_input_path,
)


def test_the_root_comes_from_the_pixi_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXI_PROJECT_ROOT", "/workspace")
    assert repo_root() == Path("/workspace")
    assert dataset_data_dir("HIW-500") == Path("/workspace/data/HIW-500")
    assert dataset_rrd_dir("hiw-500") == Path("/workspace/rrds/hiw-500")
    assert default_blueprint_path("hiw-500") == Path("/workspace/blueprints/hiw-500/default.rbl")


def test_without_the_env_var_the_manifest_marks_the_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A notebook kernel has no pixi activation; the single workspace pixi.toml is the marker."""
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)
    (tmp_path / "pixi.toml").touch()
    nested = tmp_path / "incubating" / "example" / "notebook"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert repo_root() == tmp_path


def test_outside_any_workspace_the_root_falls_back_to_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Modal worker imports the converters from no workspace at all; import must not fail."""
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert repo_root() == tmp_path


def test_input_paths_resolve_against_the_workspace_root_as_a_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Task args are typed repo-root-relative while pixi tasks run in the example directory."""
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "episode.mcap").touch()
    example_dir = tmp_path / "incubating" / "example"
    example_dir.mkdir(parents=True)
    monkeypatch.chdir(example_dir)

    assert resolve_input_path(Path("data/episode.mcap")) == tmp_path / "data" / "episode.mcap"

    # A path that exists from the working directory is taken as given.
    (example_dir / "local.mcap").touch()
    assert resolve_input_path(Path("local.mcap")) == Path("local.mcap")

    # A path that exists nowhere comes back untouched, for the caller's error message.
    assert resolve_input_path(Path("missing.mcap")) == Path("missing.mcap")
