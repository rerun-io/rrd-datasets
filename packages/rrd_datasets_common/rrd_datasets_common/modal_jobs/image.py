"""
The environment a Modal worker runs in: its image, and the HuggingFace token it needs.

Dependencies come from the caller's pyproject, so nothing here is specific to one dataset.
"""

from __future__ import annotations

import os
import tomllib
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def _split_requirements(requirements: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Split pip requirements into `(everything else, rerun)`, dropping modal.

    Modal comes from the runtime. The rerun packages install in a second pass, so a pre-release dev
    wheel found through `find_links` wins over the released one.
    """
    wanted = [dep for dep in requirements if not dep.startswith("modal")]
    return (
        [dep for dep in wanted if not dep.startswith("rerun-")],
        [dep for dep in wanted if dep.startswith("rerun-")],
    )


def image_from_pyproject(
    pyproject: Path,
    *,
    extras: Sequence[str] = (),
    apt: Sequence[str] = (),
    commands: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    python_sources: Sequence[str] = (),
    files: Mapping[str, str] | None = None,
    python_version: str = "3.12",
) -> modal.Image:
    """
    Build a worker image from a repo's `[project.dependencies]`.

    `extras` names optional-dependency groups to install as well. `apt` covers native libraries pip
    cannot supply, and `commands` runs shell steps after it, for the ones apt has no package for.
    `files` maps local directories to paths in the image, for assets a converter opens by relative
    path.

    The repo has to keep `[project.dependencies]` complete and installable by pip, since that list is
    the whole of the worker's Python environment.

    Dependencies are read on the launcher side only. A worker rebuilds this object when it imports the
    job, and by then the image it describes already exists.
    """
    pip: list[str] = []
    rerun_pip: list[str] = []
    wheel_index = os.getenv("RERUN_WHEEL_INDEX")

    if modal.is_local():
        config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = config["project"]
        requirements: list[str] = list(project["dependencies"])
        for extra in extras:
            requirements += project["optional-dependencies"][extra]
        pip, rerun_pip = _split_requirements(requirements)
        find_links = config.get("tool", {}).get("uv", {}).get("find-links", [])
        if find_links:
            wheel_index = find_links[0]

    image = modal.Image.debian_slim(python_version=python_version)
    if apt:
        image = image.apt_install(*apt)
    if commands:
        image = image.run_commands(*commands)
    image = (
        image.uv_pip_install(*pip)
        .uv_pip_install(*rerun_pip, find_links=wheel_index)
        .run_commands("rerun analytics disable")
    )
    if env:
        image = image.env(dict(env))
    for local_dir, remote_path in (files or {}).items():
        image = image.add_local_dir(local_dir, remote_path=remote_path)
    if python_sources:
        # Last, so editing the packages leaves the cached pip layers alone.
        image = image.add_local_python_source(*python_sources)
    return image


def hf_token_secret() -> modal.Secret:
    """
    The caller's HuggingFace token as a per-run secret, for reaching a gated dataset.

    Read on the launcher side from `$HF_TOKEN` or the `hf auth login` cache, then bound to the run.
    Nothing is stored on Modal, and a worker only ever sees the token in `$HF_TOKEN`.
    """
    if not modal.is_local():
        return modal.Secret.from_dict({})

    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SystemExit("No HuggingFace token found — run `hf auth login` or set HF_TOKEN.")
    return modal.Secret.from_dict({"HF_TOKEN": token})


def hf_s3_secret() -> modal.Secret:
    """
    The caller's HFAK key pair as a per-run secret, for writing an HF Storage Bucket.

    Read on the launcher side from the `RCLONE_CONFIG_HF_*` variables, then bound to the run like
    `hf_token_secret()`. Distinct from `$HF_TOKEN`: the token downloads the dataset, the key pair
    writes the bucket.
    """
    if not modal.is_local():
        return modal.Secret.from_dict({})

    from rrd_datasets_common.modal_jobs.hf_bucket import HF_S3_KEYS, missing_hf_s3_keys

    missing = missing_hf_s3_keys()
    if missing:
        raise SystemExit(f"Missing HF bucket credentials: {', '.join(missing)} — generate an HFAK key pair.")
    return modal.Secret.from_dict({key: os.environ[key] for key in HF_S3_KEYS})
