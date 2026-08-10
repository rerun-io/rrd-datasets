"""
List a HuggingFace dataset repo (once per revision, not once per run).

Listing a large repo tree is expensive, so it is cached on disk against the repo's commit sha.
"""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Bump when the on-disk cache layout changes, so older files read as a miss.
CACHE_VERSION = 2

# Env vars for a worker image that reaches the Hub.
HF_HUB_ENV = {
    # Xet and telemetry spend HuggingFace's small api quota; without them a download bills only the
    # much larger resolvers quota.
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def hf_file_index(repo_id: str, cache_path: Path) -> set[str]:
    """
    Every file path in a HuggingFace dataset repo, cached at `cache_path` and pinned to its commit.

    The cached copy is reused whenever the repo's sha still matches.
    When the sha cannot be read at all — offline, rate-limited — an existing cache is used unverified
    rather than failing the caller.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    sha: str | None = None
    try:
        sha = api.repo_info(repo_id, repo_type="dataset").sha
    except Exception as exc:
        print(f"Could not read the {repo_id} revision ({type(exc).__name__}).")

    cached = _read_cache(cache_path)
    if cached is not None and (sha is None or cached[0] == sha):
        cached_sha, files = cached
        note = f"revision {cached_sha[:8]}" if sha else "revision unverified"
        print(f"Using cached file list ({note}, {cache_path.name}).", flush=True)
        return files

    if sha is None:
        raise SystemExit(f"Cannot list {repo_id} and no usable cache at {cache_path}.")

    print(f"Listing {repo_id} (full repo tree, minutes for a large one; cached for {sha[:8]})…", flush=True)
    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    _write_cache(cache_path, sha, files)
    return files


def _read_cache(cache_path: Path) -> tuple[str, set[str]] | None:
    """The cached `(sha, files)`, or `None` when there is nothing usable."""
    try:
        cached = json.loads(gzip.decompress(cache_path.read_bytes()))
        if cached["version"] != CACHE_VERSION:
            return None
        return str(cached["sha"]), {str(path) for path in cached["files"]}
    # Missing, truncated, and older layouts all read as a miss, so a damaged cache costs a re-listing
    # instead of a crash.
    except (OSError, gzip.BadGzipFile, AttributeError, KeyError, TypeError, ValueError):
        return None


def _write_cache(cache_path: Path, sha: str, files: set[str]) -> None:
    """Write the listing for `sha`. A cache that cannot be written warns and is skipped."""
    payload = {"version": CACHE_VERSION, "sha": sha, "files": sorted(files)}
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(gzip.compress(json.dumps(payload).encode(), compresslevel=6))
    except OSError as exc:
        print(f"Could not write {cache_path} ({type(exc).__name__}) — will re-list next time.")
