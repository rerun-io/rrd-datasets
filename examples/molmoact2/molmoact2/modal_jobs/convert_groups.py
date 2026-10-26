"""
Convert MolmoAct2 file groups to RRDs on Modal, one worker per file group.

The launcher builds the file groups from the episode index, drops the ones already in the bucket,
and starts a worker for each of the rest. A worker downloads `meta/` and its group's files once
(only `meta/` when it builds just the stats layer), builds the selected layers of every episode, and
uploads each rrd under its local relative path, so a bucket synced into `rrds/molmoact2/` registers
without renaming.

Run:  pixi run -e molmo convert-on-modal --help
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from molmoact2.episode_index import HF_REPO_ID, HF_REVISION, WorkItem, discover_groups, meta_files
from molmoact2.layers import LAYERS
from molmoact2.storage import DATASET_PREFIX
from rrd_datasets_common.hf_repo import HF_HUB_ENV
from rrd_datasets_common.modal_jobs.image import image_from_pyproject
from rrd_datasets_common.modal_jobs.store import (
    check_bucket,
    extra_secrets,
    launcher_client,
    region_pin,
    s3_existing_keys,
    s3_exists,
    s3_parts,
    upload_file,
    worker_client,
    worker_env,
)
from rrd_datasets_common.paths import layer_relpath

if TYPE_CHECKING:
    from rerun.urdf import UrdfTree

REPO_ROOT = Path(__file__).resolve().parents[2]

# The vendored YAM model goes where `urdf_layer.URDF_PATH` resolves, so `load_urdfs()` works
# unchanged on a worker.
URDF_DIR_IN_IMAGE = "/root/urdf"

HOUR = 60 * 60
MEMORY_REQUEST = 2048
CPU_REQUEST = 1.0

# Caps the Hub download burst: each worker fetches `meta/` and a median of 12 data and video files.
MAX_CONTAINERS = 30

image = image_from_pyproject(
    REPO_ROOT / "pyproject.toml",
    extras=["cloud"],
    apt=["ffmpeg"],  # `LeRobotReader` re-encodes the start of a video window that begins on a delta frame
    env={**HF_HUB_ENV, **worker_env()},
    files={"urdf": URDF_DIR_IN_IMAGE},
    python_sources=("molmoact2", "rrd_datasets_common"),
    python_version="3.12",  # pyproject pins >=3.12,<3.13
)

app = modal.App("molmoact2", image=image)


def layer_dest(layer: str, recording_id: str) -> str:
    """
    The S3 URI one episode's layer is uploaded to.

    The key matches the local relative path, so a bucket synced into `rrds/molmoact2/` registers as is.
    """
    return f"{DATASET_PREFIX}{layer_relpath(layer, recording_id)}"


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------


def _build_derived_layer(layer: str, base_rrd: Path, urdfs: dict[str, UrdfTree], out_dir: Path) -> Path:
    """Build one layer that derives from an episode's base rrd, returning the written rrd."""
    from molmoact2 import camera_layer, urdf_layer

    if layer == "urdf":
        return urdf_layer.write_layer(base_rrd, urdfs, out_dir)
    if layer == "cameras":
        return camera_layer.write_layer(base_rrd, out_dir)
    raise ValueError(f"Unknown layer: {layer}")


def _hf_token_secret() -> modal.Secret:
    """The caller's HuggingFace token as a per-run secret, empty when there is none."""
    if not modal.is_local():
        return modal.Secret.from_dict({})

    from huggingface_hub import get_token

    token = get_token()
    if not token:
        # MolmoAct2 is public, so a worker can download it anonymously, with a smaller per-IP quota.
        return modal.Secret.from_dict({})
    return modal.Secret.from_dict({"HF_TOKEN": token})


@app.function(
    timeout=2 * HOUR,
    cpu=CPU_REQUEST,
    memory=MEMORY_REQUEST,
    region=region_pin(),
    secrets=[_hf_token_secret(), *extra_secrets()],
    max_containers=MAX_CONTAINERS,
)  # type: ignore[misc]
def convert_group_remote(item: WorkItem, layers: list[str], overwrite: bool, meta: list[str]) -> None:
    """
    Build the requested layers of every episode in one file group and upload each to the bucket.

    Without `overwrite`, layers already in the bucket are skipped episode by episode, so a run that
    stopped early finishes on the next one without redoing what was uploaded.
    """
    from huggingface_hub import hf_hub_download
    from rerun.experimental import LeRobotReader

    from molmoact2 import base_layer, stats_layer, urdf_layer

    print("creating storage client…")
    s3 = worker_client()
    todo = {
        recording_id: [layer for layer in layers if overwrite or not s3_exists(s3, layer_dest(layer, recording_id))]
        for recording_id in item.recording_ids
    }
    if not any(todo.values()):
        print(f"done: {item.group_id} (every selected layer is in the bucket)")
        return
    skipped = sum(not layers for layers in todo.values())

    # The stats layer reads `meta/` alone. Every other layer derives from the episode's base layer,
    # which needs the group's data and video files.
    stats_todo = [recording_id for recording_id, layers in todo.items() if stats_layer.LAYER in layers]
    base_todo = {
        recording_id: [layer for layer in layers if layer != stats_layer.LAYER] for recording_id, layers in todo.items()
    }
    files = item.files if any(base_todo.values()) else ()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        print(f"downloading {len(meta)} meta and {len(files)} data and video files of {item.group_id}…")
        for filename in [*meta, *files]:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=HF_REVISION,
                filename=filename,
                local_dir=str(source),
            )

        out_dir = root / "rrds"
        uploaded = 0
        if stats_todo:
            print(f"building {stats_layer.LAYER}: {len(stats_todo)} episode(s)…")
            stats = stats_layer.read_episode_stats(source)
            for recording_id in stats_todo:
                rrd = stats_layer.write_layer(recording_id, stats, out_dir)
                upload_file(s3, str(rrd), layer_dest(stats_layer.LAYER, recording_id))
                rrd.unlink()
                uploaded += 1

        if files:
            reader = LeRobotReader(source)
            source_meta = base_layer.read_source_meta(source)
            urdfs = urdf_layer.load_urdfs()
            for episode, recording_id in zip(item.episodes, item.recording_ids, strict=True):
                if not base_todo[recording_id]:
                    continue
                print(f"building {', '.join(base_todo[recording_id])}: {recording_id}…")
                base_rrd = base_layer.convert_episode(episode, reader, source_meta, out_dir)
                for layer in base_todo[recording_id]:
                    rrd = base_rrd if layer == "base" else _build_derived_layer(layer, base_rrd, urdfs, out_dir)
                    upload_file(s3, str(rrd), layer_dest(layer, recording_id))
                    if rrd != base_rrd:
                        rrd.unlink()
                    uploaded += 1
                base_rrd.unlink()
        print(f"done: {item.group_id} ({uploaded} rrd(s) uploaded, {skipped} episode(s) already complete)")


# --------------------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------------------


def parse_layers(value: str) -> list[str]:
    """The layers named in `value` (comma-separated, or `all`), always in `LAYERS` order."""
    if value.strip().lower() in ("", "all"):
        return list(LAYERS)
    names = {name.strip() for name in value.split(",") if name.strip()}
    unknown = sorted(names - set(LAYERS))
    if unknown:
        raise SystemExit(f"Unknown layer(s): {', '.join(unknown)}. Choose from {', '.join(LAYERS)}, or 'all'.")
    return [name for name in LAYERS if name in names]


def drop_converted(items: list[WorkItem], layers: list[str]) -> list[WorkItem] | None:
    """Drop the file groups whose selected layers are all in the bucket, or `None` when it cannot be listed."""
    from botocore.exceptions import BotoCoreError, ClientError

    # One listing under the dataset prefix answers for every layer.
    print(f"Listing {DATASET_PREFIX} to skip converted file groups…", flush=True)
    try:
        done = s3_existing_keys(launcher_client(), DATASET_PREFIX)
    except (BotoCoreError, ClientError) as exc:
        print(f"Could not list {DATASET_PREFIX} locally ({type(exc).__name__}), so the workers check instead.")
        return None

    _, key_prefix = s3_parts(DATASET_PREFIX)
    return [
        item
        for item in items
        if not all(
            f"{key_prefix}{layer_relpath(layer, recording_id)}" in done
            for recording_id in item.recording_ids
            for layer in layers
        )
    ]


@app.local_entrypoint()  # type: ignore[misc]
def main(*arglist: str) -> None:
    """List the file groups and start one Modal worker per pending group."""
    from huggingface_hub import get_token

    parser = argparse.ArgumentParser(
        description=(
            "Convert MolmoAct2 to RRDs on Modal: one worker per file group, uploading each episode's "
            ".rrd layers to the bucket. Runs detached, so watch progress in the Modal dashboard."
        ),
        epilog="Examples and prerequisites: see the README.",
    )
    parser.add_argument("--task-filter", default="", help="Keep only file groups with a task containing this text.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Max file groups to convert (<= 0 = all). Counts only groups not yet fully in the bucket. Default: 1.",
    )
    parser.add_argument(
        "--layers", default="all", help=f"Comma-separated layers to build ({', '.join(LAYERS)}), or 'all'."
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild layers that are already in the bucket.")
    parser.add_argument("--dry-run", action="store_true", help="List the file groups that would be converted.")
    args = parser.parse_args(args=arglist)

    # Check the bucket first, so a wrong target fails before the repo listing.
    check_bucket()
    layers = parse_layers(args.layers)
    matched = discover_groups(args.task_filter)
    if not matched:
        print(f"No file groups matched (task_filter={args.task_filter!r}).")
        return

    # Drop what is already converted before applying --limit, so --limit counts real work.
    pending = None if args.overwrite else drop_converted(matched, layers)
    items = matched if pending is None else pending
    if args.limit > 0:
        items = items[: args.limit]

    already = "" if pending is None else f"{len(matched) - len(pending)} already converted, "
    episodes = sum(len(item.episodes) for item in items)
    print(f"{len(matched)} file group(s) matched, {already}{len(items)} to convert ({episodes} episodes).")
    if not items:
        print(f"Nothing to do -> {DATASET_PREFIX}")
        return

    download = "only meta/" if layers == ["stats"] else "its whole file group"
    print(f"Layers: {', '.join(layers)}. Each worker downloads {download}.")
    print(f"Modal worker request: cpu={CPU_REQUEST}, memory={MEMORY_REQUEST} MiB")
    if not get_token():
        print("note: no HuggingFace token, so the workers download anonymously")

    if args.dry_run:
        print(f"Dry run: {len(items)} file group(s) would be converted -> {DATASET_PREFIX}")
        for item in items:
            print(f"  {item.group_id} ({len(item.episodes)} episodes, {len(item.files)} files)")
        return

    count = len(items)
    meta = meta_files()
    print(f"Spawning {count} worker(s) -> {DATASET_PREFIX} (overwrite={args.overwrite})")
    convert_group_remote.spawn_map(items, [layers] * count, [args.overwrite] * count, [meta] * count)
    print(f"Spawned {count} worker(s). Watch progress in the Modal dashboard.")
