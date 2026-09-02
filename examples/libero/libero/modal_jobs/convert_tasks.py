"""
Convert LIBERO demos to RRDs on Modal, one worker per task file.

The launcher lists the task files in the HuggingFace repo, drops the ones already in the bucket,
and fans the rest out across workers. Each worker downloads its file once, builds the selected
layers (the same ones as `pixi run -e libero convert`) for every demo in it, and uploads each rrd
under its local relative path, so a bucket synced into `rrds/libero/` registers without renaming.

The pixi task runs `modal run --detach`: the workers keep going after the CLI exits. Watch their
progress in the Modal dashboard.

Run `pixi run -e libero convert-on-modal --help` for flags. See the README for examples and prerequisites.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from libero.episodes import HF_REPO_ID, HF_REVISION, WorkItem, discover_task_files, recording_id
from libero.layers import LAYERS
from libero.storage import DATASET_PREFIX
from rrd_datasets_common.hf_repo import HF_HUB_ENV
from rrd_datasets_common.modal_jobs.image import hf_token_secret, image_from_pyproject
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
    from rerun.experimental import Hdf5Reader
    from rerun.urdf import UrdfTree

    from libero.properties_layer import TaskFacts

REPO_ROOT = Path(__file__).resolve().parents[2]

# Number of demos per hdf5 file, as surveyed across the dataset.
# Used for the launcher to skip a task file if all expected rrds are in the bucket.
# The worker converts whatever the file actually holds.
DEMOS_PER_TASK = 50

# The vendored `fer` model goes where `urdf_layer.URDF_PATH` resolves, so `load_urdf()` works
# unchanged on a worker.
URDF_DIR_IN_IMAGE = "/root/urdf"

HOUR = 60 * 60
MEMORY_REQUEST = 2048
CPU_REQUEST = 1.0

# Caps the Hub download burst: each worker pulls a whole task file, up to ~1.3 GB.
MAX_CONTAINERS = 30

image = image_from_pyproject(
    REPO_ROOT / "pyproject.toml",
    extras=["cloud"],
    env={**HF_HUB_ENV, **worker_env()},
    files={"urdf": URDF_DIR_IN_IMAGE},
    python_sources=("libero", "rrd_datasets_common"),
    python_version="3.12",  # pyproject pins >=3.12,<3.13
)

app = modal.App("libero", image=image)


def layer_dest(layer: str, rec_id: str) -> str:
    """
    The S3 URI one demo's layer is uploaded to.

    The key matches the local relative path, so a bucket synced into `rrds/libero/` registers as is.
    """
    return f"{DATASET_PREFIX}{layer_relpath(layer, rec_id)}"


def expected_recording_ids(item: WorkItem) -> list[str]:
    """The recording ids the launcher expects one task file to produce."""
    return [recording_id(item.task_id, f"demo_{index}") for index in range(DEMOS_PER_TASK)]


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------

_URDF: UrdfTree | None = None


def _urdf_tree() -> UrdfTree:
    """The parsed `fer` model, built once per container."""
    global _URDF
    if _URDF is None:
        from libero import urdf_layer

        _URDF = urdf_layer.load_urdf()
    return _URDF


def _build_demo_layer(layer: str, reader: Hdf5Reader, facts: TaskFacts, task: str, demo: str, out_dir: Path) -> Path:
    """Build one layer of one demo, returning the written rrd."""
    from libero import base_layer, camera_layer, properties_layer, urdf_layer

    if layer == "base":
        return base_layer.convert_demo(reader, task, demo, out_dir)
    if layer == "properties":
        return properties_layer.convert_demo(reader, facts, demo, out_dir)
    if layer == "urdf":
        return urdf_layer.convert_demo(_urdf_tree(), reader, task, demo, out_dir)
    if layer == "cameras":
        return camera_layer.convert_demo(reader, task, demo, out_dir)
    raise ValueError(f"Unknown layer: {layer}")


@app.function(
    timeout=4 * HOUR,
    cpu=CPU_REQUEST,
    memory=MEMORY_REQUEST,
    region=region_pin(),
    secrets=[hf_token_secret(), *extra_secrets()],
    max_containers=MAX_CONTAINERS,
)  # type: ignore[misc]
def convert_task_remote(item: WorkItem, layers: list[str], overwrite: bool) -> None:
    """
    Build the requested layers for every demo in one task file and upload each to the bucket.

    Without `overwrite`, layers already in the bucket are skipped demo by demo, so a cut-short
    run finishes on the next one without redoing what landed.
    """
    from huggingface_hub import hf_hub_download
    from rerun.experimental import Hdf5Reader

    from libero.base_layer import demo_keys
    from libero.properties_layer import task_facts

    print("creating storage client…")
    s3 = worker_client()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        print(f"downloading {item.path}…")
        task_file = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=HF_REVISION,
                filename=item.path,
                local_dir=str(root / "hf"),
            )
        )

        reader = Hdf5Reader(task_file)
        demos = demo_keys(reader)
        if len(demos) != DEMOS_PER_TASK:
            print(f"note: {item.path} holds {len(demos)} demo(s), not the {DEMOS_PER_TASK} the launcher assumes.")
        facts = task_facts(reader, item.task_id)

        out_dir = root / "rrds"
        uploaded = skipped = 0
        for demo in demos:
            rec_id = recording_id(item.task_id, demo)
            todo = layers if overwrite else [layer for layer in layers if not s3_exists(s3, layer_dest(layer, rec_id))]
            if not todo:
                skipped += 1
                continue
            for layer in todo:
                print(f"building {layer} layer: {rec_id}…")
                rrd = _build_demo_layer(layer, reader, facts, item.task_id, demo, out_dir)
                dest = layer_dest(layer, rec_id)
                upload_file(s3, str(rrd), dest)
                rrd.unlink()  # ~50 demos per file: keep the container disk flat.
                uploaded += 1
                print(f"uploaded: {dest}")
        print(f"done: {item.task_id} ({uploaded} rrd(s) uploaded, {skipped} demo(s) already complete)")


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


def _drop_converted(items: list[WorkItem], layers: list[str]) -> list[WorkItem] | None:
    """Drop task files whose every expected rrd (all selected layers, all DEMOS_PER_TASK demos) is in the bucket."""
    from botocore.exceptions import BotoCoreError, ClientError

    # One listing under the dataset prefix answers for every layer.
    print(f"Listing {DATASET_PREFIX} to skip already-converted task files…", flush=True)
    try:
        done = s3_existing_keys(launcher_client(), DATASET_PREFIX)
    except (BotoCoreError, ClientError) as exc:
        print(f"Could not list {DATASET_PREFIX} locally ({type(exc).__name__}) — workers will skip as they go.")
        return None

    _, key_prefix = s3_parts(DATASET_PREFIX)
    return [
        item
        for item in items
        if not all(
            f"{key_prefix}{layer_relpath(layer, rec_id)}" in done
            for rec_id in expected_recording_ids(item)
            for layer in layers
        )
    ]


@app.local_entrypoint()  # type: ignore[misc]
def main(*arglist: str) -> None:
    """Discover task files on HuggingFace and fan the conversions out across Modal workers."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert LIBERO demos to RRDs on Modal: one worker per task file (~50 demos each), "
            "uploading each demo's .rrd layers to the bucket. Runs detached — watch progress in "
            "the Modal dashboard."
        ),
        epilog="Examples and prerequisites: see the README.",
    )
    parser.add_argument(
        "--path-filter", default="", help="Keep only task files whose repo path contains this substring."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Max task files to convert (<= 0 = all). Counts only files not yet fully in the bucket. Default: 1.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help=f"Comma-separated layers to build ({', '.join(LAYERS)}), or 'all'. Default: all.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild even if the RRD already exists in the bucket."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List the task files that would be converted, then exit."
    )
    args = parser.parse_args(args=arglist)

    # Check the bucket first — a bad target should not cost the full HF tree listing.
    check_bucket()
    layers = parse_layers(args.layers)
    matched = discover_task_files(HF_REPO_ID, args.path_filter)
    if not matched:
        print(f"No task files matched (path_filter={args.path_filter!r}).")
        return

    # Drop what is already converted before applying --limit, so --limit counts real work.
    todo = None if args.overwrite else _drop_converted(matched, layers)
    items = matched if todo is None else todo
    if args.limit > 0:
        items = items[: args.limit]

    already = "" if todo is None else f"{len(matched) - len(todo)} already converted, "
    print(f"{len(matched)} task file(s) matched, {already}{len(items)} to convert (~{DEMOS_PER_TASK} demos each).")
    if not items:
        print(f"Nothing to do -> {DATASET_PREFIX}")
        return

    print(f"Layers: {', '.join(layers)}. Each worker downloads its whole task file.")
    print(f"Modal worker request: cpu={CPU_REQUEST}, memory={MEMORY_REQUEST} MiB")

    if args.dry_run:
        print(f"Dry run: {len(items)} task file(s) would be converted -> {DATASET_PREFIX}")
        for item in items:
            print(f"  {item.path} -> {item.task_id}__demo_* ({', '.join(layers)})")
        return

    count = len(items)
    print(f"Spawning {count} worker(s) -> {DATASET_PREFIX} (overwrite={args.overwrite})")
    convert_task_remote.spawn_map(items, [layers] * count, [args.overwrite] * count)
    print(f"Spawned {count} worker(s). Watch progress in the Modal dashboard.")
