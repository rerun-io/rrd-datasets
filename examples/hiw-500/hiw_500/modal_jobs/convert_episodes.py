"""
Convert HIW-500 episodes to RRDs on Modal.

The launcher is the local entrypoint. It lists the episodes in the HuggingFace repo, drops the ones
already in the bucket, and fans the rest out across workers. Each worker builds the same six layers as
`pixi run -e hiw convert` — base, urdf, odom, cameras, ir, properties — and uploads each under its
local relative path (`<layer>/<recording_id>.rrd`), so a bucket synced into `rrds/hiw-500/` registers
without renaming anything.

`--layers` narrows that set. Only the selected layers are built, and a worker downloads only what they
read: the base layer additionally embeds the calibration sidecars verbatim, the camera layer needs just
the episode's head calibration, and the properties layer its `info.json` plus the wrist calibrations —
the last two never pull the episode MCAP.

The launch is fire-and-forget: the pixi task uses `modal run --detach`, so the workers keep going
after the CLI exits. Watch their progress in the Modal dashboard.

Run `pixi run -e hiw convert-on-modal --help` for flags; see the README for examples and prerequisites.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from hiw_500.episode_index import HF_REPO_ID, HF_REVISION, WorkItem, discover_episodes
from hiw_500.layers import LAYERS
from hiw_500.storage import DATASET_PREFIX
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
    from rerun.urdf import UrdfTree

REPO_ROOT = Path(__file__).resolve().parents[2]

# Which inputs each layer actually reads, so a worker downloads no more than it needs. The base
# layer reads every calibration file: it embeds them verbatim as the episode's archival record.
# The properties layer needs the wrist calibrations only for their presence, which tells it
# whether the episode records IR.
NEEDS_MCAP = frozenset({"base", "derived_archetypes", "urdf", "odom", "ir"})
NEEDS_INFO = frozenset({"base", "properties"})
NEEDS_HEAD_CALIB = frozenset({"base", "cameras"})
NEEDS_WRIST_CALIBS = frozenset({"base", "properties"})

# Assets the converter opens by path, shipped into the image rather than downloaded per episode.
URDF_DIR_IN_IMAGE = "/root/urdf"
URDF_IN_IMAGE = f"{URDF_DIR_IN_IMAGE}/g1/g1_29dof_mode_15_with_dex1_1.urdf"

HOUR = 60 * 60
MEMORY_REQUEST = 2048
CPU_REQUEST = 1.0

# Cap the workers so their Hub requests do not arrive as one burst — HuggingFace rate-limits the
# metadata api by request count, and a worker makes about one request per file.
MAX_CONTAINERS = 50

# The native JPEG library behind pyturbojpeg.
# PyTurboJPEG 2 needs libjpeg-turbo 3, and Debian stable still packages 2.1 (`apt libturbojpeg0`).
# So take the upstream build and put its libturbojpeg where both ldconfig and PyTurboJPEG's own search list look.
LIBJPEG_TURBO_VERSION = "3.2.0"
LIBJPEG_TURBO_INSTALL = (
    "url=https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/"
    f"{LIBJPEG_TURBO_VERSION}/libjpeg-turbo-official_{LIBJPEG_TURBO_VERSION}"
    '_$(dpkg --print-architecture).deb && curl -fsSL -o /tmp/libjpeg-turbo.deb "$url"'
    " && dpkg -i /tmp/libjpeg-turbo.deb && rm /tmp/libjpeg-turbo.deb"
    " && ln -sf /opt/libjpeg-turbo/lib64/libturbojpeg.so.0 /usr/local/lib/libturbojpeg.so.0"
    " && ldconfig"
)

image = image_from_pyproject(
    REPO_ROOT / "pyproject.toml",
    extras=["cloud"],
    apt=["curl", "ca-certificates"],  # for LIBJPEG_TURBO_INSTALL download
    commands=[LIBJPEG_TURBO_INSTALL],
    env={**HF_HUB_ENV, **worker_env()},
    files={"urdf": URDF_DIR_IN_IMAGE},
    python_sources=("hiw_500", "rrd_datasets_common"),
    python_version="3.12",  # pyproject pins >=3.12,<3.13
)

app = modal.App("hiw-500", image=image)


def layer_dest(layer: str, recording_id: str) -> str:
    """
    The S3 URI one episode's layer is uploaded to.

    Layers land at `<prefix><layer>/<recording_id>.rrd` (see `storage.py` for the whole bucket
    layout) — the same relative path the local tree uses, so syncing the layer directories into
    `rrds/hiw-500/` gives `pixi run -e hiw register` exactly the paths it expects.
    """
    return f"{DATASET_PREFIX}{layer_relpath(layer, recording_id)}"


def expected_layers(item: WorkItem, layers: list[str]) -> list[str]:
    """The subset of `layers` this episode can produce: cameras needs a head calibration, ir needs IR streams."""
    return [layer for layer in layers if (layer != "cameras" or item.head_calib) and (layer != "ir" or item.has_ir)]


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------

# Built on first use and reused by every episode the container goes on to handle: parsing the
# G1 model reads 23 MB of meshes.
_URDF: UrdfTree | None = None


def _urdf_tree() -> UrdfTree:
    """The parsed G1 model, built once per container."""
    global _URDF
    if _URDF is None:
        from rerun.urdf import UrdfTree

        from hiw_500 import urdf_layer

        _URDF = UrdfTree.from_file_path(
            URDF_IN_IMAGE,
            entity_path_prefix=urdf_layer.ENTITY_PREFIX,
            static_transform_entity_path=f"{urdf_layer.ENTITY_PREFIX}/tf_static",
        )
    return _URDF


@app.function(
    timeout=2 * HOUR,
    cpu=CPU_REQUEST,
    memory=MEMORY_REQUEST,
    region=region_pin(),
    secrets=[hf_token_secret(), *extra_secrets()],
    max_containers=MAX_CONTAINERS,
)  # type: ignore[misc]
def convert_episode_remote(item: WorkItem, layers: list[str], overwrite: bool) -> None:
    """
    Build the requested layers for one HIW-500 episode and upload each to the bucket.

    Layers already in the bucket are skipped unless `overwrite`, so an episode whose upload was cut
    short finishes on the next run without redoing the layers that landed.
    """
    from huggingface_hub import hf_hub_download

    from hiw_500 import (
        base_layer,
        camera_layer,
        derived_archetypes_layer,
        ir_layer,
        odom_layer,
        properties_layer,
        urdf_layer,
    )
    from hiw_500.base_layer import Episode, EpisodeInfo

    print("creating storage client…")
    s3 = worker_client()
    wanted = expected_layers(item, layers)
    todo = (
        wanted if overwrite else [layer for layer in wanted if not s3_exists(s3, layer_dest(layer, item.recording_id))]
    )
    if not todo:
        print(f"skip (all {len(wanted)} expected layer(s) exist): {item.recording_id}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hf_dir = root / "hf"

        # Only the layers being built decide what comes down the wire.
        mcap = hf_dir / item.mcap
        if NEEDS_MCAP.intersection(todo):
            print(f"downloading {item.mcap}…")
            mcap = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    filename=item.mcap,
                    local_dir=str(hf_dir),
                )
            )
        info = EpisodeInfo()
        if item.info and NEEDS_INFO.intersection(todo):
            info_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    filename=item.info,
                    local_dir=str(hf_dir),
                )
            )
            info = EpisodeInfo.from_json(info_path)
        head_calib: Path | None = None
        if item.head_calib and NEEDS_HEAD_CALIB.intersection(todo):
            head_calib = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    filename=item.head_calib,
                    local_dir=str(hf_dir),
                )
            )
        if NEEDS_WRIST_CALIBS.intersection(todo):
            # Landing under their repo-relative paths beside the mcap, where the calibration
            # globs pick them up.
            for calib in item.wrist_calibs:
                hf_hub_download(
                    repo_id=HF_REPO_ID, repo_type="dataset", revision=HF_REVISION, filename=calib, local_dir=str(hf_dir)
                )

        episode = Episode(mcap=mcap, info=info, recording_id=item.recording_id, head_calib=head_calib)
        out_dir = root / "rrds"
        builders = {
            "base": lambda: base_layer.convert_episode(episode, out_dir),
            "derived_archetypes": lambda: derived_archetypes_layer.convert_episode(episode, out_dir),
            "urdf": lambda: urdf_layer.convert_episode(_urdf_tree(), episode, out_dir),
            "odom": lambda: odom_layer.convert_episode(episode, out_dir),
            "cameras": lambda: camera_layer.convert_episode(episode, out_dir),
            "ir": lambda: ir_layer.convert_episode(episode, out_dir),
            "properties": lambda: properties_layer.convert_episode(episode, out_dir),
        }

        for layer in todo:
            print(f"building {layer} layer…")
            rrd = builders[layer]()
            if rrd is None:
                print(f"skipped {layer} (no input): {item.recording_id}")
                continue
            dest = layer_dest(layer, item.recording_id)
            upload_file(s3, str(rrd), dest)
            print(f"uploaded: {dest}")


# --------------------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------------------


def parse_layers(value: str) -> list[str]:
    """
    The layers named in `value` (comma-separated, or `all`), in `LAYERS` order.

    Returning a fixed order keeps the logs and the build sequence the same however the flag is typed.
    """
    if value.strip().lower() in ("", "all"):
        return list(LAYERS)
    names = {name.strip() for name in value.split(",") if name.strip()}
    unknown = sorted(names - set(LAYERS))
    if unknown:
        raise SystemExit(f"Unknown layer(s): {', '.join(unknown)}. Choose from {', '.join(LAYERS)}, or 'all'.")
    return [name for name in LAYERS if name in names]


def _drop_converted(items: list[WorkItem], layers: list[str]) -> list[WorkItem] | None:
    """
    Drop episodes whose every expected layer is already in the bucket, so no worker is spawned for them.

    An episode missing even one expected layer is kept, and the worker then rebuilds only what is
    absent. Layers the episode cannot produce (`expected_layers`) are not waited for. Returns
    `None` when the bucket cannot be listed, leaving the skipping to the workers.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    # Every layer directory shares the dataset prefix, so a single listing answers for all of them.
    print(f"Listing {DATASET_PREFIX} to skip already-converted episodes…", flush=True)
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
            f"{key_prefix}{layer_relpath(layer, item.recording_id)}" in done for layer in expected_layers(item, layers)
        )
    ]


@app.local_entrypoint()  # type: ignore[misc]
def main(*arglist: str) -> None:
    """Discover episodes on HuggingFace and fan the layer builds out across Modal workers (`spawn_map`)."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert HIW-500 episodes to RRDs on Modal: discover episodes on HuggingFace, fan the "
            "layer builds out across Modal workers, and upload each .rrd to the bucket. Runs detached "
            "(fire-and-forget) — watch progress in the Modal dashboard."
        ),
        epilog="Examples and prerequisites: see the README.",
    )
    parser.add_argument("--task-filter", default="", help="Keep only episodes whose HF path contains this substring.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Max episodes to convert (<= 0 = all). Counts only episodes not yet in the bucket. Default: 1.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help=f"Comma-separated layers to build ({', '.join(LAYERS)}), or 'all'. Default: all.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild even if the RRD already exists in the bucket."
    )
    parser.add_argument("--dry-run", action="store_true", help="List the episodes that would be converted, then exit.")
    args = parser.parse_args(args=arglist)

    # Check the bucket first — a bad target should not cost the full HF tree listing.
    check_bucket()
    layers = parse_layers(args.layers)
    matched = discover_episodes(HF_REPO_ID, args.task_filter)
    if not matched:
        print(f"No episodes matched (task_filter={args.task_filter!r}).")
        return

    # Drop what is already converted before applying --limit, so --limit counts real work.
    todo = None if args.overwrite else _drop_converted(matched, layers)
    items = matched if todo is None else todo
    if args.limit > 0:
        items = items[: args.limit]

    already = "" if todo is None else f"{len(matched) - len(todo)} already converted, "
    print(f"{len(matched)} episode(s) matched, {already}{len(items)} to convert.")
    if not items:
        print(f"Nothing to do -> {DATASET_PREFIX}")
        return

    if NEEDS_MCAP.intersection(layers):
        fetches = "the episode mcap and its sidecars"
    else:
        fetches = "sidecar files only — no mcap download"
    print(f"Layers: {', '.join(layers)}. Each worker downloads {fetches}.")
    print(f"Modal worker request: cpu={CPU_REQUEST}, memory={MEMORY_REQUEST} MiB")

    if args.dry_run:
        print(f"Dry run: {len(items)} episode(s) would be converted -> {DATASET_PREFIX}")
        for item in items:
            print(f"  {item.mcap} -> {item.recording_id} ({', '.join(layers)})")
        return

    count = len(items)
    print(f"Spawning {count} worker(s) -> {DATASET_PREFIX} (overwrite={args.overwrite})")
    convert_episode_remote.spawn_map(items, [layers] * count, [args.overwrite] * count)
    print(f"Spawned {count} worker(s). Watch progress in the Modal dashboard.")
