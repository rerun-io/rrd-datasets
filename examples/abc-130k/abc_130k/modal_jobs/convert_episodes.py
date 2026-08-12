"""
Convert ABC-130k episodes to RRDs on Modal.

The launcher is the local entrypoint. It lists the episodes in the HuggingFace repo, drops the ones already in the bucket, and
fans the rest out across workers. Each worker downloads one episode, runs the same
`abc_130k.convert.convert_episode` as the local flow, and uploads the `.rrd`.

The launch is fire-and-forget: the pixi task uses `modal run --detach`, so the workers keep going
after the CLI exits. Watch their progress in the Modal dashboard.

Run `pixi run -e abc convert-on-modal --help` for flags; see the README for examples and prerequisites.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import modal

from abc_130k.episode_index import HF_REPO_ID, WorkItem, discover_episodes
from abc_130k.storage import RRD_PREFIX
from abc_130k.video_transcode import (
    DEFAULT_CRF,
    DEFAULT_GOP,
    DEFAULT_MAX_WIDTH,
)
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

REPO_ROOT = Path(__file__).resolve().parents[2]

HOUR = 60 * 60
MEMORY_REQUEST = 2048
CPU_REQUEST = 2.0

# Cap the workers so their Hub requests do not arrive as one burst — HuggingFace rate-limits the
# metadata api by request count, and a worker makes about one request per file.
MAX_CONTAINERS = 1000

image = image_from_pyproject(
    REPO_ROOT / "pyproject.toml",
    extras=["cloud"],
    apt=["ffmpeg"],  # video codecs for the RRD video streams
    env={**HF_HUB_ENV, **worker_env()},
    python_sources=("abc_130k", "rrd_datasets_common"),
)

app = modal.App("abc-130k", image=image)


@app.function(
    timeout=2 * HOUR,
    cpu=CPU_REQUEST,
    memory=MEMORY_REQUEST,
    region=region_pin(),
    secrets=[hf_token_secret(), *extra_secrets()],
    max_containers=MAX_CONTAINERS,
)  # type: ignore[misc]
def convert_episode_remote(item: WorkItem, overwrite: bool, crf: int) -> None:
    """
    Download one episode from HuggingFace, convert it, and upload the RRD to the bucket.

    Returns without converting when the RRD is already in the bucket, unless `overwrite`.
    """
    from huggingface_hub import hf_hub_download

    from abc_130k.convert import convert_episode, episode_from_mcap
    from abc_130k.video_transcode import VideoSettings

    print("creating storage client…")
    s3 = worker_client()
    dest = f"{RRD_PREFIX}{item.recording_id}.rrd"
    if not overwrite and s3_exists(s3, dest):
        print(f"skip (exists): {dest}")
        return

    print(f"downloading {item.episode_dir}…")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        filenames = [f"{item.episode_dir}/episode.mcap"]
        if item.has_annotation:
            filenames.append(f"{item.episode_dir}/annotation.mcap")
        for filename in filenames:
            hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename, local_dir=str(root / "hf"))
        print("download done.")

        video_setting = VideoSettings(crf=crf)
        print(
            f"converting to rrd… (gop={video_setting.gop_size}, crf={video_setting.crf}, "
            f"max_width={video_setting.max_width})"
        )
        mcap = root / "hf" / item.episode_dir / "episode.mcap"
        rrd = convert_episode(
            episode_from_mcap(mcap),
            root / "out",
            video_setting,
            verbose=True,  # log source codec + re-encode target/downscale
        )
        print("conversion done.")

        print("uploading…")
        upload_file(s3, str(rrd), dest)
    print(f"uploaded: {dest}")


def _drop_converted(items: list[WorkItem]) -> list[WorkItem] | None:
    """Drop episodes whose RRD is already in the bucket, so no worker is spawned for them."""
    from botocore.exceptions import BotoCoreError, ClientError

    print(f"Listing {RRD_PREFIX} to skip already-converted episodes…", flush=True)
    try:
        done = s3_existing_keys(launcher_client(), RRD_PREFIX)
    except (BotoCoreError, ClientError) as exc:
        print(f"Could not list {RRD_PREFIX} locally ({type(exc).__name__}) — workers will skip as they go.")
        return None

    _, key_prefix = s3_parts(RRD_PREFIX)
    return [it for it in items if f"{key_prefix}{it.recording_id}.rrd" not in done]


@app.local_entrypoint()  # type: ignore[misc]
def main(*arglist: str) -> None:
    """Discover episodes on HuggingFace and fan out conversion across Modal workers (`spawn_map`)."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert ABC-130k episodes to RRDs on Modal: discover episodes on HuggingFace, fan the "
            "conversion out across Modal workers, and upload each .rrd to the bucket. Runs detached "
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
        "--overwrite", action="store_true", help="Reconvert even if the RRD already exists in the bucket."
    )
    parser.add_argument("--dry-run", action="store_true", help="List the episodes that would be converted, then exit.")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help=f"libx264 CRF (default {DEFAULT_CRF}).")
    args = parser.parse_args(args=arglist)

    # Check the bucket first — a bad target should not cost the full HF tree listing.
    check_bucket()
    matched = discover_episodes(HF_REPO_ID, args.task_filter)
    if not matched:
        print(f"No episodes matched (task_filter={args.task_filter!r}).")
        return

    # Drop what is already converted before applying --limit, so --limit counts real work.
    todo = None if args.overwrite else _drop_converted(matched)
    items = matched if todo is None else todo
    if args.limit > 0:
        items = items[: args.limit]

    already = "" if todo is None else f"{len(matched) - len(todo)} already converted, "
    print(f"{len(matched)} episode(s) matched, {already}{len(items)} to convert.")
    if not items:
        print(f"Nothing to do -> {RRD_PREFIX}")
        return

    print(f"Modal worker request: cpu={CPU_REQUEST}, memory={MEMORY_REQUEST} MiB")

    if args.dry_run:
        print(f"Dry run: {len(items)} episode(s) would be converted -> {RRD_PREFIX}")
        for it in items:
            print(f"  {it.episode_dir} -> {it.recording_id}.rrd")
        return

    count = len(items)
    note = f"gop {DEFAULT_GOP}, crf {args.crf}, max_width {DEFAULT_MAX_WIDTH}"
    print(f"Spawning {count} worker(s) -> {RRD_PREFIX} (overwrite={args.overwrite}, {note})")
    convert_episode_remote.spawn_map(items, [args.overwrite] * count, [args.crf] * count)
    print(f"Spawned {count} worker(s). Watch progress in the Modal dashboard.")
