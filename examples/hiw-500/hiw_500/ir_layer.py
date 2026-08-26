"""
Build an IR layer with the wrist infrared streams of episodes that record them.

Some sessions carry four extra 30 Hz JPEG streams — an ir1/ir2 stereo pair per wrist camera
(`/camera/{left,right}_wrist/ir{1,2}/compressed`). The base layer excludes them so its size does
not double for imagery most workflows never look at; this layer passes them through unchanged, so
IR can be added to (or rebuilt for) already-converted episodes on its own. Episodes without IR
streams skip this layer.

Like the wrist color cameras, the IR streams stay 2D image views: the wrist cameras have no
mounting extrinsic, and the per-serial wrist calibrations carry only
intra-camera IR<->color extrinsics.

Run:  pixi run -e hiw convert-ir            # all episodes under data/HIW-500/
      pixi run -e hiw convert-ir <ep.mcap>  # a single episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.experimental import LazyChunkStream, McapReader, OptimizationProfile

from hiw_500.base_layer import (
    APPLICATION_ID,
    DATASET_ROOT,
    IR_TOPICS,
    RRD_ROOT,
    Episode,
    camera_fields_stream,
    discover_episodes,
    episode_from_mcap,
    media_type_lens,
)
from rrd_datasets_common.paths import layer_relpath

# Reader bookkeeping entities, emitted for every mcap even when no topic matches.
MCAP_BOOKKEEPING = ["/__mcap_metadata", "/__mcap_properties"]


def ir_stream(path: Path) -> LazyChunkStream:
    """The wrist IR streams as EncodedImage entities at their topic paths, tagged as JPEG, with every message field kept."""
    stream = McapReader(str(path), include_topic_regex=IR_TOPICS).stream()
    # forward_all so the blob (the lens's input, hence "consumed") survives alongside the new tag.
    stream = stream.lenses(media_type_lens(), content="/camera/**", output_mode="forward_all")
    return LazyChunkStream.merge(stream.drop(content=MCAP_BOOKKEEPING), camera_fields_stream(path, IR_TOPICS))


def convert_episode(ep: Episode, rrd_root: Path) -> Path | None:
    """Write the episode's ir layer, or return `None` when it records no IR streams."""
    chunks = ir_stream(ep.mcap).to_chunks()
    if not chunks:
        return None
    out_path = rrd_root / layer_relpath("ir", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LazyChunkStream.from_iter(chunks).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main() -> None:
    """Convert a single episode mcap (positional) or every episode under `DATASET_ROOT`."""
    parser = argparse.ArgumentParser(description="Build the HIW-500 wrist IR layer (episodes without IR skip).")
    parser.add_argument(
        "mcap", nargs="?", type=Path, help="A single episode mcap (default: every episode under data/HIW-500/)."
    )
    args = parser.parse_args()

    episodes = [episode_from_mcap(args.mcap)] if args.mcap is not None else discover_episodes(DATASET_ROOT)
    if not episodes:
        print(f"No episodes found under {DATASET_ROOT}")
        print("-> download some first: 'pixi run -e hiw download' (see README).")
        return
    print(f"Building ir layer for {len(episodes)} episode(s) -> {RRD_ROOT / 'ir'}/")
    for ep in episodes:
        out = convert_episode(ep, RRD_ROOT)
        if out is None:
            print(f"  {ep.recording_id}: skipped — no IR streams")
        else:
            print(f"  {ep.recording_id}: {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
