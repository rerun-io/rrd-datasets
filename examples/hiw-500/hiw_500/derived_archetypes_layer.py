"""
Build the *derived archetypes* layer: what the viewer needs typed and the raw messages do not give it.

The base layer keeps `/wbc_lerobot` as the JSON text it was recorded as. This layer parses that
text once into a struct per message for the blueprint to plot, and adds the four end-effector
position markers as `Transform3D`. Written as a separate `.rrd` per episode sharing the base
`recording_id`; episodes without the topic skip it.

Run:  pixi run -e hiw convert-derived-archetypes            # all episodes under data/HIW-500/
      pixi run -e hiw convert-derived-archetypes <ep.mcap>  # a single episode
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import rerun as rr
from rerun.experimental import Chunk, DeriveLens, LazyChunkStream, McapReader, OptimizationProfile, Selector

from hiw_500.base_layer import (
    APPLICATION_ID,
    DATASET_ROOT,
    RRD_ROOT,
    TEXT,
    VEC3,
    Episode,
    discover_episodes,
    episode_from_mcap,
)
from rrd_datasets_common.paths import layer_relpath

WBC_TOPIC = "/wbc_lerobot"
# The JSON parsed into a struct, named the way the reader names decoded messages.
MSG_WBC = "wbc_lerobot:message"
MARKER_ROOT = "/lerobot"

# Series order of the width-12 `ee_state` / `ee_action` arrays: the left arm then the right, six
# pose fields each.
EE_NAMES = [f"{arm}/{pose}" for arm in ("left", "right") for pose in ("px", "py", "pz", "rx", "ry", "rz")]


def json_struct(textcol: pa.Array) -> pa.Array:
    """Each JSON message -> one struct row; pyarrow infers the struct type from the parsed dicts."""
    return pa.array([json.loads(text) for text in textcol.to_pylist()])


def json_pos(key: str, lo: int) -> Callable[[pa.Array], pa.Array]:
    """A 3-slice of a JSON array field -> Transform3D translation (EE position)."""
    return lambda textcol: pa.array([json.loads(text)[key][lo : lo + 3] for text in textcol.to_pylist()], type=VEC3)


def wbc_lenses() -> list[DeriveLens]:
    """
    The `/wbc_lerobot` JSON as one struct per message, plus the four end-effector position markers.

    The struct keeps every key (`pivot`, `ee_state`, `ee_action`, `gripper_controls`) for the
    blueprint to plot. The markers are the part the 3D view needs typed: a `Transform3D`
    translation per arm, for measured and commanded pose alike.
    """
    lenses = [DeriveLens(TEXT, output_entity=WBC_TOPIC).to_component(MSG_WBC, Selector(".").pipe(json_struct))]
    for kind in ("ee_state", "ee_action"):
        for arm, lo in (("left", 0), ("right", 6)):
            lenses.append(
                DeriveLens(TEXT, output_entity=f"{MARKER_ROOT}/{kind}/{arm}").to_component(
                    rr.Transform3D.descriptor_translation(), Selector(".").pipe(json_pos(kind, lo))
                )
            )
    return lenses


def ee_names_chunk() -> Chunk:
    """Series labels for the `ee_state` / `ee_action` arrays, static beside the struct."""
    return Chunk.from_columns(WBC_TOPIC, indexes=[], columns=rr.AnyValues.columns(ee_names=[EE_NAMES]))


def derived_stream(path: Path) -> LazyChunkStream:
    """The struct, the markers and the labels; the text and its bookkeeping stay in the base layer."""
    stream = McapReader(str(path), include_topic_regex=[f"^{WBC_TOPIC}$"]).stream()
    stream = stream.lenses(wbc_lenses(), content=WBC_TOPIC, output_mode="drop_unmatched")
    derived = stream.filter(content=[WBC_TOPIC, f"{MARKER_ROOT}/**"])
    return LazyChunkStream.merge(derived, LazyChunkStream.from_iter([ee_names_chunk()]))


def convert_episode(ep: Episode, rrd_root: Path) -> Path | None:
    """Write the episode's derived archetypes layer, or return `None` when it records no `/wbc_lerobot`."""
    if all(channel.topic != WBC_TOPIC for channel in McapReader(str(ep.mcap)).info().channels):
        return None
    out_path = rrd_root / layer_relpath("derived_archetypes", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    derived_stream(ep.mcap).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main(argv: list[str]) -> None:
    episodes = [episode_from_mcap(Path(argv[1]))] if len(argv) > 1 else discover_episodes(DATASET_ROOT)
    print(f"Building derived archetypes layer for {len(episodes)} episode(s) -> {RRD_ROOT / 'derived_archetypes'}/")
    for ep in episodes:
        out = convert_episode(ep, RRD_ROOT)
        if out is None:
            print(f"  {ep.recording_id}: skipped — no {WBC_TOPIC}")
        else:
            print(f"  {ep.recording_id}: {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv)
