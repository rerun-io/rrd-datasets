"""
Convert ABC-130k (two-arm YAM bimanual) MCAP episodes into per-episode Rerun RRDs.

This writes the *base* layer: the raw ABC-130k `episode.mcap` (plus the optional sibling
`annotation.mcap`) as Rerun entities, one optimized RRD per episode. Each `recording_id` is a stable
`<task>__<uuid>`, so every episode is its own catalog segment. The camera video is re-encoded to
H.264 with a fixed GOP, and 1920x1200 cameras are downscaled with their `Pinhole` rescaled to match.

Run:  pixi run abc-convert                 # every episode found under data/ABC-130k/data/
      pixi run abc-convert <episode.mcap>  # a single episode mcap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import rerun as rr
from rerun.experimental import (
    Chunk,
    DeriveLens,
    LazyChunkStream,
    McapReader,
    OptimizationProfile,
    Selector,
)

from abc_130k.video_transcode import (
    DEFAULT_CRF,
    VideoSettings,
    regop_camera_streams,
)
from rrd_datasets_common.paths import dataset_data_dir, dataset_rrd_dir, resolve_input_path

# Shared workspace layout (rrd_datasets_common.paths): the raw dataset under the
# data/ root, base-layer recordings under rrds/abc-130k/base/.
DATA_ROOT = dataset_data_dir("ABC-130k") / "data"  # <split>/<task>/episode_<uuid>/{episode,annotation}.mcap
OUT_DIR = dataset_rrd_dir("abc-130k") / "base"
APPLICATION_ID = "abc_130k"

# Component identifiers McapReader produces for the custom protobuf messages.
MSG_ROBOT = "RobotState:message"  # arm state/action: position/velocity/torque, 6-DoF
MSG_GRIPPER = "GripperState:message"  # gripper state/action: position/velocity/torque, 1-DoF
MSG_INSTR = "Instructions:message"  # /instruction: whole-episode task string
MSG_ANNOT = "Annotation:message"  # /subtask-annotation (annotation.mcap): timestamped subtask labels

# YAM arm joints names. The index increases as moving away from the robot base toward the wrist.
# Used only for blueprint legend labels.
ARM_JOINT_NAMES = [f"joint_{i}" for i in range(6)]

# A 7-wide arm array carries the gripper as its last joint; 6 means the gripper has its own topic.
GRIPPER_IN_ARM_WIDTH = 7


class Side(StrEnum):
    """Which arm of the bimanual setup."""

    LEFT = "left"
    RIGHT = "right"


class Kind(StrEnum):
    """A signal stream: measured `state` (q/dq/tau) or commanded `action` (q only)."""

    STATE = "state"
    ACTION = "action"


class GripperSource(StrEnum):
    """Which topic carries a gripper's velocity/torque. Varies per episode, so probe each one."""

    ARM = "arm"  # the arm array is width-7 and holds the gripper at index 6
    GRIPPER_TOPIC = "gripper_topic"


# --------------------------------------------------------------------------------------
# lens builders
# --------------------------------------------------------------------------------------


def _scalar_lens(msg: str, entity: str, selector: Selector | str) -> DeriveLens:
    """One numeric field of a `:message` struct -> a Scalars entity."""
    sel = selector if isinstance(selector, Selector) else Selector(selector)
    return DeriveLens(msg, output_entity=entity).to_component(rr.Scalars.descriptor_scalars(), sel)


def _first6(field: str) -> Selector:
    """
    Selector: a message list field truncated to its first 6 elements, as an array `Scalars`.

    The selector can't slice (`[0:6]` doesn't parse), so `list_slice` truncates the list, then `[]`
    iterates it into scalars. Keeps the arm at its 6 joints on both the width-6 and width-7 episodes.
    """
    # TODO(michael): looks like we should add slice-support in Selectors?
    return Selector(f".{field}").pipe(lambda a: pc.list_slice(a, 0, 6)).pipe(Selector("[]"))  # type: ignore[attr-defined]


def arm_lenses(side: Side, kind: Kind, gripper_from_arm: bool) -> list[DeriveLens]:
    """
    Array `Scalars` for the 6 arm joints per signal, plus the gripper's dq/tau when they ride the arm.

    Every signal is truncated to the first 6 (`_first6`).
    Note that some episodes carry the gripper's dq/tau at index 6. This will be split out onto `/{side}/gripper/...` by the converter,
    so the arm is always 6-wide.
    """
    if kind == Kind.ACTION:
        # action carries only the commanded position (no velocity or torque)
        signals = [("q", "position")]
    else:  # kind == Kind.STATE
        signals = [("q", "position"), ("dq", "velocity"), ("tau", "torque")]

    lenses = [_scalar_lens(MSG_ROBOT, f"/{side}/arm/{kind}/{short}", _first6(field)) for short, field in signals]
    # if the gripper's dq/tau states are carried by the arm, add them here
    if kind == Kind.STATE and gripper_from_arm:
        lenses += [
            _scalar_lens(MSG_ROBOT, f"/{side}/gripper/state/{short}", f".{field}[6]")
            for short, field in (("dq", "velocity"), ("tau", "torque"))
        ]
    return lenses


def gripper_lenses(side: Side, kind: Kind, gripper_from_topic: bool) -> list[DeriveLens]:
    """
    Gripper position scalar for a gripper topic (`/{side}-ee-{kind}`), plus dq/tau when they live here.

    Gripper position is always in this topic for state and command.
    Some episodes carry the gripper's velocity/torque state as part of gripper topic. In this case, they are read from the topic.
    """
    signals = [("q", "position")]  # if kind == Kind.ACTION, position is the only signal.
    if kind == Kind.STATE and gripper_from_topic:
        signals += [("dq", "velocity"), ("tau", "torque")]
    return [_scalar_lens(MSG_GRIPPER, f"/{side}/gripper/{kind}/{short}", f".{field}[0]") for short, field in signals]


def instruction_lens() -> DeriveLens:
    """The whole-episode task string (`/instruction`) -> a TextDocument."""
    return DeriveLens(MSG_INSTR, output_entity="/instruction").to_component(
        rr.TextDocument.descriptor_text(), Selector(".data")
    )


def subtask_lens() -> DeriveLens:
    """Timestamped subtask labels (`/subtask-annotation`, from annotation.mcap) -> a StateChange timeline."""
    return DeriveLens(MSG_ANNOT, output_entity="/task/subtask").to_component(
        rr.StateChange.descriptor_state(), Selector(".data")
    )


# --------------------------------------------------------------------------------------
# stream assembly
# --------------------------------------------------------------------------------------

# Topics kept by the reader: arms, grippers, instruction, and every camera (video + calibration).
KEEP_TOPICS = [r"^/instruction$", r"^/(left|right)-(arm|ee)-(state|action)$", r"-camera(-info)?$"]

# The custom-message topics are consumed by the lenses, so drop those whole entities plus the two
# synthetic reader-bookkeeping entities. `/instruction` and the cameras are kept, so their leftover
# `Mcap*` metadata columns are removed at the component level instead (drop needs exact ids — a
# `McapChannel:*` wildcard silently matches nothing).
ARM_GRIPPER_TOPICS = [f"/{side}-{group}-{kind}" for side in Side for group in ("arm", "ee") for kind in Kind]
DROP_ENTITIES = [*ARM_GRIPPER_TOPICS, "/__mcap_metadata", "/__mcap_properties"]
DROP_COMPONENTS = [f"McapChannel:{c}" for c in ("id", "message_encoding", "metadata", "topic")] + [
    f"McapSchema:{c}" for c in ("data", "encoding", "id", "name")
]


def _unwrap(value: Any) -> Any:
    """The reader wraps each row's message struct in a single-element list; peel it off."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


@dataclass
class EpisodeMeta:
    """The `/__mcap_metadata` fields the conversion uses. Keys the episode omits stay empty/zero."""

    task: str = ""
    session_id: str = ""
    duration_sec: float = 0.0
    operator_id: str = ""
    top_camera_type: str = ""

    @classmethod
    def from_kv(cls, kv: dict[str, str]) -> EpisodeMeta:
        """Pick the used fields out of the raw `episode-metadata` key/values."""
        duration = kv.get("duration", "")
        return cls(
            task=kv.get("task_name", ""),
            session_id=kv.get("session_id", ""),
            duration_sec=float(duration) if duration else 0.0,
            operator_id=kv.get("operator_id", ""),
            top_camera_type=kv.get("top_camera_type", ""),
        )

    @property
    def station(self) -> str:
        """Station type (`RealSense`/`ZED-X`) from `top_camera_type` (raw if unknown)."""
        cam = self.top_camera_type.lower()
        return "ZED-X" if "zed" in cam else "RealSense" if "realsense" in cam else cam


def _metadata_from_chunk(ch: Chunk) -> dict[str, str]:
    """
    Flatten a `/__mcap_metadata` chunk's `episode-metadata` key/value pairs into a plain dict.

    The component is a list of `{first, second}` structs inside the per-row list wrapper; unwrap to
    the inner list, then map `first -> second`.
    """
    rb = ch.to_record_batch()
    if "episode-metadata" not in rb.schema.names:
        return {}
    kv = rb.column("episode-metadata").to_pylist()[0]
    while isinstance(kv, list) and len(kv) == 1 and isinstance(kv[0], list):
        kv = kv[0]
    return {p["first"]: p["second"] for p in kv}


def probe_episode(mcap_path: Path) -> tuple[EpisodeMeta, dict[Side, GripperSource]]:
    """
    One read of the small arm-state topics for the two things the conversion needs up front.

    Returns the episode's `EpisodeMeta` and each side's `GripperSource` — `GRIPPER_TOPIC` unless that
    side's arm array is width-7.
    """
    # allow materialization of the small selected topic for quick probing (specifically to query the gripper source)
    store = McapReader(str(mcap_path), include_topic_regex=[r"^/(left|right)-arm-state$"]).stream().collect()

    # TODO(RR-5280): McapReader should have a property access API
    kv = next(
        (_metadata_from_chunk(ch) for ch in store.stream().to_chunks() if ch.entity_path == "/__mcap_metadata"),
        {},
    )
    meta = EpisodeMeta.from_kv(kv)

    # Gripper source: query the arm state message columns via the store's DataFrame reader.
    # Read one row per state message; we only need the first of each.
    table = store.reader("message_log_time", contents=[f"/{side}-arm-state" for side in Side]).to_arrow_table()

    def first_state_message(entity: str) -> dict[str, Any]:
        """First decoded message struct logged on `entity` ({} if none)."""
        col = next((n for n in table.schema.names if n.startswith(f"{entity}:") and n.endswith(":message")), None)
        if col is None:
            return {}
        return next((m for m in map(_unwrap, table.column(col).to_pylist()) if isinstance(m, dict)), {})

    gripper_source: dict[Side, GripperSource] = {}
    for side in Side:
        arm = first_state_message(f"/{side}-arm-state")
        width_7 = len(arm.get("velocity") or []) == GRIPPER_IN_ARM_WIDTH
        gripper_source[side] = GripperSource.ARM if width_7 else GripperSource.GRIPPER_TOPIC

    return meta, gripper_source


def base_stream(
    mcap_path: Path,
    gripper_source: dict[Side, GripperSource],
    video_setting: VideoSettings | None = None,
    verbose: bool = True,
) -> LazyChunkStream:
    """
    Convert one episode.mcap into the base entity stream.

    Custom `:message` topics become scalars/text via content-scoped lenses (state carries q/dq/tau,
    action only q); cameras and calibration are never matched and pass through as VideoStream/Pinhole.
    The arm is normalized to its 6 joints and the gripper's dq/tau are routed onto `/{side}/gripper/...`
    from whichever source holds them (`gripper_source`, from `probe_episode`). Consumed raw topics and reader
    bookkeeping are dropped at the end.
    """
    stream = McapReader(str(mcap_path), include_topic_regex=KEEP_TOPICS).stream()
    for side in Side:
        for kind in Kind:
            arm = arm_lenses(side, kind, gripper_from_arm=gripper_source[side] == GripperSource.ARM)
            stream = stream.lenses(arm, content=f"/{side}-arm-{kind}", output_mode="forward_unmatched")
            gripper = gripper_lenses(side, kind, gripper_from_topic=gripper_source[side] == GripperSource.GRIPPER_TOPIC)
            stream = stream.lenses(gripper, content=f"/{side}-ee-{kind}", output_mode="forward_unmatched")
    stream = stream.lenses([instruction_lens()], content="/instruction", output_mode="forward_unmatched")
    stream = stream.drop(content=DROP_ENTITIES).drop(components=DROP_COMPONENTS)
    if video_setting is not None:
        stream = regop_camera_streams(mcap_path, stream, video_setting, verbose=verbose)
    return stream


def sidecar_stream(meta: EpisodeMeta) -> LazyChunkStream:
    """Episode-level metadata (task / session id / duration / operator) as a static `/episode` chunk."""
    chunk = Chunk.from_columns(
        "/episode",
        indexes=[],  # no timeline -> static
        columns=rr.AnyValues.columns(
            task=[meta.task],
            session_id=[meta.session_id],
            duration_sec=[meta.duration_sec],
            operator_id=[meta.operator_id],
        ),
    )
    return LazyChunkStream.from_iter([chunk])
    # TODO(michael): could be useful to have a stream method to inject a single chunk.
    # Constructing a stream for a single chunk looks always like overkill,
    # sth like `stream().add_chunk(Chunk.from_columns(…))` might be nicer for these simple ones ...?


SPLITS = ("train", "val", "test")


def _split(mcap_path: Path) -> str:
    """Dataset split (`train`/`val`/`test`) from the `.../data/<split>/<task>/...` path (empty if absent)."""
    parts = mcap_path.parts
    return next((cur for prev, cur in zip(parts, parts[1:]) if prev == "data" and cur in SPLITS), "")


def properties_stream(props: dict[str, str]) -> LazyChunkStream:
    """
    Static `/__properties/<name>` recording properties (one entity each) for catalog filter/search.

    One canonical entity per property, matching `rr.send_property` (e.g. `split`, `station`,
    `instruction`); the mapping keys become the property names.
    """
    return LazyChunkStream.from_iter(
        # TODO(RR-5243): use `Chunk.from_property()`
        Chunk.from_columns(
            f"/__properties/{name}",
            indexes=[],
            columns=rr.AnyValues.columns(**{name: [value]}),  # type: ignore[arg-type]  # dynamic kwarg name
        )
        for name, value in props.items()
    )


def annotation_stream(ann_path: Path) -> LazyChunkStream:
    """
    Convert annotation.mcap into a /task/subtask StateChange stream (timestamped subtask labels).

    Its one topic (`/subtask-annotation`) is consumed by `subtask_lens`; the raw topic and reader
    bookkeeping are removed with the same two-level cleanup as `base_stream`.
    """
    stream = McapReader(str(ann_path)).stream()
    stream = stream.lenses([subtask_lens()], content="/subtask-annotation", output_mode="forward_unmatched")
    return stream.drop(content=["/subtask-annotation", "/__mcap_metadata", "/__mcap_properties"]).drop(
        components=DROP_COMPONENTS
    )


# --------------------------------------------------------------------------------------
# episode driver
# --------------------------------------------------------------------------------------


@dataclass
class Episode:
    """One episode: its mcap, the optional sibling annotation, and a stable recording id."""

    mcap: Path
    annotation: Path | None
    recording_id: str


def episode_from_mcap(mcap: Path) -> Episode:
    """
    Build an `Episode` from one episode.mcap (or its `episode_<uuid>/` directory).

    Attaches the sibling `annotation.mcap` when it exists and derives `recording_id = "<task>__<uuid>"`
    from the `<task>/episode_<uuid>/` path (readable, stable; the uuid also equals `session_id`).
    The path may be absolute, or relative to either the working directory or the workspace root.
    """
    mcap = resolve_input_path(mcap)
    if mcap.is_dir():  # accept the episode dir too; resolve the mcap inside it.
        mcap = mcap / "episode.mcap"
    if not mcap.is_file():
        raise FileNotFoundError(f"no episode.mcap at '{mcap}'")
    annotation = mcap.with_name("annotation.mcap")
    task = mcap.parent.parent.name
    uuid = mcap.parent.name.removeprefix("episode_")
    return Episode(mcap=mcap, annotation=annotation if annotation.exists() else None, recording_id=f"{task}__{uuid}")


def discover_episodes(root: Path) -> list[Episode]:
    """Every `episode.mcap` under `root`, sorted (recursively across splits/tasks)."""
    return [episode_from_mcap(mcap) for mcap in sorted(root.rglob("episode.mcap"))]


def convert_episode(
    ep: Episode,
    out_dir: Path,
    video_setting: VideoSettings | None = None,
    verbose: bool = True,
) -> Path:
    """
    Convert one episode into an optimized `.rrd` and return its path.

    Merges the base entity stream, the `/episode` metadata sidecar, the `split` / `station` /
    `instruction` recording properties, and -- when present -- the subtask StateChange stream, then
    writes a single object-store-optimized recording stamped with `application_id` / `recording_id`.

    With `video_setting` set, the camera `VideoStream`s are re-encoded to its fixed GOP (`video_transcode`);
    without it they are copied through untouched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ep.recording_id}.rrd"
    meta, gripper_source = probe_episode(ep.mcap)
    base = base_stream(ep.mcap, gripper_source, video_setting, verbose=verbose)
    streams = [
        base,
        sidecar_stream(meta),
        properties_stream({
            "split": _split(ep.mcap),
            "station": meta.station,
            "instruction": meta.task,
        }),
    ]
    if ep.annotation is not None:
        streams.append(annotation_stream(ep.annotation))
    merged = LazyChunkStream.merge(*streams)
    merged.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main() -> None:
    """Convert a single episode mcap (positional) or every episode under `DATA_ROOT`."""
    parser = argparse.ArgumentParser(description="Convert ABC-130k episode.mcap files into per-episode Rerun RRDs.")
    parser.add_argument(
        "mcap", nargs="?", type=Path, help="A single episode.mcap (default: every episode under data/)."
    )
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help=f"libx264 CRF (default {DEFAULT_CRF}).")
    args = parser.parse_args()

    episodes = [episode_from_mcap(args.mcap)] if args.mcap is not None else discover_episodes(DATA_ROOT)
    if not episodes:
        print(f"No episodes found under {DATA_ROOT}")
        print("-> download some first: 'pixi run abc-download' (see README).")
        return
    video_setting = VideoSettings(crf=args.crf)
    print(
        f"Converting {len(episodes)} episode(s) -> {OUT_DIR} "
        f"(gop {video_setting.gop_size}, crf {video_setting.crf}, max_width {video_setting.max_width})"
    )
    for ep in episodes:
        out = convert_episode(ep, OUT_DIR, video_setting)
        tag = "annotated" if ep.annotation is not None else "no annotation"
        print(f"  {ep.recording_id}: {out.name} ({out.stat().st_size / 1e6:.1f} MB, {tag})")


if __name__ == "__main__":
    main()
