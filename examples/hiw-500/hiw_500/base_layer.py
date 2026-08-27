"""
Convert HIW-500 (Unitree G1 bimanual) MCAP episodes into per-episode Rerun RRDs.

The *base* layer: the whole MCAP as Rerun entities, one optimized RRD per episode under a stable
`recording_id`. Decoded messages stay whole, the file's own records come along, and only the wrist
IR streams are left to their own layer. The stereo head image is split into its two eyes.

Sidecars logged beside it: the subtask boundaries from `info.json`, the calibration files and the
series labels; the other `info.json` fields are the properties layer's. A channel census flags
episodes with undecodable messages.

Run:  pixi run -e hiw convert-base            # all episodes under data/HIW-500/
      pixi run -e hiw convert-base <ep.mcap>  # a single episode mcap
"""

from __future__ import annotations

import argparse
import io
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun as rr
import yaml
from PIL import Image
from rerun.experimental import Chunk, DeriveLens, LazyChunkStream, McapInfo, McapReader, OptimizationProfile, Selector
from turbojpeg import TurboJPEG

from rrd_datasets_common.paths import dataset_data_dir, dataset_rrd_dir, layer_relpath, resolve_input_path

# Shared workspace layout (rrd_datasets_common.paths): the raw dataset under the
# data/ root; each layer in its own directory, rrds/hiw-500/<layer>/.
DATASET_ROOT = dataset_data_dir("HIW-500")
RRD_ROOT = dataset_rrd_dir("hiw-500")
APPLICATION_ID = "hiw_500"

# The recording property every per-episode field hangs off, so the catalog columns read
# `property:episode:<name>`. `rr.RecordingStream.send_property(PROPERTY, …)` writes this path.
PROPERTY = "episode"
PROPERTY_PATH = f"/__properties/{PROPERTY}"
# The MCAP header as a recording property: `property:mcap:<name>` columns in the catalog.
MCAP_PROPERTY = "mcap"

# Component identifiers of the decoded messages, one struct column per topic. `homies/*` wraps a
# std_msgs Header around a unitree_hg payload; the definitions are at
# https://github.com/unitreerobotics/unitree_ros2/blob/master/cyclonedds_ws/src/unitree/unitree_hg/msg.
MSG_LOWSTATE = "homies.msg.LowStateStamped:message"
MSG_LOWCMD = "homies.msg.LowCmdStamped:message"
MSG_MOTOR_STATE = "homies.msg.MotorStateStamped:message"
MSG_MOTOR_CMD = "homies.msg.MotorCmdStamped:message"
MSG_IMU = "homies.msg.IMUStateStamped:message"
MSG_ODOM = "unitree_go.msg.SportModeState:message"
TEXT = "TextDocument:text"
BLOB = "EncodedImage:blob"
# The camera message as `ros2_reflection` decodes it, and the two fields kept from it per row.
CAMERA_MSG = "sensor_msgs.msg.CompressedImage:message"
CAMERA_FORMAT = "sensor_msgs.msg.CompressedImage:format"
CAMERA_HEADER = "sensor_msgs.msg.CompressedImage:header"

# The wrist IR streams have their own layer (`ir_layer`); every other topic comes through.
IR_TOPICS = ["^/camera/(left|right)_wrist/ir[12]/compressed$"]
RGB_CAMERA_TOPICS = ["^/camera/(head|left_wrist|right_wrist)/image/compressed$"]
HEAD_TOPIC = "/camera/head/image/compressed"

# Unitree G1 29-DoF motor order (G1JointIndex).
# LowState.motor_state and LowCmd.motor_cmd are fixed 35-element arrays in Unitree SDK2.
# For the 29-DoF G1, indices 0-28 correspond to the 29 body joints below; indices 29-34 are unused.
# Ordering follows Unitree's official G1 DDS joint-index table and SDK2
# JointIndex definition, and is consistent with the official G1 29-DoF URDF.
G1_JOINT_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]
N_JOINTS = len(G1_JOINT_NAMES)

# Arrow type of a Transform3D translation.
VEC3 = pa.list_(pa.float32(), 3)

_TJ = TurboJPEG()  # libjpeg-turbo handle for lossless compressed-domain cropping


# --------------------------------------------------------------------------------------
# lens pipe callbacks (PyArrow transforms run inside Selector.pipe — the idiomatic place)
# --------------------------------------------------------------------------------------


def const_str(value: str) -> Callable[[pa.Array], pa.Array]:
    """A pipe that emits a constant string per row (for constant parent/child frame names)."""
    return lambda arr: pa.array([value] * len(arr), type=pa.utf8())


def crop_jpegs(left: bool) -> Callable[[pa.Array], pa.Array]:
    """
    Split each side-by-side stereo JPEG into its left/right half in the compressed domain.

    `_TJ.crop` keeps the DCT coefficients (no pixel decode/encode), so every block decodes as before
    except the two columns at the cut, where 4:2:0 chroma is interpolated against a new image edge.
    Bytes move through numpy buffers rather than Python int lists, which keeps the Arrow
    `list<uint8>` round-trip cheap.
    """

    def run(blob: pa.Array) -> pa.Array:
        values: np.ndarray = blob.values.to_numpy(zero_copy_only=False)  # all jpegs, concatenated
        offsets: np.ndarray = blob.offsets.to_numpy()

        def frame(i: int) -> bytes:
            return values[offsets[i] : offsets[i + 1]].tobytes()

        w, h = Image.open(io.BytesIO(frame(0))).size  # lazy header read; resolution is constant
        x = 0 if left else w // 2
        out = [_TJ.crop(frame(i), x, 0, w // 2, h) for i in range(len(blob))]
        flat = np.frombuffer(b"".join(out), dtype=np.uint8)
        new_offsets = np.zeros(len(out) + 1, dtype=np.int32)
        np.cumsum([len(b) for b in out], out=new_offsets[1:])
        return pa.ListArray.from_arrays(pa.array(new_offsets), pa.array(flat, type=pa.uint8())).cast(blob.type)

    return run


# --------------------------------------------------------------------------------------
# lens builders
# --------------------------------------------------------------------------------------


def head_split_lenses() -> list[DeriveLens]:
    """
    Split the side-by-side stereo head image into left/right halves (lossless).

    The output uses `descriptor_blob()` (not the bare `"EncodedImage:blob"` string) so it carries
    the EncodedImage archetype tag — otherwise the viewer sees a loose component ("Without
    archetype") and won't render it spatially.
    """
    return [
        DeriveLens(BLOB, output_entity="/camera/head/left").to_component(
            rr.EncodedImage.descriptor_blob(), Selector(".").pipe(crop_jpegs(left=True))
        ),
        DeriveLens(BLOB, output_entity="/camera/head/right").to_component(
            rr.EncodedImage.descriptor_blob(), Selector(".").pipe(crop_jpegs(left=False))
        ),
    ]


def media_type_lens() -> DeriveLens:
    """
    Tag every camera EncodedImage with its codec.

    McapReader leaves `media_type` unset and the viewer needs it ("No codec specified"); all
    camera streams here are JPEG. Emitted via `descriptor_media_type()` so it carries the
    EncodedImage archetype tag.
    """
    return DeriveLens(BLOB).to_component(
        rr.EncodedImage.descriptor_media_type(), Selector(".").pipe(const_str("image/jpeg"))
    )


# --------------------------------------------------------------------------------------
# stream assembly
# --------------------------------------------------------------------------------------


def camera_fields_stream(path: Path, topics: list[str]) -> LazyChunkStream:
    """
    The camera messages' `header` and `format`, one row per image, beside the decoded blob.

    `ros2msg` folds `header.stamp` into the `ros2_timestamp` timeline and discards `format`; a
    reflection-only read of the same topics keeps both as columns, so every message field is
    recoverable. Only the lens outputs leave here; the bookkeeping rows arrive with the main stream.
    """
    fields = (
        DeriveLens(CAMERA_MSG)
        .to_component(CAMERA_FORMAT, Selector(".format"))
        .to_component(CAMERA_HEADER, Selector(".header"))
    )
    stream = McapReader(str(path), decoders=["ros2_reflection"], include_topic_regex=topics).stream()
    return stream.lenses(fields, content="/camera/**", output_mode="drop_unmatched").filter(content="/camera/**")


def base_stream(path: Path) -> LazyChunkStream:
    """
    One McapReader stream over every topic but the wrist IR; the decoded topics pass through whole.

    No scalars are derived: the blueprint maps its series onto the struct fields. Both lenses forward
    their input: the side-by-side head blob stays beside the two eyes split from it, because the
    crop is not byte-exact at the seam, and every camera blob keeps its codec tag next to it.
    """
    stream = McapReader(str(path), exclude_topic_regex=IR_TOPICS).stream()
    stream = stream.lenses(head_split_lenses(), content=HEAD_TOPIC, output_mode="forward_all")
    stream = stream.lenses(media_type_lens(), content="/camera/**", output_mode="forward_all")
    return stream


# --------------------------------------------------------------------------------------
# sidecars
# --------------------------------------------------------------------------------------


@dataclass
class Subtask:
    """One subtask boundary: its label and the episode timestamp (ns) where it starts."""

    task: str
    timestamp_ns: int


@dataclass
class EpisodeInfo:
    """
    The `info.json` fields, less `duration_ns`, which `duration_sec` recovers exactly.

    Keys the episode omits stay empty/zero.
    """

    task: str = ""
    scene: int = -1
    duration_sec: float = 0.0
    episode_name: str = ""
    start_timestamp_ns: int = 0
    end_timestamp_ns: int = 0
    subtasks: list[Subtask] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> EpisodeInfo:
        """Parse an `info.json` sidecar; a missing file reads as all defaults."""
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(
            task=raw.get("task", ""),
            scene=raw.get("scene", -1),
            duration_sec=float(raw.get("duration_sec", 0.0)),
            episode_name=raw.get("episode_name", ""),
            start_timestamp_ns=int(raw.get("start_timestamp_ns", 0)),
            end_timestamp_ns=int(raw.get("end_timestamp_ns", 0)),
            subtasks=[Subtask(task=s["task"], timestamp_ns=s["timestamp_ns"]) for s in raw.get("subtasks", [])],
        )


def subtask_chunks(info: EpisodeInfo) -> list[Chunk]:
    """
    The subtask boundaries from `info.json` on `/task/subtask`; nothing for an episode without them.

    Each boundary is a discrete state that holds until the next, so a `StateChange` renders as colored
    lanes in a StateTimelineView (vs. append-only TextLog events). The other `info.json` fields are
    recording properties, written by the properties layer.
    """
    if not info.subtasks:
        return []
    ts = np.array([s.timestamp_ns for s in info.subtasks], dtype="datetime64[ns]")
    return [
        Chunk.from_columns(
            "/task/subtask",
            indexes=[rr.TimeColumn("message_publish_time", timestamp=ts)],
            columns=rr.StateChange.columns(state=[s.task for s in info.subtasks]),
        )
    ]


def names_chunks() -> list[Chunk]:
    """
    Joint labels for the motor arrays, static beside their structs.

    Element i of `motor_state` / `motor_cmd` is the joint `joint_names[i]`. The blueprint carries
    the display labels; this is the machine-readable mapping.
    """
    return [
        Chunk.from_columns(entity, indexes=[], columns=rr.AnyValues.columns(joint_names=[G1_JOINT_NAMES]))
        for entity in ("/stamped/lowstate", "/stamped/lowcmd")
    ]


CALIBRATION_ARCHETYPE = "CalibrationFile"

# Reserved for the sidecar's own filename, which no vendor field may take.
_CALIBRATION_PATH = "path"

_CALIBRATION_LOADERS: dict[str, Callable[[str], Any]] = {
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
    ".json": json.loads,
}
_CALIBRATION_ARROW_TYPES: dict[type, pa.DataType] = {
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
}


def _calibration_leaves(parsed: Any, prefix: str = "") -> dict[str, Any]:
    """Every leaf of a parsed sidecar, keyed by its dotted path; only dicts recurse, so a list keeps its shape."""
    leaves: dict[str, Any] = {}
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            leaves.update(_calibration_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        leaves[prefix] = parsed
    return leaves


def _calibration_value(value: Any) -> Any:
    """`value` with anything arrow has no type for replaced by its text, so no field is dropped."""
    if isinstance(value, list):
        return [_calibration_value(item) for item in value]
    if value is None or type(value) in _CALIBRATION_ARROW_TYPES:
        return value
    return str(value)


def _calibration_arrow_type(value: Any) -> pa.DataType:
    """The arrow type of one leaf; a nested list nests its element type, and `None` reads as text."""
    if isinstance(value, list):
        first = next((item for item in value if item is not None), None)
        return pa.list_(pa.float64() if first is None else _calibration_arrow_type(first))
    return _CALIBRATION_ARROW_TYPES.get(type(value), pa.string())


def _calibration_row(value: Any) -> pa.Array:
    """One row holding `value`, typed explicitly: an inferred empty list arrives as `list<null>`."""
    normalized = _calibration_value(value)
    return pa.array([normalized], type=_calibration_arrow_type(normalized))


def calibration_components(file: Path, rel: Path) -> dict[str, pa.Array]:
    """
    The sidecar's contents as `CalibrationFile` components, plus `path`; an unparsed suffix keeps its `text`.

    `path` is reserved for the source filename, so a sidecar carrying its own is a hard error rather
    than a silent overwrite.
    """
    loader = _CALIBRATION_LOADERS.get(file.suffix)
    leaves = _calibration_leaves(loader(file.read_text())) if loader else {"text": file.read_text()}
    if _CALIBRATION_PATH in leaves:
        raise ValueError(f"{rel} has its own `{_CALIBRATION_PATH}` field, which the source filename would overwrite")
    components = {name: _calibration_row(value) for name, value in leaves.items()}
    components[_CALIBRATION_PATH] = _calibration_row(rel.as_posix())
    return components


def calibration_chunks(ep: Episode) -> list[Chunk]:
    """
    The episode's calibration sidecars as `CalibrationFile` components, one static chunk per file.

    Wrist files become `wrist_camera<N>` in sorted filename order: a serial-named entity would move
    between episodes, and the serial-to-side mapping is only in the file content.
    """
    calib_dir = ep.mcap.parent / "calibration"
    if not calib_dir.is_dir():
        return []
    chunks: list[Chunk] = []
    wrist_count = 0
    for file in sorted(path for path in calib_dir.rglob("*") if path.is_file()):
        rel = file.relative_to(ep.mcap.parent)
        if file.name.startswith("camera_") and file.suffix == ".json":
            wrist_count += 1
            stem = f"wrist_camera{wrist_count}"
        else:
            stem = file.stem
        chunks.append(
            Chunk.from_columns(
                "/" + "/".join((*rel.parts[:-1], stem)),
                indexes=[],
                columns=rr.DynamicArchetype.columns(
                    archetype=CALIBRATION_ARCHETYPE,
                    components=calibration_components(file, rel),
                ),
            )
        )
    return chunks


def has_ir(ep: Episode) -> bool:
    """
    Whether the episode records the wrist IR streams.

    Inferred from the per-serial wrist calibrations beside the episode — the IR streams and those
    files arrived on the rig together — so metadata-only layers can answer it without downloading
    the mcap. `ir_layer` reads the streams themselves and decides for itself.
    """
    return any((ep.mcap.parent / "calibration" / "params").glob("camera_*.json"))


# --------------------------------------------------------------------------------------
# channel census
# --------------------------------------------------------------------------------------


def expected_message_counts(info: McapInfo, exclude: Iterable[str] = ()) -> dict[str, int]:
    """
    Per-topic message counts from the MCAP summary, without the topics matching an `exclude` pattern.

    A channel the summary gives no count for cannot be checked and is left out.
    """
    skipped = [re.compile(pattern) for pattern in exclude]
    return {
        channel.topic: channel.message_count
        for channel in info.channels
        if channel.message_count is not None and not any(pattern.match(channel.topic) for pattern in skipped)
    }


def undecodable_topics(expected: Mapping[str, int], chunks: Iterable[Chunk]) -> list[str]:
    """
    Topics whose decoded rows fall short of the MCAP's own message counts.

    A message that fails CDR decoding is skipped without a row (e.g. the Feb 2026 Clothes-Washing
    sessions declare `MotorStateStamped` on the left dex1 state channel but carry smaller `MotorCmd`
    payloads), so the recording silently under-reports the channel. Rows are counted per component:
    a topic entity carries several chunk families (the decoded message, a lens output, a second
    reader's columns), each with one row per decoded message, so summing chunks would count a
    message several times.
    """
    decoded: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        if chunk.is_static:
            continue
        per_component = decoded.setdefault(chunk.entity_path, {})
        for name in chunk.to_record_batch().schema.names:
            if name != "rerun.controls.RowId" and name not in chunk.timeline_names:
                per_component[name] = per_component.get(name, 0) + chunk.num_rows
    return sorted(topic for topic, count in expected.items() if max(decoded.get(topic, {}).values(), default=0) < count)


def raw_stream(path: Path, topics: list[str]) -> LazyChunkStream:
    """The messages of `topics` as their CDR bytes (`McapMessage:data`), for channels the decoders drop rows from."""
    patterns = [f"^{re.escape(topic)}$" for topic in topics]
    return McapReader(str(path), decoders=["raw"], include_topic_regex=patterns).stream()


def census_chunk(topics: list[str]) -> Chunk:
    """The census verdict as a recording property, so the catalog can filter on it."""
    return Chunk.from_property(
        PROPERTY,
        rr.AnyValues(
            has_undecodable=bool(topics),
            # Typed explicitly: an inferred empty list arrives as `list<null>`.
            undecodable_topics=pa.array(topics, type=pa.string()),
        ),
    )


def mcap_chunk(profile: str, library: str, codecs: list[str]) -> Chunk:
    """The MCAP header as a recording property; the empty codec name MCAP uses for uncompressed chunks reads `none`."""
    compression = pa.array([codec or "none" for codec in codecs], type=pa.string())
    return Chunk.from_property(MCAP_PROPERTY, rr.AnyValues(profile=profile, library=library, compression=compression))


# --------------------------------------------------------------------------------------
# episode driver
# --------------------------------------------------------------------------------------


@dataclass
class Episode:
    """One episode: its mcap, its parsed `info.json` and optional head calibration, and a stable recording id."""

    mcap: Path
    info: EpisodeInfo
    recording_id: str
    head_calib: Path | None


def recording_id_for(mcap: Path) -> str:
    """
    `<task>__<session>__<episode_NNNN>` for an episode MCAP.

    Those three directories above the file are the dataset's layout, so the id comes out the same
    whether the episode was found by scanning `HIW-500/` or named on the command line, and it
    matches what the Modal job derives from the HuggingFace path (`episode_index.recording_id`).
    Deriving it any other way (say from the episode directory alone) collides across sessions and
    tasks — every `episode_0001` would land on one catalog segment.
    """
    return "__".join(mcap.resolve().parent.parts[-3:])


def episode_from_mcap(mcap: Path) -> Episode:
    """
    Build an `Episode` from one episode mcap (or its `episode_NNNN/` directory).

    Parses the sibling `info.json` when it exists (episode metadata + subtask labels), picks up
    the sibling head stereo calibration when the episode ships one, and derives the recording id
    from the episode path (`recording_id_for`). The path may be absolute, or relative to either
    the working directory or the workspace root.
    """
    mcap = resolve_input_path(mcap)
    if mcap.is_dir():  # accept the episode dir too; the mcap inside is named after it.
        mcap = mcap / f"{mcap.name}.mcap"
    if not mcap.is_file():
        raise FileNotFoundError(f"no episode mcap at '{mcap}'")
    info = EpisodeInfo.from_json(mcap.with_name("info.json"))
    head_calib = mcap.with_name("calibration") / "params" / "head_camera_params.yaml"
    return Episode(
        mcap=mcap,
        info=info,
        recording_id=recording_id_for(mcap),
        head_calib=head_calib if head_calib.is_file() else None,
    )


def discover_episodes(root: Path) -> list[Episode]:
    """Every `episode_*.mcap` under `root`, sorted (recursively across tasks/sessions)."""
    return [episode_from_mcap(mcap) for mcap in sorted(root.rglob("episode_*.mcap"))]


def convert_episode(ep: Episode, rrd_root: Path) -> Path:
    """
    Convert one episode into an optimized base-layer `.rrd` and return its path.

    Merges the base entity stream with the camera fields and the sidecars (the subtask boundaries from
    `info.json`, the parsed calibration files, the series labels), then checks the collected store against the MCAP
    summary: a channel that came up short keeps its raw bytes. The census verdict and the MCAP
    header ride along as recording properties.
    """
    out_path = rrd_root / layer_relpath("base", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = LazyChunkStream.merge(
        base_stream(ep.mcap),
        camera_fields_stream(ep.mcap, RGB_CAMERA_TOPICS),
        LazyChunkStream.from_iter(subtask_chunks(ep.info) + calibration_chunks(ep) + names_chunks()),
    )
    store = merged.collect(optimize=OptimizationProfile.OBJECT_STORE)
    info = McapReader(str(ep.mcap)).info()
    short = undecodable_topics(expected_message_counts(info, exclude=IR_TOPICS), store.stream())
    properties = [census_chunk(short), mcap_chunk(info.profile, info.library, [c.codec for c in info.compression])]
    streams = [store.stream(), LazyChunkStream.from_iter(properties)]
    if short:
        streams.append(raw_stream(ep.mcap, short))
    LazyChunkStream.merge(*streams).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id
    )
    return out_path


def main() -> None:
    """Convert a single episode mcap (positional) or every episode under `DATASET_ROOT`."""
    parser = argparse.ArgumentParser(description="Convert HIW-500 episode MCAPs into per-episode Rerun RRDs.")
    parser.add_argument(
        "mcap", nargs="?", type=Path, help="A single episode mcap (default: every episode under data/HIW-500/)."
    )
    args = parser.parse_args()

    episodes = [episode_from_mcap(args.mcap)] if args.mcap is not None else discover_episodes(DATASET_ROOT)
    if not episodes:
        print(f"No episodes found under {DATASET_ROOT}")
        print("-> download some first: 'pixi run -e hiw download' (see README).")
        return
    print(f"Converting {len(episodes)} episode(s) -> {RRD_ROOT / 'base'}/")
    for ep in episodes:
        out = convert_episode(ep, RRD_ROOT)
        print(f"  {ep.recording_id}: {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
