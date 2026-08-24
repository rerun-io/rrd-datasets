"""
Convert HIW-500 (Unitree G1 bimanual) MCAP episodes into per-episode Rerun RRDs.

This writes the *base* layer: the raw ROS2 streams as Rerun entities, one optimized RRD per
episode with a stable `recording_id`, so every episode is its own catalog segment. Nothing
kinematic is computed here — URDF FK is a separate layer.

One `McapReader` stream, shaped by lenses. The reader decodes the well-known ROS2 types on its
own (CompressedImage -> `EncodedImage`, std_msgs/String -> `TextDocument`); the custom
`homies/*` / `unitree_go/*` messages arrive as `<name>:message` structs that `DeriveLens` +
`Selector` turn into scalars and transforms. Hand-built chunks are the sidecars — `info.json`
(episode metadata + subtask labels) and the calibration files, one `CalibrationFile` component
per value — plus the static joint-name mapping and controller gains beside the joint arrays. A channel census compares decoded rows against the
MCAP's own per-channel counts and flags episodes with undecodable messages as recording properties.

Run:  pixi run -e hiw convert-base            # all episodes under data/HIW-500/
      pixi run -e hiw convert-base <ep.mcap>  # a single episode mcap
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
import yaml
from PIL import Image
from rerun.experimental import (
    Chunk,
    DeriveLens,
    LazyChunkStream,
    McapReader,
    OptimizationProfile,
    Selector,
)
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

# Component identifiers produced by McapReader's decoders.
# Note that the package name "homies/* = Header + a unitree_hg payload"
# See https://github.com/unitreerobotics/unitree_ros2/blob/master/cyclonedds_ws/src/unitree/unitree_hg/msg for the message definitions.
MSG_LOWSTATE = "homies.msg.LowStateStamped:message"
MSG_LOWCMD = "homies.msg.LowCmdStamped:message"
MSG_MOTOR_STATE = "homies.msg.MotorStateStamped:message"
MSG_MOTOR_CMD = "homies.msg.MotorCmdStamped:message"
MSG_IMU = "homies.msg.IMUStateStamped:message"
MSG_ODOM = "unitree_go.msg.SportModeState:message"
TEXT = "TextDocument:text"
BLOB = "EncodedImage:blob"

# Only decode the topics we use (excludes e.g. the wrist ir1/ir2 streams some episodes carry).
KEEP_TOPICS = [
    "^/stamped/lowstate$",
    "^/stamped/lowcmd$",
    "^/stamped/secondary_imu$",
    "^/stamped/dex1/(left|right)/(state|cmd)$",
    "^/lf/odommodestate$",
    "^/wbc_lerobot$",
    "^/annotation$",
    "^/camera/(head|left_wrist|right_wrist)/image/compressed$",
]
# Raw sources consumed by lenses (and reader bookkeeping) — by census time only their channel and
# schema statics remain (`forward_unmatched` drops a consumed column and its rows); removed before
# writing.
DROP_RAWS = [
    "/stamped/**",
    "/lf/odommodestate",
    "/wbc_lerobot",
    "/camera/head/image/compressed",
    "/__mcap_metadata",
    "/__mcap_properties",
]
# The wrist cameras and /annotation are kept, so their leftover Mcap* metadata columns are removed
# at the component level instead (drop needs exact ids — a `McapChannel:*` wildcard silently
# matches nothing).
DROP_COMPONENTS = [f"McapChannel:{c}" for c in ("id", "message_encoding", "metadata", "topic")] + [
    f"McapSchema:{c}" for c in ("data", "encoding", "id", "name")
]

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

# LowState carries two temperature sensors per motor. The source does not document which is which,
# so each lands on its own width-29 entity, named by index.
N_MOTOR_TEMPS = 2

# Series order of the width-12 ee_state/ee_action arrays: the /wbc_lerobot JSON lays out the left
# arm then the right, six pose fields each.
EE_FIELDS = ("px", "py", "pz", "rx", "ry", "rz")
EE_NAMES = [f"{arm}/{field}" for arm in ("left", "right") for field in EE_FIELDS]

# Series order of the width-3 IMU arrays, per source field.
IMU_NAMES = {"rpy": ["r", "p", "y"], "gyroscope": ["x", "y", "z"], "accelerometer": ["x", "y", "z"]}

# The robot carries two IMUs: one embedded in LowState (pelvis) and one standalone. They are
# separate sensors reading different orientations, so both are kept, each under its own name.
# The value is the message id and the struct path its 3-axis fields hang off.
IMU_SOURCES = {"pelvis": (MSG_LOWSTATE, ".data.imu_state"), "secondary": (MSG_IMU, ".data")}

# Arrow types expected by Transform3D translation/quaternion components.
VEC3 = pa.list_(pa.float32(), 3)
QUAT = pa.list_(pa.float32(), 4)

# pyarrow.compute ships incomplete stubs; alias once (used in the joint-array selector).
list_slice = pc.list_slice  # type: ignore[attr-defined]

_TJ = TurboJPEG()  # libjpeg-turbo handle for lossless compressed-domain cropping


# --------------------------------------------------------------------------------------
# lens pipe callbacks (PyArrow transforms run inside Selector.pipe — the idiomatic place)
# --------------------------------------------------------------------------------------


def to_vec3(arr: pa.Array) -> pa.Array:
    """A list<float>[3] field -> Transform3D translation type."""
    return pa.array([list(v) for v in arr.to_pylist()], type=VEC3)


def quat_wxyz_to_xyzw(arr: pa.Array) -> pa.Array:
    """Unitree quaternions are [w, x, y, z]; Rerun wants [x, y, z, w]."""
    return pa.array([[v[1], v[2], v[3], v[0]] for v in arr.to_pylist()], type=QUAT)


def const_str(value: str) -> Callable[[pa.Array], pa.Array]:
    """A pipe that emits a constant string per row (for constant parent/child frame names)."""
    return lambda arr: pa.array([value] * len(arr), type=pa.utf8())


def crop_jpegs(left: bool) -> Callable[[pa.Array], pa.Array]:
    """
    Split each side-by-side stereo JPEG into its left/right half, losslessly.

    `_TJ.crop` crops in the JPEG compressed domain (no pixel decode/encode); the head halves are
    640 wide (MCU-aligned), so the crop is exact and lossless. Bytes move through numpy buffers
    rather than Python int lists, which keeps the Arrow `list<uint8>` round-trip cheap.
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


def _records(textcol: pa.Array) -> list[dict[str, Any]]:
    return [json.loads(t) for t in textcol.to_pylist()]


def json_index(key: str, i: int) -> Callable[[pa.Array], pa.Array]:
    """One element of a JSON array field -> a single Scalar per row (the pivot channels stay per-element)."""
    return lambda c: pa.array([r[key][i] for r in _records(c)], type=pa.float64())


def json_array(key: str, width: int) -> Callable[[pa.Array], pa.Array]:
    """
    A JSON array field -> one list row per message, truncated to `width`.

    A list row is not a valid `Scalars` value on its own; a following `Selector("[]")` fans it
    into a width-`width` array `Scalars`. The truncation pins the width the series labels assume.
    """
    return lambda c: pa.array([r[key][:width] for r in _records(c)], type=pa.list_(pa.float64()))


def json_pos(key: str, lo: int) -> Callable[[pa.Array], pa.Array]:
    """A 3-slice of a JSON array field -> Transform3D translation (EE position)."""
    return lambda c: pa.array([r[key][lo : lo + 3] for r in _records(c)], type=VEC3)


def gripper_field(name: str) -> Callable[[pa.Array], pa.Array]:
    """One `gripper_controls` field -> a single Scalar per row."""
    return lambda c: pa.array([r["gripper_controls"][name] for r in _records(c)], type=pa.float64())


# --------------------------------------------------------------------------------------
# lens builders (one set per source topic)
# --------------------------------------------------------------------------------------


def _scalar_lens(msg: str, entity: str, selector: Selector | str) -> DeriveLens:
    """One numeric field of a `:message` struct -> a Scalars entity."""
    sel = selector if isinstance(selector, Selector) else Selector(selector)
    return DeriveLens(msg, output_entity=entity).to_component(rr.Scalars.descriptor_scalars(), sel)


def _motors(field: str, element: str) -> Selector:
    """
    One element field of a motor array -> a width-29 array `Scalars` row, in G1 motor order.

    The motor arrays are fixed 35-wide (indices 29-34 unused) and the selector can't slice, so
    `list_slice` truncates to the 29 body joints before `[]` iterates the elements. The f32
    source values are cast to f64 — the `Scalars` datatype — because the viewer plots only the
    first instance of a float32 array.
    """
    return (
        Selector(f".data.{field}")
        .pipe(lambda arr: list_slice(arr, 0, N_JOINTS))
        .pipe(Selector(f"[].{element}"))
        .pipe(lambda arr: arr.cast(pa.float64()))
    )


def joint_lenses() -> list[DeriveLens]:
    """Per-joint lowstate signals: one width-29 `Scalars` entity per signal, in G1 motor order."""
    lenses = [
        _scalar_lens(MSG_LOWSTATE, f"/state/joint/{short}", _motors("motor_state", element))
        for short, element in (
            ("q", "q"),
            ("dq", "dq"),
            ("tau", "tau_est"),
            ("voltage", "vol"),
            # Per-motor status bitfield. Zero on healthy hardware; a set bit marks a motor fault.
            ("status", "motorstate"),
            # Reported per-motor mode. Constant for most episodes, but it does switch mid-episode,
            # so it rides the timeline rather than a static row.
            ("mode", "mode"),
        )
    ]
    lenses += [
        _scalar_lens(MSG_LOWSTATE, f"/state/joint/temperature/{i}", _motors("motor_state", f"temperature[{i}]"))
        for i in range(N_MOTOR_TEMPS)
    ]
    return lenses


def cmd_lenses() -> list[DeriveLens]:
    """Commanded joint arrays from lowcmd, aligned to the same motor order."""
    return [
        _scalar_lens(MSG_LOWCMD, f"/cmd/joint/{short}", _motors("motor_cmd", element))
        for short, element in (("q", "q"), ("tau", "tau"))
    ]


def imu_lenses(imu: str) -> list[DeriveLens]:
    """
    One IMU's signals: a width-3 `Scalars` entity per 3-axis field, plus its die temperature.

    `imu` keys `IMU_SOURCES`, which picks the message and the struct path. f32 values are cast to
    f64 for the same reason as `_motors`.
    """
    msg, root = IMU_SOURCES[imu]
    lenses = [
        _scalar_lens(
            msg,
            f"/state/imu/{imu}/{name}",
            Selector(f"{root}.{name}").pipe(Selector("[]")).pipe(lambda arr: arr.cast(pa.float64())),
        )
        for name in IMU_NAMES
    ]
    lenses.append(
        _scalar_lens(
            msg,
            f"/state/imu/{imu}/temperature",
            Selector(f"{root}.temperature").pipe(lambda arr: arr.cast(pa.float64())),
        )
    )
    return lenses


def odom_lenses() -> list[DeriveLens]:
    """Base odometry: per-axis position/velocity + height/yaw scalars, plus a 3D Transform3D pose."""
    lenses: list[DeriveLens] = []
    for name in ("position", "velocity"):
        for i, ax in enumerate("xyz"):
            lenses.append(_scalar_lens(MSG_ODOM, f"/state/base/{name}/{ax}", f".{name}[{i}]"))
    lenses += [
        _scalar_lens(MSG_ODOM, "/state/base/body_height", ".body_height"),
        _scalar_lens(MSG_ODOM, "/state/base/yaw_speed", ".yaw_speed"),
        DeriveLens(MSG_ODOM, output_entity="/state/base")
        .to_component(rr.Transform3D.descriptor_translation(), Selector(".position").pipe(to_vec3))
        .to_component(
            rr.Transform3D.descriptor_quaternion(), Selector(".imu_state.quaternion").pipe(quat_wxyz_to_xyzw)
        ),
    ]
    return lenses


def _wbc_scalar_lens(entity: str, pipe: Callable[[pa.Array], pa.Array]) -> DeriveLens:
    return DeriveLens(TEXT, output_entity=entity).to_component(
        rr.Scalars.descriptor_scalars(), Selector(".").pipe(pipe)
    )


def lerobot_lenses() -> list[DeriveLens]:
    """Parse /wbc_lerobot JSON into EE state/action arrays + 3D EE poses, gripper, pivot."""
    lenses: list[DeriveLens] = []
    for kind in ("ee_state", "ee_action"):
        # The 12 pose values as one array Scalars on the parent; series order in EE_NAMES.
        lenses.append(
            DeriveLens(TEXT, output_entity=f"/lerobot/{kind}").to_component(
                rr.Scalars.descriptor_scalars(),
                Selector(".").pipe(json_array(kind, len(EE_NAMES))).pipe(Selector("[]")),
            )
        )
        for arm, lo in (("left", 0), ("right", 6)):
            # 3D end-effector position marker (translation only).
            lenses.append(
                DeriveLens(TEXT, output_entity=f"/lerobot/{kind}/{arm}").to_component(
                    rr.Transform3D.descriptor_translation(), Selector(".").pipe(json_pos(kind, lo))
                )
            )
    for k in ("left_trigger", "left_squeeze", "right_trigger", "right_squeeze"):
        lenses.append(_wbc_scalar_lens(f"/lerobot/gripper/{k}", gripper_field(k)))
    for i in range(7):
        lenses.append(_wbc_scalar_lens(f"/lerobot/pivot/{i}", json_index("pivot", i)))
    return lenses


def gripper_lenses() -> list[tuple[str, list[DeriveLens]]]:
    """(content, lenses) pairs for the four dex1 gripper topics (single joint each)."""
    out: list[tuple[str, list[DeriveLens]]] = []
    for side in ("left", "right"):
        out.append((
            f"/stamped/dex1/{side}/state",
            [
                _scalar_lens(MSG_MOTOR_STATE, f"/state/gripper/{side}/{short}", f".data.states[0].{element}")
                for short, element in (("q", "q"), ("dq", "dq"), ("tau", "tau_est"))
            ],
        ))
        out.append((
            f"/stamped/dex1/{side}/cmd",
            [_scalar_lens(MSG_MOTOR_CMD, f"/cmd/gripper/{side}/q", ".data.cmds[0].q")],
        ))
    return out


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


# Camera entities that stay in the output: the split head eyes and the wrist passthroughs. The
# raw side-by-side head stream is shed once its split lens ran.
KEPT_CAMERAS = [
    "/camera/head/left",
    "/camera/head/right",
    "/camera/left_wrist/image/compressed",
    "/camera/right_wrist/image/compressed",
]


def base_stream(path: Path) -> LazyChunkStream:
    """
    One McapReader stream, shaped by content-scoped lenses.

    The raw source entities leave here as skeletons: each lens consumes its input column, and
    `forward_unmatched` drops what was consumed, leaving the static `McapChannel`/`McapSchema`
    rows the channel census reads. `convert_episode` drops the skeletons and the reader
    bookkeeping after the census.
    """
    stream = McapReader(str(path), include_topic_regex=KEEP_TOPICS).stream()
    # The joint arrays and the pelvis IMU both come out of lowstate, so they share one pass: a
    # second `lenses` call would find the message column already consumed.
    stream = stream.lenses(
        joint_lenses() + imu_lenses("pelvis"), content="/stamped/lowstate", output_mode="forward_unmatched"
    )
    stream = stream.lenses(cmd_lenses(), content="/stamped/lowcmd", output_mode="forward_unmatched")
    for content, lenses in gripper_lenses():
        stream = stream.lenses(lenses, content=content, output_mode="forward_unmatched")
    stream = stream.lenses(imu_lenses("secondary"), content="/stamped/secondary_imu", output_mode="forward_unmatched")
    stream = stream.lenses(odom_lenses(), content="/lf/odommodestate", output_mode="forward_unmatched")
    stream = stream.lenses(lerobot_lenses(), content="/wbc_lerobot", output_mode="forward_unmatched")
    stream = stream.lenses(
        head_split_lenses(), content="/camera/head/image/compressed", output_mode="forward_unmatched"
    )
    # The kept camera EncodedImages need a codec tag for the viewer. forward_all so the blob (the
    # lens's input, hence "consumed") survives alongside the new tag.
    stream = stream.lenses(media_type_lens(), content=KEPT_CAMERAS, output_mode="forward_all")
    # Wrist cameras (/camera/{left,right}_wrist/image/compressed) and /annotation pass through.
    return stream


@dataclass
class Subtask:
    """One subtask boundary: its label and the episode timestamp (ns) where it starts."""

    task: str
    timestamp_ns: int


@dataclass
class EpisodeInfo:
    """The `info.json` fields the conversion uses. Keys the episode omits stay empty/zero."""

    task: str = ""
    scene: int = -1
    duration_sec: float = 0.0
    episode_name: str = ""
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
            subtasks=[Subtask(task=s["task"], timestamp_ns=s["timestamp_ns"]) for s in raw.get("subtasks", [])],
        )


# --------------------------------------------------------------------------------------
# static configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticConfig:
    """One configuration field that holds for a whole episode, logged once beside its signals."""

    topic: str
    message: str
    path: str  # dotted field path inside the `:message` struct
    entity: str
    name: str  # AnyValues component name
    width: int  # leading values kept per message (motor arrays are 35-wide, only 29 are real)


STATIC_CONFIGS = (
    StaticConfig("/stamped/lowcmd", MSG_LOWCMD, "data.motor_cmd.kp", "/cmd/joint", "kp", N_JOINTS),
    StaticConfig("/stamped/lowcmd", MSG_LOWCMD, "data.motor_cmd.kd", "/cmd/joint", "kd", N_JOINTS),
    *(
        StaticConfig(f"/stamped/dex1/{side}/cmd", MSG_MOTOR_CMD, f"data.cmds.{gain}", f"/cmd/gripper/{side}", gain, 1)
        for side in ("left", "right")
        for gain in ("kp", "kd")
    ),
)


def _message_field(column: pa.Array, path: str) -> pa.Array:
    """Follow a dotted path into a `:message` struct column, flattening every list level on the way."""
    arr = column
    while not pa.types.is_struct(arr.type):
        arr = arr.flatten()
    for part in path.split("."):
        arr = arr.field(part)
        while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
            arr = arr.flatten()
    return arr


def static_config_chunks(path: Path) -> list[Chunk]:
    """
    Controller gains and the motor enable mask, read once and logged as static rows.

    These hold for a whole episode, so one static row carries what would otherwise repeat on every
    message. Constancy is verified while reading, so a source that varies one of them raises here
    instead of silently recording only the first value.
    """
    by_topic: dict[str, list[StaticConfig]] = defaultdict(list)
    for config in STATIC_CONFIGS:
        by_topic[config.topic].append(config)
    values: dict[str, dict[str, list[float]]] = defaultdict(dict)
    stream = McapReader(str(path), include_topic_regex=[f"^{topic}$" for topic in by_topic]).stream()
    for chunk in stream.to_chunks():
        configs = by_topic.get(chunk.entity_path)
        if configs is None:
            continue
        batch = chunk.to_record_batch()
        column = next((name for name in batch.schema.names if name.endswith(":message")), None)
        if column is None or batch.num_rows == 0:
            continue
        for config in configs:
            raw = _message_field(batch.column(column), config.path).to_numpy(zero_copy_only=False)
            rows = np.asarray(raw, dtype=np.float64).reshape(batch.num_rows, -1)[:, : config.width]
            if not bool(np.all(rows == rows[0])):
                row, column = (int(index[0]) for index in np.nonzero(rows != rows[0])[:2])
                raise ValueError(
                    f"{path.name}: {config.topic} {config.path} varies within the episode "
                    f"(row {row} index {column}: {rows[0][column]} -> {rows[row][column]}); "
                    "a field that changes has to be logged on the timeline, not as a static row"
                )
            first: list[float] = rows[0].tolist()
            seen = values[config.entity].get(config.name)
            if seen is not None and seen != first:
                raise ValueError(f"{path.name}: {config.topic} {config.path} disagrees between chunks")
            values[config.entity][config.name] = first
    return [
        Chunk.from_columns(
            entity,
            indexes=[],
            # The stub cannot express that these dynamic keys never collide with the leading
            # `drop_untyped_nones` parameter, so it reads every value as that flag.
            columns=rr.AnyValues.columns(**{name: [value] for name, value in fields.items()}),  # type: ignore[arg-type]
        )
        for entity, fields in values.items()
    ]


def joint_names_chunks() -> list[Chunk]:
    """
    Motor index -> joint name, static beside the joint arrays.

    Series i of every width-29 joint array is the joint `joint_names[i]`. Logged on the array
    parents, so one component per side covers q/dq/tau. The blueprint carries the display labels;
    this component is the machine-readable mapping.
    """
    return [
        Chunk.from_columns(entity, indexes=[], columns=rr.AnyValues.columns(joint_names=[G1_JOINT_NAMES]))
        for entity in ("/state/joint", "/cmd/joint")
    ]


def ee_names_chunks() -> list[Chunk]:
    """Series index -> pose field, static on the width-12 ee arrays (same order for state and action)."""
    return [
        Chunk.from_columns(entity, indexes=[], columns=rr.AnyValues.columns(ee_names=[EE_NAMES]))
        for entity in ("/lerobot/ee_state", "/lerobot/ee_action")
    ]


def imu_names_chunks() -> list[Chunk]:
    """Series index -> axis, static on each width-3 IMU array, for both IMUs."""
    return [
        Chunk.from_columns(f"/state/imu/{imu}/{name}", indexes=[], columns=rr.AnyValues.columns(imu_names=[axes]))
        for imu in IMU_SOURCES
        for name, axes in IMU_NAMES.items()
    ]


def sidecar_stream(info: EpisodeInfo) -> LazyChunkStream:
    """Episode metadata + subtask labels from info.json — the genuine hand-built sidecar."""
    chunks: list[Chunk] = [
        Chunk.from_columns(
            "/episode",
            indexes=[],
            columns=rr.AnyValues.columns(
                task=[info.task],
                scene=[info.scene],
                duration_sec=[info.duration_sec],
                episode_name=[info.episode_name],
            ),
        )
    ]
    if info.subtasks:
        # Subtasks are discrete states: each boundary is a StateChange that holds until the next,
        # rendered as colored lanes by a StateTimelineView (vs. append-only TextLog events).
        ts = np.array([s.timestamp_ns for s in info.subtasks], dtype="datetime64[ns]")
        chunks.append(
            Chunk.from_columns(
                "/task/subtask",
                indexes=[rr.TimeColumn("message_publish_time", timestamp=ts)],
                columns=rr.StateChange.columns(state=[s.task for s in info.subtasks]),
            )
        )
    return LazyChunkStream.from_iter(chunks)


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

# Emitted by McapReader: the file's per-channel message counts (on `/__mcap_properties`) and each
# topic entity's channel id — the join key between the two.
STAT_CHANNEL_COUNTS = "McapStatistics:channel_message_counts"
CHANNEL_ID = "McapChannel:id"

# Where each raw topic's decoded rows survive to be counted: one lens output per topic, row-aligned
# with its input (`forward_unmatched` drops the consumed raw column and with it the raw rows).
# Topics not listed keep rows on their own entity (the camera and /annotation passthroughs).
CENSUS_PROXIES = {
    "/stamped/lowstate": "/state/joint/q",
    "/stamped/lowcmd": "/cmd/joint/q",
    "/stamped/secondary_imu": "/state/imu/secondary/rpy",
    "/stamped/dex1/left/state": "/state/gripper/left/q",
    "/stamped/dex1/left/cmd": "/cmd/gripper/left/q",
    "/stamped/dex1/right/state": "/state/gripper/right/q",
    "/stamped/dex1/right/cmd": "/cmd/gripper/right/q",
    "/lf/odommodestate": "/state/base/position/x",
    "/wbc_lerobot": "/lerobot/gripper/left_trigger",
}


def undecodable_topics(chunks: Iterable[Chunk]) -> list[str]:
    """
    Topics whose decoded rows fall short of the MCAP's own per-channel message counts.

    A message that fails CDR decoding is discarded without a row or an exception (e.g. the Feb
    2026 Clothes-Washing sessions declare `MotorStateStamped` on the left dex1 state channel but
    carry smaller `MotorCmd` payloads), so the RRD silently under-reports the channel. The MCAP's
    authoritative counts arrive in the stream itself (statistics on `/__mcap_properties`), keyed
    to each topic entity by its static `McapChannel:id` row; decoded rows are counted on the
    topic's `CENSUS_PROXIES` entity — no second read of the file.
    """
    expected: dict[int, int] = {}
    channel_ids: dict[str, list[int]] = {}
    decoded: Counter[str] = Counter()
    for chunk in chunks:
        if not chunk.is_static:
            decoded[chunk.entity_path] += chunk.num_rows
            continue
        batch = chunk.to_record_batch()
        for index, column_field in enumerate(batch.schema):
            if column_field.name == STAT_CHANNEL_COUNTS:
                for row in batch.column(index).to_pylist():
                    for entry in row[0] if row and isinstance(row[0], list) else row:
                        expected[int(entry["channel_id"])] = int(entry["message_count"])
            elif column_field.name == CHANNEL_ID:
                for row in batch.column(index).to_pylist():
                    channel_ids.setdefault(chunk.entity_path, []).extend(int(value) for value in row)
    return sorted(
        topic
        for topic, ids in channel_ids.items()
        if decoded[CENSUS_PROXIES.get(topic, topic)] < sum(expected.get(channel_id, 0) for channel_id in ids)
    )


def census_chunk(topics: list[str]) -> Chunk:
    """The census verdict as a recording property, so the catalog can filter on it."""
    return Chunk.from_columns(
        PROPERTY_PATH,
        indexes=[],
        columns=rr.AnyValues.columns(
            has_undecodable=[bool(topics)],
            # Typed explicitly: an inferred empty list arrives as `list<null>` and then drops
            # the topics of every episode that does have failures.
            undecodable_topics=pa.array([topics], type=pa.list_(pa.string())),
        ),
    )


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

    Merges the base entity stream with the sidecars (`info.json` metadata, the parsed
    calibration files, and the static joint-name mapping), runs the channel census on the
    collected store, then writes a single
    object-store-optimized recording — census verdict as a recording property, raw skeletons and reader
    bookkeeping dropped — stamped with `application_id` / `recording_id`.
    """
    out_path = rrd_root / layer_relpath("base", ep.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = LazyChunkStream.merge(
        base_stream(ep.mcap),
        sidecar_stream(ep.info),
        LazyChunkStream.from_iter(
            calibration_chunks(ep)
            + joint_names_chunks()
            + ee_names_chunks()
            + imu_names_chunks()
            + static_config_chunks(ep.mcap)
        ),
    )
    store = merged.collect(optimize=OptimizationProfile.OBJECT_STORE)
    census = census_chunk(undecodable_topics(store.stream().to_chunks()))
    final = LazyChunkStream.merge(
        store.stream().drop(content=DROP_RAWS).drop(components=DROP_COMPONENTS),
        LazyChunkStream.from_iter([census]),
    )
    final.write_rrd(str(out_path), application_id=APPLICATION_ID, recording_id=ep.recording_id)
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
