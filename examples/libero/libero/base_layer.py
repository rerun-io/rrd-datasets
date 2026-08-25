"""
Convert LIBERO demos into per-demo Rerun RRDs.

This writes the *base* layer: everything is converted to one RRD per demo with a stable
`recording_id` (`<suite>/<task>/<demo>`). The mapping and the three dropped items are
documented in `observations.md`.

One `Hdf5Reader` stream per demo, shaped by `DeriveLens`es. The base keeps the raw
representations — lossless format changes only; anything derived (e.g. the rotation vectors as a
`Transform3D`) belongs to later layers. Cameras are discovered from their `[N, H, W, 3]` uint8
shape and flipped upright (robosuite stores OpenGL bottom-up buffers); the float arrays become
`Scalars`; the `model_file` and `init_state` attributes land statically at `/replay/*`, enough to
replay the demo in robosuite.

Run:  pixi run -e libero convert-base              # every downloaded task file
      pixi run -e libero convert-base <task.hdf5>  # a single task file
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from rerun.experimental import (
    Chunk,
    DeriveLens,
    Hdf5Reader,
    LazyChunkStream,
    OptimizationProfile,
    Selector,
)

from libero.episodes import LOCAL_DIR, discover_local_task_files, recording_id, task_id
from rrd_datasets_common.paths import dataset_rrd_dir, layer_relpath, resolve_input_path

RRD_ROOT = dataset_rrd_dir("libero")
APPLICATION_ID = "libero"

# Redundant or uninterpretable — see observations.md.
IGNORED_DATASETS = ["states", "robot_states", "obs/ee_states"]

# The verified sim timebase: the first sample at 0.25 s, then exactly 20 Hz, in every surveyed demo.
SIM_T0_NS = 250_000_000
SIM_DT_NS = 50_000_000


@dataclass
class Camera:
    """One camera item in a demo group, discovered from its `[N, H, W, 3]` uint8 shape."""

    name: str  # entity leaf, e.g. "agentview"
    source: str  # source item name, e.g. "agentview_rgb"
    height: int
    width: int


def demo_keys(reader: Hdf5Reader) -> list[str]:
    """The file's demo groups, numerically sorted — HDF5 iterates them lexicographically (`demo_10` before `demo_2`)."""
    keys = [group.removeprefix("/data/") for group in reader.groups("/data")]
    demos = [key for key in keys if "/" not in key and key.startswith("demo_")]
    return sorted(demos, key=lambda key: int(key.removeprefix("demo_")))


def discover_cameras(reader: Hdf5Reader, demo: str) -> list[Camera]:
    """Every `[N, H, W, 3]` uint8 item in the demo group, so a suite that adds a camera needs no name list."""
    cameras = []
    for info in reader.datasets(f"/data/{demo}"):
        if info.dtype == "uint8" and len(info.shape) == 4 and info.shape[3] == 3:
            item = info.path.rsplit("/", 1)[-1]
            cameras.append(Camera(item.removesuffix("_rgb"), item, int(info.shape[1]), int(info.shape[2])))
    return sorted(cameras, key=lambda camera: camera.name)


def task_language(reader: Hdf5Reader) -> str:
    """The task's language instruction, from the file-level `problem_info` attribute."""
    problem_info = json.loads(str(reader.attributes("/data")["problem_info"]))
    return str(problem_info["language_instruction"])


# Lens outputs must use non-nullable element types: the viewer deserializes components strictly,
# and pyarrow's default nullable elements read as a different datatype that renders as nothing.
_BLOB = pa.list_(pa.field("item", pa.uint8(), nullable=False))
_SCALARS = pa.list_(pa.field("item", pa.float64(), nullable=False))


def flip_vertical(height: int, width: int) -> Callable[[pa.Array], pa.Array]:
    """Image frames arrive as flat `H·W·3` blobs stored bottom-up; return them top-down."""
    frame_bytes = height * width * 3

    def flip(arr: pa.Array) -> pa.Array:
        images = arr.flatten().to_numpy(zero_copy_only=False).reshape(-1, height, width, 3)
        upright = np.ascontiguousarray(images[:, ::-1])
        offsets = pa.array(np.arange(len(arr) + 1) * frame_bytes, type=pa.int32())
        return pa.ListArray.from_arrays(offsets, pa.array(upright.reshape(-1), type=pa.uint8())).cast(_BLOB)

    return flip


def to_scalars(arr: pa.Array) -> pa.Array:
    """Any numeric column — plain or list-valued — as the f64 lists `Scalars` stores."""
    if pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
        return pc.cast(arr, _SCALARS)
    values = pc.cast(arr, pa.float64())
    return pa.ListArray.from_arrays(pa.array(np.arange(len(arr) + 1), type=pa.int32()), values).cast(_SCALARS)


def demo_lenses(cameras: list[Camera]) -> list[DeriveLens]:
    """The full source-to-entity mapping of one demo, per the table in the README."""
    scalar_targets = {
        "joint_states": "/robot/joint_states",
        "gripper_states": "/robot/gripper_states",
        # The rotation vector stays as stored (its magnitude may exceed π and carries the winding);
        # deriving a Transform3D from it belongs to a later layer.
        "ee_pos": "/robot/ee_pos",
        "ee_ori": "/robot/ee_ori",
        "actions": "/action",
        "rewards": "/reward",
        "dones": "/done",
    }
    lenses = [
        DeriveLens(camera.source, output_entity=f"/camera/{camera.name}").to_component(
            rr.Image.descriptor_buffer(), Selector(".").pipe(flip_vertical(camera.height, camera.width))
        )
        for camera in cameras
    ]
    lenses += [
        DeriveLens(source, output_entity=entity).to_component(
            rr.Scalars.descriptor_scalars(), Selector(".").pipe(to_scalars)
        )
        for source, entity in scalar_targets.items()
    ]
    lenses += [
        DeriveLens("model_file", output_entity="/replay/model_file").to_component(
            rr.ComponentDescriptor("model_file"), Selector(".")
        ),
        DeriveLens("init_state", output_entity="/replay/init_state").to_component(
            rr.ComponentDescriptor("init_state"), Selector(".")
        ),
    ]
    return lenses


def with_sim_time(chunk: Chunk) -> list[Chunk]:
    """
    Attach the `sim_time` timeline to a temporal chunk.

    The source stores no time item, but the timebase is exact (observations.md), so the timeline
    derives from each chunk's own `row_index` values — a chunk may start at any step.
    """
    if chunk.is_static:
        return [chunk]
    batch = chunk.to_record_batch()
    if "rerun.controls.RowId" in batch.schema.names:
        batch = batch.drop_columns(["rerun.controls.RowId"])
    sim_ns = batch.column("row_index").to_numpy() * SIM_DT_NS + SIM_T0_NS
    batch = batch.append_column(pa.field("sim_time", pa.duration("ns")), pa.array(sim_ns, type=pa.duration("ns")))
    return Chunk.from_record_batch(batch, index=["row_index", "sim_time"])


def image_format_chunks(cameras: list[Camera]) -> list[Chunk]:
    """The static `Image:format` beside each camera's buffers."""
    return [
        Chunk.from_columns(
            f"/camera/{camera.name}",
            indexes=[],
            columns=rr.Image.columns(
                format=[
                    rr.components.ImageFormat(
                        width=camera.width,
                        height=camera.height,
                        color_model=rr.ColorModel.RGB,
                        channel_datatype=rr.ChannelDatatype.U8,
                    )
                ]
            ),
        )
        for camera in cameras
    ]


def convert_demo(reader: Hdf5Reader, task: str, demo: str, rrd_root: Path) -> Path:
    """Write one demo's base layer; returns the written path."""
    cameras = discover_cameras(reader, demo)
    stream = reader.stream(root_group=f"/data/{demo}", ignore_datasets=IGNORED_DATASETS, use_structs=False)
    stream = stream.lenses(demo_lenses(cameras), output_mode="drop_unmatched").flat_map(with_sim_time)
    statics = image_format_chunks(cameras)
    statics.append(
        Chunk.from_columns(
            "/task/instruction", indexes=[], columns=rr.TextDocument.columns(text=[task_language(reader)])
        )
    )
    # Entities that share a plot need per-entity legend labels, which ride statically in the data.
    series_labels: dict[str, list[str]] = {
        "/reward": ["reward"],
        "/done": ["done"],
        "/robot/ee_pos": ["x", "y", "z"],
        "/robot/ee_ori": ["rx", "ry", "rz"],
    }
    statics += [
        Chunk.from_columns(entity, indexes=[], columns=rr.SeriesLines.columns(names=names).partition([len(names)]))
        for entity, names in series_labels.items()
    ]
    merged = LazyChunkStream.merge(stream, LazyChunkStream.from_iter(statics))

    rec_id = recording_id(task, demo)
    out_path = rrd_root / layer_relpath("base", rec_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=rec_id
    )
    return out_path


def task_files(argv: list[str]) -> list[tuple[Path, str]]:
    """The task files to convert as `(filesystem path, task id)`: one user-passed file, or every downloaded one."""
    if len(argv) > 1:
        path = resolve_input_path(Path(argv[1]))
        return [(path, task_id(f"{path.parent.name}/{path.name}"))]
    return [(LOCAL_DIR / item.path, item.task_id) for item in discover_local_task_files()]


def main(argv: list[str]) -> None:
    inputs = task_files(argv)
    print(f"Building base layer for {len(inputs)} task file(s) -> {RRD_ROOT / 'base'}/")
    for path, task in inputs:
        reader = Hdf5Reader(path)
        for demo in demo_keys(reader):
            out = convert_demo(reader, task, demo, RRD_ROOT)
            print(f"  {recording_id(task, demo)}: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv)
