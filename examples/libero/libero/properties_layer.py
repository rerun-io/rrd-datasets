"""
Build a properties layer: per-demo metadata that shows up as catalog columns.

Recording properties (logged under `/__properties/episode`) attach to the segment when
registered. They are filterable/sortable columns reading `property:episode:<name>`. The
values come from the file attributes (`problem_info`, `env_name`), the file path (suite, task,
scene prefix), and the demo's `num_samples`.

Run:  pixi run -e libero convert-properties              # every downloaded task file
      pixi run -e libero convert-properties <task.hdf5>  # a single task file
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.experimental import Chunk, Hdf5Reader, LazyChunkStream, OptimizationProfile

from libero.base_layer import APPLICATION_ID, RRD_ROOT, demo_keys, task_files, task_language
from libero.episodes import recording_id
from rrd_datasets_common.paths import layer_relpath

PROPERTY = "episode"

# libero_10/90 filenames carry a scene prefix the task language drops, e.g. `KITCHEN_SCENE3`.
_SCENE_PREFIX = re.compile(r"([A-Z0-9]+_SCENE\d+)_")
_SCENE_PREFIX = re.compile(r"([A-Z0-9_]+_SCENE\d+)_")


@dataclass
class TaskFacts:
    """The per-file property values, shared by every demo in the file."""

    suite: str
    task: str
    scene: str  # empty when the filename carries no scene prefix
    language: str
    env_name: str
    source_file: str


def task_facts(reader: Hdf5Reader, task: str) -> TaskFacts:
    """The property values one task file contributes, from its attributes and its path."""
    suite, _, name = task.partition("/")
    scene_match = _SCENE_PREFIX.match(name)
    return TaskFacts(
        suite=suite,
        task=name,
        scene=scene_match.group(1) if scene_match else "",
        language=task_language(reader),
        env_name=str(reader.attributes("/data")["env_name"]),
        source_file=f"{task}_demo.hdf5",
    )


def convert_demo(reader: Hdf5Reader, facts: TaskFacts, demo: str, rrd_root: Path) -> Path:
    """Write one demo's properties layer; returns the written path."""
    num_samples = int(str(reader.attributes(f"/data/{demo}")["num_samples"]))
    rec_id = recording_id(f"{facts.suite}/{facts.task}", demo)
    out_path = rrd_root / layer_relpath("properties", rec_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One property holding every field, so the catalog columns read `property:episode:<name>`.
    chunk = Chunk.from_property(
        PROPERTY,
        rr.AnyValues(
            suite=[facts.suite],
            task=[facts.task],
            scene=[facts.scene],
            language=[facts.language],
            env_name=[facts.env_name],
            num_samples=np.array([num_samples], dtype=np.int64),
            source_file=[facts.source_file],
        ),
    )
    LazyChunkStream.from_iter([chunk]).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(out_path), application_id=APPLICATION_ID, recording_id=rec_id
    )
    return out_path


def main(argv: list[str]) -> None:
    inputs = task_files(argv)
    print(f"Building properties layer for {len(inputs)} task file(s) -> {RRD_ROOT / 'properties'}/")
    for path, task in inputs:
        reader = Hdf5Reader(path)
        facts = task_facts(reader, task)
        for demo in demo_keys(reader):
            out = convert_demo(reader, facts, demo, RRD_ROOT)
            print(f"  {recording_id(task, demo)}: {out}")


if __name__ == "__main__":
    main(sys.argv)
