"""
Build both layers (base, properties) for every demo in one command.

Each layer module owns its conversion and runs on its own (`convert-base` / `convert-properties`);
this module runs them in order, opening each task file once.

Run:  pixi run -e libero convert              # every downloaded task file
      pixi run -e libero convert <task.hdf5>  # a single task file
"""

from __future__ import annotations

import sys

from rerun.experimental import Hdf5Reader

from libero import base_layer, properties_layer
from libero.base_layer import RRD_ROOT, demo_keys, task_files
from libero.episodes import recording_id


def main(argv: list[str]) -> None:
    inputs = task_files(argv)
    print(f"Converting {len(inputs)} task file(s) -> {RRD_ROOT}/<layer>/ (base + properties)")
    for path, task in inputs:
        reader = Hdf5Reader(path)
        facts = properties_layer.task_facts(reader, task)
        for demo in demo_keys(reader):
            base_layer.convert_demo(reader, task, demo, RRD_ROOT)
            properties_layer.convert_demo(reader, facts, demo, RRD_ROOT)
            print(f"  {recording_id(task, demo)}: 2 layers written")


if __name__ == "__main__":
    main(sys.argv)
