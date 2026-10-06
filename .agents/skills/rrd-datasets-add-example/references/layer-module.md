# Layer module template

Every layer is written by one module, `[pkg]/[layer]_layer.py`, that runs on its own.
This file is the template those modules share.
`references/project-layout.md` says where their output goes.

The base layer follows the same template, with two differences: it reads the source by definition, and it derives the recording id that every other layer then inherits.

## Read the layer you depend on

Derive a layer from the layer it depends on, not from the source.
Take the recording id from that input instead of deriving it a second time, so the layers of an episode always stack.
Treat a reach back to the source as a signal: either the input layer's conversion is missing something, or the derivation is wrong.

Expect one narrow exception, an outside resource that no layer contains, such as a robot model.
Read `rerun-urdf` for such a resource: how to stream it, and what belongs in a dataset asset rather than in every episode's `.rrd`.
Keep reading the input layer for everything the episodes themselves contain.

## One machine or many

On one machine, read the input layers' `.rrd` files, build the layer for every episode, then register all of them at once.
That is the local `convert-[layer]` task followed by `register`, and the skeleton below follows it.

Across many cloud workers, an append job builds the layer one segment at a time from the registered dataset, as the "Append job" section of `references/modal-job.md` describes.

Keep the derivation itself a pure function over chunks or a dataframe, such as `[layer]_stream` below.
Both ways then call the same code and differ only in how they read the input and where they write.

---

```python
"""
Build the *[layer]* layer: [what it adds that its input layer does not contain].

Derived from the [input] layer, so the source is never read: [what it computes].

Run:  pixi run -e [env] convert-[layer]              # every [input] RRD under rrds/[dataset]/[input]/
      pixi run -e [env] convert-[layer] <in.rrd>     # one recording
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.chunk import ChunkStore, LazyChunkStream, OptimizationProfile, RrdReader

from [pkg].base_layer import APPLICATION_ID
from rrd_datasets_common.paths import dataset_rrd_dir, layer_relpath

LAYER = "[layer]"
INPUT_LAYER = "base"
RRD_ROOT = dataset_rrd_dir("[dataset]")


def [layer]_stream(store: ChunkStore) -> LazyChunkStream:
    """The derivation: [input entities] -> [output entities]. Pure, so a test runs it on a store."""


def has_input(store: ChunkStore) -> bool:
    """Whether this recording has what the layer derives from."""


def write_layer(input_rrd: Path, out_dir: Path) -> Path | None:
    """Write one recording's [layer] layer under `out_dir`, or None when its input is absent."""
    reader = RrdReader(str(input_rrd))
    recordings = reader.recordings()
    if len(recordings) != 1:
        raise ValueError(f"{input_rrd}: expected one recording, found {len(recordings)}")
    recording_id = recordings[0].recording_id
    store = reader.store()
    if not has_input(store):
        return None
    out_path = out_dir / layer_relpath(LAYER, recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    derived = [layer]_stream(store).collect(optimize=OptimizationProfile.OBJECT_STORE)
    derived.write_rrd(str(out_path), application_id=APPLICATION_ID, recording_id=recording_id)
    return out_path


def discover_inputs(path: Path) -> list[Path]:
    """The input RRDs at `path`: the file itself, or a directory's top level."""
    return [path] if path.is_file() else sorted(path.glob("*.rrd"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the [layer] layer from [input] RRDs.")
    parser.add_argument(
        "input", nargs="?", type=Path, default=RRD_ROOT / INPUT_LAYER, help="An RRD, or a directory of them."
    )
    parser.add_argument("--out-dir", type=Path, default=RRD_ROOT, help="Root the <layer>/ folder is written under.")
    args = parser.parse_args()

    inputs = discover_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No .rrd at {args.input} — run `pixi run -e [env] convert-{INPUT_LAYER}` first.")
    print(f"Building {LAYER} layer for {len(inputs)} recording(s) -> {args.out_dir / LAYER}/")
    for input_rrd in inputs:
        out = write_layer(input_rrd, args.out_dir)
        if out is None:
            print(f"  {input_rrd.stem}: skipped — [what was missing]")
            continue
        print(f"  {out.stem}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
```

A layer that reads the source, such as the base layer, replaces `write_layer` with the function below.
Its docstring then names the source it reads, instead of saying that the source is never read.

```python
def convert_item(item: WorkItem, rrd_root: Path) -> Path | None:
    """Write one [episode]'s [layer] layer from the source, or None when the input is absent."""
    if not records_[thing](item.[main]):
        return None
    out_path = rrd_root / layer_relpath(LAYER, item.recording_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = [layer]_stream(item.[main]).collect(optimize=OptimizationProfile.OBJECT_STORE)
    store.write_rrd(str(out_path), application_id=APPLICATION_ID, recording_id=item.recording_id)
    return out_path
```

A layer built on an outside resource takes it as an argument, so the parse happens once per run:

```python
def load_resource() -> [Resource]:
    """Parse the vendored resource. One call per run serves every episode."""


def write_asset(resource: [Resource], out_dir: Path) -> Path:
    """Write the dataset-wide part once, under its own recording id."""


def write_layer(input_rrd: Path, resource: [Resource], out_dir: Path) -> Path | None:
    """As above, with the resource passed in rather than loaded per episode."""
```

`main` loads the resource, calls `write_asset` once, then passes the resource through the loop.
