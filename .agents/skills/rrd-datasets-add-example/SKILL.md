---
name: rrd-datasets-add-example
description:
  "Convert a multimodal robotics dataset (MCAP, HDF5, LeRobot, Parquet,
  raw video, or similar formats) into layered Rerun recordings (.rrd)
  and a catalog-ready dataset, following the conventions and workflow
  of the [rerun-io/rrd-datasets](https://github.com/rerun-io/rrd-datasets) project. Use when adding a new dataset
  example to rrd-datasets, implementing its conversion and ingestion
  workflow, or creating a new dataset-conversion project modeled on
  rrd-datasets."
---

# RRD Datasets: Add Example

## Purpose

This skill adds a dataset example to [rerun-io/rrd-datasets](https://github.com/rerun-io/rrd-datasets).
The example converts the source into layered Rerun recordings (`.rrd`), with the repo's standard download, convert, view, and register tasks.
This skill covers the repo's conventions.
The `rerun-dataset-conversion` skill covers the conversion principles, and the other `rerun-*` skills cover the Rerun SDK.

Read `references/project-layout.md` and `references/readme-template.md` before starting.
Read each other reference at the step that names it.

Existing examples may predate the current practice, so follow this skill where they differ.
The Modal jobs and the append workflow are still changing.
Where a dataset needs something different, explain why in the module's docstring.

## Workflow

The agent implements each step and reports progress.
It leaves uncertain design decisions to the user.
The user provides design choices, validation, and domain-specific knowledge as needed.

### 1. Review the repository structure and existing rrd-datasets examples (agent)

Share the overall workflow with the user, and give a heads-up that this is a long process that will require multiple steps and user decisions.

### 2. Add download tooling (agent)

Two modules are written at this step: (1) the episode index, which turns the source's file listing into work items, and (2) `download.py`, which fetches a sample of them into `data/[Dataset Name]/` through `dataset_data_dir`.
Follow `references/episode-index.md` for the index module.
Follow `references/data-sources.md` for what to pin, cache, and avoid when listing and downloading.

Keep `download.py` sequential.
Parallel downloads belong to the Modal job in step 9, where each worker converts what it downloads.

- Name the sample [unit]s in `download.py`'s `SAMPLES`, each with its size in a comment, and keep the total to a few GB.
- Open the module docstring with what the dataset is, how big the sample is, and the `pixi run -e [env] download` line.

### 3. Inspect the source dataset (agent and user)

Read the `rerun-dataset-conversion` skill's "understand your data — expect the unexpected" section carefully before proceeding.

Write what the survey finds in `observations.md`, beside the example's README.
Its outline is in `references/observations-template.md`.

Once the user approves, put the most interesting and representative samples in `download.py`'s `SAMPLES`.

- Run the inspection inside the pixi env (`pixi run -e [env] …`), against the samples under `data/[Dataset Name]/`.
- Read the source's inventory from the episode index's cached listing rather than querying the source again.
- Record which streams or sidecars only some [episode]s have: that later decides where a layer module skips an [episode] instead of failing.

### 4. Design the Rerun representation and implement base conversion (agent and user)

Read the `rerun-dataset-conversion` skill's "conversion - base" section and suggest the conversion mapping.
It owns the mapping rules, the properties, the layer split, and the sign-off before any code is written.
The rest of this step is what the repo adds on top.

- Keep the entity paths and names the reader emits, such as `/__hdf5_properties`.
  Document them in the README instead of renaming them in the converter, and raise a request upstream when a name is wrong.
- Keep an episode's recording id identical across the episode index, local discovery, the id written into each `.rrd`, the file stem of every layer, and the catalog segment.
- Never round-trip a numeric Arrow column through Python objects.
  Flatten the buffer and reshape it, rather than calling `.tolist()` or `np.asarray` on the column.

### 5. Validate base conversion (agent and user)

Read the `rerun-dataset-conversion` skill's "conversion check" section, which owns the round-trip test and the source-versus-base size comparison.
Once it passes, its "initial blueprint" section covers giving the user something to inspect.

- Never compare `.rrd` bytes, since two writes of the same data differ.
  Rebuild the source message from the layer's columns and compare it against the source bytes.
- Write the tests as the "Tests" section of `references/project-layout.md` describes.

### 6. Add derived / augmented layers (agent and user)

Read the `rerun-dataset-conversion` skill's "enrich the data" step and its "splitting into layers" guideline, which decide what earns a layer of its own.
Follow `references/layer-module.md` for the module itself.
Build it on one machine first.
The same derivation runs later as an append job on Modal (step 9) once the dataset is registered and too large for one machine.

### 7. Add blueprint and visualizations (agent and user)

Read the `rerun-dataset-conversion` skill's "initial blueprint" step, then `rerun-blueprint` for the layout itself.

- Write the blueprint in `blueprint.py`, saved to `blueprints/[dataset]/default.rbl` through `default_blueprint_path`.
  The `blueprint` pixi task regenerates it, and `upload-blueprint` copies it to the bucket.
- Plot from the message structs through component mappings rather than materialising scalars into a layer.
- Map a repeated field once with a `[]` selector, such as `.data.motor_state[].q`.
  A mapping per index copies the whole struct per series per frame and collapses the frame rate.
- Expect a file-based view to show no shared asset.
  The asset can be viewed once the dataset is registered.
- Ask the user to inspect the result before moving on.

### 8. Add catalog registration (agent)

Read the `rerun-dataset-conversion` skill's "use the data" step and its `references/registering.md`, then `rerun-catalog-queries` for reading back.

- Follow `references/catalog-module.md` for `catalog.py`, which registers the layers, the shared asset, and the default blueprint in one run.
- Serve the catalog with `pixi run serve` first.

### 9. Add remote execution on Modal (agent and user)

Follow `references/modal-job.md`.
The conversion job builds the base layer from the source, and an append job adds derived layers to the registered dataset.
Test each job as its "Tests" section describes, then leave the full run (`--limit 0`) to the user.
Ask the user to test a small run.

### 10. Document the example (agent and user)

Most of the README exists by now, since each earlier step added its sections (see the table in `references/readme-template.md`).
Write the parts that depend on the finished example:

- the opening paragraph, which names every layer
- the status line, kept only while the bucket is unpublished
- the "Converted `.rrd` Dataset" section, with the bucket, the source revision, and the SDK version once the bucket is published
- the References section
- the example's row in the dataset table of the root README
