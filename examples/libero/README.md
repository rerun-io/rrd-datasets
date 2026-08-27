# LIBERO

[LIBERO](https://libero-project.github.io/) is a lifelong-learning manipulation benchmark simulated in [robosuite](https://robosuite.ai/)-based environments: 130 tabletop tasks across five suites, each task shipping ~50 teleoperated demos as one HDF5 file.
This example converts each demo into two Rerun recordings (`.rrd`).
One recording corresponds to one layer: the demo itself, and its metadata properties.

**Status: incubating.** Download, conversion, and the default blueprint work; the urdf layer (URDFs are available from Franka homepage), catalog registration, and the remote conversion are still to come.

Below is the viewer showing a converted demo with the default blueprint.

![LIBERO in the Rerun viewer](init_blueprint.png)

The [default blueprint](#3-view) puts the two camera panes on the left with the task instruction above them, and the joint, gripper, action, and end-effector plots on the right.

> **Note:** this example uses Pixi. Get it [here](https://pixi.prefix.dev/latest/installation/).
> Everything runs inside the pixi env: prefix task commands with `pixi run`, and direct tool commands (`hf`, `rerun`) with `pixi run -e libero`.
> File paths in the commands below are relative to the repository root.

## Dataset

- **Source**: [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) on Hugging Face
- **License**: Apache 2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on five sample task files (~3.4 GB), one per suite, all listed in [observations.md](observations.md#example-files).
- **Access**: public, not gated.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

### Converted `.rrd` Dataset

Converted recordings will be published to a Hugging Face bucket when ready.
They are built from source revision [`f13aa24a`](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a), dated 2025-05-18, with rerun-sdk 0.36.1.

## Local Runs

The source data revision is pinned in this example.
Bump `HF_REVISION` in [`libero/episodes.py`](libero/episodes.py) to pick up dataset changes.

### 1. Download

Download the five sample demo files (~3.4 GB, one per suite) into `data/LIBERO/<suite>/`:

```bash
pixi run -e libero download
```

To download different files, edit `SAMPLES` in [`libero/download.py`](libero/download.py).

### 2. Convert (HDF5 → RRD)

Convert downloaded task files into per-demo Rerun recordings (`.rrd`) that share a `recording_id`.
The viewer/catalog stacks them as **layers** of one logical recording: a base layer that carries the full demo, plus a properties layer that carries the catalog metadata.
Each layer can be added, replaced, or re-run without touching the other.

Build both layers, for every downloaded file or one task file:

```bash
pixi run -e libero convert                    # every downloaded task file
pixi run -e libero convert <task.hdf5>        # one task file (~50 demos)
```

> **Note:** This example also includes its own task for each layer (`convert-base`, `convert-properties`) writing the corresponding `.rrd`.

### 3. View

Generate the default blueprint, then view a demo with it:

```bash
pixi run -e libero blueprint
pixi run -e libero rerun rrds/libero/*/libero_goal/turn_on_the_stove/demo_0.rrd blueprints/libero/default.rbl
```

> Keep the `*` to load all layers.
> To modify the layout, edit `libero/blueprint.py` and rerun the `blueprint` task.

### 4. Local Catalog

_Not built yet — this will register the converted demos as a `libero` dataset with named layers and install the default blueprint._

## Remote Convert Example on Modal

_Not built yet — this will fan the 130 task files out across [Modal](https://modal.com/) workers and upload the `.rrd` layers to the Hugging Face bucket._

## Observations

We share our survey on the source dataset in [observations.md](observations.md).

## Mapping to Rerun

The base layer keeps the demo group as `Hdf5Reader` emits it: every dataset is a column named after itself, every attribute a static column, dtypes and array widths unchanged.
Only the two camera datasets are reshaped, into upright `Image`s.
The table shows where each source item lands, and in which layer.
The datasets that repeat others are listed in [observations.md](observations.md#redundancies); they are kept, not plotted.

| Source                                                                                | Entity path                       | Component / archetype  | Layer      | Notes                                                                           |
| ------------------------------------------------------------------------------------- | --------------------------------- | ---------------------- | ---------- | ------------------------------------------------------------------------------- |
| `actions`, `rewards`, `dones`, `states`, `robot_states`                               | `/demo`                           | one column per dataset | base       | `actions[7]` in −1…1, `rewards`/`dones` u8, `states[47…110]`, `robot_states[9]` |
| `obs/joint_states`, `obs/gripper_states`, `obs/ee_pos`, `obs/ee_ori`, `obs/ee_states` | `/demo/obs`                       | one column per dataset | base       | radians, metres and the raw rotation vector, as stored                          |
| `obs/agentview_rgb`, `obs/eye_in_hand_rgb`                                            | `/camera/{agentview,eye_in_hand}` | `Image`                | base       | flipped upright; the static `Image:format` carries the frame shape              |
| demo attrs `model_file`, `init_state`, `num_samples`                                  | `/demo/__hdf5_properties`         | static columns         | base       | MuJoCo scene XML and initial state, enough to replay the demo                   |
| file attrs `problem_info`, `env_args`, `bddl_file_name`, `env_name`, …                | `/__hdf5_properties`              | static columns         | base       | the two JSON attrs also parsed, as `problem_info:parsed` and `env_args:parsed`  |
| `problem_info.language_instruction`                                                   | `/task/instruction`               | `TextDocument`         | base       | static, for the instruction pane                                                |
| file attrs, `num_samples`, filename                                                   | segment properties                | —                      | properties | suite, scene, task language, num_samples, source file                           |

No `Scalars` are derived.
The default blueprint plots the arrays straight from their columns through component mappings and names the series there ([`libero/blueprint.py`](libero/blueprint.py)); a hand-made view shows the same arrays with index labels.

### Round trip test (HDF5 → RRD → HDF5)

Everything the source file holds is in the base layer, so a demo can be written back to HDF5.
A test ([`tests/test_base_layer.py`](tests/test_base_layer.py)) compares the result with the original, dataset by dataset and attribute by attribute, on the synthesized fixture and on a downloaded demo.
This is a value-level identity test, not a byte-level one.

## Rerun APIs demonstrated

- [`Hdf5Reader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.Hdf5Reader) reads each demo group into chunk streams as-is, and the task file's attributes through a second stream over `/data` (`base_layer.py`).
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) turn the camera blobs into upright `Image` buffers and parse the JSON attributes into structs (`base_layer.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout; component mappings plot the array columns without derived `Scalars` and carry the series labels (`blueprint.py`).

## References

- [Chunk processing API](https://rerun.io/docs/concepts/logging-and-ingestion/chunk-processing-api)
  — the reader + lens pipeline this conversion is built on.
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) — reshaping/deriving
  components in-stream.
