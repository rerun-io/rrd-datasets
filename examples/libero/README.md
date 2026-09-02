# LIBERO

[LIBERO](https://libero-project.github.io/) is a lifelong-learning manipulation benchmark simulated in [robosuite](https://robosuite.ai/)-based environments: 130 tabletop tasks across five suites, each task shipping ~50 teleoperated demos as one HDF5 file.
This example converts each demo into four Rerun recordings (`.rrd`).
One recording corresponds to one layer: the demo itself, its metadata properties, the posed robot model, and the two cameras placed in the scene.

**Status: incubating.** Download, conversion, the urdf and cameras layers, the default blueprint, catalog registration, and the remote Modal conversion work; publishing the converted bucket is still to come.

Below is the viewer showing a converted demo with the default blueprint.

![LIBERO in the Rerun viewer](init_blueprint.png)

The [default blueprint](#3-view) puts the task instruction and the posed arm on the left, with both camera frustums in the scene, the two camera panes on the right, and the joint, gripper, action, and end-effector plots along the bottom.

> **Note:** this example uses Pixi. Get it [here](https://pixi.prefix.dev/latest/installation/).
> Everything runs inside the pixi env: prefix task commands with `pixi run`, and direct tool commands (`hf`, `rerun`, `modal`) with `pixi run -e libero`.
> File paths in the commands below are relative to the repository root.

## Dataset

- **Source**: [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) on Hugging Face
- **License**: Apache 2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on five sample task files (~3.4 GB), one per suite, all listed in [observations.md](observations.md#sample-files).
- **Access**: public, not gated.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

### Converted `.rrd` Dataset

Converted recordings will be published to a Hugging Face bucket when ready.
They are built from source revision [`f13aa24a`](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a), dated 2025-05-18, with rerun-sdk 0.37.0.

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
The viewer/catalog stacks them as **layers** of one logical recording: a base layer that carries the full demo, a properties layer that carries the catalog metadata, a urdf layer that carries the posed robot model, and a cameras layer that places the two cameras in the scene.
Each layer can be added, replaced, or re-run without touching the others.

Build every layer, for every downloaded file or one task file:

```bash
pixi run -e libero convert                    # every downloaded task file
pixi run -e libero convert <task.hdf5>        # one task file (~50 demos)
```

> **Note:** This example also includes its own task for each layer (`convert-base`, `convert-properties`, `convert-urdf`, `convert-cameras`) writing the corresponding `.rrd`.

### 3. View

Generate the default blueprint, then view a demo with it:

```bash
pixi run -e libero blueprint
pixi run -e libero rerun rrds/libero/*/libero_goal/turn_on_the_stove__demo_0.rrd blueprints/libero/default.rbl
```

> Keep the `*` to load all layers.
> To modify the layout, edit `libero/blueprint.py` and rerun the `blueprint` task.

### 4. Local Catalog

Register the converted demos to a [catalog server](https://rerun.io/docs/concepts/how-does-rerun-work#catalog-server), then browse, sort, filter, and query the segments as one dataset.
Once registered, demos become queryable segments with named layers.

Start a local server:

```bash
pixi run serve                # start an in-memory catalog server (leave running)
```

In another shell, register converted demos and the default blueprint to a local catalog:

```bash
pixi run -e libero register   # register all demos as the `libero` dataset
```

> **Note:** On a catalog, each demo becomes one segment, keyed by its `recording_id`. Each `.rrd` of that demo attaches as one named layer of the segment (`base`, `properties`, `urdf`, `cameras`).
> A layer name is an argument to the register call (`layer_name=` in `libero/catalog.py`).
> The `register` task creates the dataset, attaches each demo's RRDs as its named layers, and installs `blueprints/libero/default.rbl` as the default blueprint (generate it first with `pixi run -e libero blueprint`).

Browse them in the Rerun Viewer:

```sh
pixi run -e libero rerun rerun+http://127.0.0.1:51234
```

## Remote Convert Example on Modal

The steps above run locally on the sample files.
To convert the full dataset off-box, the [Modal](https://modal.com/) job under `libero/modal_jobs/` fans the 130 task files out across workers: each worker downloads one task file, converts every demo in it, and uploads the `.rrd` layers to a Hugging Face bucket.
It runs detached and returns immediately.
Watch progress in the Modal dashboard.

### 1. Prerequisite: storage backend

Converted RRDs land in a bucket, and `STORAGE_BACKEND` picks which kind.
This example converts to a [Hugging Face Storage Bucket](https://huggingface.co/docs/hub/main/en/storage-buckets-s3)
behind its S3-compatible gateway, reached with `boto3` like any S3 bucket.
The `libero` environments default `STORAGE_BACKEND` to `hf` and `HF_BUCKET` to `libero`, so the namespace that owns the bucket is the one value you have to set.
Access uses [HF S3 credentials](https://huggingface.co/docs/hub/storage-buckets-s3#generating-s3-credentials): an access key ID prefixed `HFAK…` and a secret access key.
Generate them from a fine-grained HF token scoped to the bucket.
The launcher passes them to the workers as an ephemeral per-run secret.

Set the env vars for your backend, or edit the placeholders in
[`rrd_datasets_common/storage.py`](../../packages/rrd_datasets_common/rrd_datasets_common/storage.py) (buckets) and
[`rrd_datasets_common/modal_jobs/store.py`](../../packages/rrd_datasets_common/rrd_datasets_common/modal_jobs/store.py) (role ARN).
The dataset's own layout under the bucket lives in [`storage.py`](libero/storage.py):

| Env var                                           | Backend | What it is                                                            |
| ------------------------------------------------- | ------- | --------------------------------------------------------------------- |
| `HF_NAMESPACE`                                    | hf      | The user or org that owns the bucket — set this one                   |
| `HF_BUCKET`                                       | hf      | Bucket the RRDs are written to, `libero` by default — create it first |
| `HF_BUCKET_ACCESS_KEY_ID` / `…_SECRET_ACCESS_KEY` | hf      | The HF S3 credentials                                                 |

`HF_NAMESPACE` has no default, and the bucket has to exist.

> **Note:** to store in an AWS S3 bucket instead, set `STORAGE_BACKEND=s3` and follow the
> [S3 prerequisite in the ABC-130k example](../abc-130k/README.md#1-prerequisite-s3-storage).

### 2. Prerequisite: Modal setup

- `pixi run -e libero modal setup` — authenticate the Modal CLI (one-time).
- `pixi run -e libero hf auth login`, or set `$HF_TOKEN` — optional for this public dataset, but anonymous callers share a smaller per-IP download quota.

### 3. Run Convert

Run `pixi run -e libero convert-on-modal --help` to see all options.

```bash
# One new task file (~50 demos), every layer (the default when no flags are given):
pixi run -e libero convert-on-modal

# Every task file (--limit 0 removes the cap):
pixi run -e libero convert-on-modal --limit 0

# Rebuild one suite:
pixi run -e libero convert-on-modal --path-filter libero_goal/ --limit 0 --overwrite

# See what would run, without spawning anything:
pixi run -e libero convert-on-modal --dry-run --limit 10
```

> **Note:** Without `--overwrite`, anything already in the bucket is skipped.
> The launcher spawns no worker for a task file whose demos all have every selected layer, assuming 50 demos per file ([observations.md](observations.md#source-hdf5-layout)).
> A task file missing even one `.rrd` still gets a worker, which downloads the file once and builds only what is missing.

#### Picking layers

`--layers` lets you choose which layers to build:

```bash
# Only the base layer:
pixi run -e libero convert-on-modal --layers base --limit 0

# Rebuild layers after changing them:
pixi run -e libero convert-on-modal --layers urdf,cameras --limit 0 --overwrite
```

Every layer reads the same task file, so a worker downloads it whatever the selection.

### 4. Upload the blueprint

`pixi run -e libero blueprint` writes `blueprints/libero/default.rbl`.
To upload it to your HF bucket (`s3://<bucket>/blueprints/`), run:

```bash
pixi run -e libero upload-blueprint
```

## Observations

We share our summary of the source dataset in [observations.md](observations.md).

## Mapping to Rerun

The base layer keeps the demo group as `Hdf5Reader` emits it: every dataset is a column named after itself, every attribute a static column, dtypes and array widths unchanged.
Two exceptions: the camera datasets are reshaped into upright `Image`s, and `states` and `init_state` become variable-length lists — their width is the scene's MuJoCo state size (47…110), and a catalog dataset needs one schema across every demo.
The table shows where each source item lands, and in which layer.
The datasets that repeat others are listed in [observations.md](observations.md#redundancies); they are kept, not plotted.

| Source                                                                                | Entity path                       | Component / archetype                       | Layer      | Notes                                                                           |
| ------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------- | ---------- | ------------------------------------------------------------------------------- |
| `actions`, `rewards`, `dones`, `states`, `robot_states`                               | `/demo`                           | one column per dataset                      | base       | `actions[7]` in −1…1, `rewards`/`dones` u8, `states[47…110]`, `robot_states[9]` |
| `obs/joint_states`, `obs/gripper_states`, `obs/ee_pos`, `obs/ee_ori`, `obs/ee_states` | `/demo/obs`                       | one column per dataset                      | base       | radians, metres and the raw rotation vector, as stored                          |
| `obs/agentview_rgb`, `obs/eye_in_hand_rgb`                                            | `/camera/{agentview,eye_in_hand}` | `Image`                                     | base       | flipped upright; the static `Image:format` carries the frame shape              |
| demo attrs                                                                            | `/demo/__hdf5_properties`         | static columns                              | base       | MuJoCo scene XML and initial state, enough to replay the demo                   |
| file attrs                                                                            | `/__hdf5_properties`              | static columns                              | base       | the two JSON attrs also parsed, as `problem_info:parsed` and `env_args:parsed`  |
| `problem_info.language_instruction`                                                   | `/task/instruction`               | `TextDocument`                              | base       | static, for the instruction pane                                                |
| file attrs, `num_samples`, filename                                                   | segment properties                | —                                           | properties | suite, scene, task language, num_samples, source file                           |
| `fer.urdf` meshes and fixed joints                                                    | `/urdf/fer/**`                    | `Asset3D`                                   | urdf       | static; the arm model, ~4 MB per recording                                      |
| `obs/joint_states`, `obs/gripper_states` → FK                                         | `/urdf/transforms`                | `Transform3D`                               | urdf       | one row per joint per step, named frames                                        |
| `model_file` `robot0_base` pose                                                       | `/urdf/world_from_base`           | `Transform3D`                               | urdf       | static; places the arm where the scene had it                                   |
| `model_file` `<camera>` elements                                                      | `/camera/{agentview,eye_in_hand}` | `Transform3D`, `Pinhole`, `CoordinateFrame` | cameras    | static; places the two images in the scene, see [below](#the-cameras-layer)     |

No `Scalars` are derived.
The default blueprint plots the arrays straight from their columns through component mappings and names the series there ([`libero/blueprint.py`](libero/blueprint.py)); a hand-made view shows the same arrays with index labels.

### Round trip test (HDF5 → RRD → HDF5)

Everything the source file holds is in the base layer, so a demo can be written back to HDF5.
A test ([`tests/test_base_layer.py`](tests/test_base_layer.py)) compares the result with the original, dataset by dataset and attribute by attribute, on the synthesized fixture and on a downloaded demo.
This is a value-level identity test, not a byte-level one.

### The urdf layer

The urdf layer poses the vendored Franka `fer` model ([`urdf/fer/`](urdf/fer/), provenance and regeneration recipe in [its README](urdf/fer/README.md)) with the base layer's joint columns.
The URDF carries no world position, so the model alone would render at the world origin; a static `Transform3D` from the demo's scene XML sets the arm's base pose in the world.
Forward kinematics to `fer_hand_tcp` reproduces the recorded `obs/ee_pos`, up to the fixed offset between Franka's tool center point and robosuite's grip site.

### The cameras layer

Each demo's MuJoCo XML lists its cameras with a pose and a vertical field of view.
`agentview` is fixed in the world; `robot0_eye_in_hand` moves with the arm.
The intrinsics follow [robosuite's camera utilities](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/camera_utils.py), and the `Pinhole` declares MuJoCo's `RUB` camera axes.

## Rerun APIs demonstrated

- [`Hdf5Reader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.Hdf5Reader) reads each demo group into chunk streams as-is, and the task file's attributes through a second stream over `/data` (`base_layer.py`).
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) turn the camera blobs into upright `Image` buffers and parse the JSON attributes into structs (`base_layer.py`).
- `rerun.urdf.UrdfTree` streams the robot model and solves forward kinematics from the joint columns, scattered into per-joint `Transform3D` rows (`urdf_layer.py`).
- `Pinhole`, `CoordinateFrame` and a static `Transform3D` on the image entities place the base layer's frames in 3D, `camera_xyz=RUB` matching MuJoCo's camera frame (`camera_layer.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout; component mappings plot the array columns without derived `Scalars` and carry the series labels (`blueprint.py`).

## References

- [Chunk processing API](https://rerun.io/docs/concepts/logging-and-ingestion/chunk-processing-api)
  — the reader + lens pipeline this conversion is built on.
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) — reshaping/deriving
  components in-stream.
