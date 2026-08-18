# HIW-500

[BitRobot/HIW-500](https://huggingface.co/datasets/BitRobot/HIW-500) is an open humanoid teleoperation dataset: 23,000+ episodes of a **Unitree G1** with two **Dex1** grippers doing household tasks in 12 real homes, recorded as ROS 2 MCAP.
This example converts each episode into multiple Rerun recordings (`.rrd`). One recording corresponds to one layer: raw streams, optional data (IR camera), the robot model, its odometry, the camera geometry, and the episode metadata. Each can be built and rebuilt on its own.
Below is the viewer showing a converted episode with the default blueprint.

![HIW-500 in the Rerun viewer](screenshot.png)

The [default blueprint](#3-view) puts a 3D scene in the `odom` frame on the left with the subtask timeline beneath it, the cameras on the right, and joint and end-effector plots along the bottom.
The camera pane shows the head pair above the wrists, where an `RGB` and an `IR` tab switch between the two modalities of the same cameras.

There are two ways to run it.
The [local version](#local-runs) downloads four sample episodes, converts them, and registers them to a catalog you can query.
The [Modal](https://modal.com/)-based [remote version](#remote-convert-example-on-modal) converts the whole dataset into a storage bucket.

> **Note:** this example uses Pixi. Get it [here](https://pixi.prefix.dev/latest/installation/).
> Everything runs inside the pixi env: prefix task commands with `pixi run`, and direct tool commands (`hf`, `rerun`, `modal`) with `pixi run -e hiw`.
> File paths in the commands below are relative to the repository root.

## Dataset

- **Source**: [BitRobot/HIW-500](https://huggingface.co/datasets/BitRobot/HIW-500) on Hugging Face
- **License**: CC BY 4.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on four sample episodes (~450 MB) spanning four tasks (TODO add observations.md). Three carry the full sensor set; the fourth predates the wrist IR cameras and the calibration sidecars, so it exercises the skip paths. The Modal job converts the full dataset.
- **Access**: public, no gating. A Hugging Face login only raises the download rate limit.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

A survey of the source episodes lists every source topic with its rate and where the converter puts it, and records what varies across episodes:
(TODO add observations.md)

### Converted `.rrd` Dataset

Converted recordings will be published to a Hugging Face bucket when ready.

## Local Runs

### 1. Download

Download the four sample episodes (~450 MB) into `data/HIW-500/`:

```bash
pixi run -e hiw download
```

To download different episodes, edit `SAMPLES` in [`hiw_500/download.py`](hiw_500/download.py).

The downloader picks the sample files out of the repo's file listing, made once and cached against the revision.
Avoid using `hf download --include` for this dataset: the Hugging Face CLI enumerates the entire repository (~106,000 entries) on every call before applying the pattern, which can look like a hang.

### 2. Convert (MCAP → RRD)

Convert downloaded episode MCAP files into multiple Rerun recordings (`.rrd`) that share a `recording_id`.
The viewer/catalog stacks them as **layers** of one logical recording: a base layer that carries faithful raw streams
plus other layers that augment the base recording (robot model, odometry, cameras, wrist IR, metadata properties).
Each layer can be added, replaced, or re-run without touching the others.

Build every layer for every episode in one command:

```bash
pixi run -e hiw convert            # all episodes under data/HIW-500/
pixi run -e hiw convert <ep.mcap>  # a single episode
```

> **Note:** This example also includes its own task for each layer (`convert-base`, `convert-urdf`, `convert-odom`,
> `convert-cameras`, `convert-ir`, `convert-properties`) writing the corresponding `.rrd`.

What each layer holds and where its entities land is in [More about Layers](#more-about-layers).

### 3. View

An episode has up to six layers, and the `cameras` and `ir` layers exist only where the episode carries their inputs.

Open the recordings straight in the viewer.
Matching `recording_id`s merge into one recording per episode, so all layers line up:

```bash
pixi run -e hiw rerun rrds/hiw-500/*/*.rrd        # every episode
pixi run -e hiw rerun rrds/hiw-500/*/<id>.rrd     # one episode's layers
```

Good for a quick look. No server, no registration.

> Keep the `*` to load all layers. Opening only `base/<id>.rrd` gives you
> the base layer alone, and the 3D scene comes up empty.

View an episode with the default blueprint:

```bash
pixi run -e hiw rerun rrds/hiw-500/*/<id>.rrd blueprints/hiw-500/default.rbl
```

> **Notes:** only the left head eye appears in the 3D view. The right one stays in its own 2D pane, where it does not overlap the scene.
> Regenerate the blueprint with `pixi run -e hiw blueprint`, which overwrites the committed
> [`blueprints/hiw-500/default.rbl`](../../blueprints/hiw-500/default.rbl).

### 4. Local Catalog

Register the converted episodes to a [catalog server](https://rerun.io/docs/concepts/how-does-rerun-work#catalog-server), then browse, sort, filter, and query the segments as one dataset.
Once registered, episodes become queryable segments with named layers.

```bash
pixi run serve             # start the in-memory Rerun catalog on :51234 (leave running)
pixi run -e hiw register   # in another shell: register all episodes as the `hiw_500` dataset
```

> **Note:** each episode becomes one segment, keyed by its `recording_id`, and each `.rrd` of that episode attaches as one named layer of the segment.
> A layer name is an argument to the register call (`layer_name=` in `hiw_500/catalog.py`).
> The `register` task creates the dataset, attaches each episode's RRDs as its named layers paired by `recording_id`, and installs [`blueprints/hiw-500/default.rbl`](../../blueprints/hiw-500/default.rbl) as the default blueprint.

Browse them in the Rerun Viewer:

```sh
pixi run -e hiw rerun rerun+http://127.0.0.1:51234
```

https://github.com/user-attachments/assets/5180a716-9ceb-4209-a795-626053d59ba5

## Remote Convert Example on Modal

The steps above run locally.
To convert the full dataset off-box, the [Modal](https://modal.com/) job under `hiw_500/modal_jobs/` fans episodes out across workers: each worker downloads one episode, converts it, and uploads the `.rrd` files to a Hugging Face bucket.
It runs detached and returns immediately. Watch progress in the Modal dashboard.

### 1. Prerequisite: storage backend

Converted RRDs land in a bucket, and `STORAGE_BACKEND` picks which kind.
This example converts to a [HuggingFace Storage Bucket](https://huggingface.co/docs/hub/main/en/storage-buckets-s3)
behind its S3-compatible gateway, reached with `boto3` like any S3 bucket.
The `hiw` environments default `STORAGE_BACKEND` to `hf` and `HF_BUCKET` to `hiw-500`, so the namespace that owns the bucket is the one value you have to set.
Access uses an HFAK key pair, generated once from a fine-grained HF token scoped to the bucket and shipped to the workers as an ephemeral per-run secret.

Set the env vars for your backend, or edit the placeholders in
[`rrd_datasets_common/storage.py`](../../packages/rrd_datasets_common/rrd_datasets_common/storage.py) (buckets) and
[`rrd_datasets_common/modal_jobs/store.py`](../../packages/rrd_datasets_common/rrd_datasets_common/modal_jobs/store.py) (role ARN);
the dataset's own layout under the bucket lives in [`storage.py`](hiw_500/storage.py):

| Env var                                                  | Backend | What it is                                                             |
| -------------------------------------------------------- | ------- | ---------------------------------------------------------------------- |
| `HF_NAMESPACE`                                           | hf      | The user or org that owns the bucket — set this one                    |
| `HF_BUCKET`                                              | hf      | Bucket the RRDs are written to, `hiw-500` by default — create it first |
| `RCLONE_CONFIG_HF_ACCESS_KEY_ID` / `…_SECRET_ACCESS_KEY` | hf      | The HFAK key pair (the names double as an `rclone` remote config)      |

`HF_NAMESPACE` has no default, and the bucket has to exist: `hf buckets create <namespace>/hiw-500 --private`.

> **Note:** to store in an AWS S3 bucket instead, set `STORAGE_BACKEND=s3` and follow the
> [S3 prerequisite in the ABC-130k example](../abc-130k/README.md#1-prerequisite-s3-storage).

### 2. Prerequisite: Modal setup

- `pixi run -e hiw modal setup` — authenticate the Modal CLI (one-time).
- HuggingFace login (`pixi run -e hiw hf auth login`, or set `$HF_TOKEN`). The dataset is public, so
  this is about rate limits rather than access: anonymous callers share a small per-IP quota,
  which a wide fan-out exhausts quickly. The token ships as an ephemeral per-run secret, so
  there is nothing stored on Modal to refresh.

Discovering which episodes exist means listing the whole HuggingFace tree (~106,000 files). That
list is cached in `.cache/hf_files.json.gz` and pinned to the dataset's commit sha, so only the
first launch per revision pays for it. Delete the file to force a re-listing.

### 3. Run Convert

Run `pixi run -e hiw convert-on-modal --help` for the full flag reference.

```bash
# One new episode, every layer it can have (the default when no flags are given):
pixi run -e hiw convert-on-modal

# Every episode (--limit 0 removes the cap):
pixi run -e hiw convert-on-modal --limit 0

# Rebuild the first 5 episodes of one task:
pixi run -e hiw convert-on-modal --task-filter Sweep-Floor --limit 5 --overwrite

# See what would run, without spawning anything:
pixi run -e hiw convert-on-modal --dry-run --limit 10
```

Unless `--overwrite`, layers already in the bucket are dropped **before** any worker is spawned,
so re-runs and incremental backfills don't pay container time for work already done. An episode
missing even one selected layer is still spawned, and the worker rebuilds only what is absent,
so an upload cut short finishes on the next run.

#### Picking layers

`--layers` narrows the build to a subset, which is the cheap way to add or re-do one derived
view across the dataset:

```bash
# Only the base layer:
pixi run -e hiw convert-on-modal --layers base --limit 0

# Rebuild two derived layers after changing them:
pixi run -e hiw convert-on-modal --layers cameras,properties --limit 0 --overwrite
```

A worker downloads only what the selected layers read. The camera layer needs just the episode's
head calibration yaml and the properties layer only its sidecars, so neither pulls the episode
MCAP, which runs to hundreds of MB. Rebuilding those two across the dataset costs almost nothing.

## Observations

We share interesting observations on a subset of episodes, completing the official dataset card:
(TODO add observations.md)

## More about Layers

One module per layer, each writing its own `.rrd`.

### Base layer

`hiw_500/base_layer.py` — a faithful conversion of the raw streams, no kinematics. A single
`McapReader` stream shaped by lenses: cameras decode to `EncodedImage` (the stereo head is split
into `/camera/head/{left,right}`), the custom `homies/*` / `unitree_go/*` messages become
per-joint and end-effector scalars/transforms (`/state/…`, `/cmd/…`, `/lerobot/…`), and
`info.json` becomes the `/episode` + `/task/subtask` sidecar. Entities keep the MCAP-native
timelines.
The episode's `calibration/` files ride along verbatim under `/calibration/…`, so the RRD is a self-contained record of the episode.
A channel census compares decoded rows against the MCAP's own message counts and stamps `has_undecodable` / `undecodable_topics` on `/episode`, since a message that fails to decode is dropped silently.

### URDF layer

`hiw_500/urdf_layer.py` — the animated Unitree G1 mesh, driven by forward kinematics from the
`/stamped/lowstate` joint positions (emitted to `/robot/transforms`). Its 29 revolute joints
match the documented Unitree motor order. (The Dex1 finger joints stay at rest: no 1:1 mapping
in the URDF.)

The URDF is **not** part of the HF dataset. We use `g1_29dof_mode_15_with_dex1_1.urdf` from
Unitree's [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description)
(the Dex1 variant), vendored with its meshes under `urdf/g1/`.

### Odometry layer

`hiw_500/odom_layer.py` — connects the robot to the world. The URDF layer roots the G1 at
`pelvis`, so on its own it animates in place; this layer adds the time-varying `odom → pelvis`
transform from `/lf/odommodestate` so the whole robot moves through the scene.

### Camera layer

`hiw_500/camera_layer.py` — places the head camera in 3D. The head camera mount is a fixed
joint in the URDF, so its position is already known; this layer adds the optical-frame
transform plus per-eye `Pinhole` intrinsics from the episode's own stereo calibration
(`calibration/params/head_camera_params.yaml`) so both stereo eyes project onto their image
planes. Episodes without that calibration (the older sessions) skip this layer and keep their
head images 2D. The **wrist cameras stay 2D**: the dataset provides no camera→robot
(hand-eye) calibration for them.

### Wrist IR layer

`hiw_500/ir_layer.py` — the four wrist infrared streams (`/camera/{left,right}_wrist/ir{1,2}/compressed`), passed through as JPEG images on their own topic paths.
The base layer leaves them out so its size does not double for imagery most workflows never look at.
They stay 2D like the wrist color images, and sessions recorded before the IR cameras reached the rig skip this layer.
The default blueprint shows them under the `IR` tab of the wrist camera pane, beside the `RGB` tab carrying the colour streams.

### Properties layer

`hiw_500/properties_layer.py` — per-episode metadata logged as recording properties, which the
catalog surfaces as **columns** to filter, sort, and search on: `task`, `task_group`,
`duration_sec`, `num_subtasks`, `subtask_labels`, `scene`, `has_ir`, `robot`. Values come from
`info.json`, the calibration sidecars, the dataset path, and a constant. `scene` is `-1` when the
episode names none, and `has_ir` reads the presence of the wrist calibrations, which arrived on
the rig together with the IR streams.

## Rerun APIs demonstrated

- [`McapReader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.McapReader) decodes the episode topics, including the custom `homies/*` messages, into chunk streams (`base_layer.py`).
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) turn the raw messages into typed components: `Scalars` for the joint signals, `Transform3D` for the base pose, `EncodedImage` for the split stereo head (`base_layer.py`).
- `rerun.urdf.UrdfTree` loads the vendored G1 model and runs forward kinematics from the joint states (`urdf_layer.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout, including the frame-targeted 3D view (`blueprint.py`).
- [`CatalogClient`](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model) registers each episode as a dataset segment with named layers and installs the default blueprint (`catalog.py`).

## References

- [Rerun Learn](https://rerun.io/learn) — a hands-on course covering a similar example.
- [Chunk processing API](https://rerun.io/docs/concepts/logging-and-ingestion/chunk-processing-api)
  — the reader + lens pipeline this conversion is built on.
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) — reshaping/deriving
  components in-stream.
- [`robot_data_preprocessing` example](https://github.com/rerun-io/rerun/tree/main/examples/python/robot_data_preprocessing)
  — end-to-end MCAP + URDF + sidecar pipeline.
