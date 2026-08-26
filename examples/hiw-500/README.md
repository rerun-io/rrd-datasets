# HIW-500

[BitRobot/HIW-500](https://huggingface.co/datasets/BitRobot/HIW-500) is an open humanoid teleoperation dataset: 23,000+ episodes of a **Unitree G1** with two **Dex1** grippers doing household tasks in 12 real homes, recorded as ROS 2 MCAP.
This example converts each episode into multiple Rerun recordings (`.rrd`). One recording corresponds to one layer: raw streams, optional data (IR camera), the robot model, its odometry, the camera geometry, and the episode metadata. Each can be built and rebuilt on its own.
Below is the viewer showing a converted episode with the default blueprint.

![HIW-500 in the Rerun viewer](screenshot.png)

The [default blueprint](#3-view) puts a 3D scene in the `odom` frame on the left with the subtask timeline beneath it, the cameras on the right, and joint, end-effector and gripper plots along the bottom.
The camera pane shows the head pair above the wrists, where an `RGB` and an `IR` tab switch between the two modalities of the same cameras.

There are two ways to run it.
The [local version](#local-runs) downloads four sample episodes, converts them, and registers them to a catalog you can query.
The [Modal](https://modal.com/)-based [remote version](#remote-convert-example-on-modal) converts the whole dataset into a storage bucket.

> **Note:** this example uses Pixi. Get it [here](https://pixi.prefix.dev/latest/installation/).
> Everything runs inside the pixi env: prefix task commands with `pixi run`, and direct tool commands (`hf`, `rerun`, `modal`) with `pixi run -e hiw`.
> File paths in the commands below are relative to the repository root.

> **Note:** this is an extended version of our previous repo, https://github.com/rerun-io/hiw-500_demo/. In this version, we include the [IR layer](#ir-layer) and a [remote cloud-based conversion example](#remote-convert-example-on-modal).

## Dataset

- **Source**: [BitRobot/HIW-500](https://huggingface.co/datasets/BitRobot/HIW-500) on Hugging Face
- **License**: CC BY 4.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on four sample episodes (~450 MB) spanning four tasks, all listed in [observations.md](observations.md#example-episodes). The Modal job supports the full dataset conversion with options to filter or limit episodes.
- **Access**: public, not gated.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

### Converted `.rrd` Dataset

Converted recordings will be published to a Hugging Face bucket when ready.
They are built from source revision [`2ca7ffcd`](https://huggingface.co/datasets/BitRobot/HIW-500/tree/2ca7ffcd85ec5212f81ae08491a4076bf48ea841), dated 2026-06-29, with rerun-sdk 0.36.1.

## Local Runs

File paths below are relative to the repository root.
The source data revision is pinned in this example.
Bump `HF_REVISION` in [`hiw_500/episode_index.py`](hiw_500/episode_index.py) to pick up newer episodes.

### 1. Download

Download the four sample episodes (~450 MB) into `data/HIW-500/`:

```bash
pixi run -e hiw download
```

To download different episodes, edit `SAMPLES` in [`hiw_500/download.py`](hiw_500/download.py).

The downloader caches a listing of the whole repo in `.cache/hf_files.json.gz`.
Building it takes a while on the first run, and it is rebuilt when the dataset revision changes.

### 2. Convert (MCAP → RRD)

Convert downloaded episode MCAP files into multiple Rerun recordings (`.rrd`) that share a `recording_id`.
The viewer/catalog stacks them as **layers** of one logical recording: a base layer that carries the raw source
plus other layers that augment it (typed archetypes derived from the messages, robot model, odometry, cameras, wrist IR, metadata properties).
Each layer can be added, replaced, or re-run without touching the others.

Build every layer, for the whole set or one episode:

```bash
pixi run -e hiw convert            # all episodes under data/HIW-500/
pixi run -e hiw convert <ep.mcap>  # a single episode
```

> **Note:** This example also includes its own task for each layer (`convert-base`, `convert-derived-archetypes`,
> `convert-urdf`, `convert-odom`, `convert-cameras`, `convert-ir`, `convert-properties`) writing the corresponding `.rrd`.

See [More about Layers](#more-about-layers) for what each layer carries.

### 3. View

View a result in the Rerun Viewer:

```bash
pixi run -e hiw rerun rrds/hiw-500/*/*.rrd        # every episode
pixi run -e hiw rerun rrds/hiw-500/*/<id>.rrd     # one episode
```

> Keep the `*` to load all layers.

Generate the default blueprint, then view an episode with it:

```bash
pixi run -e hiw blueprint
pixi run -e hiw rerun rrds/hiw-500/*/<id>.rrd blueprints/hiw-500/default.rbl
```

> **Notes:** only the left head eye appears in the 3D view.
> To modify the layout, edit `hiw_500/blueprint.py` and rerun the `blueprint` task.

### 4. Local Catalog

Register the converted episodes to a [catalog server](https://rerun.io/docs/concepts/how-does-rerun-work#catalog-server), then browse, sort, filter, and query the segments as one dataset.
Once registered, episodes become queryable segments with named layers.

Start a local server:

```bash
pixi run serve              # start an in-memory catalog server (leave running)
```

In another shell, register converted episodes and the default blueprint to a local catalog:

```bash
pixi run -e hiw register   # register all episodes as the `hiw_500` dataset
```

> **Note:** On a catalog, each episode becomes one segment, keyed by its `recording_id`. Each `.rrd` of that episode attaches as one named layer of the segment.
> A layer name is an argument to the register call (`layer_name=` in `hiw_500/catalog.py`).
> The `register` task creates the dataset, attaches each episode's RRDs as its named layers, and installs `blueprints/hiw-500/default.rbl` as the default blueprint (generate it first with `pixi run -e hiw blueprint`).

Browse them in the Rerun Viewer:

```sh
pixi run -e hiw rerun rerun+http://127.0.0.1:51234
```

The video below shows what it looks like.

https://github.com/user-attachments/assets/b4396d45-d4ba-4f30-9763-1aa9057c66f7

## Remote Convert Example on Modal

The steps above run locally.
To convert the full dataset off-box, the [Modal](https://modal.com/) job under `hiw_500/modal_jobs/` fans episodes out across workers: each worker downloads one episode, converts it, and uploads the `.rrd` files to a Hugging Face bucket.
It runs detached and returns immediately. Watch progress in the Modal dashboard.

### 1. Prerequisite: storage backend

Converted RRDs land in a bucket, and `STORAGE_BACKEND` picks which kind.
This example converts to a [HuggingFace Storage Bucket](https://huggingface.co/docs/hub/main/en/storage-buckets-s3)
behind its S3-compatible gateway, reached with `boto3` like any S3 bucket.
The `hiw` environments default `STORAGE_BACKEND` to `hf` and `HF_BUCKET` to `hiw-500`, so the namespace that owns the bucket is the one value you have to set.
Access uses [HF S3 credentials](https://huggingface.co/docs/hub/storage-buckets-s3#generating-s3-credentials): an access key ID prefixed `HFAK…` and a secret access key.
Generate them from a fine-grained HF token scoped to the bucket. These credentials are shipped to the Modal workers as an ephemeral per-run secret.

Set the env vars for your backend, or edit the placeholders in
[`rrd_datasets_common/storage.py`](../../packages/rrd_datasets_common/rrd_datasets_common/storage.py) (buckets) and
[`rrd_datasets_common/modal_jobs/store.py`](../../packages/rrd_datasets_common/rrd_datasets_common/modal_jobs/store.py) (role ARN);
the dataset's own layout under the bucket lives in [`storage.py`](hiw_500/storage.py):

| Env var                                           | Backend | What it is                                                             |
| ------------------------------------------------- | ------- | ---------------------------------------------------------------------- |
| `HF_NAMESPACE`                                    | hf      | The user or org that owns the bucket — set this one                    |
| `HF_BUCKET`                                       | hf      | Bucket the RRDs are written to, `hiw-500` by default — create it first |
| `HF_BUCKET_ACCESS_KEY_ID` / `…_SECRET_ACCESS_KEY` | hf      | The HF S3 credentials                                                  |

`HF_NAMESPACE` has no default, and the bucket has to exist.

> **Note:** to store in an AWS S3 bucket instead, set `STORAGE_BACKEND=s3` and follow the
> [S3 prerequisite in the ABC-130k example](../abc-130k/README.md#1-prerequisite-s3-storage).

### 2. Prerequisite: Modal setup

- `pixi run -e hiw modal setup` — authenticate the Modal CLI (one-time).
- `pixi run -e hiw hf auth login`, or set `$HF_TOKEN` — not required since the dataset is public, but it raises the download quota.
  (Anonymous callers share a smaller per-IP quota.)

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

> **Note:** Without `--overwrite`, anything already in the bucket is skipped.
> The launcher spawns no worker for an episode whose layers are all present.
> An episode missing even one layer still gets a worker, which then builds only what is missing.

#### Picking layers

`--layers` lets you choose which layers to build:

```bash
# Only the base layer:
pixi run -e hiw convert-on-modal --layers base --limit 0

# Rebuild layers after changing them:
pixi run -e hiw convert-on-modal --layers cameras,properties --limit 0 --overwrite
```

A worker downloads only what the selected layers read.

### 4. Upload the blueprint

`pixi run -e hiw blueprint` writes `blueprints/hiw-500/default.rbl`.
To upload it to your HF bucket (`s3://<bucket>/blueprints/`), run:

```bash
pixi run -e hiw upload-blueprint
```

## Observations

We share our observations including useful details beyond the dataset card.
See [observations.md](observations.md) for the full survey.

## Mapping to Rerun

The table below shows what each source becomes in the recording, whether the base layer carries it or which layer adds it, and which view of the default blueprint shows it.
Every topic lands at its own path unless a row says otherwise.
`<side>` is `left` or `right`.
Every message stays whole: a custom `homies/*` or `unitree_go/*` message is one struct column named after its schema, with every field, and the blueprint picks its series out of those structs.

| Source                                    | Archetype or component                                                                                                           | In base              | Shown in                              |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------- |
| `/camera/head/image/compressed`           | `EncodedImage`, `CompressedImage:{header,format}`; the eyes split to `/camera/head/{left,right}`                                 | ✅                   | Scene, Head L, Head R                 |
| `/camera/<side>_wrist/image/compressed`   | `EncodedImage`, `CompressedImage:{header,format}`                                                                                | ✅                   | Wrist L, Wrist R                      |
| `/stamped/lowstate`                       | `homies.msg.LowStateStamped:message`                                                                                             | ✅                   | Joints                                |
| `/stamped/lowcmd`                         | `homies.msg.LowCmdStamped:message`                                                                                               | ✅                   | —                                     |
| `/stamped/secondary_imu`                  | `homies.msg.IMUStateStamped:message`                                                                                             | ✅                   | —                                     |
| `/stamped/dex1/<side>/{cmd,state}`        | `homies.msg.Motor{Cmd,State}Stamped:message`                                                                                     | ✅                   | Gripper(Dex1)                         |
| `/lf/odommodestate`                       | `unitree_go.msg.SportModeState:message`                                                                                          | ✅                   | —                                     |
| `/wbc_lerobot`                            | `TextDocument`                                                                                                                   | ✅                   | —                                     |
| `/annotation`                             | `TextDocument`                                                                                                                   | ✅                   | —                                     |
| every topic                               | `McapSchema`, `McapChannel` (static)                                                                                             | ✅                   | —                                     |
| MCAP metadata, statistics, header         | `rosbag2`, `McapStatistics`, `profile`/`library`/`compression` at `/__mcap_metadata`, `/__mcap_properties`, `/__properties/mcap` | ✅                   | —                                     |
| `info.json` sidecar                       | `AnyValues` at `/episode`, `StateChange` at `/task/subtask`                                                                      | ✅                   | Subtasks                              |
| `calibration/` sidecars                   | `CalibrationFile` at `/calibration/…`                                                                                            | ✅                   | —                                     |
| `/camera/<side>_wrist/ir{1,2}/compressed` | `EncodedImage`, `CompressedImage:{header,format}`                                                                                | `ir`                 | IR tab                                |
| `/wbc_lerobot`                            | `wbc_lerobot:message`; `Transform3D` at `/lerobot/{ee_state,ee_action}/<side>`                                                   | `derived_archetypes` | End-effector, Gripper(LeRobot), Scene |
| `/stamped/lowstate`, URDF                 | `Asset3D`, `Transform3D` at `/robot/**`                                                                                          | `urdf`               | Scene                                 |
| `/lf/odommodestate`                       | `Transform3D` at `/odom/pelvis`                                                                                                  | `odom`               | Scene                                 |
| head calibration                          | `Pinhole`, `Transform3D` on `/camera/head/{left,right}`                                                                          | `cameras`            | Scene                                 |

### Also in the recording

The blueprint plots the joint `q`/`dq`/`tau_est`, the end-effector arrays, the dex1 jaw angles and the teleop gripper inputs.
Every other field sits in the same structs, ready to plot: add the entity to a time series view, or map a series onto a field the way `blueprint.py` does.

- **Motor health** — `data.motor_state[i]` in `/stamped/lowstate`.
- **Commands and gains** — `data.motor_cmd[i]` in `/stamped/lowcmd`; `data.cmds[0]` in the dex1 command topics.
- **Gripper rates** — `data.states[0]` in the dex1 state topics.
- **IMUs** — `data.imu_state` in `/stamped/lowstate`; `/stamped/secondary_imu`.
- **Base motion** — `position`, `velocity`, `body_height`, `yaw_speed` and the `foot_*` arrays in `/lf/odommodestate`.
- **Teleop pivot** — `pivot[0…6]` in the `/wbc_lerobot` struct.

## More about Layers

Each layer is a separate module that writes its own .rrd.

### Base layer

`hiw_500/base_layer.py` — Everything in the MCAP, as recorded.
Each decoded message stays one struct, the file's own records (schemas, channel QoS, metadata, statistics) come along, and only the wrist IR streams go to their own layer.
The stereo head image is split into its two eyes; the original stays too.
Beside the MCAP it logs the sidecars: `info.json`, the calibration files as `CalibrationFile` components, and the joint labels.
A census compares the decoded rows with the MCAP summary; a channel that lost messages is flagged (`has_undecodable`, `undecodable_topics`) and its raw bytes are kept.

### Derived archetypes layer

`hiw_500/derived_archetypes_layer.py` — This layer holds what the viewer needs typed and the raw messages do not give it: the `/wbc_lerobot` JSON parsed into one struct per message, for the blueprint to plot, and the four end-effector positions as `Transform3D` markers for the 3D scene.
Episodes without the topic skip this layer.

### URDF layer

`hiw_500/urdf_layer.py` — This layer carries the animated Unitree G1 mesh, driven by forward kinematics from the joint positions.

The URDF is **not** part of the HF dataset. We use `g1_29dof_mode_15_with_dex1_1.urdf` from
Unitree's [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description)
(the Dex1 variant), vendored with its meshes under `urdf/g1/`.
Unitree distributes it under the [BSD-3-Clause license](https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE); a copy is included at [`urdf/g1/LICENSE`](urdf/g1/LICENSE).

### Odometry layer

`hiw_500/odom_layer.py` — This layer connects the robot to the world by adding the `odom → pelvis`
transform so the whole robot moves through the scene.

### Camera layer

`hiw_500/camera_layer.py` — This layer places the head camera in 3D and adds the optical-frame
transform. Episodes without that calibration skip this layer.

### IR layer

`hiw_500/ir_layer.py` — This layer includes the four wrist IR streams only for episodes that recorded IR, with the same `header` and `format` columns and schema rows as the colour cameras in the base layer.
The default blueprint shows them under the `IR` tab of the wrist camera pane, beside the `RGB` tab carrying the colour streams.

### Properties layer

`hiw_500/properties_layer.py` — This layer adds per-episode metadata logged as recording properties, which the
catalog shows as columns to filter, sort, and search on: `task`,
`duration_sec`, `num_subtasks`, `subtask_labels`, `scene`, `has_ir`, `robot`.

## Rerun APIs demonstrated

- [`McapReader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.McapReader) decodes the episode topics into chunk streams — the custom `homies/*` messages by reflection into structs, the cameras and text topics into archetypes — and its summary drives the channel census (`base_layer.py`).
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) turn the raw messages into what the viewer needs typed: `EncodedImage` halves for the split stereo head, `Transform3D` for the base pose and the end-effector markers, a struct from the `/wbc_lerobot` JSON (`base_layer.py`, `odom_layer.py`, `derived_archetypes_layer.py`).
- [Component mappings](https://rerun.io/docs/howto/visualization/plot-any-scalar) plot the joint, gripper and end-effector series straight out of the message structs, one `VisualizerComponentMapping` per series, so no `Scalars` are materialised (`blueprint.py`).
- `rerun.urdf.UrdfTree` loads the vendored G1 model and runs forward kinematics from the joint states (`urdf_layer.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout, including the frame-targeted 3D view and the struct-mapped series (`blueprint.py`).
- [`CatalogClient`](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model) registers each episode as a dataset segment with named layers and installs the default blueprint (`catalog.py`).

## References

- [Rerun Learn](https://rerun.io/learn) — a hands-on course covering a similar example.
- [Chunk processing API](https://rerun.io/docs/concepts/logging-and-ingestion/chunk-processing-api)
  — the reader + lens pipeline this conversion is built on.
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) — reshaping/deriving
  components in-stream.
- [`robot_data_preprocessing` example](https://github.com/rerun-io/rerun/tree/main/examples/python/robot_data_preprocessing)
  — end-to-end MCAP + URDF + sidecar pipeline.
