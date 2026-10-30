# MolmoAct2-BimanualYAM

[MolmoAct2-BimanualYAM](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) is a bimanual manipulation dataset recorded on two I2RT YAM arms: 32,246 teleoperated episodes (about 704 hours) with three cameras, in the LeRobot v3 format.
This example converts each episode into four Rerun recordings (`.rrd`).
One recording corresponds to one layer: the episode as recorded, the two posed arms, the three cameras placed in the scene, and the source's statistics of the episode.

**Status: incubating.** The conversion works end to end, but the converted dataset is not published yet.

Below is the viewer showing a converted episode with the default blueprint.

![MolmoAct2 in the Rerun viewer](screenshot.png)

The [default blueprint](#3-view) puts the episode's instruction at the top, the posed arms next to the top and wrist cameras below it, and each arm's state and action plots, as two tabs, along the bottom.

There are two ways to run it.
The [local version](#local-runs) downloads three sample file groups, converts them, and registers them to a catalog you can query.
The [Modal](https://modal.com/)-based [remote version](#remote-convert-example-on-modal) converts the whole dataset into a storage bucket.

> [!Note]
> This example uses Pixi.
> Get it [here](https://pixi.prefix.dev/latest/installation/).
> Everything runs inside the pixi env: prefix task commands with `pixi run`, and direct tool commands (`hf`, `rerun`, `modal`) with `pixi run -e molmo`.
> File paths in the commands below are relative to the repository root.

## Dataset

- **Source**: [allenai/MolmoAct2-BimanualYAM-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) on Hugging Face
- **License**: Apache 2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on three sample file groups (~1.1 GB, 19 episodes) spanning three tasks, all listed in [observations.md](observations.md#surveyed-file-groups).
  The Modal job supports the full dataset conversion with options to filter or limit file groups.
- **Access**: public, not gated.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

### Converted `.rrd` Dataset

Converted recordings are published at [`rerun/molmoact2`](https://huggingface.co/buckets/rerun/molmoact2) — download them directly if you only want the `.rrd` data.
They are built from source revision [`e9f21ae1`](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/tree/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c), dated 2026-06-11, with rerun-sdk 0.38.1.

## Local Runs

The source data revision is pinned in this example.
Bump `HF_REVISION` in [`molmoact2/episode_index.py`](molmoact2/episode_index.py) to pick up newer episodes.

### 1. Download

Download the dataset's `meta/` folder (72 MB) and three sample file groups (~1.1 GB, 19 episodes) into `data/MolmoAct2/`.
The dataset stores many episodes in each video file, so the samples are file groups: runs of consecutive episodes whose files hold no other episode.

```bash
pixi run -e molmo download
```

To download different file groups (see below [A note on episodes and files](#a-note-on-episodes-and-files)), edit `SAMPLES` in [`molmoact2/download.py`](molmoact2/download.py).

The downloader caches a listing of the whole repo in `.cache/hf_files.json.gz`.
Building it takes a while on the first run, and it is rebuilt when the dataset revision changes.

#### A note on episodes and files

Each data parquet contains a median of 10 consecutive episodes, and each mp4 holds different numbers of episodes, a median of 12 to 19 depending on the camera.
This is expected in the [LeRobot v3 format](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3): the data and each camera start a new file separately, whenever the current file reaches a size limit.
So the files of one episode also hold parts of its neighbors.

To handle this efficiently, we define a `file group`: A file group is the smallest set of consecutive episodes that can be downloaded without parts of other episodes.
The figure below shows a specific example, covering episodes from 947 to 959.
Each bar represents a file in the figure.

![Episodes 947–959 form one file group](file-groups.png)

The dataset has 786 file groups, and a group holds a median of 40 episodes, at most 130, and about 3 GB.
No two groups share a file, so each group can be downloaded and converted on its own, for example on separate machines.

### 2. Convert (LeRobot → RRD)

Convert each downloaded episode into a Rerun recording (`.rrd`), the base layer, with the recording id `episode_NNNNN`:

```bash
pixi run -e molmo convert-base                       # every downloaded episode under data/MolmoAct2/
pixi run -e molmo convert-base --from 800 --to 809   # downloaded episodes 800 to 809
```

The recordings are written to `rrds/molmoact2/base/`.

Then build the urdf, cameras, and stats layers for the converted episodes.
The urdf layer poses the two arms, the cameras layer places the three cameras in the scene, and the stats layer holds the source's statistics of the episode.
All three keep each episode's recording id:

```bash
pixi run -e molmo convert-urdf                                             # every recording under rrds/molmoact2/base/
pixi run -e molmo convert-urdf rrds/molmoact2/base/episode_00800.rrd       # one recording
pixi run -e molmo convert-cameras                                          # every recording under rrds/molmoact2/base/
pixi run -e molmo convert-cameras rrds/molmoact2/base/episode_00800.rrd    # one recording
pixi run -e molmo convert-stats                                            # every recording under rrds/molmoact2/base/
pixi run -e molmo convert-stats rrds/molmoact2/base/episode_00800.rrd      # one recording
```

The layers are written to `rrds/molmoact2/urdf/`, `rrds/molmoact2/cameras/`, and `rrds/molmoact2/stats/`, and the arm model is written once for the whole dataset, as one asset per arm: `rrds/molmoact2/assets/urdf-model-{left,right}.rrd`.
See [More about Layers](#more-about-layers) for what each layer contains.

### 3. View

View a result in the Rerun Viewer:

```bash
pixi run -e molmo rerun rrds/molmoact2/base/*.rrd                # every episode
pixi run -e molmo rerun rrds/molmoact2/base/episode_00800.rrd    # one episode
```

Generate the default blueprint, then view one episode with it:

```bash
pixi run -e molmo blueprint
pixi run -e molmo rerun rrds/molmoact2/*/episode_00800.rrd blueprints/molmoact2/default.rbl
```

> Keep the `*` to load all layers.

> [!Note]
> The viewer opens local files through its Viewer catalog by default, and it can then ignore the blueprint file.
> Turn off **Settings** → **Viewer catalog** → **Load files via Viewer catalog** before running the command above.

> [!Note]
> Viewed from files alone, the Arms view shows the three cameras but not the arm meshes.
> The arm meshes are in the dataset's model assets, which have their own recording ids, so the viewer does not add them to the episode.
> The [local catalog](#4-local-catalog) registers the assets and shows the posed arms, as in the screenshot above.

> [!Note]
> Each arm's action plot is in the tab next to its state plot.
> Each plot shows the 7 values of one arm.
> The other arm's values are hidden, and the legend lists them as a single `left arm` or `right arm` entry, which shows them when clicked.
> A joint has the same color in every plot, so its state and action can be compared.
> To modify the layout, edit `molmoact2/blueprint.py` and rerun the `blueprint` task.

### 4. Local Catalog

Register the converted episodes to a [catalog server](https://rerun.io/docs/concepts/how-does-rerun-work#catalog-server), then browse, sort, filter, and query them as one dataset.

Start a local server:

```bash
pixi run serve              # start an in-memory catalog server (leave running)
```

In another shell, register the converted episodes:

```bash
pixi run -e molmo register   # register all episodes as the `molmoact2` dataset
```

Browse them in the Rerun Viewer:

```bash
pixi run -e molmo rerun rerun+http://127.0.0.1:51234
```

See [`notebook/filter_episodes.ipynb`](notebook/filter_episodes.ipynb) for a query that leaves out the short episodes and the episodes in which neither arm moves.
See [`notebook/compare_arm_speed.ipynb`](notebook/compare_arm_speed.ipynb) for queries that compare how fast the arms move in each episode and each data file.
Pick `.pixi/envs/molmo/bin/python` as their kernel, which includes the notebook viewer (`rerun-notebook`).

## Remote Convert Example on Modal

The steps above run locally on the downloaded sample files.
The following steps convert the full dataset on cloud workers.

### 1. Prerequisite: storage backend

Set the env vars for your storage backend in the shell, or edit the defaults in
[`rrd_datasets_common/storage.py`](../../packages/rrd_datasets_common/rrd_datasets_common/storage.py).
The dataset's own layout under the bucket lives in [`storage.py`](molmoact2/storage.py):

| Env var                                           | Backend | What it is                                                              |
| ------------------------------------------------- | ------- | ----------------------------------------------------------------------- |
| `STORAGE_BACKEND`                                 | both    | Which bucket kind, `hf` or `s3`, `hf` in the `molmo` environments       |
| `HF_NAMESPACE`                                    | hf      | The user or org that owns the bucket, set this one                      |
| `HF_BUCKET`                                       | hf      | Bucket the RRDs are written to, `molmoact2` by default, create it first |
| `HF_BUCKET_ACCESS_KEY_ID` / `…_SECRET_ACCESS_KEY` | hf      | The HF S3 credentials                                                   |

This example converts to a [Hugging Face Storage Bucket](https://huggingface.co/docs/hub/main/en/storage-buckets-s3) behind its S3-compatible gateway.
Access uses [HF S3 credentials](https://huggingface.co/docs/hub/storage-buckets-s3#generating-s3-credentials), generated from a fine-grained HF token scoped to the bucket.
The launcher passes them to the workers as a per-run secret.

> [!Note]
> To store in an AWS S3 bucket instead, `export STORAGE_BACKEND=s3` and follow the [S3 prerequisite in the ABC-130k example](../abc-130k/README.md#1-prerequisite-s3-storage).

### 2. Prerequisite: Modal setup

The [Modal](https://modal.com/) job under `molmoact2/modal_jobs/` starts one worker per [file group](#a-note-on-episodes-and-files).
Each worker downloads `meta/` and its group's files, converts every episode in it, and uploads the `.rrd` layers to the bucket.

To set it up:

- `pixi run -e molmo modal setup` — authenticate the Modal CLI (one-time).
- `pixi run -e molmo hf auth login`, or set `$HF_TOKEN` — optional for this public dataset, but anonymous callers share a smaller per-IP download quota.

### 3. Run Convert

Run `pixi run -e molmo convert-on-modal --help` to see all options.

```bash
# One new file group, every layer (the default when no flags are given):
pixi run -e molmo convert-on-modal

# Every file group (--limit 0 removes the cap):
pixi run -e molmo convert-on-modal --limit 0

# The file groups of one task:
pixi run -e molmo convert-on-modal --task-filter "Fold the cloth" --limit 0

# See what would run, without starting any worker:
pixi run -e molmo convert-on-modal --dry-run --limit 10
```

The `convert-on-modal` task runs detached and returns immediately.
Watch progress in the Modal dashboard.
A full run writes about 2.4 TB to the bucket, about the size of the source.

> [!Note]
> Without `--overwrite`, anything already in the bucket is skipped.

#### Picking layers

`--layers` lets you choose which layers to build:

```bash
# Only the base layer:
pixi run -e molmo convert-on-modal --layers base --limit 0

# Rebuild layers after changing them:
pixi run -e molmo convert-on-modal --layers urdf,cameras --limit 0 --overwrite

# Add the stats layer to a converted bucket:
pixi run -e molmo convert-on-modal --layers stats --limit 0
```

The stats layer reads only `meta/`, so `--layers stats` downloads no data or video file.
Every other layer derives from the base layer, so selecting any of them makes a worker download its whole file group.

### 4. Upload the blueprint

`pixi run -e molmo blueprint` writes `blueprints/molmoact2/default.rbl`.
To upload it to your bucket (`s3://<bucket>/blueprints/`), run:

```bash
pixi run -e molmo upload-blueprint
```

### 5. Upload the arm model assets

`pixi run -e molmo convert-urdf` writes the two arm models as `rrds/molmoact2/assets/urdf-model-{left,right}.rrd`.
To upload them to your bucket (`s3://<bucket>/assets/`), run:

```bash
pixi run -e molmo upload-asset
```

## Observations

We share our observations including useful details beyond the dataset card.
See [observations.md](observations.md) for the full survey.

## Mapping to Rerun

The base layer keeps each episode as `LeRobotReader` emits it: one entity per feature, named after the feature, with the values and the video samples unchanged.
It adds two things: a `timestamp` timeline for the source column the reader drops, and the episode's recording properties.
The urdf layer and the model assets add the two posed arms, computed from `/observation.state`.
The stats layer adds the source's statistics of each episode, and how far each arm moves.

| Source                                                              | Entity path                                                 | Component / archetype                       | Layer   | Notes                                                                                                                                                                                            |
| ------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `action` [14]                                                       | `/action`                                                   | `Scalars`, static `SeriesLines:names`       | base    | commanded joints in radians, grippers from 0 (closed) to 1 (open), named as in `info.json`                                                                                                       |
| `observation.state` [14]                                            | `/observation.state`                                        | `Scalars`, static `SeriesLines:names`       | base    | measured joints, in the same units                                                                                                                                                               |
| `task_index`                                                        | `/task`                                                     | `TextDocument`                              | base    | the task text, repeated on every frame                                                                                                                                                           |
| `observation.images.{top,left,right}`                               | `/observation.images.{top,left,right}`                      | `VideoStream`                               | base    | the AV1 samples of the episode, copied without re-encoding, except that a window starting on a delta frame has its frames up to the next keyframe re-encoded (see `keyframe_misaligned_cameras`) |
| `frame_index`                                                       | `frame_index` timeline                                      | —                                           | base    | the frame number within the episode                                                                                                                                                              |
| `timestamp`                                                         | `timestamp` timeline                                        | —                                           | base    | `float32(frame_index / 30)` seconds, as whole nanoseconds                                                                                                                                        |
| `meta/episodes`, `meta/tasks_annotated.parquet`, `info.json`        | segment properties                                          | —                                           | base    | see [Base layer](#base-layer)                                                                                                                                                                    |
| `yam.urdf` meshes and joints                                        | `/urdf_{left,right}/yam/**`, `/urdf_{left,right}/tf_static` | `Asset3D`, `Transform3D`                    | asset   | static; one model rrd per arm, 3.3 MB each, once per dataset                                                                                                                                     |
| `observation.state` [14] → forward kinematics                       | `/urdf_{left,right}/transforms`                             | `Transform3D`                               | urdf    | 8 joints per arm per frame, between named frames                                                                                                                                                 |
| estimated arm placement                                             | `/urdf_{left,right}/world_from_base`                        | `Transform3D`                               | urdf    | static; see [URDF layer](#urdf-layer)                                                                                                                                                            |
| i2rt's wrist camera mount                                           | `/observation.images.{left,right}`                          | `Transform3D`, `Pinhole`, `CoordinateFrame` | cameras | static; on the gripper flange, so the video moves with the arm                                                                                                                                   |
| estimated top camera pose                                           | `/observation.images.top`                                   | `Transform3D`, `Pinhole`, `CoordinateFrame` | cameras | static; see [Cameras layer](#cameras-layer)                                                                                                                                                      |
| `meta/episodes` statistics of `observation.state`, `action`         | `/stats/observation.state`, `/stats/action`                 | `FeatureStats`                              | stats   | static; one value (`count`) is the number of frames, every other statistic has 14 values, and the 14 element names come from `info.json`                                                         |
| `meta/episodes` statistics of `observation.images.{top,left,right}` | `/stats/observation.images.{top,left,right}`                | `FeatureStats`                              | stats   | static; one value (`count`) is the number of frames the source sampled, and every other statistic has one value per RGB channel                                                                  |
| `meta/episodes` `q01` and `q99` of `action`                         | segment properties                                          | —                                           | stats   | see [Stats layer](#stats-layer)                                                                                                                                                                  |

### Round trip test (LeRobot → RRD → LeRobot)

Every data column of the source can be rebuilt from the base layer.
A test ([`tests/test_base_layer.py`](tests/test_base_layer.py)) converts a synthesized dataset, rebuilds `action`, `observation.state`, `timestamp`, `frame_index`, `episode_index`, `index`, and `task_index` from the recordings, and compares them bit for bit with the source.
`timestamp` comes from the `timestamp` timeline, `task_index` from the `/task` text, and `index` from the `length` of the earlier episodes.
The same test checks every property against the source metadata.
A second test converts a downloaded episode and checks its video by count, timestamps, and keyframe position: every camera holds one sample per frame, on the same `timestamp` values as the data, and starts on a keyframe.

## More about Layers

Each layer is a separate module that writes its own `.rrd`.

### Base layer

`molmoact2/base_layer.py` — Everything in the episode's data rows and videos, as recorded.
The reader keeps `frame_index` as the only timeline, so the base layer adds `timestamp` as a second one.
It computes each value as `float32(frame_index / 30)`, which also gives a time to video frames that have no data row (episode 31749).
The source `timestamp` column holds the same values: its per-episode statistics in `meta/episodes` (count, min, max, mean) match `frame_index / 30` in all 32,246 episodes of the pinned revision.
The episode's metadata is logged as recording properties, which the catalog shows as columns to filter, sort, and search on:

| Property                      | Type            | Source                                                                                                                                                                      |
| ----------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `episode_index`               | int64           | `meta/episodes`                                                                                                                                                             |
| `task`                        | string          | `meta/episodes`, where the source stores a one-element list                                                                                                                 |
| `instruction`                 | string          | `meta/tasks_annotated.parquet`                                                                                                                                              |
| `length`                      | int64           | `meta/episodes`, in frames                                                                                                                                                  |
| `robot_type`                  | string          | `meta/info.json`                                                                                                                                                            |
| `source_revision`             | string          | the pinned dataset revision                                                                                                                                                 |
| `source_files`                | list of strings | the episode's data parquet, then the video file of each camera (left, right, top), as paths in the source repo, e.g. `videos/observation.images.top/chunk-000/file-823.mp4` |
| `num_frames_mismatched`       | bool            | true when the state, the action, or a camera holds a number of frames other than `length` (see [observations.md](observations.md#edge-cases-and-data-bugs))                 |
| `keyframe_misaligned_cameras` | list of strings | the cameras (left, right, top) whose video window disagrees with the keyframes of its file: the window starts on a delta frame, or its first two frames are keyframes       |

`keyframe_misaligned_cameras` comes from the keyframes alone.
A window can be shifted in a way the keyframes do not show, and a flagged window's picture can still look aligned.

### URDF layer

`molmoact2/urdf_layer.py` — Both arms, posed by forward kinematics from `/observation.state` with the model in [`urdf/yam/`](urdf/yam/).
The source dataset does not provide the arm placement. We use an estimate.

### Cameras layer

`molmoact2/camera_layer.py` — The three cameras, placed in the scene so their videos show in the Arms view.
The source dataset does not provide camera calibration.
The wrist mounts come from i2rt's station model, and the top camera's pose and all intrinsics are estimates.

> [!Note]
> The authors posted partial calibration in [allenai/molmoact2#2](https://github.com/allenai/molmoact2/issues/2): a nominal layout drawing and the top camera pose.
> The sample intrinsics posted in the issue are for 640×480, so they do not apply to the 640×360 videos.
> Our hand-fitted values match its arm spacing (0.50 m) and top camera tilt (72°).

### Stats layer

`molmoact2/stats_layer.py` — The source's statistics of the episode, from `meta/episodes`, and how far each arm moves.

The state, the action, and each camera get a static `FeatureStats` entity under `/stats/`, with one component per statistic: `min`, `max`, `mean`, `std`, `q01`, `q10`, `q50`, `q90`, `q99`, and `count`.
The values are copied as the source stores them, errors included (see [observations.md](observations.md#statistics)).
The statistics of `timestamp`, `frame_index`, `episode_index`, `index`, and `task_index` are left out.

Four recording properties hold each arm's spread: `q99 − q01` of its commanded `action` over the episode.
The catalog shows them as `property:stats:<name>`.

| Property                                      | Type    | Value                                                  |
| --------------------------------------------- | ------- | ------------------------------------------------------ |
| `left_joint_spread`, `right_joint_spread`     | float64 | the largest spread of the arm's six joints, in radians |
| `left_gripper_spread`, `right_gripper_spread` | float64 | the gripper's spread, from 0 (closed) to 1 (open)      |

Across the 32,246 episodes, all but 25 of the 64,492 joint spreads (one per arm) are below 0.01 rad or above 1 rad, so any threshold in between gives nearly the same split.

## Rerun APIs demonstrated

The conversion is built on the [chunk processing API](https://rerun.io/docs/concepts/logging-and-ingestion/chunk-processing-api) and [lenses](https://rerun.io/docs/concepts/query-and-transform/lenses).

- [`LeRobotReader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.LeRobotReader) reads each episode of the [LeRobot v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) source (`base_layer.py`).
- [`LazyChunkStream.map`](https://ref.rerun.io/docs/python/stable/chunk/#rerun.chunk.LazyChunkStream.map) adds the `timestamp` timeline (`base_layer.py`).
- [`Chunk.from_property`](https://ref.rerun.io/docs/python/stable/chunk/#rerun.chunk.Chunk.from_property) logs the episode metadata as recording properties (`base_layer.py`).
- [`rerun.urdf.UrdfTree`](https://ref.rerun.io/docs/python/stable/urdf/#rerun.urdf.UrdfTree) poses the arms by forward kinematics, as in the [`robot_data_preprocessing` example](https://github.com/rerun-io/rerun/tree/main/examples/python/robot_data_preprocessing) (`urdf_layer.py`).
- [`Pinhole`](https://rerun.io/docs/reference/types/archetypes/pinhole) and [`Transform3D`](https://rerun.io/docs/reference/types/archetypes/transform3d) place the cameras in 3D (`camera_layer.py`).
- [`DynamicArchetype`](https://ref.rerun.io/docs/python/stable/rerun/#rerun.DynamicArchetype) logs the source's statistics under [custom component names](https://rerun.io/docs/howto/logging-and-ingestion/custom-data) (`stats_layer.py`).
- [`CatalogClient`](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model) registers the episodes, the arm assets, and the blueprint (`catalog.py`).
- The [segment table](https://ref.rerun.io/docs/python/stable/catalog/#rerun.catalog.DatasetEntry.segment_table) and [`filter_segments`](https://ref.rerun.io/docs/python/stable/catalog/#rerun.catalog.DatasetEntry.filter_segments) select episodes by their recording properties (`notebook/filter_episodes.ipynb`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API builds the default layout (`blueprint.py`).
