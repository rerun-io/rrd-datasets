# MolmoAct2-BimanualYAM

Below is the viewer showing a converted episode with the default blueprint.

![MolmoAct2 in the Rerun viewer](screenshot.png)

The [default blueprint](#3-view) puts the episode's instruction at the top, the posed arms next to the top and wrist cameras below it, and each arm's state and action plots, as two tabs, along the bottom.

## Dataset

- **Source**: [allenai/MolmoAct2-BimanualYAM-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset) on Hugging Face
- **License**: Apache 2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on three sample file groups (~1.1 GB, 19 episodes) spanning three tasks, all listed in [observations.md](observations.md#surveyed-file-groups).
- **Access**: public, not gated.

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

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

Then build the urdf and cameras layers from the base recordings.
The urdf layer poses the two arms, and the cameras layer places the three cameras in the scene.
Both keep each episode's recording id:

```bash
pixi run -e molmo convert-urdf                                             # every recording under rrds/molmoact2/base/
pixi run -e molmo convert-urdf rrds/molmoact2/base/episode_00800.rrd       # one recording
pixi run -e molmo convert-cameras                                          # every recording under rrds/molmoact2/base/
pixi run -e molmo convert-cameras rrds/molmoact2/base/episode_00800.rrd    # one recording
```

The layers are written to `rrds/molmoact2/urdf/` and `rrds/molmoact2/cameras/`, and the arm model is written once for the whole dataset, as one asset per arm: `rrds/molmoact2/assets/urdf-model-{left,right}.rrd`.
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

> **Note:** The viewer opens local files through its Viewer catalog by default, and it can then ignore the blueprint file.
> Turn off **Settings** → **Viewer catalog** → **Load files via Viewer catalog** before running the command above.

> **Note:** Viewed from files alone, the **Arms** view shows the three cameras but not the arm meshes.
> The arm meshes are in the dataset's model assets, which have their own recording ids, so the viewer does not add them to the episode.
> A catalog with the assets registered shows the posed arms, as in the screenshot above.

> **Note:** Each arm's action plot is in the tab next to its state plot.
> Each plot shows the 7 values of one arm.
> The other arm's values are hidden, and the legend lists them as a single `left arm` or `right arm` entry, which shows them when clicked.
> A joint has the same color in every plot, so its state and action can be compared.
> To modify the layout, edit `molmoact2/blueprint.py` and rerun the `blueprint` task.

## Observations

We share our observations including useful details beyond the dataset card.
See [observations.md](observations.md) for the full survey.

## Mapping to Rerun

The base layer keeps each episode as `LeRobotReader` emits it: one entity per feature, named after the feature, with the values and the video samples unchanged.
It adds two things: a `timestamp` timeline for the source column the reader drops, and the episode's recording properties.
The urdf layer and the model assets add the two posed arms, computed from `/observation.state`.

| Source                                                       | Entity path                                                 | Component / archetype                       | Layer   | Notes                                                                                      |
| ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------- | ------- | ------------------------------------------------------------------------------------------ |
| `action` [14]                                                | `/action`                                                   | `Scalars`, static `SeriesLines:names`       | base    | commanded joints in radians, grippers from 0 (closed) to 1 (open), named as in `info.json` |
| `observation.state` [14]                                     | `/observation.state`                                        | `Scalars`, static `SeriesLines:names`       | base    | measured joints, in the same units                                                         |
| `task_index`                                                 | `/task`                                                     | `TextDocument`                              | base    | the task text, repeated on every frame                                                     |
| `observation.images.{top,left,right}`                        | `/observation.images.{top,left,right}`                      | `VideoStream`                               | base    | the AV1 samples of the episode, copied without re-encoding                                 |
| `frame_index`                                                | `frame_index` timeline                                      | —                                           | base    | the frame number within the episode                                                        |
| `timestamp`                                                  | `timestamp` timeline                                        | —                                           | base    | `float32(frame_index / 30)` seconds, as whole nanoseconds                                  |
| `meta/episodes`, `meta/tasks_annotated.parquet`, `info.json` | segment properties                                          | —                                           | base    | see [Base layer](#base-layer)                                                              |
| `yam.urdf` meshes and joints                                 | `/urdf_{left,right}/yam/**`, `/urdf_{left,right}/tf_static` | `Asset3D`, `Transform3D`                    | asset   | static; one model rrd per arm, 3.3 MB each, once per dataset                               |
| `observation.state` [14] → forward kinematics                | `/urdf_{left,right}/transforms`                             | `Transform3D`                               | urdf    | 8 joints per arm per frame, between named frames                                           |
| estimated arm placement                                      | `/urdf_{left,right}/world_from_base`                        | `Transform3D`                               | urdf    | static; see [URDF layer](#urdf-layer)                                                      |
| i2rt's wrist camera mount                                    | `/observation.images.{left,right}`                          | `Transform3D`, `Pinhole`, `CoordinateFrame` | cameras | static; on the gripper flange, so the video moves with the arm                             |
| estimated top camera pose                                    | `/observation.images.top`                                   | `Transform3D`, `Pinhole`, `CoordinateFrame` | cameras | static; see [Cameras layer](#cameras-layer)                                                |

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

| Property             | Type   | Source                                                                                                                                                      |
| -------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `episode_index`      | int64  | `meta/episodes`                                                                                                                                             |
| `task`               | string | `meta/episodes`, where the source stores a one-element list                                                                                                 |
| `instruction`        | string | `meta/tasks_annotated.parquet`                                                                                                                              |
| `length`             | int64  | `meta/episodes`, in frames                                                                                                                                  |
| `robot_type`         | string | `meta/info.json`                                                                                                                                            |
| `source_revision`    | string | the pinned dataset revision                                                                                                                                 |
| `has_frame_mismatch` | bool   | true when the state, the action, or a camera holds a number of frames other than `length` (see [observations.md](observations.md#edge-cases-and-data-bugs)) |

### URDF layer

`molmoact2/urdf_layer.py` — Both arms, posed by forward kinematics from `/observation.state` with the model in [`urdf/yam/`](urdf/yam/).
The source dataset does not provide the arm placement. We use an estimate.

### Cameras layer

`molmoact2/camera_layer.py` — The three cameras, placed in the scene so their videos show in the **Arms** view.
The source dataset does not provide camera calibration.
The wrist mounts come from i2rt's station model, and the top camera's pose and all intrinsics are estimates.

## Rerun APIs demonstrated

- [`LeRobotReader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.LeRobotReader) reads one episode at a time into chunk streams, with each camera's video cut to the episode and copied without re-encoding (`base_layer.py`).
- A `LazyChunkStream.map` step adds the `timestamp` timeline to every chunk as an extra index column (`base_layer.py`).
- `Chunk.from_property` logs the episode's metadata as recording properties (`base_layer.py`).
- `rerun.urdf.UrdfTree` streams the arm model and solves forward kinematics (`urdf_layer.py`).
- `Pinhole`, `CoordinateFrame` and a static `Transform3D` place the camera videos in 3D (`camera_layer.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout.
  A component mapping shows the `instruction` property as text, and `SeriesLines` overrides split the 14 state and action values by arm (`blueprint.py`).
