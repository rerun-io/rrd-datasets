# MolmoAct2

Below is the viewer showing a converted episode with the default blueprint.

![MolmoAct2 in the Rerun viewer](screenshot.png)

The [default blueprint](#3-view) puts the episode's instruction at the top, the top and wrist cameras below it, and the state and action plots of each arm along the bottom.

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

To download different file groups, edit `SAMPLES` in [`molmoact2/download.py`](molmoact2/download.py).

The downloader caches a listing of the whole repo in `.cache/hf_files.json.gz`.
Building it takes a while on the first run, and it is rebuilt when the dataset revision changes.

### 2. Convert (LeRobot → RRD)

Convert each downloaded episode into a Rerun recording (`.rrd`), the base layer, with the recording id `episode_NNNNN`:

```bash
pixi run -e molmo convert-base                       # every downloaded episode under data/MolmoAct2/
pixi run -e molmo convert-base --from 800 --to 809   # downloaded episodes 800 to 809
```

The recordings are written to `rrds/molmoact2/base/`.
See [More about Layers](#more-about-layers) for what the base layer contains.

### 3. View

View a result in the Rerun Viewer:

```bash
pixi run -e molmo rerun rrds/molmoact2/base/*.rrd                # every episode
pixi run -e molmo rerun rrds/molmoact2/base/episode_00800.rrd    # one episode
```

Generate the default blueprint, then view one episode with it:

```bash
pixi run -e molmo blueprint
pixi run -e molmo rerun rrds/molmoact2/base/episode_00800.rrd blueprints/molmoact2/default.rbl
```

> **Note:** The viewer opens local files through its Viewer catalog by default, and it can then ignore the blueprint file.
> Turn off **Settings** → **Viewer catalog** → **Load files via Viewer catalog** before running the command above.

> **Note:** Each plot shows the 7 values of one arm.
> The other arm's values are hidden, and the legend lists them as a single `left arm` or `right arm` entry, which shows them when clicked.
> A joint has the same color in every plot, so its state and action can be compared.
> To modify the layout, edit `molmoact2/blueprint.py` and rerun the `blueprint` task.

## Observations

We share our observations including useful details beyond the dataset card.
See [observations.md](observations.md) for the full survey.

## Mapping to Rerun

The base layer keeps each episode as `LeRobotReader` emits it: one entity per feature, named after the feature, with the values and the video samples unchanged.
It adds two things: a `timestamp` timeline, which brings back the source column the reader drops, and the episode's recording properties.

| Source                                                       | Entity path                            | Component / archetype                 | Layer | Notes                                                                                      |
| ------------------------------------------------------------ | -------------------------------------- | ------------------------------------- | ----- | ------------------------------------------------------------------------------------------ |
| `action` [14]                                                | `/action`                              | `Scalars`, static `SeriesLines:names` | base  | commanded joints in radians, grippers from 0 (closed) to 1 (open), named as in `info.json` |
| `observation.state` [14]                                     | `/observation.state`                   | `Scalars`, static `SeriesLines:names` | base  | measured joints, in the same units                                                         |
| `task_index`                                                 | `/task`                                | `TextDocument`                        | base  | the task text, repeated on every frame                                                     |
| `observation.images.{top,left,right}`                        | `/observation.images.{top,left,right}` | `VideoStream`                         | base  | the AV1 samples of the episode, copied without re-encoding                                 |
| `frame_index`                                                | `frame_index` timeline                 | —                                     | base  | the frame number within the episode                                                        |
| `timestamp`                                                  | `timestamp` timeline                   | —                                     | base  | the source's float32 seconds, as whole nanoseconds                                         |
| `meta/episodes`, `meta/tasks_annotated.parquet`, `info.json` | segment properties                     | —                                     | base  | see [Base layer](#base-layer)                                                              |

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
The reader keeps `frame_index` as the only timeline, so the base layer adds the source's `timestamp` back as a second one.
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

## Rerun APIs demonstrated

- [`LeRobotReader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.LeRobotReader) reads one episode at a time into chunk streams, with each camera's video cut to the episode and copied without re-encoding (`base_layer.py`).
- A `LazyChunkStream.map` step adds the `timestamp` timeline to every chunk as an extra index column (`base_layer.py`).
- `Chunk.from_property` logs the episode's metadata as recording properties (`base_layer.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout.
  A component mapping shows the `instruction` property as text, and `SeriesLines` overrides split the 14 state and action values by arm (`blueprint.py`).
