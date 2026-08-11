# ABC-130k

[XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) is a large open-source bi-manual teleoperation dataset: 130k episodes across 197 household manipulation tasks, recorded as MCAP.
This example converts each episode MCAP into a Rerun recording (`.rrd`).
The [default blueprint](#running) arranges a converted episode across three rows.
Below is the viewer displaying a sample episode.

https://github.com/user-attachments/assets/3bcafe00-1f66-47c5-97a1-9d20be273664

The top and wrist cameras run across the top row, the instruction and subtask timeline share the middle row, and the bottom row plots arm and gripper positions, velocities, and torques.

This example includes (1) a [local version](#local-runs) that downloads, converts, registers, and queries the data for a small subset, and (2) a [Modal](https://modal.com/)-based [remote version](#remote-convert-example-on-modal) that downloads, converts, and stores all episodes in an AWS S3 bucket.

## Dataset

- **Source**: [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) on Hugging Face
- **License**: Apache-2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on one sample episode from the `fold_and_stack_the_t_shirts` task; the Modal job converts the full dataset.
- **Access**: gated — accept the terms on Hugging Face, then authenticate (`pixi run -e abc hf auth login`, or set `$HF_TOKEN`).

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

### Converted `.rrd` Dataset

Converted recordings are published at [`rerun/abc-130k`](https://huggingface.co/buckets/rerun/abc-130k) — download them directly if you only want the `.rrd` data.

# Running

File paths below are relative to the repository root.

## Local Runs

### 1. Download

Download one sample episode (~450 MB) into `data/ABC-130k/`:

```sh
pixi run -e abc download
```

To inspect the dataset metadata and pick episodes more selectively, see [`notebook/download_episodes.ipynb`](notebook/download_episodes.ipynb).

### 2. Convert (MCAP → RRD)

Convert downloaded episode MCAP files into per-episode Rerun recordings under `rrds/abc-130k/base/`:

```bash
pixi run -e abc convert                            # every downloaded episode
pixi run -e abc convert <path to episode.mcap>     # just one
```

#### Notes on Video Transcode

The convert task always re-encodes the camera video to H.264 with a fixed config: keyframe interval (GOP) 60 for finer seeking (source's GOP is ~250), and 1920x1200 cameras downscaled to 640x400 (aspect-preserving).
The camera intrinsic parameters are updated accordingly.
`--crf` tunes quality (default 23).

> **Note:** the re-encode is CPU-heavy.
> So the convert task can take minutes per episode (versus sub-second for a plain pass-through).
> This project does not use GPU acceleration.

### 3. View

View a result in the Rerun Viewer:

```bash
pixi run -e abc rerun rrds/abc-130k/base/*.rrd        # every episode
pixi run -e abc rerun rrds/abc-130k/base/<id>.rrd     # one episode
```

View the result with the default blueprint:

```sh
pixi run -e abc rerun rrds/abc-130k/base/<episode>.rrd blueprints/abc-130k/default.rbl
```

> **Notes:** The blueprint produces the three-row layout described above.
> Two views sit behind tabs and are easy to miss: the top-camera slot has a second tab for the ZED stereo pair, and the instruction view has one for the subtask timeline.
> It also carries the per-joint legend labels (array entities would otherwise show bare indices) and splits `q`, `dq`, and `tau` into separate plots so each keeps its own value range.
> Regenerate it with `pixi run -e abc blueprint`, which overwrites the committed `.rbl`.

### 4. Local Catalog

The converted episodes become more useful when registered to a **[catalog server](https://rerun.io/docs/concepts/how-does-rerun-work#catalog-server)**. You can browse, sort, filter, and query segments of your dataset.

Start a local server:

```sh
pixi run serve          # start an in-memory catalog server (leave running)
```

In another shell, register converted episodes and the default blueprint to a local catalog:

```sh
pixi run -e abc register               # register every converted episode under rrds/
pixi run -e abc register --recreate    # delete the existing dataset and rebuild it from scratch
```

Browse them on the Rerun viewer.

```sh
pixi run -e abc rerun rerun+http://127.0.0.1:51234
```

See [`notebook/query_local_catalog.ipynb`](notebook/query_local_catalog.ipynb) for example catalog queries — episodes with subtask annotations, and subtask labels containing 'mistake'.

## Remote Convert Example on Modal

The steps above run locally with a small set of data.
To convert the full dataset off-box, the [Modal](https://modal.com/) job under `abc_130k/modal_jobs/` fans episodes out across workers: each worker downloads one episode, converts it, and uploads the `.rrd` to S3.

### 1. Prerequisite (1) S3 Storage

Storage is plain S3 via [`boto3`](https://docs.aws.amazon.com/boto3/latest/), and AWS access uses [Modal's OIDC](https://modal.com/docs/guide/oidc-integration) identity exchanged for temporary credentials (no keys stored, no Modal Volumes to configure).
To point at your own infrastructure, set the env vars below

| Env var        | What it is                                                              |
| -------------- | ----------------------------------------------------------------------- |
| `S3_BUCKET`    | Bucket the RRDs are written to                                          |
| `S3_REGION`    | Bucket region (workers run in the same region)                          |
| `AWS_ROLE_ARN` | IAM role that trusts Modal's OIDC issuer and can read/write that bucket |

The defaults are placeholders.

### 2. Prerequisite (2) Modal Setup

- `pixi run -e abc modal setup` — authenticate the Modal CLI (one-time).
- Log in to Hugging Face locally (`pixi run -e abc hf auth login`, or set `$HF_TOKEN`) — needed to list the gated dataset and to hand your token to the workers (shipped as an ephemeral per-run secret, so there is nothing stored on Modal to refresh).

### 3. Run Convert

Run `pixi run -e abc convert-on-modal --help` for the full flag reference.
Video is always transcoded with the fixed config; `--crf` tunes quality — see the [transcode note](#notes-on-video-transcode).

A few common invocations:

```bash
# One new episode (the default when no flags are given):
pixi run -e abc convert-on-modal

# The entire dataset (--limit 0 removes the cap):
pixi run -e abc convert-on-modal --limit 0

# Overwrite first 5 episodes of one task:
pixi run -e abc convert-on-modal --task-filter fold_and_stack_the_t_shirts --limit 5 --overwrite
```

> **Note:** transcoding is CPU-heavy, so a higher worker `cpu` allocation speeds it up (see `CPU_REQUEST` in [convert_episodes.py](abc_130k/modal_jobs/convert_episodes.py)).

> **Note:** episode discovery (`discover_episodes` in [episode_index.py](abc_130k/episode_index.py)) lists the full Hugging Face repo tree once, which takes minutes for a repo this size.
> The listing is cached in `examples/abc-130k/.cache/hf_files.json.gz` and reused until the dataset revision changes (the [download notebook](notebook/download_episodes.ipynb) shares it).
> Delete the file to force a re-listing.

## Observations

Surveying a subset of the episodes surfaced several useful details beyond the dataset card. See [observations.md](observations.md) for the full survey.

## Mapping to Rerun

The table below shows where each source topic lands in the recording.
`<side>` is `left` or `right`; camera topics keep their names (`/top-camera`, `/left-wrist-camera`, …).

| Source                    | Entity path                          | Archetype      |
| ------------------------- | ------------------------------------ | -------------- |
| `/instruction`            | `/instruction`                       | `TextDocument` |
| `/<side>-arm-state`       | `/<side>/arm/state/{q, dq, tau}`     | `Scalars`      |
| `/<side>-arm-action`      | `/<side>/arm/action/q`               | `Scalars`      |
| `/<side>-ee-state`        | `/<side>/gripper/state/{q, dq, tau}` | `Scalars`      |
| `/<side>-ee-action`       | `/<side>/gripper/action/q`           | `Scalars`      |
| `/*-camera`               | same path                            | `VideoStream`  |
| `/*-camera-info`          | same path                            | `Pinhole`      |
| `annotation.mcap` sidecar | `/task/subtask`                      | `StateChange`  |

References of Rerun APIs demonstrated in the example:

- [`McapReader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.McapReader) decodes the episode topics, including the custom protobuf messages, into chunk streams (`convert.py`).
- [`Lenses`](https://rerun.io/docs/concepts/query-and-transform/lenses) turn the raw messages into typed components: `Scalars` for the joint signals, `TextDocument` for the instruction, `StateChange` for the subtask labels (`convert.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout, including the per-series legend overrides (`blueprint.py`).
- [`CatalogClient`](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model) registers each episode as a dataset segment and installs the default blueprint (`catalog.py`).
