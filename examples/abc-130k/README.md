# ABC-130k

[XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) is a large open-source bi-manual teleoperation dataset: 130k episodes across 197 household manipulation tasks, recorded as MCAP.
This example converts each episode MCAP into a Rerun recording (`.rrd`).

The [default blueprint](#running) arranges a converted episode across three rows.
The top and wrist cameras run across the top row, the instruction and subtask timeline share the middle row, and the bottom row plots arm and gripper positions, velocities, and torques.

## Dataset

- **Source**: [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) on Hugging Face
- **License**: Apache-2.0. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on one sample episode from the `fold_and_stack_the_t_shirts` task; the Modal job converts the full dataset.
- **Access**: gated — accept the terms on Hugging Face, then authenticate (`pixi run -e abc-130k hf auth login`, or set `$HF_TOKEN`).

This example does not redistribute the dataset.
Data is downloaded at runtime from Hugging Face.

### Converted `.rrd` Dataset

Converted recordings are published at [`rerun/abc-130k`](https://huggingface.co/buckets/rerun/abc-130k) — download them directly if you only want the `.rrd` data.

# Running

File paths below are relative to the repository root.

## Local Runs

Download one sample episode (~450 MB):

```sh
pixi run abc-download
```

To pick episodes and tasks more selectively, see [`notebook/download_episodes.ipynb`](notebook/download_episodes.ipynb).

Convert it, writing `rrds/abc-130k/base/<episode>.rrd`:

```sh
pixi run abc-convert
```

View the result with the default blueprint:

```sh
pixi run -e abc-130k rerun rrds/abc-130k/base/<episode>.rrd blueprints/abc-130k/default.rbl
```

The blueprint produces the three-row layout described above.
Two views sit behind tabs and are easy to miss: the top-camera slot has a second tab for the ZED stereo pair, and the instruction view has one for the subtask timeline.
It also carries the per-joint legend labels (array entities would otherwise show bare indices) and splits `q`, `dq`, and `tau` into separate plots so each keeps its own value range.
Regenerate it with `pixi run abc-blueprint`, which overwrites the committed `.rbl`.

To browse, filter, and query episodes, register them to a local catalog:

```sh
pixi run serve          # start an in-memory catalog server (leave running)
```

On a separate terminal,

```sh
pixi run abc-register   # register converted episodes and the default blueprint
pixi run -e abc-130k rerun rerun+http://127.0.0.1:51234
```

[`notebook/query_local_catalog.ipynb`](notebook/query_local_catalog.ipynb) shows example queries against the registered dataset.

## Modal Runs

To convert the full dataset off-box, the [Modal](https://modal.com/) job under `modal_jobs/` fans episodes out across workers: each worker downloads one episode, converts it, and uploads the `.rrd` to S3.
AWS access uses Modal's OIDC identity, and your Hugging Face token ships to the workers as an ephemeral per-run secret.
Point it at your own bucket with `S3_BUCKET`, `S3_REGION`, and `AWS_ROLE_ARN`:

```sh
pixi run abc-convert-on-modal --limit 10    # 0 for all episodes, default 1.
```



## Observations

Surveying a subset of the episodes surfaced several useful details beyond the dataset card. See [observations.md](observations.md) for the full survey.

## Mapping to Rerun

The table below shows where each source topic lands in the recording.
`<side>` is `left` or `right`; camera topics keep their names (`/top-camera`, `/left-wrist-camera`, …).

| Source | Entity path | Archetype |
| ------ | ----------- | --------- |
| `/instruction` | `/instruction` | `TextDocument` |
| `/<side>-arm-state` | `/<side>/arm/state/{q, dq, tau}` | `Scalars` |
| `/<side>-arm-action` | `/<side>/arm/action/q` | `Scalars` |
| `/<side>-ee-state` | `/<side>/gripper/state/{q, dq, tau}` | `Scalars` |
| `/<side>-ee-action` | `/<side>/gripper/action/q` | `Scalars` |
| `/*-camera` | same path | `VideoStream` |
| `/*-camera-info` | same path | `Pinhole` |
| `annotation.mcap` sidecar | `/task/subtask` | `StateChange` |

Each arm signal is one array `Scalars` entity holding all 6 joints, rather than one entity per joint; the blueprint supplies the per-joint legend labels.
The gripper's velocity and torque sit in different topics depending on the episode (see [observations.md](observations.md)); the converter probes each episode and routes them onto `/<side>/gripper/state/`.

Camera video is re-encoded to H.264 with a fixed keyframe interval (GOP 60) for finer seeking, and 1920x1200 cameras are downscaled to 640x400 with their `Pinhole` rescaled to match.
The re-encode is CPU-heavy — expect minutes per episode — and `--crf` tunes quality (default 23).

Each recording carries filterable properties: `split` (train/val), `station`, and `instruction`.

Rerun APIs demonstrated:

- [`McapReader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.McapReader) decodes the episode topics, including the custom protobuf messages, into chunk streams (`convert.py`).
- [`Lenses`](https://rerun.io/docs/concepts/query-and-transform/lenses) turn the raw messages into typed components: `Scalars` for the joint signals, `TextDocument` for the instruction, `StateChange` for the subtask labels (`convert.py`).
- `write_rrd` with `OptimizationProfile.OBJECT_STORE` writes one optimized recording per episode (`convert.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout, including the per-series legend overrides (`blueprint.py`).
- [`CatalogClient`](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model) registers each episode as a dataset segment and installs the default blueprint (`catalog.py`).
