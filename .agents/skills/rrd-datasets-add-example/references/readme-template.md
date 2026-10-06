# README template

Every example README under `examples/` follows the same skeleton.
The sentences every example shares are written out in full, and the parts that change per dataset are marked.

The template itself starts below the horizontal rule.

## Write it as you add modules

Do not copy the whole template at the start.
Each module you add brings its own sections over, in the skeleton's order and with the skeleton's wording.
The README then never holds a stub, and it never describes something that does not exist yet.

| When you add                        | Bring over these sections                                                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `download.py` and the episode index | Local Runs (the revision-pin line), 1. Download                                                                        |
| the survey, `observations.md`       | Dataset, Observations                                                                                                  |
| `base_layer.py`                     | 2. Convert, Round trip test, plus the base entries of Mapping to Rerun, More about Layers, and Rerun APIs demonstrated |
| every further layer module          | its rows in Mapping to Rerun, its More about Layers entry, its Rerun APIs demonstrated bullet                          |
| `urdf_layer.py`, `upload_asset.py`  | the model-rrd lines in 2. Convert and 4. Local Catalog, 5. Upload the shared model asset                               |
| `blueprint.py`                      | 3. View, the screenshot, 4. Upload the blueprint                                                                       |
| `catalog.py`                        | 4. Local Catalog                                                                                                       |
| `modal_jobs/`                       | Remote Convert Example on Modal, steps 1 to 3                                                                          |
| the final pass                      | the opening paragraph, the status line, Converted `.rrd` Dataset, References, the root README row                      |

Sections tied to one module are written once.
Mapping to Rerun, More about Layers, and Rerun APIs demonstrated grow by one entry per layer.
The final pass covers what depends on the whole: the opening paragraph names the layer set, and the bucket, revision, and SDK lines exist only after publication.
The root README table has the columns Dataset, Domain, Input, Rerun HF bucket, and Status.

## How to fill it

Placeholders follow the legend in the "Placeholders" section of `references/project-layout.md`: replace every `[…]`, and keep every `<…>` as it is.

Three more conventions:

- `<!-- … -->` comments are guidance for you.
  Delete them.
- A block marked optional is removed when the example has nothing for it.
  Never leave one as a stub.
- Unmarked sentences are the shared wording.
  Keep them unless the example contradicts them.

The skeleton assumes the source is hosted on Hugging Face.
`[org/repo]`, the `hf` commands, the token note, and the pinned revision all depend on it.
A source hosted elsewhere keeps the same sections, with those lines rewritten for how it is actually fetched and pinned.

## Prose rules

- Describe only what the reader will see.
  Never document what the example lacks.
- Call out every view hidden behind a tab.
  They are easy to miss.
- Open each section with a full sentence, not a fragment.
- When the source is inconsistent and the conversion unifies it, say both halves: what the source mixes, and what the output unifies.
- Put gotchas in `> **Note:**` blockquotes, right after the command they concern.
- One sentence per line.
  Use a spaced em dash ( — ) for dashes and `…` for ellipses.
- Use plain, concise wording.
  Readability beats technical detail.

---

# [Dataset Name]

[[Dataset Name]]([project or dataset URL]) is [one sentence: what was recorded, on which robot, how many [episode]s, in which format].
This example converts each [episode] into [N] Rerun recordings (`.rrd`).
One recording corresponds to one layer: [the layers in plain words, e.g. "the raw streams, the robot FK, the camera geometry, and the episode metadata"].

<!-- Optional while the bucket is unpublished:
**Status: incubating.** [What works], and [what is still to come].
-->

Below is the viewer showing a converted [episode] with the default blueprint.

![[Dataset Name] in the Rerun viewer](screenshot.png)
<!-- A GitHub user-attachments video URL on its own line works in place of the image. -->

The [default blueprint](#3-view) puts [which view is where: the 3D scene, the cameras, the plots along the bottom].
<!-- Name the views behind tabs here, e.g. "an `RGB` and an `IR` tab switch between the two modalities of the same cameras". -->

There are two ways to run it.
The [local version](#local-runs) downloads [N] sample [unit]s, converts them, and registers them to a catalog you can query.
The [Modal](https://modal.com/)-based [remote version](#remote-convert-example-on-modal) converts the whole dataset into a storage bucket.

> **Note:** this example uses Pixi.
> Get it [here](https://pixi.prefix.dev/latest/installation/).
> Everything runs inside the pixi env: prefix task commands with `pixi run`, and direct tool commands (`hf`, `rerun`, `modal`) with `pixi run -e [env]`.
> File paths in the commands below are relative to the repository root.

## Dataset

- **Source**: [[org/repo]](https://huggingface.co/datasets/[org/repo]) on Hugging Face
- **License**: [license]. Converted artifacts are derived from the dataset, so redistributing them is governed by the same terms.
- **Subset used**: the local demo runs on [N] sample [unit]s (~[size]) spanning [tasks, suites, or dates], all listed in [observations.md](observations.md#[anchor]).
  The Modal job supports the full dataset conversion with options to filter or limit [unit]s.
- **Access**: public, not gated. <!-- Gated: "gated — accept the terms on Hugging Face, then authenticate (`pixi run -e [env] hf auth login`, or set `$HF_TOKEN`)." -->

This example does not redistribute the dataset.
Data is downloaded at runtime from the original Hugging Face repo.

### Converted `.rrd` Dataset

Converted recordings are published at [`[namespace]/[dataset]`](https://huggingface.co/buckets/[namespace]/[dataset]) — download them directly if you only want the Rerun data.
They are built from source revision [`[short sha]`](https://huggingface.co/datasets/[org/repo]/tree/[full sha]), dated [YYYY-MM-DD], with rerun-sdk [version].
<!-- Before publication the first sentence reads: "Converted recordings will be published to a Hugging Face bucket when ready." -->

## Local Runs

The source data revision is pinned in this example.
Bump `HF_REVISION` in [`[pkg]/episode_index.py`]([pkg]/episode_index.py) to pick up newer [unit]s.

### 1. Download

Download the [N] sample [unit]s (~[size]) into `data/[Dataset Name]/`:

```bash
pixi run -e [env] download
```

To download different [unit]s, edit `SAMPLES` in [`[pkg]/download.py`]([pkg]/download.py).

The downloader caches a listing of the whole repo in `.cache/hf_files.json.gz`.
Building it takes a while on the first run, and it is rebuilt when the dataset revision changes.

### 2. Convert ([FORMAT] → RRD)

Convert each downloaded [episode] into multiple Rerun recordings (`.rrd`) that share a `recording_id`.
The viewer/catalog stacks them as **layers** of one logical recording: a base layer that holds the raw source, plus other layers that add to it ([the layers in plain words]).
Each layer can be added, replaced, or re-run without touching the others.
The robot meshes are written once for the whole dataset, as the shared model rrd `rrds/[dataset]/assets/urdf-model.rrd`. <!-- only with a urdf layer -->

Build every layer, for the whole set or one [unit]:

```bash
pixi run -e [env] convert            # every downloaded [unit] under data/[Dataset Name]/
pixi run -e [env] convert <path>     # a single [unit file]
```

> **Note:** This example also includes its own task for each layer (`convert-base`, `convert-[layer]`, …) writing the corresponding `.rrd`.

See [More about Layers](#more-about-layers) for what each layer contains.

<!-- Optional, when the conversion offers a lossy media option:
#### Notes on Video Transcode
Say what `convert` does by default (pass-through), what the re-encode changes (codec, GOP, resolution, intrinsics), which variant the published bucket holds, and the CPU cost per [episode].
-->

### 3. View

View a result in the Rerun Viewer:

```bash
pixi run -e [env] rerun rrds/[dataset]/*/*.rrd        # every [episode]
pixi run -e [env] rerun rrds/[dataset]/*/<id>.rrd     # one [episode]
```

> Keep the `*` to load all layers.

Generate the default blueprint, then view one [episode] with it:

```bash
pixi run -e [env] blueprint
pixi run -e [env] rerun rrds/[dataset]/*/<id>.rrd blueprints/[dataset]/default.rbl
```

> **Note:** [what a reader would not guess: which eye appears in 3D, which views are behind tabs, how array series are labelled].
> To modify the layout, edit `[pkg]/blueprint.py` and rerun the `blueprint` task.

> **Note:** Viewed from files alone, the [episode] shows no robot meshes. <!-- only with a urdf layer -->
> To see the posed robot with its meshes, move to the next step.

### 4. Local Catalog

Register the converted [episode]s to a [catalog server](https://rerun.io/docs/concepts/how-does-rerun-work#catalog-server), then browse, sort, filter, and query the segments as one dataset.
Once registered, [episode]s become queryable segments with named layers.

Start a local server:

```bash
pixi run serve              # start an in-memory catalog server (leave running)
```

In another shell, register converted [episode]s and the default blueprint to a local catalog:

```bash
pixi run -e [env] register   # register all [episode]s as the `[dataset]` dataset
```

> **Note:** On a catalog, each [episode] becomes one segment, keyed by its `recording_id`.
> Each `.rrd` of that [episode] attaches as one named layer of the segment ([the layer names]).
> The `register` task creates the dataset, attaches each [episode]'s RRDs as its named layers, and installs `blueprints/[dataset]/default.rbl` as the default blueprint (generate it first with `pixi run -e [env] blueprint`).
> It also registers the shared model rrd as the dataset's `urdf-model` [asset](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model#assets). <!-- only with a urdf layer -->

Browse them in the Rerun Viewer:

```sh
pixi run -e [env] rerun rerun+http://127.0.0.1:51234
```

<!-- Optional:
The video below shows what it looks like.

[GitHub user-attachments video URL]
-->

## Remote Convert Example on Modal

The steps above run locally on the downloaded sample [unit]s.
The following steps convert the full dataset on cloud workers.

### 1. Prerequisite: storage backend

Set the env vars for your storage backend in the shell, or edit the defaults in [`rrd_datasets_common/storage.py`](../../packages/rrd_datasets_common/rrd_datasets_common/storage.py) (backend, buckets) and [`rrd_datasets_common/modal_jobs/store.py`](../../packages/rrd_datasets_common/rrd_datasets_common/modal_jobs/store.py) (role ARN).
Your own `export` wins over the defaults.

| Env var                                           | Backend | What it is                                                               |
| ------------------------------------------------- | ------- | ------------------------------------------------------------------------ |
| `STORAGE_BACKEND`                                 | both    | Which bucket kind, `hf` or `s3` — `hf` in the `[env]` environments       |
| `HF_NAMESPACE`                                    | hf      | The user or org that owns the bucket — set this one                      |
| `HF_BUCKET`                                       | hf      | Bucket the RRDs are written to, `[dataset]` by default — create it first |
| `HF_BUCKET_ACCESS_KEY_ID` / `…_SECRET_ACCESS_KEY` | hf      | The HF S3 credentials                                                    |

This example converts to a [Hugging Face Storage Bucket](https://huggingface.co/docs/hub/main/en/storage-buckets-s3) behind its S3-compatible gateway, reached with `boto3` like any S3 bucket.
Access uses [HF S3 credentials](https://huggingface.co/docs/hub/storage-buckets-s3#generating-s3-credentials): an access key ID prefixed `HFAK…` and a secret access key.
Generate them from a fine-grained HF token scoped to the bucket.
The launcher passes them to the workers as an ephemeral per-run secret.
The dataset's own layout under the bucket is defined in [`storage.py`]([pkg]/storage.py).

> **Note:** to store in an AWS S3 bucket instead, `export STORAGE_BACKEND=s3` and follow the [S3 prerequisite in the ABC-130k example](../abc-130k/README.md#1-prerequisite-s3-storage).

### 2. Prerequisite: Modal setup

The [Modal](https://modal.com/) job under `[pkg]/modal_jobs/` runs one worker per [unit]: each worker downloads its [unit], converts it, and uploads the `.rrd` layers to a Hugging Face bucket.

To set it up:

- `pixi run -e [env] modal setup` — authenticate the Modal CLI (one-time).
- `pixi run -e [env] hf auth login`, or set `$HF_TOKEN` — optional for this public dataset, but anonymous callers share a smaller per-IP download quota.
  <!-- Gated: "The token lists the gated dataset and reaches the workers as an ephemeral per-run secret, so nothing is stored on Modal to refresh." -->

### 3. Run Convert

Run `pixi run -e [env] convert-on-modal --help` to see all options.

```bash
# One new [unit], every layer it can have (the default when no flags are given):
pixi run -e [env] convert-on-modal

# Every [unit] (--limit 0 removes the cap):
pixi run -e [env] convert-on-modal --limit 0

# Rebuild the first 5 [unit]s of one task:
pixi run -e [env] convert-on-modal --task-filter [task] --limit 5 --overwrite

# See what would run, without spawning anything:
pixi run -e [env] convert-on-modal --dry-run --limit 10
```

<!-- Name the filter flag the launcher offers, such as `--task-filter` or `--path-filter`. -->

The `convert-on-modal` task runs detached and returns immediately.
Watch progress in the Modal dashboard.

> **Note:** Without `--overwrite`, anything already in the bucket is skipped.
> The launcher spawns no worker for a [unit] whose layers are all present.
> A [unit] missing even one layer still gets a worker, which then builds only what is missing.

#### Picking layers

`--layers` lets you choose which layers to build:

```bash
# Only the base layer:
pixi run -e [env] convert-on-modal --layers base --limit 0

# Rebuild layers after changing them:
pixi run -e [env] convert-on-modal --layers [layer],[layer] --limit 0 --overwrite
```

A worker downloads only what the selected layers read.
<!-- When every layer reads one file: "Every layer reads the same [unit file], so a worker downloads it whatever the selection." -->

### 4. Upload the blueprint

`pixi run -e [env] blueprint` writes `blueprints/[dataset]/default.rbl`.
To upload it to your HF bucket (`s3://<bucket>/blueprints/`), run:

```bash
pixi run -e [env] upload-blueprint
```

<!-- Optional, only with a urdf layer: -->

### 5. Upload the shared model asset

Run `pixi run -e [env] convert-urdf` once locally.
It will build the asset model rrd as `rrds/[dataset]/assets/urdf-model.rrd`.
To upload it to your HF bucket (`s3://<bucket>/assets/`), run:

```bash
pixi run -e [env] upload-asset
```

## Observations

We share our observations including useful details beyond the dataset card.
See [observations.md](observations.md) for the full survey.

## Mapping to Rerun

The base layer keeps the [episode] as `[Reader]` emits it: [what that means for this source, e.g. "every dataset is a column named after itself, every attribute a static column, dtypes and array widths unchanged"].
<!-- State every exception with its reason, e.g. "Two exceptions: the camera datasets are reshaped into upright `Image`s, and `states` becomes a variable-length list — its width varies with the scene, and a catalog dataset needs one schema across every demo." -->

The table shows the entity path and the layer of each source item.
<!-- Optional: "The datasets that repeat others are listed in [observations.md](observations.md#redundancies); they are kept, not plotted." -->

| Source            | Entity path        | Component / archetype | Layer      | Notes                                 |
| ----------------- | ------------------ | --------------------- | ---------- | ------------------------------------- |
| `[source item]`   | `/[path]`          | `[Archetype]`         | base       | [units, ranges, shapes, as stored]    |
| `[source item]`   | `/[path]`          | `[Archetype]`         | [layer]    | [what this layer computes, from what] |
| [metadata fields] | segment properties | —                     | properties | [the property names]                  |

<!-- A "Shown in" column naming the blueprint view can replace Notes when the layout has many views. -->

No `Scalars` are derived.
The default blueprint plots the arrays straight from their columns through component mappings and names the series there ([`[pkg]/blueprint.py`]([pkg]/blueprint.py)).

<!-- Optional, when the blueprint plots a subset of what the structs hold:
### Also in the recording

Every other field is in the same structs, ready to plot: add the entity to a time series view, or map a series onto a field the way `blueprint.py` does.

- **[Group]** — `[field]` in `[struct path]`.
-->

### Round trip test ([FORMAT] → RRD → [FORMAT])

[What the tests check, and on which inputs: a synthetic fixture and a downloaded [unit].]
[Which tests compare bytes, and which only check structure or values.]

## More about Layers

Each layer is a separate module that writes its own `.rrd`.

### Base layer

`[pkg]/base_layer.py` — Everything in the [source file], as recorded.
[What stays whole, which bookkeeping comes along, and the few lenses the source demands.]
[Which sidecars are logged beside it.]
<!-- With a census: "A census compares the decoded rows with the [FORMAT] summary; a channel that lost messages is flagged (`has_undecodable`, `undecodable_topics`) and its raw bytes are kept." -->

### [Layer] layer

`[pkg]/[layer]_layer.py` — This layer [what it computes, from which base columns].
[Which [episode]s skip it, when its input is optional.]
<!-- For a urdf layer, name the model, its origin, its license, and the vendored copy:
"The URDF is **not** part of the HF dataset. We use `[file]` from [source](url), vendored with its meshes under `urdf/[robot]/`. [Vendor] distributes it under the [license](url); a copy is included at [`urdf/[robot]/LICENSE`](urdf/[robot]/LICENSE)."
-->

### Properties layer

`[pkg]/properties_layer.py` — This layer adds per-[episode] metadata logged as recording properties, which the catalog shows as columns to filter, sort, and search on.

## Rerun APIs demonstrated

<!-- Keep the bullets this example uses, one per API, each ending with the module that uses it. -->

- [`McapReader`](https://ref.rerun.io/docs/python/stable/chunk/#rerun.chunk.McapReader) decodes the [unit] topics into chunk streams (`base_layer.py`).
- [`Hdf5Reader`](https://ref.rerun.io/docs/python/stable/experimental/#rerun.experimental.Hdf5Reader) reads each [episode] group into chunk streams as-is (`base_layer.py`).
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) turn the raw messages into what the viewer needs typed: [which components, from what] (`[module].py`).
- [Component mappings](https://rerun.io/docs/howto/visualization/plot-any-scalar) plot the [which] series straight out of the stored columns, so no `Scalars` are materialised (`blueprint.py`).
- [`rerun.urdf.UrdfTree`](https://ref.rerun.io/docs/python/stable/urdf/#rerun.urdf.UrdfTree) loads the vendored [robot] model and runs forward kinematics from the joint states (`urdf_layer.py`).
- [`DatasetEntry.register_asset`](https://ref.rerun.io/docs/python/stable/catalog/#rerun.catalog.DatasetEntry.register_asset) attaches the shared model rrd as the dataset's [asset](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model#assets), which the catalog merges into every segment (`catalog.py`).
- The [blueprint](https://rerun.io/docs/concepts/visualization/blueprints) API composes the default layout (`blueprint.py`).
- [`CatalogClient`](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model) registers each [episode] as a dataset segment with named layers and installs the default blueprint (`catalog.py`).

## References

- [Chunk processing API](https://rerun.io/docs/concepts/logging-and-ingestion/chunk-processing-api) — the reader + lens pipeline this conversion is built on.
- [Lenses](https://rerun.io/docs/concepts/query-and-transform/lenses) — reshaping/deriving components in-stream.

<!-- Add the upstream example or course closest to this conversion, when there is one. -->
