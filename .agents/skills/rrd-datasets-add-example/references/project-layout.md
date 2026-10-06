# Project layout

Every example in `examples/` is built the same way.
This file lists the rules that keep them alike, says why each rule exists, and points at the code that implements it.
Code shared by the examples is in `packages/rrd_datasets_common/`.

## Terms

Episode, layer, recording id, and segment mean what the `rerun-dataset-conversion` skill defines under "Terms".
In libero, an episode is a demo.
This repo adds three terms:

- **Unit** — the piece of source data that one worker converts, written `[unit]` in the templates.
  By default it is one source file: one episode in hiw-500 and abc-130k, one task file holding many demos in libero.
- **Episode index** — the module that turns the source's file listing into the units to convert, and mints the recording id of each episode.
  `references/episode-index.md` is its template.
- **Asset** — a recording shared by every segment of a dataset, such as the robot meshes.

## Placeholders

The references write a value that changes per example in square brackets, such as `[pkg]`.
Replace each one with the example's value, so that none is left in a finished file.
Angle brackets, such as `<id>` or `<bucket>`, stand for a value that changes per run or per file, and stay as they are.

| Placeholder      | Stands for                                                                                    | Example            |
| ---------------- | --------------------------------------------------------------------------------------------- | ------------------ |
| `[Dataset Name]` | the display name, also the directory under `data/`                                            | `HIW-500`          |
| `[dataset]`      | the slug used for the example directory, the pixi feature, paths, the bucket, and the catalog | `hiw-500`          |
| `[env]`          | the pixi environment                                                                          | `hiw`              |
| `[pkg]`          | the Python package                                                                            | `hiw_500`          |
| `[org/repo]`     | the Hugging Face dataset id                                                                   | `BitRobot/HIW-500` |
| `[namespace]`    | the Hugging Face user or org that owns the bucket                                             | `rerun`            |
| `[FORMAT]`       | the source format                                                                             | `MCAP`, `HDF5`     |
| `[unit]`         | what one worker converts, as defined under "Terms"                                            | episode, task file |
| `[episode]`      | one episode, as `rerun-dataset-conversion` defines it                                         | episode, demo      |
| `[layer]`        | a layer name                                                                                  | `urdf`             |

Any other `[…]` is a one-off value to write in.

## One directory per dataset

```
examples/
    [dataset]/              one directory per dataset — the conversion code
        README.md           source, license, mapping table, how to run
        observations.md     the survey: what the samples showed beyond the dataset card
        [screenshot].png    the viewer image the README embeds
        pyproject.toml      the example's package: its dependencies and the rerun-sdk pin,
                            read by the worker image build and the pin test
        urdf/               the vendored robot model, meshes, and license — only with a urdf layer
        [pkg]/              download, episode index, layer modules, layer registry, convert,
                            blueprint, catalog, storage, upload_blueprint, upload_asset
            modal_jobs/     the Modal jobs
        notebook/           the catalog query notebook, where the example has one
        tests/
packages/
    rrd_datasets_common/    utilities shared by the datasets: path helpers,
                            HuggingFace repo access, storage backends, Modal image and jobs
        tests/              the pin test and the shared-utility tests
```

The names in square brackets change per example.
Everything else is fixed.

A new example reuses what `rrd_datasets_common/` already holds instead of writing its own copy.
A helper that a second example turns out to need moves there.

## Paths

Every input and output path comes from the helpers in `rrd_datasets_common/paths.py`.
No module builds a path relative to its own directory.

The whole pipeline rests on one rule: a layer of an episode has exactly one relative path, `<layer>/<recording_id>.rrd`.
That path is the same under the local output root `rrds/[dataset]/` and under the dataset's bucket prefix, and the asset files under `assets/` follow it too.
`layer_relpath(layer, recording_id)` is the one function that builds it.
Because both sides agree, a layer directory syncs between disk and bucket without renaming, and `register` reads either side as it is.

## The layer registry

`layers.py` holds one thing: the `LAYERS` tuple, which names the layers in build order.

Each layer is written by its own module, and each module can run on its own.
When an episode lacks the input a module needs, the module skips that episode instead of failing.
For example, hiw-500's `ir` module skips episodes that recorded no infrared streams.

`convert` runs the layer modules in registry order.
`register` attaches each layer directory to the dataset under the layer's name.
Only the catalog stores that name, not the `.rrd` file.

## Dataset assets

An asset is a recording that every segment of a dataset shares, such as the robot meshes.

A dataset can have several, each at `assets/<name>.rrd`.
hiw-500 and libero have one each, `assets/urdf-model.rrd`, written by `urdf_layer.py`.
`convert-urdf` builds it locally and `upload-asset` copies it to the bucket.
`register` attaches it with `register_asset`, and the catalog merges it into every segment.

## Stage names

Every example names its stages the same way.
The environment flag picks which example they run on (`-e hiw`, `-e libero`).

| Task               | What it does                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `download`         | Downloads the sample [unit]s into `data/[Dataset Name]/`.                                                        |
| `convert`          | Writes every layer for the downloaded episodes.                                                                  |
| `blueprint`        | Writes the viewer layouts, at least `default.rbl`.                                                               |
| `register`         | Registers the episodes on a catalog, attaches the assets, and installs `default.rbl` as the dataset's blueprint. |
| `convert-on-modal` | Converts the whole dataset on Modal workers, straight into the bucket.                                           |
| `upload-blueprint` | Copies the blueprints into the bucket.                                                                           |
| `upload-asset`     | Copies the shared assets into the bucket.                                                                        |

A task exists only where the dataset needs it.
abc-130k has no shared asset and so no `upload-asset`, and it adds a `demo` task of its own.
hiw-500 and libero also have one task per layer, named `convert-[layer]`.

## Dependencies and environments

An example's dependencies are declared in two files.
The root `pixi.toml` is the repository's only pixi manifest: every example registers a feature, a tasks feature, and two environments there.
The feature is named after the example directory (`hiw-500`), the environments are short (`hiw` and `hiw-dev`), and both environments also include `common`.
The example's `pyproject.toml` holds package metadata and the pip-installable dependency list, nothing else.

What goes where in the example's feature:

- `[feature.[dataset].dependencies]` — conda-forge packages: native libraries, type stubs, notebook kernels.
- `[feature.[dataset].pypi-dependencies]` — PyPI-only packages, and the example itself as an editable path dependency with its `cloud` extra, so the conversion modules run from the source tree.
- `[feature.[dataset].activation.env]` — `PACKAGE_DIR`, the directory the generic `py-fmt`, `py-lint`, and `tests` tasks run in.

How the two environments fit together:

- Both share one `solve-group`.
- `[env]-dev` lists the example feature before `dev`, because the first feature's `activation.env` wins and `PACKAGE_DIR` must be the example's.
- `[env]-dev` is added to the `py-lint-all` and `tests-all` task lists, which is how CI picks the example up.
- Dependencies only the tests need go in a third feature that just `[env]-dev` includes, as `hiw-500-dev` and `libero-test` do.

## The SDK pin

The bytes in an `.rrd` depend on the `rerun-sdk` version that wrote it.
If the local environment and the worker image run different versions, they write different recordings, and nothing reports an error.

So the pin is exact, and it is repeated wherever an environment is built.
`packages/rrd_datasets_common/tests/test_version_pins.py` fails when the copies drift apart.
The worker image is built from the example's `pyproject.toml`, so that dependency list has to be complete and pip-installable.

## Tests

An example keeps its tests in `examples/[dataset]/tests/`, and the shared utilities keep theirs in `packages/rrd_datasets_common/tests/`.
`pixi run -e [env]-dev tests` runs one package's tests, and `tests-all` runs every package's.
CI runs `py-fmt-check`, `py-lint-all`, and `tests-all`.
`check-all` runs those plus the markdown, typo, and link checks.

Add each layer module together with its test.
Nothing is downloaded before the tests run, so no test may assume that `data/` holds anything.
libero and abc-130k build a small synthetic input in the test itself, and hiw-500 reads a downloaded episode through a fixture that skips when `data/` is empty.
No sample file is committed.

## Storage

Each dataset owns one prefix, and everything it produces is stored under it.
On s3 that prefix is `s3://<bucket>/[dataset]/`, and on an HF bucket the dataset has the bucket to itself.
The dataset defines only what goes below it: the layer directories, `assets/`, and `blueprints/`.
A dataset can have several blueprints, and `register` installs `default.rbl` as the dataset's own.

The backend and the bucket come from environment variables (`STORAGE_BACKEND`, then `HF_NAMESPACE` and `HF_BUCKET`, or `S3_BUCKET` and `S3_REGION`).
The code holds placeholders for them.
No account name and no credential goes into the repo.

## Cloud jobs

The Modal jobs in `[pkg]/modal_jobs/` follow `references/modal-job.md`.

## abc-130k, an older layout

abc-130k predates the layer registry: it has no `layers.py`, and one recording per episode holds everything.
Read it for its decisions, not for its structure.
