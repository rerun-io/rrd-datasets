# Modal job template

The jobs in `[pkg]/modal_jobs/` process a dataset on [Modal](https://modal.com) and upload the results directly to the bucket.
Each job is one Python file with two parts:

- The **launcher** runs on your machine.
  It finds the items that still need work and hands each one to a worker.
- The **worker** runs on Modal and processes one item per call.

There are two kinds of job:

|              | Conversion job                                                | Append job                                                 |
| ------------ | ------------------------------------------------------------- | ---------------------------------------------------------- |
| File         | `convert_[unit]s.py`                                          | `append_[layer].py`                                        |
| Builds       | the base layer, and any layer that is cheap to derive from it | one or more layers derived from registered layers          |
| Worker reads | the source files                                              | the input layer, through the catalog                       |
| One item is  | a [unit]                                                      | a segment                                                  |
| Pending when | a selected layer is missing from the bucket                   | the input layer is registered and the derived layer is not |

`references/project-layout.md` defines a [unit] under "Terms", and its "Storage" section says where the output goes.
Step 6 of `SKILL.md` says when a layer runs as an append job.
Reuse the shared code in `rrd_datasets_common/modal_jobs/`, which builds the worker image and handles both storage backends.

Each kind of job below ends with a skeleton that leaves out the imports and constants.
Adapt it to the dataset instead of copying it whole.

## Rules for both kinds

### Launcher order

1. Call `check_bucket()`.
   On the `hf` backend it stops the launch when the bucket does not exist, before the launcher fetches the source listing.
   On s3 it checks nothing, because the launcher may have no AWS credentials.
2. Select the pending items.
3. Apply `--limit` to the pending items.
   Finished work is already removed at this point, so `--limit 5` starts five items that still need work.

### Flags

Every example's jobs take these flags, with the same meaning:

| Flag          | Meaning                                                              |
| ------------- | -------------------------------------------------------------------- |
| `--limit N`   | Process at most N pending items (default 1, 0 for all).              |
| `--dry-run`   | Print the pending items and exit.                                    |
| `--layers`    | Conversion job only. The layers to build, comma-separated, or `all`. |
| `--overwrite` | Rebuild layers that are already in the bucket.                       |

### Worker image

The worker does not use pixi: it installs the example's `[project.dependencies]` plus the `cloud` extra with pip.
A package that pixi takes only from conda-forge is missing on the worker, so list every package in `pyproject.toml`.

Install a native library that pip cannot provide with the `apt` argument of `image_from_pyproject`.
If Debian packages the wrong version, install the upstream build with `commands` instead, at the version pixi pins.
Then add a comment next to the pixi dependency that says where the image gets its copy.

Add files that the converter opens by path, such as a robot model, with `files`, at the path where the converter looks for them.

Bind every secret per run: the dataset token, and the bucket keys when the backend is `hf`.
On s3 the worker assumes a role instead, so it holds no long-lived key.
Never put a credential in the image.

### Tests

Test the launcher's selection without a dataset, a bucket, or a catalog, by patching the bucket listing or the segment table.
`tests/test_modal_prefilter.py` in the existing examples shows how.
Test a worker by running it: `--dry-run` first, then a run with no flags.
Then open the result in the viewer, after registering it for a conversion job.

### Pixi tasks

Add a task for each job next to the example's other tasks.
`--detach` keeps the workers running after the command returns.

```toml
[feature.[dataset]-tasks.tasks.convert-on-modal]
cmd = "modal run --detach -q -m [pkg].modal_jobs.convert_[unit]s"
cwd = "examples/[dataset]"
description = "Convert the dataset's [unit]s on Modal"

[feature.[dataset]-tasks.tasks.append-[layer]-on-modal]
cmd = "modal run --detach -q -m [pkg].modal_jobs.append_[layer]"
cwd = "examples/[dataset]"
description = "Append the [layer] layer to the registered dataset on Modal"
```

## Conversion job

### Selecting pending [unit]s

Build the work items from the episode index, filtered by `--task-filter`.
Then drop every [unit] whose selected layers are all in the bucket already.
If the bucket listing fails, start the workers anyway: each worker skips the layers that exist.

### Worker

Download the [unit]'s source files once, into a temporary directory, and build the base layer from them.
Download only the files the selected layers read.
Compare the recording id read from the source with the one the episode index assigned, and raise an error if they differ.
If another layer is cheap to derive, build it in the same worker from the base `.rrd` the worker just wrote.

Upload each layer to the same relative path it has locally under `rrds/[dataset]/`.
A copy of the bucket synced into that directory can then be registered without renaming any file.
Without `--overwrite`, skip every layer already in the bucket, so a rerun builds only the missing layers.

### Downloads

Give the worker enough memory, disk, and `timeout` for the largest [unit].
A worker that fails near the end of a long download has to start the download again.
If one source file is very large, consider a smaller [unit] in the episode index instead of a bigger worker.

Limit how many workers run at once with `MAX_CONTAINERS`, because Hugging Face rate-limits by the number of requests and each worker fetches several files.
`references/data-sources.md` covers the source's quotas and how to pass its token to the workers.
A source that is not on Hugging Face needs its own download call in the worker and its own secret.

### Skeleton

```python
"""Convert [Dataset Name] [unit]s to RRDs on Modal, one worker per [unit]."""

image = image_from_pyproject(
    REPO_ROOT / "pyproject.toml",
    extras=["cloud"],
    env={**HF_HUB_ENV, **worker_env()},
    python_sources=("[pkg]", "rrd_datasets_common"),
)
app = modal.App("[dataset]", image=image)


def layer_dest(layer: str, recording_id: str) -> str:
    """The bucket URI of a layer file, matching its local relative path."""
    return f"{DATASET_PREFIX}{layer_relpath(layer, recording_id)}"


@app.function(
    timeout=2 * HOUR,
    cpu=1.0,
    memory=2048,  # MiB, sized for the largest [unit]
    region=region_pin(),
    secrets=[hf_token_secret(), *extra_secrets()],
    max_containers=MAX_CONTAINERS,
)  # type: ignore[misc]
def convert_remote(item: WorkItem, layers: list[str], overwrite: bool) -> None:
    """Build the requested layers of one [unit] and upload each to the bucket."""
    s3 = worker_client()
    todo = [layer for layer in layers if overwrite or not s3_exists(s3, layer_dest(layer, item.recording_id))]
    if not todo:
        return
    with tempfile.TemporaryDirectory() as tmp:
        # [Download the source files the layers in `todo` read, and check the recording id.]
        # [Build "base", then the layers derived from it.]
        for layer in todo:
            rrd = ...  # [one layer's .rrd under Path(tmp)]
            upload_file(s3, str(rrd), layer_dest(layer, item.recording_id))


def parse_layers(value: str) -> list[str]:
    """The layers named in `value` (comma-separated, or `all`), in `LAYERS` order. An unknown name stops the launch."""


def drop_converted(items: list[WorkItem], layers: list[str]) -> list[WorkItem] | None:
    """Drop [unit]s whose selected layers are all in the bucket, or `None` when the bucket cannot be listed."""
    try:
        done = s3_existing_keys(launcher_client(), DATASET_PREFIX)
    except (BotoCoreError, ClientError):
        return None
    _, key_prefix = s3_parts(DATASET_PREFIX)
    return [
        item
        for item in items
        if not all(f"{key_prefix}{layer_relpath(layer, item.recording_id)}" in done for layer in layers)
    ]


@app.local_entrypoint()  # type: ignore[misc]
def main(*arglist: str) -> None:
    """List the [unit]s and start one Modal worker per pending [unit]."""
    parser = argparse.ArgumentParser(description="Convert [Dataset Name] to RRDs on Modal.")
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(args=arglist)

    check_bucket()
    layers = parse_layers(args.layers)
    matched = discover_[unit]s(HF_REPO_ID, args.task_filter)
    pending = None if args.overwrite else drop_converted(matched, layers)
    items = matched if pending is None else pending
    if args.limit > 0:
        items = items[: args.limit]
    if args.dry_run or not items:
        print("\n".join(item.recording_id for item in items))
        return
    convert_remote.spawn_map(items, [layers] * len(items), [args.overwrite] * len(items))
```

## Append job

An append job needs a catalog that Modal workers can reach, with its token bound per run.
The local `pixi run serve` catalog is not reachable from Modal.
Without a reachable catalog, the worker can read the input layer's `.rrd` from the bucket with `RrdReader`, and the `register` task registers the results afterwards.

### Selecting pending segments

Read each segment's layers from the `rerun_layer_names` column of `segment_table()`.
A segment is pending when it has the input layer but not the derived layer.
If the list is null, treat the layer as missing.
With `--overwrite`, segments that already have the derived layer stay selected.

### Worker

One container handles many segments.
Open the catalog and S3 clients, and load any model, once per container in `@modal.enter()`.

For each segment:

1. Query the input layer, and read only the entities the layer needs (`filter_segments`, then `filter_contents`, then `reader`).
   The `rerun-catalog-queries` skill covers query costs.
2. Build the layer with the same derivation function the local layer module uses.
3. Write the `.rrd` with the segment id as its recording id, and upload it to the same relative path the conversion job uses.
4. Register the file with `on_duplicate=REPLACE`, wait for the registration, and check that the segment now lists the layer.

When the file is already in the bucket and `--overwrite` is not set, skip steps 1 to 3 but still register the file.
An earlier run may have stopped between the upload and the registration, and the launcher keeps selecting the segment until the catalog lists the layer.

With `--overwrite`, the segment cannot be read between the upload and the end of its registration.
Do not run another job on the same dataset at that time.

Record where the layer came from in a property chunk (`provenance()` in the skeleton): the input layer, and the model and its pinned revision when a model built the layer.
Do not pin a region for GPU workers.
GPUs are scarce outside the large regions, and a pinned job waits in the queue without an error.

### Skeleton

The append job's two helpers and its worker:

```python
from datafusion import col, lit
from datafusion import functions as F


def pending_segments(dataset: DatasetEntry, input_layer: str, layer: str, overwrite: bool) -> list[str]:
    """Segments that have `input_layer` registered and, unless `overwrite`, not yet `layer`."""
    table = dataset.segment_table().filter(F.array_has(col("rerun_layer_names"), lit(input_layer)))
    if not overwrite:
        # A null layer list counts as missing, so the segment stays pending.
        table = table.filter(~F.coalesce(F.array_has(col("rerun_layer_names"), lit(layer)), lit(False)))
    return sorted(table.collect_column("rerun_segment_id").to_pylist())


def layer_names(dataset: DatasetEntry, segment_id: str) -> list[str]:
    """The layers the catalog lists for one segment."""
    names = dataset.filter_segments(segment_id).segment_table().collect_column("rerun_layer_names").to_pylist()
    return (names[0] or []) if names else []


@app.cls(
    gpu="[gpu]",
    timeout=HOUR,
    secrets=[catalog_token_secret(), *extra_secrets()],
    max_containers=MAX_CONTAINERS,
)  # type: ignore[misc]
class AppendWorker:
    """One container, many segments: the clients and any model load once."""

    @modal.enter()  # type: ignore[misc]
    def setup(self) -> None:
        self.dataset = CatalogClient(CATALOG_URL, token=os.environ["[TOKEN_VAR]"]).get_dataset(DATASET_NAME)
        self.s3 = worker_client()

    @modal.method()  # type: ignore[misc]
    def append(self, segment_id: str, overwrite: bool) -> None:
        dest = layer_dest(LAYER, segment_id)
        if overwrite or not s3_exists(self.s3, dest):
            view = self.dataset.filter_segments(segment_id).filter_contents([INPUT_ENTITY])
            frames = pa.table(view.reader(index=INDEX))  # 1. query the input layer
            chunks = [layer]_chunks(frames)  # 2. dataframe -> chunks -> transform
            with tempfile.TemporaryDirectory() as tmp:
                rrd = Path(tmp) / f"{LAYER}.rrd"
                stream = LazyChunkStream.from_iter([provenance(), *chunks])
                stream.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(  # 3. write and upload
                    str(rrd), application_id=APPLICATION_ID, recording_id=segment_id
                )
                upload_file(self.s3, str(rrd), dest)
        self.dataset.register([dest], layer_name=LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        if LAYER not in layer_names(self.dataset, segment_id):  # 4. register and confirm
            raise RuntimeError(f"{segment_id}: {LAYER} uploaded to {dest} but not registered")
```

The launcher calls `pending_segments()`, applies `--limit`, and starts the workers with `AppendWorker().append.spawn_map(...)`.
