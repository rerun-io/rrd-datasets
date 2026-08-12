# RRD Datasets

<p align="center">
  <img src="assets/rerun-wordmark-animation.gif" alt="Rerun wordmark" width="500">
</p>

Physical AI teams win by iterating quickly on data composition and modeling while scaling data and compute.

Open datasets are an important part of that process, but they often arrive in different formats and schemas.
Using them effectively can require substantial dataset-specific work before they are ready to feed into training pipelines.

This repository provides examples for converting open-source datasets into the [Rerun file format (`.rrd`)](https://rerun.io/docs/concepts/logging-and-ingestion/rrd-format), making them easier to inspect, combine, and use in training workflows.
Converted datasets are published on Hugging Face, with more to come.

Already using `.rrd`? You're in the right place.

New to Rerun? Start with [What is Rerun?](https://rerun.io/docs/overview/what-is-rerun), [How does Rerun work?](https://rerun.io/docs/concepts/how-does-rerun-work), and [A new data layer for robot learning](https://rerun.io/blog/data-layer-for-robot-learning).

## Quickstart

Dependencies are managed with [Pixi](https://pixi.sh). Install it:

    curl -fsSL https://pixi.sh/install.sh | sh

Clone the repository:

    git clone https://github.com/rerun-io/rrd-datasets.git
    cd rrd-datasets

For a quick demo, run an ABC-130k dataset example end to end — download sample episodes, convert them, and open the results in the Rerun Viewer:

    pixi run -e abc demo

Note that the dataset is gated on Hugging Face — accept its terms and authenticate first, see [examples/abc-130k](examples/abc-130k#dataset).

It downloads files about 900 MB in total under `data/`.

Run `pixi task list` to see all available tasks.

### Agent Skills

This repository was built using [Rerun's agent skills](https://github.com/rerun-io/rerun/tree/main/skills).
The set used is listed in [`skills-lock.json`](skills-lock.json); install them with:

```bash
npx skills experimental_install
```

## Datasets

The repository currently supports the following datasets, with additional integrations planned.

| Dataset                       | Domain        | Input | Rerun HF bucket                                                   | Status |
| ----------------------------- | ------------- | ----- | ----------------------------------------------------------------- | ------ |
| [ABC-130k](examples/abc-130k) | Bi-manual arm | MCAP  | [`rerun/abc-130k`](https://huggingface.co/buckets/rerun/abc-130k) | ✅     |
| HIW-500                       | Humanoid      | MCAP  | N/A                                                               | 🚧     |
| _More to come_                |               |       |                                                                   |        |

## Repository structure

The repository is organized around self-contained dataset examples, with shared utilities and committed viewer blueprints kept alongside them.

    examples/       one directory per dataset — the conversion code
    packages/       shared utilities used by the examples
    blueprints/     each dataset's committed default viewer layout
    data/  rrds/    created locally by the tasks: downloads and converted recordings (gitignored)

Each example is a small Python package in the following structure:

    examples/<dataset>/
        README.md       source, license, mapping to Rerun, how to run
        <dataset>/      the conversion code: download.py, convert.py, blueprint.py, catalog.py, …
        tests/

registering its pixi tasks in `pixi.toml`, and committing its default blueprint under `blueprints/<dataset>/`.

## How it works

Each example follows the same pipeline:

    📥 Download a sample of the source dataset
          ↓
    🔄 Convert it to .rrd and generate a blueprint
          ↓
    👀 Inspect the recordings in the Rerun Viewer
          ↓
    🗄️ Serve the recordings and register them to a catalog
          ↓
    🔎 Query the catalog for your curation/training

Each stage is a pixi task named after the stage, run in the dataset's environment — for example `pixi run -e abc download` or `pixi run -e abc convert`.

The examples intentionally favor readable conversion code over abstraction.

## License

Code in this repository is dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Individual datasets retain their original licenses.
See each dataset's README for source and licensing information.
