# RRD Datasets

Physical AI teams win by iterating quickly on data composition and modeling while scaling data and compute.

Open datasets are an important part of that process, but they often arrive in different formats and schemas.
Using them effectively can require substantial dataset-specific work before they are ready to feed into training pipelines.

This repository provides examples for converting open-source datasets into the [Rerun file format (`.rrd`)](https://rerun.io/docs/concepts/logging-and-ingestion/rrd-format), making them easier to inspect, combine, and use in training workflows.
Converted datasets are published on Hugging Face, with more to come.

Already using `.rrd`? You're in the right place.

New to Rerun? Start with [What is Rerun?](https://rerun.io/docs/overview/what-is-rerun), [How does Rerun work?](https://rerun.io/docs/concepts/how-does-rerun-work), and [A new data layer for robot learning](https://rerun.io/blog/data-layer-for-robot-learning).


## Quickstart

Install Pixi:

    curl -fsSL https://pixi.sh/install.sh | sh

Clone the repository:

    git clone https://github.com/rerun-io/rrd-datasets.git
    cd rrd-datasets

Run an example end to end — download a sample episode, convert it, and open the result in the Rerun Viewer:

    pixi run abc-demo

Run `pixi task list` to see all available tasks.


## Datasets

| Dataset | Domain | Input | Rerun HF bucket | Status |
|---------|--------|-------|-----------------|--------|
| [ABC-130k](examples/abc-130k) | Bi-manual arm | MCAP | [`rerun/abc-130k`](https://huggingface.co/buckets/rerun/abc-130k) | ✅ |
| [HIW-500](examples/hiw-500) | Humanoid | MCAP | N/A | 🚧 |
| *More to come* | | | | |


## How it works

Each example follows the same pipeline:

    Download a sample of the source dataset
          ↓
    Convert it to .rrd and generate a blueprint
          ↓
    Inspect the recordings in the Rerun Viewer
          ↓
    Serve the recordings and register them to a catalog
          ↓
    Query the catalog from your training code

Each stage is a pixi task named `<dataset>-<stage>`, such as `abc-download` or `abc-convert`.

The examples intentionally favor readable conversion code over abstraction.


## Repository structure

    examples/       one directory per dataset
    packages/       shared utilities used by the examples

Each example contains its own conversion code and a README with the dataset-specific details: source, license, mapping to Rerun, and how to run it.


## Adding a dataset

A new example is a directory under `examples/`:

    examples/<dataset>/
        README.md
        download.py
        convert.py

Start from [`examples/_template`](examples/_template).


## License

Code in this repository is dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Individual datasets retain their original licenses.
See each dataset's README for source and licensing information.
