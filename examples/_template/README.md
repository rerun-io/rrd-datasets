# <Dataset name>

One or two sentences: what the dataset contains and what this example converts it into.

Converted recordings are published at <S3-compatible or Hugging Face bucket URL> — download them directly if you only want the `.rrd` data.

<Optional: screenshot or short clip of the resulting recording in the Rerun Viewer.>

## Dataset

- **Source**: <upstream URL, e.g. the Hugging Face dataset page>
- **License**: <upstream license, attribution requirements, and any redistribution constraints>
- **Subset used**: <which episodes/samples this example uses, and why>
- **Access**: <anonymous, or gated — how to authenticate>

This example does not redistribute the dataset.
<State whether data is downloaded at runtime, and from where.>

## Observations

What surveying the episodes revealed that the dataset card does not say.

- <what is consistent: container format, topic set, schemas — and where episodes deviate>
- <what varies: frame rates, resolutions, codecs, durations, file sizes>
- <quirks that shaped the conversion, e.g. two timing regimes forcing per-topic alignment>

<Link to a fuller survey document if one exists.>

## Mapping to Rerun

Explain how the source schema maps to entity paths, archetypes, and timelines.

| Source         | Entity path  | Archetype                              |
| -------------- | ------------ | -------------------------------------- |
| <topic/column> | <`/robot/…`> | <`Transform3D`, `Image`, `Scalars`, …> |

Rerun APIs demonstrated: <e.g. `McapReader`, lenses, `send_chunks`, blueprint>.

## Running

Download a small sample:

```sh
pixi run <dataset>-download
```

Convert it — writes <e.g. `rrds/<dataset>/<episode>.rrd`>:

```sh
pixi run <dataset>-convert
```

View the result:

```sh
pixi run rerun <path to .rrd>
```

<One line on the sample's approximate download size, conversion time, and memory.>

<Optional: Modal section — what the Modal job provisions, required secrets by name, where results are stored.>

## Known limitations

<Optional — drop the section if there are none.>

- <conversion gaps, unsupported topics, approximations>
