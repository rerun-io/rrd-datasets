# Observations template

`observations.md` is the survey: what the sample [unit]s showed that the dataset card does not say.
It is published with the example, so write it for someone who has the source dataset and no Rerun in the picture at all.
The README cites it section by section, and the conversion code refers to it whenever it handles a case that only some [episode]s have.

The template itself starts below the horizontal rule.

## Write it while you survey

Start the file in step 3, with the Summary table and the Example [episode]s table.
Add a section each time a later step turns up something the converter has to handle, such as a stream that only some [episode]s record, or two fields that disagree.
A finding that changed the conversion belongs here even when it is a single sentence.

## How to fill it

Placeholders follow the legend in the "Placeholders" section of `references/project-layout.md`: replace every `[…]`, and keep every `<…>` as it is.

Sections in the skeleton that a uniform source can drop: Example [episode]s, and Edge cases and data bugs.
Keep Summary and Source layout whatever the source looks like — "uniform, the same tree everywhere" is itself the finding.

Be clear that the survey is based on a subset, not the whole dataset.

The prose rules in `readme-template.md` apply here too.

---

# [Dataset Name] Observations

These findings come from surveying a subset of the source [unit]s, with a focus on details that aren't covered in the [dataset card]([dataset card URL]).
We're sharing them because they may be useful to others working with the dataset, especially when validating assumptions or building data pipelines around it.

The source revision is [`[short sha]`]([repo tree URL at that sha]).
The survey covers [N] [unit]s: [tasks, sessions, size, robot time], out of [the full dataset's counts].

## Summary

[One paragraph: the source's own counts, where they came from, and whether they disagree with the paper or project site.]
[Then one sentence saying whether the data is uniform or heterogeneous, which sets up the table.]

| Dimension           | Variation                            | Notes                                 |
| ------------------- | ------------------------------------ | ------------------------------------- |
| [Duration and size] | [Diverse / Identical / Two variants] | [the range, or what the variants are] |
| [Frame rates]       |                                      |                                       |
| [Resolution, codec] |                                      |                                       |
| [Calibration]       |                                      |                                       |
| [What else varies]  |                                      |                                       |

## Source layout

```
[the directory tree, topic list, or HDF5 hierarchy, one comment per line saying what it holds]
```

## Example [episode]s

The table below shows representative converted segments, with one example for each notable case.
The `Tag` column provides a short name for referring to each segment in the sections that follow.

| Tag     | Segment id | Demonstrates                        |
| ------- | ---------- | ----------------------------------- |
| [`tag`] | [id]       | [the one case this [episode] shows] |

## [One heading per finding]

[What varies, what two sources disagree about, or what is redundant.]
[Name the tag of an [episode] that shows it, and say what the conversion does about it.]

## Edge cases and data bugs

### [The bug, named as the reader would hit it]

[What the source does, how often, and what the converter does in response.]

## [Appendix: full task list, or other long listings]
