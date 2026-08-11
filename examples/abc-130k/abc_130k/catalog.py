"""
Register the converted ABC-130k RRDs into a local Rerun catalog as one dataset.

The flow mirrors the cloud ingestion pattern, with two pixi tasks:

    pixi run serve           # start the in-memory Rerun catalog (leave running)
    pixi run abc-register    # in another shell: register every converted episode

Each episode's `.rrd` becomes one *segment* (id = its `recording_id`), and the default blueprint is
installed on the dataset. Connect a viewer to the catalog URL to browse, filter, and query across
episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from abc_130k.blueprint import BLUEPRINT_PATH
from abc_130k.convert import APPLICATION_ID, OUT_DIR

DEFAULT_CATALOG_URL = "rerun+http://127.0.0.1:51234"
DATASET_NAME = APPLICATION_ID


def _episode_rrds(rrd_dir: Path) -> list[Path]:
    """Every converted episode in `rrd_dir` — the `<recording_id>.rrd` files the converter writes."""
    return sorted(rrd_dir.glob("*.rrd"))


def register_episodes(
    catalog_url: str,
    dataset_name: str,
    rrd_dir: Path,
    blueprint: Path,
    *,
    recreate: bool = False,
) -> DatasetEntry:
    """
    Register every episode as one segment of the dataset and set the blueprint.

    The dataset is created if missing, and segments that are already there are replaced.
    With `recreate`, an existing dataset of the same name is deleted first and rebuilt from scratch.
    """
    rrds = _episode_rrds(rrd_dir)
    if not rrds:
        raise FileNotFoundError(f"No episode .rrd files in {rrd_dir} — run `pixi run abc-convert` first.")

    client = CatalogClient(catalog_url)
    if recreate and dataset_name in client.dataset_names():
        client.get_dataset(dataset_name).delete()
    dataset = client.create_dataset(dataset_name, exist_ok=True)

    uris = [p.resolve().as_uri() for p in rrds]
    dataset.register(uris, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    print(f"Registered {len(uris)} episode(s)")

    if blueprint.exists():
        dataset.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
        print(f"Default blueprint: {blueprint}")
    else:
        print(f"No blueprint at {blueprint} (run `pixi run abc-blueprint`)")
    return dataset


def main() -> None:
    """Register every converted episode into the local catalog as one dataset."""
    parser = argparse.ArgumentParser(description="Register converted ABC-130k RRDs into a local Rerun catalog.")
    parser.add_argument("--rrd-dir", type=Path, default=OUT_DIR, help="Directory holding the per-episode .rrd files.")
    parser.add_argument(
        "--catalog-url", default=DEFAULT_CATALOG_URL, help="gRPC URL of the local catalog (`pixi run serve`)."
    )
    parser.add_argument(
        "--dataset-name", default=DATASET_NAME, help=f"Dataset to (re)create (default: {DATASET_NAME})."
    )
    parser.add_argument(
        "--blueprint", type=Path, default=BLUEPRINT_PATH, help="Blueprint (.rbl) to install as the dataset default."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete an existing dataset first and rebuild it from scratch.",
    )
    args = parser.parse_args()

    dataset = register_episodes(
        args.catalog_url, args.dataset_name, args.rrd_dir, args.blueprint, recreate=args.recreate
    )
    segments = dataset.segment_ids()
    print(f"Dataset '{dataset.name}' on {args.catalog_url} holds {len(segments)} segment(s):")
    for segment in segments:
        print(f"  - {segment}")


if __name__ == "__main__":
    main()
