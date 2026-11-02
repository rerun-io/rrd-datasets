"""The stats layer of downloaded episodes: the source statistics come back unchanged, and the spreads tell the arms apart."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from rerun.chunk import LazyStore, RrdReader

from molmoact2.episode_index import LOCAL_DIR, recording_id
from molmoact2.stats_layer import PROPERTY, STATS, STATS_ARCHETYPE, read_episode_stats, write_layer

# Episodes in which only the left arm moves, only the right arm moves, and neither moves (`no-motion`).
LEFT_ONLY = 295
RIGHT_ONLY = 503
STILL = 1832


def flat_values(values: pa.Array) -> np.ndarray:
    """Every value in a list column, with the nesting removed."""
    while pa.types.is_list(values.type):
        values = values.flatten()
    flat: np.ndarray = values.to_numpy()
    return flat


def static_batch(store: LazyStore, entity: str) -> pa.RecordBatch:
    (chunk,) = store.stream().filter(content=entity).to_chunks()
    return chunk.to_record_batch()


@pytest.mark.skipif(not (LOCAL_DIR / "meta" / "info.json").exists(), reason="needs `pixi run -e molmo download`")
def test_downloaded_stats_round_trip(tmp_path: Path) -> None:
    stats = read_episode_stats(LOCAL_DIR)
    spreads = {}
    for episode in (LEFT_ONLY, RIGHT_ONLY, STILL):
        store = RrdReader(str(write_layer(recording_id(episode), stats, tmp_path))).store()
        row = stats.episodes.slice(episode, 1)
        for feature in stats.features:
            batch = static_batch(store, f"/stats/{feature}")
            for stat in STATS:
                source = flat_values(row[f"stats/{feature}/{stat}"].combine_chunks())
                written = flat_values(batch.column(f"{STATS_ARCHETYPE}:{stat}"))
                np.testing.assert_array_equal(written, source, err_msg=f"{episode} {feature} {stat}")
        state = static_batch(store, "/stats/observation.state")
        assert state.column(f"{STATS_ARCHETYPE}:names").to_pylist() == [stats.names["observation.state"]]

        properties = static_batch(store, f"/__properties/{PROPERTY}")
        spreads[episode] = {name: flat_values(properties.column(name))[0] for name in stats.spreads}

    for moving, arm, other in ((LEFT_ONLY, "left", "right"), (RIGHT_ONLY, "right", "left")):
        assert spreads[moving][f"{arm}_joint_spread"] > 1
        assert spreads[moving][f"{arm}_gripper_spread"] > 0.5
        assert spreads[moving][f"{other}_joint_spread"] < 0.01
        assert spreads[moving][f"{other}_gripper_spread"] < 0.01
    assert all(value < 0.01 for value in spreads[STILL].values())
