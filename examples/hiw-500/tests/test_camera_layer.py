"""Tests for the head-camera calibration parse and the no-calibration skip."""

from __future__ import annotations

from pathlib import Path

from hiw_500.base_layer import Episode, EpisodeInfo
from hiw_500.camera_layer import HeadCalibration, convert_episode

CALIB_YAML = """\
camera_matrix_left:
- [322.0, 0.0, 297.2]
- [0.0, 320.2, 246.1]
- [0.0, 0.0, 1.0]
camera_matrix_right:
- [320.4, 0.0, 305.6]
- [0.0, 319.1, 242.3]
- [0.0, 0.0, 1.0]
image_size: [640, 480]
baseline: 60.53
"""


def test_head_calibration_reads_per_eye_k_and_baseline(tmp_path: Path) -> None:
    path = tmp_path / "head_camera_params.yaml"
    path.write_text(CALIB_YAML)
    calib = HeadCalibration.from_yaml(path)
    assert calib.k_left[0, 0] == 322.0
    assert calib.k_right[0, 0] == 320.4
    assert (calib.width, calib.height) == (640, 480)
    assert abs(calib.baseline_m - 0.06053) < 1e-9


def test_an_episode_without_calibration_skips_the_layer(tmp_path: Path) -> None:
    episode = Episode(mcap=tmp_path / "e.mcap", info=EpisodeInfo(), recording_id="t__t__e", head_calib=None)
    assert convert_episode(episode, tmp_path) is None
    assert list(tmp_path.rglob("*.rrd")) == []
