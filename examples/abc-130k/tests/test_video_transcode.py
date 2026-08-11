"""Tests for `video_transcode`: re-GOP a synthetic in-memory clip (no dataset needed)."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from typing import cast

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from av.video.codeccontext import VideoCodecContext

from abc_130k.video_transcode import (
    DEFAULT_PRESET,
    _group_gops,
    _make_encoder,
    _scale_intrinsics,
    _set_resolution,
    _target_size,
    transcode_to_gop,
)

_IDR = 5  # H.264 NAL unit type for an IDR (keyframe) slice


def _encode_synth(enc: VideoCodecContext, n_frames: int) -> list[bytes]:
    """Encode `n_frames` synthetic frames with `enc` to Annex B samples, one per frame."""
    samples: list[bytes] = []
    for i in range(n_frames):
        arr = np.full((enc.height, enc.width, 3), (i * 17) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
        frame.pts = i
        frame.time_base = Fraction(1, 1)
        samples += [bytes(p) for p in enc.encode(frame)]
    samples += [bytes(p) for p in enc.encode(None)]  # flush
    return samples


def _synth_clip(n_frames: int, gop: int = 10, width: int = 64, height: int = 48) -> list[bytes]:
    """H.264 Annex B samples, one per frame."""
    return _encode_synth(_make_encoder(width, height, gop, 23, DEFAULT_PRESET, threads=1), n_frames)


def _synth_h265_clip(n_frames: int, gop: int = 10, width: int = 64, height: int = 48) -> list[bytes]:
    """H.265 Annex B samples (test input only; production never encodes H.265)."""
    enc = cast(VideoCodecContext, av.CodecContext.create("libx265", "w"))
    enc.width, enc.height, enc.pix_fmt, enc.time_base = width, height, "yuv420p", Fraction(1, 1)
    enc.options = {"x265-params": f"keyint={gop}:min-keyint={gop}:log-level=error"}
    return _encode_synth(enc, n_frames)


def _nal_types(sample: bytes) -> list[int]:
    """NAL unit types (`nal_unit_type & 0x1F`) in an Annex B `sample`, in order."""
    types: list[int] = []
    i, n = 0, len(sample)
    while i < n:
        if sample[i : i + 3] == b"\x00\x00\x01":
            start = i + 3
        elif sample[i : i + 4] == b"\x00\x00\x00\x01":
            start = i + 4
        else:
            i += 1
            continue
        if start < n:
            types.append(sample[start] & 0x1F)
        i = start + 1
    return types


def test_frame_count_preserved() -> None:
    out = list(transcode_to_gop(_synth_clip(25, gop=10), gop_size=5))
    assert len(out) == 25


def test_exact_gop() -> None:
    gop = 5
    out = list(transcode_to_gop(_synth_clip(23, gop=10), gop_size=gop))
    keyframes = [i for i, s in enumerate(out) if _IDR in _nal_types(s.data)]
    assert keyframes == list(range(0, len(out), gop))


def test_is_keyframe_matches_idr() -> None:
    out = list(transcode_to_gop(_synth_clip(23, gop=10), gop_size=5))
    for s in out:
        assert s.is_keyframe == (_IDR in _nal_types(s.data))


def test_output_decodes() -> None:
    out = list(transcode_to_gop(_synth_clip(12, gop=10), gop_size=4))
    decoder = av.CodecContext.create("h264", "r")
    decoded = sum(len(decoder.decode(av.Packet(s.data))) for s in out)
    decoded += len(decoder.decode(None))  # flush
    assert decoded == len(out)


def test_gop_ge_clip_length_single_keyframe() -> None:
    out = list(transcode_to_gop(_synth_clip(8, gop=10), gop_size=30))
    assert [i for i, s in enumerate(out) if s.is_keyframe] == [0]


def test_streams_gop_by_gop_without_buffering() -> None:
    """The first GOP must arrive before the input is fully drained (no whole-video buffering)."""
    samples = _synth_clip(40, gop=10)
    pulled = 0

    def tracked() -> Iterator[bytes]:
        nonlocal pulled
        for s in samples:
            pulled += 1
            yield s

    gops = _group_gops(transcode_to_gop(tracked(), gop_size=5))
    first = next(gops)

    assert first[0].is_keyframe and len(first) == 5
    assert pulled < len(samples)  # a regression to list(...) would drain all 40 before yielding


def _decoded_size(samples: list[bytes]) -> tuple[int, int]:
    """Decode the first output sample and return its (width, height)."""
    dec = av.CodecContext.create("h264", "r")
    for s in samples:
        for f in dec.decode(av.Packet(s)):
            return f.width, f.height
    raise AssertionError("no frame decoded")


def test_max_width_downscales_1920x1200() -> None:
    """`max_width=640` downscales a 1920x1200 source to 640x400 (aspect-preserving), count preserved."""
    out = list(transcode_to_gop(_synth_clip(8, gop=10, width=1920, height=1200), gop_size=5, max_width=640))
    assert len(out) == 8
    assert _decoded_size([s.data for s in out]) == (640, 400)


def test_max_width_leaves_other_resolutions() -> None:
    """`max_width` only touches 1920x1200; an 848x480 source is unchanged."""
    out = list(transcode_to_gop(_synth_clip(8, gop=10, width=848, height=480), gop_size=5, max_width=640))
    assert _decoded_size([s.data for s in out]) == (848, 480)


def test_target_size() -> None:
    assert _target_size(1920, 1200, 640) == (640, 400)  # aspect-preserving
    assert _target_size(848, 480, 640) == (848, 480)  # not 1920x1200 -> unchanged
    assert _target_size(1920, 1200, None) == (1920, 1200)  # no max_width -> unchanged


def test_set_resolution() -> None:
    arr = pa.array([[1920.0, 1200.0]], type=pa.list_(pa.float32(), 2))
    out = _set_resolution((640, 400))(arr)
    assert out.to_pylist() == [[640.0, 400.0]]
    assert out.type == arr.type


def test_scale_intrinsics_column_major() -> None:
    # column-major K: fx@0, fy@4, cx@6, cy@7; the rest (zeros and the 1) stay put.
    k = [600.0, 0.0, 0.0, 0.0, 600.0, 0.0, 960.0, 600.0, 1.0]
    out = _scale_intrinsics(1 / 3)(pa.array([k], type=pa.list_(pa.float32(), 9))).to_pylist()[0]
    assert (out[0], out[4], out[6], out[7]) == (200.0, 200.0, 320.0, 200.0)
    assert out[8] == 1.0 and out[1] == 0.0


def test_h265_source_reencodes_to_h264() -> None:
    """An H.265 source decodes and re-encodes to H.264 with closed-GOP IDR keyframes, count preserved."""
    clip = _synth_h265_clip(23, gop=10)
    out = list(transcode_to_gop(clip, gop_size=5, codec=rr.VideoCodec.H265))
    assert len(out) == 23
    for s in out:  # output is H.264; labeled keyframes are real IDRs
        assert s.is_keyframe == (_IDR in _nal_types(s.data))
    assert [i for i, s in enumerate(out) if s.is_keyframe] == list(range(0, 23, 5))
    dec = av.CodecContext.create("h264", "r")
    decoded = sum(len(dec.decode(av.Packet(s.data))) for s in out) + len(dec.decode(None))
    assert decoded == len(out)
