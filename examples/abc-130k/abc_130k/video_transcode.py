"""
Re-encode camera `VideoStream`s to a fixed GOP.

Decode H.264 and H.265 sources and re-encode them to H.264 via PyAV.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple, cast

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from av.video.codeccontext import VideoCodecContext
from rerun.experimental import Chunk, LazyChunkStream, McapReader, MutateLens, Selector

DEFAULT_GOP = 60
DEFAULT_CRF = 23
DEFAULT_PRESET = "veryfast"  # x264 speed preset (faster = less CPU, larger files)
DEFAULT_THREADS = 1  # one thread per encoder (if set to 0, use all available cores)
DEFAULT_MAX_WIDTH = 640  # downscale 1920x1200 cameras to this width


@dataclass
class VideoSettings:
    """How to re-encode the camera streams. Pass one of these instead of five loose knobs."""

    gop_size: int = DEFAULT_GOP
    crf: int = DEFAULT_CRF
    preset: str = DEFAULT_PRESET
    threads: int = DEFAULT_THREADS
    max_width: int | None = DEFAULT_MAX_WIDTH


# VideoStream component ids.
_SAMPLE = "VideoStream:sample"
_CODEC = "VideoStream:codec"
_IS_KEYFRAME = "VideoStream:is_keyframe"
_VIDEO_COMPONENTS = [_SAMPLE, _CODEC, _IS_KEYFRAME]

# (Only) this resolution is downscaled by `max_width` (the ZED cameras)
_RESIZE_SOURCE = (1920, 1200)

# Pinhole components rescaled with a downscaled video.
_RESOLUTION = "Pinhole:resolution"
_IMAGE_FROM_CAMERA = "Pinhole:image_from_camera"  # 3x3 intrinsics, column-major: fx@0 fy@4 cx@6 cy@7

# Always re-encode to H.264; the source is decoded with the matching decoder.
_ENCODER = "libx264"
_DECODER: dict[rr.VideoCodec, str] = {rr.VideoCodec.H264: "h264", rr.VideoCodec.H265: "hevc"}  # PyAV decoder names

# `VideoStream:codec` reads back as a fourcc int.
_FOURCC: dict[int, rr.VideoCodec] = {
    0x61766331: rr.VideoCodec.H264,  # "avc1"
    0x68657631: rr.VideoCodec.H265,  # "hev1"
}

# Log labels (not decoder names -- H.265's decoder is "hevc").
_CODEC_LABEL: dict[rr.VideoCodec, str] = {rr.VideoCodec.H264: "H264", rr.VideoCodec.H265: "H265"}


# --------------------------------------------------------------------------------------
# downscale helpers
# --------------------------------------------------------------------------------------


def _target_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """Downscale a 1920x1200 source to `max_width` (aspect-preserving, even dims); others unchanged."""
    if max_width is None or (width, height) != _RESIZE_SOURCE:
        return width, height
    tw = max_width - max_width % 2
    th = round(height * tw / width)
    return tw, th - th % 2


def _set_resolution(target: tuple[int, int]) -> Any:
    """Lens callback: overwrite every `Pinhole:resolution` row with `target`."""
    wh = [float(target[0]), float(target[1])]
    return lambda arr: pa.array([wh] * len(arr), type=arr.type)


def _scale_intrinsics(scale: float) -> Any:
    """Lens callback: scale fx/fy/cx/cy of a column-major 3x3 `Pinhole:image_from_camera` by `scale`."""

    def fn(arr: Any) -> Any:
        rows = []
        for k in arr.to_pylist():
            if k is not None:
                k = list(k)
                for i in (0, 4, 6, 7):
                    k[i] *= scale
            rows.append(k)
        return pa.array(rows, type=arr.type)

    return fn


# --------------------------------------------------------------------------------------
# transcode
# --------------------------------------------------------------------------------------


def _column_samples(arr: Any) -> list[bytes | None]:
    """
    Per-row sample bytes from the `list<list<uint8>>` column (`None` if empty).

    Slices the flat uint8 buffer directly; `.to_pylist()` boxes every byte (~1000x slower).
    """
    outer_off = arr.offsets.to_numpy()  # row i -> [outer_off[i], outer_off[i + 1]) inner-list slots
    inner = arr.flatten()  # per-row byte lists, concatenated
    inner_off = inner.offsets.to_numpy()
    data = inner.values.to_numpy(zero_copy_only=True)  # one contiguous buffer for all rows
    valid = arr.is_valid().to_numpy(zero_copy_only=False)
    out: list[bytes | None] = []
    for i in range(len(arr)):
        if not valid[i] or outer_off[i + 1] == outer_off[i]:
            out.append(None)
        else:
            j = int(outer_off[i])  # size-1 wrapper: one inner list per row
            out.append(data[inner_off[j] : inner_off[j + 1]].tobytes())
    return out


class EncodedSample(NamedTuple):
    """One re-encoded sample: Annex B bytes and whether it is a keyframe."""

    data: bytes
    is_keyframe: bool


def _set_threads(ctx: Any, threads: int) -> None:
    """Pin codec threading (1 = single-threaded, 0 = all cores); frame-parallel is safe -- no B-frames (#10090)."""
    ctx.thread_count = threads
    if threads != 1:
        ctx.thread_type = "FRAME"


def _make_encoder(width: int, height: int, gop_size: int, crf: int, preset: str, threads: int) -> VideoCodecContext:
    """H.264 encoder pinned to an exact closed GOP of `gop_size` (every keyframe an IDR, as Rerun needs)."""
    enc = cast(VideoCodecContext, av.CodecContext.create(_ENCODER, "w"))
    enc.width = width
    enc.height = height
    enc.pix_fmt = "yuv420p"
    enc.time_base = Fraction(1, 1)
    _set_threads(enc, threads)
    enc.options = {
        "preset": preset,
        "crf": str(crf),
        # Keyframe every N, no scene-cut keyframes, no B-frames (#10090).
        "x264-params": f"keyint={gop_size}:min-keyint={gop_size}:scenecut=0:bframes=0",
    }
    return enc


def _decode_frames(samples: Iterable[bytes], decoder: str, threads: int) -> Iterator[Any]:
    """Decode `samples` with the `decoder` codec, in order."""
    dec = cast(VideoCodecContext, av.CodecContext.create(decoder, "r"))
    _set_threads(dec, threads)
    for sample in samples:
        yield from dec.decode(av.Packet(sample))
    yield from dec.decode(None)  # flush


def _encode_frames(
    frames: Iterable[Any], gop_size: int, crf: int, preset: str, threads: int, max_width: int | None = None
) -> Iterator[EncodedSample]:
    """Re-encode `frames` to H.264 at a fixed GOP (optionally downscaled); yield `EncodedSample`s in order."""
    encoder: Any = None
    for index, frame in enumerate(frames):
        tw, th = _target_size(frame.width, frame.height, max_width)
        if encoder is None:
            encoder = _make_encoder(tw, th, gop_size, crf, preset, threads)
        if (frame.width, frame.height) != (tw, th) or frame.format.name != "yuv420p":
            frame = frame.reformat(width=tw, height=th, format="yuv420p")
        frame.pts = index
        frame.time_base = Fraction(1, 1)
        # Drop the source picture type so keyframes land by keyint, not the source's GOP.
        frame.pict_type = av.video.frame.PictureType.NONE
        for packet in encoder.encode(frame):
            yield EncodedSample(bytes(packet), bool(packet.is_keyframe))
    if encoder is not None:
        for packet in encoder.encode(None):  # flush
            yield EncodedSample(bytes(packet), bool(packet.is_keyframe))


def transcode_to_gop(
    samples: Iterable[bytes],
    gop_size: int,
    crf: int = DEFAULT_CRF,
    codec: rr.VideoCodec = rr.VideoCodec.H264,
    preset: str = DEFAULT_PRESET,
    threads: int = DEFAULT_THREADS,
    max_width: int | None = None,
) -> Iterator[EncodedSample]:
    """
    Decode `codec` `samples` and re-encode to H.264 at an exact GOP of `gop_size`; yield in order.

    Lazy: yields each sample as it is encoded. No B-frames (#10090), so output is 1:1 and in order.
    `max_width` downscales 1920x1200 sources (see `_target_size`).
    """
    frames = _decode_frames(samples, _DECODER[codec], threads)
    return _encode_frames(frames, gop_size, crf, preset, threads, max_width)


# --------------------------------------------------------------------------------------
# stream assembly
# --------------------------------------------------------------------------------------


def _time_column(name: str, arrow_type: pa.DataType, values_ns: list[int]) -> Any:
    """Rebuild a `TimeColumn` for `name` from int-ns `values_ns`, matching the original Arrow type."""
    # `timestamp=`/`duration=` read bare ints as seconds; pass ns explicitly as numpy ns dtypes.
    if pa.types.is_timestamp(arrow_type):
        return rr.TimeColumn(name, timestamp=np.array(values_ns, dtype="datetime64[ns]"))
    if pa.types.is_duration(arrow_type):
        return rr.TimeColumn(name, duration=np.array(values_ns, dtype="timedelta64[ns]"))
    return rr.TimeColumn(name, sequence=values_ns)


def _video_entities(mcap_path: Path) -> tuple[dict[str, int | None], dict[str, tuple[int, int]]]:
    """One camera-topic scan: `({video entity: codec fourcc}, {-camera-info entity: (w, h)})`."""
    reader = McapReader(str(mcap_path), include_topic_regex=[r"-camera(-info)?$"]).stream()
    entities: set[str] = set()
    codecs: dict[str, int] = {}
    resolutions: dict[str, tuple[int, int]] = {}
    for ch in reader.to_chunks():
        rb = ch.to_record_batch()
        names = rb.schema.names
        if any(n.endswith(_SAMPLE) for n in names):
            entities.add(ch.entity_path)
        codec_col = next((n for n in names if n.endswith(_CODEC)), None)
        if codec_col is not None and ch.entity_path not in codecs:
            for v in rb.column(codec_col).to_pylist():
                iv = v[0] if isinstance(v, list) and v else v
                if iv is not None:
                    codecs[ch.entity_path] = int(iv)
                    break
        res_col = next((n for n in names if n.endswith(_RESOLUTION)), None)
        if res_col is not None and ch.entity_path not in resolutions:
            for v in rb.column(res_col).to_pylist():
                wh = v
                while isinstance(wh, list) and wh and isinstance(wh[0], list):
                    wh = wh[0]  # unwrap list nesting
                if isinstance(wh, list) and len(wh) == 2 and wh[0] is not None:
                    resolutions[ch.entity_path] = (int(wh[0]), int(wh[1]))
                    break
    return {e: codecs.get(e) for e in sorted(entities)}, resolutions


class _TimelineIndex:
    """
    Records each sample's timeline values as its bytes stream through, so the output keeps the original timestamps.

    `capture` yields the bytes and records the values; `columns` rebuilds the `TimeColumn`s.
    """

    def __init__(self) -> None:
        self.names: list[str] = []  # timeline names
        self.types: dict[str, Any] = {}  # timeline name -> Arrow type
        self.rows: list[list[int]] = []  # per-row ns values, aligned to `names`

    def capture(self, stream: Any) -> Iterator[bytes]:
        """Yield each sample's bytes in arrival (= timeline) order; record its timeline values."""
        for ch in stream.to_chunks():
            rb = ch.to_record_batch()
            sample_col = next((n for n in rb.schema.names if n.endswith(_SAMPLE)), None)
            if sample_col is None:
                continue
            if not self.names:
                self.names = [n for n in ch.timeline_names if n in rb.schema.names]
                self.types = {n: rb.schema.field(n).type for n in self.names}
            cols = {n: rb.column(n).cast(pa.int64()).to_pylist() for n in self.names}
            samples = _column_samples(rb.column(sample_col))
            for i in range(rb.num_rows):
                sample = samples[i]
                if sample is not None:
                    self.rows.append([cols[n][i] for n in self.names])
                    yield sample

    def columns(self, which: list[int] | None = None) -> list[Any]:
        """`TimeColumn`s for all rows, or only the row indices in `which`."""
        rows = self.rows if which is None else [self.rows[i] for i in which]
        return [_time_column(n, self.types[n], [r[j] for r in rows]) for j, n in enumerate(self.names)]


def _group_gops(samples: Iterable[EncodedSample]) -> Iterator[list[EncodedSample]]:
    """Group an encoded sample stream into GOPs, starting a new group at each keyframe (one GOP held at a time)."""
    gop: list[EncodedSample] = []
    for sample in samples:
        if sample.is_keyframe and gop:
            yield gop
            gop = []
        gop.append(sample)
    if gop:
        yield gop


def _transcode_entity(
    mcap_path: Path, entity: str, codec: rr.VideoCodec, video_setting: VideoSettings
) -> Iterator[Chunk]:
    """Stream one camera's samples through the re-GOP transcode; yield fresh `VideoStream` chunks, one GOP per chunk."""
    stream = McapReader(str(mcap_path), include_topic_regex=[f"^{re.escape(entity)}$"]).stream()
    index = _TimelineIndex()
    samples = transcode_to_gop(
        index.capture(stream),
        video_setting.gop_size,
        video_setting.crf,
        codec,
        video_setting.preset,
        video_setting.threads,
        video_setting.max_width,
    )
    gops = _group_gops(samples)

    emitted = 0
    for gop in gops:
        if emitted == 0:  # codec is static; emit once, ahead of the first sample chunk.
            yield Chunk.from_columns(entity, indexes=[], columns=rr.VideoStream.columns(codec=[rr.VideoCodec.H264]))
        rows = list(range(emitted, emitted + len(gop)))
        yield Chunk.from_columns(
            entity, indexes=index.columns(rows), columns=rr.VideoStream.columns(sample=[e.data for e in gop])
        )
        # Each GOP opens with exactly one IDR (bframes=0); log it sparsely at the GOP's first row.
        yield Chunk.from_columns(
            entity, indexes=index.columns([emitted]), columns=rr.VideoStream.columns(is_keyframe=[True])
        )
        emitted += len(gop)

    if emitted and emitted != len(index.rows):
        raise ValueError(f"{entity}: {emitted} packets from {len(index.rows)} samples (source has B-frames?)")


def _rescale_pinholes(
    stream: LazyChunkStream, resolutions: dict[str, tuple[int, int]], max_width: int | None
) -> LazyChunkStream:
    """Rescale each 1920x1200 `Pinhole` (resolution + intrinsics) to match the downscaled video."""
    if max_width is None:
        return stream
    info = [e for e, wh in resolutions.items() if wh == _RESIZE_SOURCE]
    if not info:
        return stream
    target = _target_size(*_RESIZE_SOURCE, max_width)
    scale = target[0] / _RESIZE_SOURCE[0]
    return stream.lenses(
        [
            MutateLens(_RESOLUTION, Selector(".").pipe(_set_resolution(target))),
            MutateLens(_IMAGE_FROM_CAMERA, Selector(".").pipe(_scale_intrinsics(scale))),
        ],
        content=info,
        output_mode="forward_unmatched",
    )


def regop_camera_streams(
    mcap_path: Path,
    base: LazyChunkStream,
    video_setting: VideoSettings | None = None,
    verbose: bool = False,
) -> LazyChunkStream:
    """
    Re-encode every camera `VideoStream` to an H.264 stream with a fixed GOP; pass the rest of `base` through.

    Non-video data (incl. camera `CoordinateFrame`) passes through lazily; each camera's samples are
    read from its own topic and transcoded on the fly. `VideoSettings.threads` sets the per-encoder core
    count (1 = single-threaded; 0 = all cores). `verbose` logs the codecs.
    """
    video_setting = video_setting if video_setting is not None else VideoSettings()
    entities, resolutions = _video_entities(mcap_path)
    if not entities:
        return base
    rest = base.drop(components=_VIDEO_COMPONENTS)
    rest = _rescale_pinholes(rest, resolutions, video_setting.max_width)
    videos = []
    labels: set[str] = set()
    for entity, fourcc in entities.items():
        codec = _FOURCC.get(fourcc) if fourcc is not None else None
        if codec is None:
            found = f"0x{fourcc:08x}" if fourcc is not None else "none"
            raise ValueError(f"{entity}: unsupported codec ({found}); only H.264 (avc1) and H.265 (hev1) supported")
        labels.add(_CODEC_LABEL[codec])
        videos.append(LazyChunkStream.from_iter(_transcode_entity(mcap_path, entity, codec, video_setting)))
    if verbose:
        downscaled = video_setting.max_width is not None and _RESIZE_SOURCE in resolutions.values()
        resize = (
            f", downscale {_RESIZE_SOURCE[0]}x{_RESIZE_SOURCE[1]} to {video_setting.max_width}w" if downscaled else ""
        )
        print(f"source video codec: {', '.join(sorted(labels))} -> re-encode to H264 ({_ENCODER}){resize}")
    return LazyChunkStream.merge(rest, *videos)
