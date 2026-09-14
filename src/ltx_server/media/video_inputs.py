"""Cancellable, bounded video preparation and strict retake source validation."""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import Any, BinaryIO

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import check_cancelled
from ltx_server.media.ffmpeg import inspect_output
from ltx_server.media.processes import run_bounded
from ltx_server.media.validation import container
from ltx_server.schemas.generation import GenerationSpec


@dataclass(frozen=True)
class VideoSource:
    has_audio: bool
    codec: str


def _input_safety(source: Path) -> list[str]:
    demuxer, _, _ = container(source)
    return [
        "-protocol_whitelist",
        "file,pipe",
        "-format_whitelist",
        demuxer,
        "-probesize",
        "5000000",
        "-analyzeduration",
        "5000000",
        "-threads",
        "2",
    ]


def _probe(
    path: Path, settings: Settings, cancel: Event, *, count_frames: bool = False
) -> dict[str, Any]:
    raw = run_bounded(
        [
            settings.ffprobe_path,
            "-v",
            "error",
            *_input_safety(path),
            *(["-count_frames"] if count_frames else []),
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames:"
            "stream_tags=rotate:stream_side_data=rotation:stream_disposition=attached_pic",
            "-of",
            "json",
            str(path),
        ],
        cancel,
        settings.media_probe_timeout_seconds,
    )
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or not isinstance(value.get("streams"), list):
            raise ValueError("Invalid probe")
        return value
    except (ValueError, TypeError):
        raise ServiceError(ErrorCode.INVALID_MEDIA, "Invalid video metadata", 422) from None


def _normalize(
    source: Path,
    destination: Path,
    spec: GenerationSpec,
    settings: Settings,
    cancel: Event,
    *,
    preserve_audio: bool,
) -> None:
    has_audio = preserve_audio and any(
        s.get("codec_type") == "audio" for s in _probe(source, settings, cancel)["streams"]
    )
    duration = spec.frames / spec.fps
    # FFmpeg applies display rotation before these filters. fps/aresample anchor both
    # streams at zero while retaining any delay between them. Padding never stretches time.
    # Crop in source coordinates first: equivalent to filling/center-cropping the
    # target, without allocating an enormous intermediate for extreme aspect ratios.
    filters = (
        f"fps={spec.fps}:start_time=0,"
        f"crop=w='min(iw,ih*{spec.width}/{spec.height}/sar)':"
        f"h='min(ih,iw*sar*{spec.height}/{spec.width})',"
        f"scale={spec.width}:{spec.height},setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={duration},trim=end_frame={spec.frames}"
    )
    limit = settings.max_prepared_video_mb * 1024 * 1024
    args = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-filter_threads",
        "1",
        *_input_safety(source),
        "-i",
        str(source),
        "-map",
        "0:V:0",
        "-vf",
        filters,
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        args += [
            "-map",
            "0:a:0",
            "-af",
            f"aresample=48000:async=1:first_pts=0,apad,atrim=duration={duration}",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    else:
        args += ["-an"]
    args += [
        "-t",
        str(duration),
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-fs",
        str(limit),
        "-f",
        "mp4",
        str(destination),
    ]
    run_bounded(
        args,
        cancel,
        settings.media_prepare_timeout_seconds,
        destination=destination,
        max_bytes=limit,
    )
    check_cancelled(cancel)
    inspect_output(
        destination,
        settings,
        width=spec.width,
        height=spec.height,
        frames=spec.frames,
        fps=spec.fps,
        has_audio=has_audio,
        cancel=cancel,
    )
    check_cancelled(cancel)


def prepare_reference(
    source: Path, destination: Path, spec: GenerationSpec, settings: Settings, cancel: Event
) -> None:
    _normalize(source, destination, spec, settings, cancel, preserve_audio=False)


def prepare_retake(
    source: Path, destination: Path, spec: GenerationSpec, settings: Settings, cancel: Event
) -> None:
    _normalize(source, destination, spec, settings, cancel, preserve_audio=True)


def validate_retake_source(
    path: Path, spec: GenerationSpec, settings: Settings, cancel: Event | None = None
) -> VideoSource:
    """Reject implicit rotation, resizing, FPS changes and frame trimming."""
    probe = _probe(path, settings, cancel or Event(), count_frames=True)
    try:
        stream = next(
            s
            for s in probe["streams"]
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        )
        rotations = [stream.get("tags", {}).get("rotate", 0)] + [
            s.get("rotation", 0) for s in stream.get("side_data_list", [])
        ]
        if (
            (stream["width"], stream["height"]) != (spec.width, spec.height)
            or Fraction(stream["avg_frame_rate"]) != spec.fps
            or Fraction(stream["r_frame_rate"]) != spec.fps
            or int(stream["nb_read_frames"]) != spec.frames
            or any(float(rotation) % 360 != 0 for rotation in rotations)
        ):
            raise ValueError("Mismatched source grid")
        return VideoSource(
            has_audio=any(s.get("codec_type") == "audio" for s in probe["streams"]),
            codec=str(stream["codec_name"]),
        )
    except (ValueError, KeyError, IndexError, TypeError, StopIteration, ZeroDivisionError):
        raise ServiceError(
            ErrorCode.INVALID_INPUT,
            "Retake source must be upright and match the requested dimensions, FPS and "
            "normalized frame count; opt into retake.normalize_source to prepare it",
            422,
        ) from None


def encode_source_video(
    source: Path,
    destination: Path,
    spec: GenerationSpec,
    settings: Settings,
    cancel: Event,
    *,
    codec: str,
    audio: BinaryIO | None,
    audio_rate: int,
) -> None:
    """Keep source video for audio-only edits; copy prepared H.264 without a VAE round trip."""
    args = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        *_input_safety(source),
        "-i",
        str(source),
    ]
    passed: tuple[int, ...] = ()
    if audio is not None:
        audio.seek(0)
        passed = (audio.fileno(),)
        args += ["-f", "f32le", "-ar", str(audio_rate), "-ac", "2", "-i", f"pipe:{audio.fileno()}"]
    args += ["-map", "0:V:0", "-c:v", "copy" if codec == "h264" else "libx264"]
    if codec != "h264":
        args += [
            "-threads",
            "2",
            "-preset",
            settings.video_preset,
            "-crf",
            str(settings.video_crf),
            "-pix_fmt",
            "yuv420p",
        ]
    if audio is not None:
        args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-af", "apad"]
    else:
        args += ["-an"]
    limit = settings.max_prepared_video_mb * 1024 * 1024
    args += [
        "-t",
        str(spec.frames / spec.fps),
        "-map_metadata",
        "-1",
        "-fs",
        str(limit),
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(destination),
    ]
    run_bounded(
        args,
        cancel,
        settings.media_prepare_timeout_seconds,
        destination=destination,
        max_bytes=limit,
        pass_fds=passed,
    )
