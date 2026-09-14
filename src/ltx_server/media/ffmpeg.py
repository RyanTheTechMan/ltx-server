"""Bounded streaming MP4 encoding. No torch imports or shell commands."""

import json
import os
import select
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import BinaryIO

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import check_cancelled
from ltx_server.jobs.state import OutputInfo
from ltx_server.media.processes import run_bounded


def decode_source_audio(path: Path, *, duration: float, cancel: Event, settings: Settings) -> bytes:
    """Decode only the requested span; stereo 48 kHz PCM, trim/pad without time stretching."""
    samples = round(duration * 48000)
    expected = samples * 2 * 4
    with tempfile.TemporaryFile() as pcm, tempfile.TemporaryFile() as stderr:
        process = None
        try:
            check_cancelled(cancel)
            process = subprocess.Popen(
                [
                    settings.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-t",
                    str(duration),
                    "-af",
                    f"aresample=48000,apad=whole_len={samples},atrim=end_sample={samples}",
                    "-ac",
                    "2",
                    "-ar",
                    "48000",
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                stdin=subprocess.DEVNULL,
                stdout=pcm,
                stderr=stderr,
            )
            deadline = time.monotonic() + settings.media_probe_timeout_seconds
            while process.poll() is None:
                check_cancelled(cancel)
                if time.monotonic() >= deadline or os.fstat(pcm.fileno()).st_size > expected:
                    raise ServiceError(
                        ErrorCode.INVALID_MEDIA, "Source audio decoding exceeded limits", 422
                    )
                cancel.wait(0.05)
            check_cancelled(cancel)
            pcm.seek(0)
            data = pcm.read(expected + 1)
            if process.returncode != 0 or len(data) != expected:
                raise ServiceError(
                    ErrorCode.INVALID_MEDIA, "Source audio could not be decoded", 422
                )
            return data
        except FileNotFoundError:
            raise ServiceError(
                ErrorCode.MEDIA_UNAVAILABLE, "FFmpeg is not installed", 503
            ) from None
        except OSError:
            raise ServiceError(
                ErrorCode.INVALID_MEDIA, "Source audio decoding failed", 422
            ) from None
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()


def _write(process: subprocess.Popen[bytes], data: bytes, cancel: Event, seconds: float) -> None:
    assert process.stdin is not None
    fd = process.stdin.fileno()
    remaining = memoryview(data)
    deadline = time.monotonic() + seconds
    while remaining:
        check_cancelled(cancel)
        if process.poll() is not None or time.monotonic() >= deadline:
            raise ServiceError(ErrorCode.OUTPUT_ENCODING_FAILED, "FFmpeg exited or stopped reading")
        if not select.select([], [fd], [], 0.1)[1]:
            continue
        try:
            count = os.write(fd, remaining[:65536])
        except BlockingIOError:
            continue
        remaining = remaining[count:]
        deadline = time.monotonic() + seconds


def encode_mp4(
    chunks: Iterable[bytes],
    path: Path,
    *,
    width: int,
    height: int,
    frames: int,
    fps: int,
    cancel: Event,
    settings: Settings,
    audio: BinaryIO | None = None,
    audio_rate: int = 48000,
    audio_channels: int = 2,
    on_flush: Callable[[], None] = lambda: None,
) -> None:
    """Read planar BT.709 limited-range YUV420 bytes and optional interleaved f32le audio.

    Audio is in an anonymous temporary file shared by descriptor, never a new named
    artifact. Decoding and encoding overlap one video chunk at a time. FFmpeg must
    fully exit before the caller may publish or remove the partial path.
    """
    args = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv420p",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
    ]
    passed: tuple[int, ...] = ()
    if audio is not None:
        audio.seek(0)
        passed = (audio.fileno(),)
        args += [
            "-f",
            "f32le",
            "-ar",
            str(audio_rate),
            "-ac",
            str(audio_channels),
            "-i",
            f"pipe:{audio.fileno()}",
        ]
    args += [
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        settings.video_preset,
        "-crf",
        str(settings.video_crf),
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
    ]
    if audio is not None:
        args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-af", "apad"]
    args += ["-t", str(frames / fps), "-movflags", "+faststart", "-f", "mp4", str(path)]
    process = None
    with tempfile.TemporaryFile() as stderr:
        try:
            check_cancelled(cancel)
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                pass_fds=passed,
                bufsize=0,
            )
            assert process.stdin is not None
            os.set_blocking(process.stdin.fileno(), False)
            size = 0
            for chunk in chunks:
                size += len(chunk)
                if size > frames * width * height * 3 // 2:
                    raise ServiceError(ErrorCode.OUTPUT_ENCODING_FAILED, "Too many decoded pixels")
                _write(process, chunk, cancel, settings.ffmpeg_stall_timeout_seconds)
            if size != frames * width * height * 3 // 2:
                raise ServiceError(
                    ErrorCode.OUTPUT_ENCODING_FAILED, "Unexpected decoded frame count"
                )
            process.stdin.close()
            on_flush()
            deadline = time.monotonic() + settings.ffmpeg_stall_timeout_seconds
            while process.poll() is None:
                check_cancelled(cancel)
                if time.monotonic() >= deadline:
                    raise ServiceError(
                        ErrorCode.OUTPUT_ENCODING_FAILED, "FFmpeg finalization timed out"
                    )
                cancel.wait(0.05)
            if process.returncode != 0:
                raise ServiceError(ErrorCode.OUTPUT_ENCODING_FAILED, "FFmpeg could not encode MP4")
            check_cancelled(cancel)
        except FileNotFoundError:
            raise ServiceError(
                ErrorCode.MEDIA_UNAVAILABLE, "FFmpeg is not installed", 503
            ) from None
        except (BrokenPipeError, OSError):
            raise ServiceError(ErrorCode.OUTPUT_ENCODING_FAILED, "FFmpeg output failed") from None
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if process.stdin is not None:
                    process.stdin.close()


def inspect_output(
    path: Path,
    settings: Settings,
    *,
    width: int,
    height: int,
    frames: int,
    fps: int,
    has_audio: bool,
    cancel: Event | None = None,
) -> OutputInfo:
    try:
        raw = run_bounded(
            [
                settings.ffprobe_path,
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,"
                "nb_read_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            cancel or Event(),
            settings.media_probe_timeout_seconds,
        )
        probe = json.loads(raw)
        videos = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
        audios = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
        video = videos[0]
        duration = float(video.get("duration", probe["format"]["duration"]))
        if (
            len(videos) != 1
            or bool(audios) != has_audio
            or video["codec_name"] != "h264"
            or (int(video["width"]), int(video["height"])) != (width, height)
            or int(video["nb_read_frames"]) != frames
            or Fraction(video["avg_frame_rate"]) != fps
            or abs(duration - frames / fps) > 1 / fps
        ):
            raise ValueError("Output metadata differs from request")
        if has_audio and (
            len(audios) != 1
            or audios[0]["codec_name"] != "aac"
            or abs(float(audios[0]["duration"]) - duration) > 0.15
        ):
            raise ValueError("Output audio is not synchronized")
        return OutputInfo(
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            duration=duration,
            has_audio=has_audio,
            size_bytes=path.stat().st_size,
        )
    except ServiceError as exc:
        if exc.detail.code in {ErrorCode.GENERATION_CANCELLED, ErrorCode.MEDIA_UNAVAILABLE}:
            raise
        raise ServiceError(
            ErrorCode.OUTPUT_VALIDATION_FAILED, "Encoded MP4 failed video/audio validation"
        ) from None
    except FileNotFoundError:
        raise ServiceError(ErrorCode.MEDIA_UNAVAILABLE, "ffprobe is not installed", 503) from None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        raise ServiceError(
            ErrorCode.OUTPUT_VALIDATION_FAILED, "Encoded MP4 failed video/audio validation"
        ) from None
