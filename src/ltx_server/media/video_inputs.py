"""Bounded reference normalization and strict retake source validation."""

import json
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from threading import Event

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import check_cancelled
from ltx_server.media.ffmpeg import inspect_output
from ltx_server.schemas.generation import GenerationSpec


def prepare_reference(
    source: Path, destination: Path, spec: GenerationSpec, settings: Settings, cancel: Event
) -> None:
    filters = (
        f"fps={spec.fps},scale={spec.width}:{spec.height}:force_original_aspect_ratio=increase,"
        f"crop={spec.width}:{spec.height},setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={spec.frames / spec.fps}"
    )
    with tempfile.TemporaryFile() as stderr:
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
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    filters,
                    "-frames:v",
                    str(spec.frames),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-f",
                    "mp4",
                    str(destination),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
            deadline = time.monotonic() + settings.media_probe_timeout_seconds
            while process.poll() is None:
                check_cancelled(cancel)
                if time.monotonic() >= deadline:
                    raise ServiceError(
                        ErrorCode.INVALID_MEDIA, "Reference normalization timed out", 422
                    )
                cancel.wait(0.05)
            check_cancelled(cancel)
            if process.returncode:
                raise ServiceError(
                    ErrorCode.INVALID_MEDIA, "Reference video could not be normalized", 422
                )
        except FileNotFoundError:
            raise ServiceError(
                ErrorCode.MEDIA_UNAVAILABLE, "FFmpeg is not installed", 503
            ) from None
        except OSError:
            raise ServiceError(
                ErrorCode.INVALID_MEDIA, "Reference conversion failed", 422
            ) from None
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
    inspect_output(
        destination,
        settings,
        width=spec.width,
        height=spec.height,
        frames=spec.frames,
        fps=spec.fps,
        has_audio=False,
    )


def validate_retake_source(path: Path, spec: GenerationSpec, settings: Settings) -> None:
    """Retake preserves the source grid; reject implicit resizes, FPS changes or trimming."""
    try:
        result = subprocess.run(
            [
                settings.ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=settings.media_probe_timeout_seconds,
        )
        stream = json.loads(result.stdout)["streams"][0]
        if (
            (stream["width"], stream["height"]) != (spec.width, spec.height)
            or Fraction(stream["avg_frame_rate"]) != spec.fps
            or Fraction(stream["r_frame_rate"]) != spec.fps
            or int(stream["nb_read_frames"]) != spec.frames
        ):
            raise ValueError("Mismatched source grid")
    except FileNotFoundError:
        raise ServiceError(ErrorCode.MEDIA_UNAVAILABLE, "ffprobe is not installed", 503) from None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        raise ServiceError(
            ErrorCode.INVALID_INPUT,
            "Retake source must match the requested dimensions, FPS and normalized frame count",
            422,
        ) from None
