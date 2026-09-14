import asyncio
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError


@dataclass(frozen=True)
class MediaInfo:
    type: Literal["image", "audio", "video"]
    mime_type: str
    width: int | None = None
    height: int | None = None
    duration: float | None = None


def inspect_image(path: Path, max_pixels: int) -> MediaInfo | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                if img.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ServiceError(ErrorCode.INVALID_MEDIA, "Use a PNG, JPEG or WebP image")
                width, height = img.size
                if width * height > max_pixels or getattr(img, "n_frames", 1) != 1:
                    raise ServiceError(ErrorCode.INVALID_MEDIA, "Image is too large or animated")
                mime = Image.MIME[img.format]
                img.verify()
            with Image.open(path) as decoded:
                decoded.load()
            return MediaInfo("image", mime, width, height)
    except UnidentifiedImageError:
        return None
    except (
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
        OSError,
        ValueError,
        SyntaxError,
    ):
        raise ServiceError(ErrorCode.INVALID_MEDIA, "Image cannot be safely decoded") from None


def container(path: Path) -> tuple[str, str, str]:
    """Sniff a narrow container allowlist; never feed playlists/URLs to FFmpeg."""
    with path.open("rb") as file:
        header = file.read(64)
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "mov", "video/mp4", "audio/mp4"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "matroska,webm", "video/webm", "audio/webm"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "wav", "", "audio/wav"
    if header.startswith(b"fLaC"):
        return "flac", "", "audio/flac"
    if header.startswith(b"OggS"):
        return "ogg", "video/ogg", "audio/ogg"
    if header.startswith(b"ID3") or (
        len(header) > 1 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
    ):
        return "mp3", "", "audio/mpeg"
    raise ServiceError(ErrorCode.INVALID_MEDIA, "Unsupported or unreadable media container", 415)


async def run_media_tool(executable: str, args: list[str], timeout_seconds: float) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise ServiceError(
            ErrorCode.MEDIA_UNAVAILABLE, "FFmpeg/ffprobe is not installed", 503
        ) from None
    try:
        async with asyncio.timeout(timeout_seconds):
            stdout, _ = await process.communicate()
        if process.returncode:
            raise ServiceError(ErrorCode.INVALID_MEDIA, "Media could not be decoded", 415)
        return stdout
    except TimeoutError:
        raise ServiceError(ErrorCode.INVALID_MEDIA, "Media validation timed out", 415) from None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def validate_media(path: Path, settings: Settings) -> MediaInfo:
    info = await asyncio.to_thread(inspect_image, path, settings.max_image_pixels)
    if info:
        return info
    demuxer, video_mime, audio_mime = await asyncio.to_thread(container, path)
    safety = [
        "-protocol_whitelist",
        "file,pipe",
        "-format_whitelist",
        demuxer,
        "-probesize",
        "5000000",
        "-analyzeduration",
        "5000000",
        "-threads",
        "1",
    ]
    raw = await run_media_tool(
        settings.ffprobe_path,
        [
            "-v",
            "error",
            *safety,
            "-show_entries",
            "format=duration,format_name:stream=codec_type,width,height:stream_disposition=attached_pic",
            "-of",
            "json",
            str(path),
        ],
        settings.media_probe_timeout_seconds,
    )
    try:
        probe: dict[str, Any] = json.loads(raw)
        duration = float(probe["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration")
        streams = probe.get("streams", [])
        video = next(
            (
                s
                for s in streams
                if s.get("codec_type") == "video"
                and not s.get("disposition", {}).get("attached_pic")
            ),
            None,
        )
        if video and video_mime:
            width, height = int(video["width"]), int(video["height"])
            if width <= 0 or height <= 0 or width * height > settings.max_image_pixels:
                raise ValueError("Invalid video size")
            info = MediaInfo("video", video_mime, width, height, duration)
        elif any(s.get("codec_type") == "audio" for s in streams):
            info = MediaInfo("audio", audio_mime, duration=duration)
        else:
            raise ValueError("No audio or video stream")
    except (ValueError, KeyError, TypeError):
        raise ServiceError(ErrorCode.INVALID_MEDIA, "Invalid audio/video metadata", 415) from None
    # Header probing alone accepts broken media. Decode a bounded sample of the selected stream.
    await run_media_tool(
        settings.ffmpeg_path,
        [
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            *safety,
            "-i",
            str(path),
            "-map",
            "0:v:0" if info.type == "video" else "0:a:0",
            "-t",
            "1",
            "-threads",
            "1",
            "-f",
            "null",
            "-",
        ],
        settings.media_probe_timeout_seconds,
    )
    return info
