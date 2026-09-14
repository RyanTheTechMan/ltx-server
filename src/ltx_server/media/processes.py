"""Bounded synchronous media processes for the dedicated worker thread."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Event

from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import check_cancelled


def run_bounded(
    args: list[str],
    cancel: Event,
    seconds: float,
    *,
    destination: Path | None = None,
    max_bytes: int = 0,
    pass_fds: tuple[int, ...] = (),
) -> bytes:
    """Reap every process before returning; cap probe output and prepared media size."""
    with tempfile.TemporaryFile() as stdout:
        process = None
        try:
            check_cancelled(cancel)
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.DEVNULL,
                pass_fds=pass_fds,
            )
            deadline = time.monotonic() + seconds
            while True:
                finished = process.poll() is not None
                check_cancelled(cancel)
                if time.monotonic() >= deadline:
                    raise ServiceError(ErrorCode.INVALID_MEDIA, "Video preparation timed out", 422)
                if os.fstat(stdout.fileno()).st_size > 1024 * 1024 or (
                    destination is not None
                    and destination.exists()
                    and destination.stat().st_size > max_bytes
                ):
                    raise ServiceError(
                        ErrorCode.INVALID_MEDIA, "Video preparation exceeded size limits", 422
                    )
                if finished:
                    break
                cancel.wait(0.05)
            if process.returncode:
                raise ServiceError(ErrorCode.INVALID_MEDIA, "Video preparation failed", 422)
            stdout.seek(0)
            return stdout.read(1024 * 1024 + 1)
        except FileNotFoundError:
            raise ServiceError(
                ErrorCode.MEDIA_UNAVAILABLE, "FFmpeg/ffprobe is not installed", 503
            ) from None
        except OSError:
            raise ServiceError(ErrorCode.INVALID_MEDIA, "Video preparation failed", 422) from None
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
