import fcntl
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

from ltx_server.config import Settings

Area = Literal["outputs", "tmp", "assets"]
MANAGED_NAME = re.compile(r"(?:gen|asset)_[0-9a-f]{32}\.(?:mp4|partial|bin)$")


def new_id(prefix: Literal["gen", "asset"]) -> str:
    return f"{prefix}_{uuid4().hex}"


class Storage:
    """Flat, service-owned directories. Client filenames are never used as paths.

    Storage must be writable only by the service account. This is not a boundary
    against another process with that account's permissions.
    """

    def __init__(self, settings: Settings) -> None:
        assert settings.output_dir and settings.temp_dir and settings.asset_dir
        self.roots: dict[Area, Path] = {
            "outputs": settings.output_dir,
            "tmp": settings.temp_dir,
            "assets": settings.asset_dir,
        }
        self._locks: list[BinaryIO] = []

    def open(self) -> None:
        try:
            for root in sorted(self.roots.values()):
                # Reject symlinks at every path component, including existing ancestors.
                if any(p.is_symlink() for p in (root, *root.parents)):
                    raise ValueError("Storage directories must not contain symlinks")
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
                fd = os.open(
                    root / ".ltx-server.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
                )
                lock = os.fdopen(fd, "rb+")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    lock.close()
                    raise RuntimeError(
                        "Storage is in use; run exactly one server process"
                    ) from None
                self._locks.append(lock)
            if self.roots["tmp"].stat().st_dev != self.roots["outputs"].stat().st_dev:
                raise ValueError("TEMP_DIR and OUTPUT_DIR must be on the same filesystem")
            if self.roots["tmp"].stat().st_dev != self.roots["assets"].stat().st_dev:
                raise ValueError("TEMP_DIR and ASSET_DIR must be on the same filesystem")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for lock in self._locks:
            lock.close()
        self._locks.clear()

    def path(self, area: Area, name: str) -> Path:
        if not MANAGED_NAME.fullmatch(name):
            raise ValueError("Invalid managed filename")
        path = self.roots[area] / name
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in managed storage")
        return path

    def create_partial(self, identifier: str) -> Path:
        path = self.path("tmp", f"{identifier}.partial")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        return path

    def publish(self, identifier: str, area: Literal["outputs", "assets"]) -> Path:
        source = self.path("tmp", f"{identifier}.partial")
        target = self.path(area, f"{identifier}.{'mp4' if area == 'outputs' else 'bin'}")
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("Cannot publish an empty or missing artifact")
        # Mark the start of retention at publication, not when encoding began.
        os.utime(source, None)
        os.replace(source, target)
        return target

    def remove(self, area: Area, name: str) -> None:
        self.path(area, name).unlink(missing_ok=True)

    def files(self, area: Area) -> Iterator[Path]:
        for path in self.roots[area].iterdir():
            if MANAGED_NAME.fullmatch(path.name) and not path.is_symlink() and path.is_file():
                yield path
