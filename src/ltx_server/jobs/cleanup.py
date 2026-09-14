import asyncio
import logging
from datetime import datetime, timedelta

from ltx_server.config import Settings
from ltx_server.jobs.manager import JobManager
from ltx_server.jobs.state import TERMINAL, utcnow
from ltx_server.media.assets import AssetManager
from ltx_server.media.storage import Area, Storage

logger = logging.getLogger(__name__)


class CleanupWorker:
    def __init__(
        self, settings: Settings, storage: Storage, assets: AssetManager, jobs: JobManager
    ) -> None:
        self.settings, self.storage, self.assets, self.jobs = settings, storage, assets, jobs
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def sweep(self, now: datetime | None = None) -> None:
        now = now or utcnow()
        for job in self.jobs.repository.all():
            self.jobs.expire(job, now)
            if (
                job.info.status in TERMINAL
                and job.info.completed_at
                and not job.readers
                and job.info.output is None
                and job.info.completed_at + timedelta(seconds=self.settings.job_record_ttl_seconds)
                <= now
            ):
                self.jobs.repository.remove(job.info.id)
        for identifier, record in list(self.assets.records.items()):
            if record.info.expires_at <= now and not record.leases:
                self.assets.delete(identifier)
        protected: dict[Area, set[str]] = {
            "assets": set(self.assets.records),
            "tmp": set(self.assets.uploading),
            "outputs": set(),
        }
        for job in self.jobs.repository.all():
            if job.info.status not in TERMINAL:
                protected["tmp"].add(job.info.id)
            if job.info.output is not None or job.readers:
                protected["outputs"].add(job.info.id)
        # On restart in-memory records are lost. Reap only recognizable abandoned files
        # according to mtime; unknown files, directories and symlinks are left alone.
        areas: list[tuple[Area, int]] = [
            ("tmp", self.settings.temp_ttl_seconds),
            ("outputs", self.settings.output_ttl_seconds),
            ("assets", self.settings.asset_ttl_seconds),
        ]
        for area, ttl in areas:
            for path in self.storage.files(area):
                if (
                    path.name.split(".", 1)[0] not in protected[area]
                    and path.stat().st_mtime + ttl <= now.timestamp()
                ):
                    path.unlink(missing_ok=True)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="ltx-cleanup-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), self.settings.cleanup_interval_seconds)
            except TimeoutError:
                try:
                    self.sweep()
                except Exception:
                    logger.exception("cleanup_failed")
