import asyncio
import logging
import secrets
import time
from collections import deque
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

from pydantic import BaseModel

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ErrorDetail, ServiceError
from ltx_server.inference.backend import GenerationBackend, GenerationContext
from ltx_server.jobs.repository import Job, JobRepository, MemoryJobRepository
from ltx_server.jobs.state import TERMINAL, JobInfo, JobStatus, utcnow
from ltx_server.media.assets import AssetManager
from ltx_server.media.storage import Storage, new_id
from ltx_server.schemas.generation import GenerationRequest, normalize_request

logger = logging.getLogger(__name__)


class QueueInfo(BaseModel):
    running: str | None
    queued: list[str]
    length: int
    max_size: int
    accepting: bool


class JobManager:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        assets: AssetManager,
        backend: GenerationBackend,
        repository: JobRepository | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.assets = assets
        self.backend = backend
        self.repository = repository if repository is not None else MemoryJobRepository()
        self.pending: deque[str] = deque()
        self.running: str | None = None
        self.accepting = False
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("Worker already started")
        self.accepting = True
        self._worker = asyncio.create_task(self._run(), name="ltx-generation-worker")

    async def stop(self) -> None:
        self.accepting = False
        self.backend.begin_shutdown()
        for identifier in list(self.pending):
            self.delete(identifier)
        if self.running:
            self.delete(self.running)
        self._wake.set()
        if self._worker:
            await self._worker
            self._worker = None
        await self.backend.close()

    def submit(self, request: GenerationRequest) -> JobInfo:
        if not self.accepting:
            raise ServiceError(ErrorCode.SERVER_STOPPING, "Server is not accepting jobs", 503)
        spec = normalize_request(request, self.settings, secrets.randbits(32))
        self.backend.validate_request(spec)
        if len(self.pending) >= self.settings.max_queue_size:
            raise ServiceError(ErrorCode.QUEUE_FULL, "Generation queue is full", 429)
        identifier = new_id("gen")
        self.assets.acquire(identifier, spec.assets())
        info = JobInfo(
            id=identifier,
            seed=spec.seed,
            width=spec.width,
            height=spec.height,
            frames=spec.frames,
            fps=spec.fps,
        )
        self.repository.add(Job(info=info, spec=spec))
        self.pending.append(identifier)
        self._wake.set()
        # Snapshot ensures the HTTP 202 response always represents admission.
        return info.model_copy(deep=True)

    def get(self, identifier: str) -> Job:
        job = self.repository.get(identifier)
        if job is None:
            raise ServiceError(ErrorCode.GENERATION_NOT_FOUND, "Generation not found", 404)
        self.expire(job, utcnow())
        return job

    def queue_info(self) -> QueueInfo:
        return QueueInfo(
            running=self.running,
            queued=list(self.pending),
            length=len(self.pending),
            max_size=self.settings.max_queue_size,
            accepting=self.accepting,
        )

    def delete(self, identifier: str) -> None:
        job = self.repository.get(identifier)
        if job is None:
            return
        if job.info.status not in TERMINAL:
            job.info.cancellation_requested = True
            job.cancel.set()
            if identifier == self.running:
                # Keep the worker slot and asset leases until the backend actually returns.
                return
            self.pending.remove(identifier)
            job.info.transition(JobStatus.CANCELLED)
            job.info.error = ErrorDetail(
                code=ErrorCode.GENERATION_CANCELLED,
                message="Generation cancelled",
            )
            self.assets.release(identifier)
        elif job.info.status == JobStatus.COMPLETE:
            job.info.transition(JobStatus.CANCELLED)
        job.info.output = None
        job.info.expires_at = None
        job.delete_pending = True
        self._delete_output(job)

    def _delete_output(self, job: Job) -> None:
        if job.readers == 0:
            self.storage.remove("outputs", f"{job.info.id}.mp4")
            job.delete_pending = False

    def expire(self, job: Job, now: datetime) -> None:
        if (
            job.info.status == JobStatus.COMPLETE
            and job.info.expires_at
            and job.info.expires_at <= now
            and not job.readers
        ):
            self._delete_output(job)
            job.info.transition(JobStatus.EXPIRED, now)
            job.info.output = None

    def begin_download(self, identifier: str) -> tuple[Job, Path]:
        job = self.get(identifier)
        if job.info.status == JobStatus.EXPIRED or (
            job.info.expires_at and job.info.expires_at <= utcnow()
        ):
            raise ServiceError(ErrorCode.OUTPUT_EXPIRED, "Output has expired", 410)
        if job.info.status != JobStatus.COMPLETE:
            raise ServiceError(ErrorCode.OUTPUT_NOT_READY, "Output is not available", 409)
        path = self.storage.path("outputs", f"{identifier}.mp4")
        if not path.is_file():
            job.info.transition(JobStatus.EXPIRED)
            job.info.output = None
            raise ServiceError(ErrorCode.OUTPUT_EXPIRED, "Output file is no longer available", 410)
        job.readers += 1
        return job, path

    def end_download(self, job: Job, successful: bool, now: datetime | None = None) -> None:
        job.readers -= 1
        if successful and job.downloaded_at is None and job.info.status == JobStatus.COMPLETE:
            job.downloaded_at = now or utcnow()
            if self.settings.delete_after_download and job.info.expires_at:
                job.info.expires_at = min(
                    job.info.expires_at,
                    job.downloaded_at + timedelta(seconds=self.settings.download_grace_seconds),
                )
        if job.delete_pending:
            self._delete_output(job)

    async def _run(self) -> None:
        try:
            await self.backend.initialize()
        except ServiceError:
            # Liveness/diagnostics remain available; jobs can retry after model files are installed.
            logger.warning("inference_initialization_failed")
        while self.accepting or self.pending:
            if not self.pending:
                self._wake.clear()
                await self._wake.wait()
                continue
            identifier = self.pending.popleft()
            self.running = identifier
            job = self.get(identifier)
            started = time.monotonic()
            try:
                job.info.transition(JobStatus.LOADING)
                partial_path = self.storage.create_partial(identifier)
                context = GenerationContext(
                    id=identifier,
                    spec=job.spec,
                    partial_path=partial_path,
                    asset_paths={
                        key: self.storage.path("assets", f"{key}.bin") for key in job.spec.assets()
                    },
                    cancel=job.cancel,
                    report_stage=partial(self._report_stage, job),
                    report_metrics=job.info.metrics.update,
                )
                output = await self.backend.generate(context)
                if not job.cancel.is_set():
                    final = self.storage.publish(identifier, "outputs")
                    output.size_bytes = final.stat().st_size
                    output.content_url = f"/v1/generations/{identifier}/content"
                    job.info.output = output
                    job.info.transition(JobStatus.COMPLETE)
                    job.info.expires_at = utcnow() + timedelta(
                        seconds=self.settings.output_ttl_seconds,
                    )
            except ServiceError as exc:
                job.info.error = exc.detail
                if not job.cancel.is_set():
                    job.info.transition(JobStatus.FAILED)
                logger.info(
                    "generation_rejected",
                    extra={"job_id": identifier, "error_code": exc.detail.code},
                )
            except Exception:
                logger.exception("generation_failed", extra={"job_id": identifier})
                job.info.error = ErrorDetail(
                    code=ErrorCode.GENERATION_FAILED,
                    message="Generation failed; see server logs",
                )
                if not job.cancel.is_set():
                    job.info.transition(JobStatus.FAILED)
            finally:
                if job.cancel.is_set():
                    # Cancellation wins over a backend error or late result.
                    job.info.transition(JobStatus.CANCELLED)
                    job.info.error = ErrorDetail(
                        code=ErrorCode.GENERATION_CANCELLED,
                        message="Generation cancelled",
                    )
                try:
                    self.storage.remove("tmp", f"{identifier}.partial")
                    if job.info.status != JobStatus.COMPLETE:
                        self.storage.remove("outputs", f"{identifier}.mp4")
                except OSError:
                    logger.exception("artifact_cleanup_failed", extra={"job_id": identifier})
                self.assets.release(identifier)
                self.running = None
                logger.info(
                    "generation_finished",
                    extra={
                        "job_id": identifier,
                        "status": job.info.status,
                        "width": job.spec.width,
                        "height": job.spec.height,
                        "frames": job.spec.frames,
                        "fps": job.spec.fps,
                        "seed": job.spec.seed,
                        "total_seconds": round(time.monotonic() - started, 3),
                        "metrics": job.info.metrics,
                    },
                )

    @staticmethod
    def _report_stage(job: Job, status: JobStatus) -> None:
        if status in TERMINAL or status == JobStatus.QUEUED:
            raise ValueError("Backend may report only active stages")
        job.info.transition(status)
