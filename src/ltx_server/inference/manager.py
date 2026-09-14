import asyncio
import logging
import platform
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Lock
from typing import Protocol, TypeVar

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import (
    BackendStatus,
    GenerationBackend,
    GenerationContext,
    UnavailableBackend,
    check_cancelled,
    validate_capabilities,
)
from ltx_server.inference.models import ModelInventory
from ltx_server.inference.runtime import LTXRuntime
from ltx_server.jobs.state import JobStatus, OutputInfo
from ltx_server.schemas.generation import GenerationSpec

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Runtime(Protocol):
    timings: dict[str, float]

    def load(self, preload: bool = False, spec: GenerationSpec | None = None) -> None: ...
    def generate(self, context: GenerationContext) -> OutputInfo: ...
    def close(self) -> None: ...
    def is_oom(self, exc: BaseException) -> bool: ...


class PipelineManager:
    """Own one CUDA thread and one cached distilled pipeline across all resolutions."""

    def __init__(
        self,
        settings: Settings,
        runtime_factory: Callable[[Settings, ModelInventory, Event], Runtime] = LTXRuntime,
    ):
        self.settings = settings
        self.inventory = ModelInventory(settings)
        self.factory = runtime_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx-cuda")
        self._stop = Event()
        self._status_lock = Lock()
        self._status = BackendStatus(enabled=True, state="unloaded")
        self._runtime: Runtime | None = None
        self._runtime_key: tuple[object, ...] = ()
        self._closed = False

    def status(self) -> BackendStatus:
        with self._status_lock:
            return self._status.model_copy()

    def _set_status(self, state: str, error: str | None = None) -> None:
        with self._status_lock:
            self._status = BackendStatus(
                enabled=True, state=state, model_ready=state == "ready", error=error
            )

    def validate_request(self, spec: GenerationSpec) -> None:
        validate_capabilities(spec, self.inventory.loras)

    async def initialize(self) -> None:
        if self.settings.warm_model_on_start:
            await self._execute(lambda: self._prepare(True))

    def _prepare(self, preload: bool = False, spec: GenerationSpec | None = None) -> Runtime:
        check_cancelled(self._stop)
        key = self.inventory.loras.key(spec)
        if self._runtime is not None and key != self._runtime_key:
            previous, self._runtime = self._runtime, None
            self._discard(previous)
        if self._runtime is not None:
            return self._runtime
        self._set_status("loading_model")
        runtime = self.factory(self.settings, self.inventory, self._stop)
        try:
            runtime.load(preload=preload, spec=spec)
            check_cancelled(self._stop)
        except Exception as exc:
            oom = runtime.is_oom(exc)
            self._discard(runtime)
            code = (
                ErrorCode.CUDA_OUT_OF_MEMORY
                if oom
                else exc.detail.code
                if isinstance(exc, ServiceError)
                else ErrorCode.MODEL_LOAD_FAILED
            )
            self._set_status("error", code)
            logger.exception("model_load_failed")
            if isinstance(exc, ServiceError) and not oom:
                raise
            raise ServiceError(code, "Model initialization failed; see server logs", 503) from None
        self._runtime = runtime
        self._runtime_key = key
        self._set_status("ready")
        return runtime

    @staticmethod
    def _discard(runtime: Runtime) -> None:
        try:
            runtime.close()
        except Exception:
            # Preserve the original failure and keep the queue worker alive.
            logger.exception("inference_cleanup_failed")

    async def _execute(self, fn: Callable[[], T], cancel: Event | None = None) -> T:
        if self._closed:
            raise ServiceError(ErrorCode.SERVER_STOPPING, "Inference worker is closed", 503)
        future = asyncio.get_running_loop().run_in_executor(self._executor, fn)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            (cancel or self._stop).set()
            # Never release assets/worker while a cancelled asyncio wrapper still has CUDA work.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if future.done() and not future.cancelled():
                future.exception()
            raise

    async def generate(self, context: GenerationContext) -> OutputInfo:
        self.validate_request(context.spec)
        loop = asyncio.get_running_loop()

        def report_stage(stage: JobStatus) -> None:
            loop.call_soon_threadsafe(context.report_stage, stage)

        def report_metrics(metrics: dict[str, float]) -> None:
            loop.call_soon_threadsafe(context.report_metrics, metrics)

        threaded = replace(
            context,
            report_stage=report_stage,
            report_metrics=report_metrics,
        )
        return await self._execute(lambda: self._generate(threaded), context.cancel)

    def _generate(self, context: GenerationContext) -> OutputInfo:
        check_cancelled(context.cancel)
        start = time.monotonic()
        cold = self._runtime is None or self._runtime_key != self.inventory.loras.key(context.spec)
        runtime = self._prepare(spec=context.spec)
        load_seconds = time.monotonic() - start
        try:
            check_cancelled(context.cancel)
            return runtime.generate(context)
        except Exception as exc:
            oom = runtime.is_oom(exc)
            # Failure invalidates tensors/iterators. Next queued job gets a clean runtime.
            self._runtime = None
            self._discard(runtime)
            if context.cancel.is_set() or self._stop.is_set():
                self._set_status("unloaded")
                raise ServiceError(
                    ErrorCode.GENERATION_CANCELLED, "Generation cancelled", 409
                ) from None
            code = (
                ErrorCode.CUDA_OUT_OF_MEMORY
                if oom
                else exc.detail.code
                if isinstance(exc, ServiceError)
                else ErrorCode.GENERATION_FAILED
            )
            self._set_status("error", code)
            logger.exception("inference_failed", extra={"job_id": context.id, "error_code": code})
            if isinstance(exc, ServiceError) and not oom:
                raise
            raise ServiceError(code, "LTX generation failed; see server logs") from None
        finally:
            context.report_metrics(
                {
                    **runtime.timings,
                    "pipeline_init_seconds": load_seconds,
                    "cold_pipeline": float(cold),
                    "total_seconds": time.monotonic() - start,
                }
            )

    def begin_shutdown(self) -> None:
        self._stop.set()
        self._set_status("stopping")

    async def close(self) -> None:
        if self._closed:
            return
        self.begin_shutdown()

        def close_runtime() -> None:
            if self._runtime:
                self._runtime.close()
                self._runtime = None

        await self._execute(close_runtime)
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True)


def create_backend(settings: Settings) -> GenerationBackend:
    enabled = settings.inference_backend == "ltx" or (
        settings.inference_backend == "auto"
        and platform.system() == "Linux"
        and platform.machine() == "x86_64"
    )
    return PipelineManager(settings) if enabled else UnavailableBackend(settings)
