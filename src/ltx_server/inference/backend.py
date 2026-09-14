from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Protocol

from pydantic import BaseModel

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.loras import LoraRegistry
from ltx_server.jobs.state import JobStatus, OutputInfo
from ltx_server.schemas.generation import GenerationSpec


class BackendStatus(BaseModel):
    enabled: bool = False
    state: str = "disabled"
    model_ready: bool = False
    error: str | None = None


def check_cancelled(cancel: Event) -> None:
    if cancel.is_set():
        raise ServiceError(ErrorCode.GENERATION_CANCELLED, "Generation cancelled", 409)


@dataclass(frozen=True)
class GenerationContext:
    id: str
    spec: GenerationSpec
    partial_path: Path
    asset_paths: dict[str, Path]
    cancel: Event
    report_stage: Callable[[JobStatus], None]
    report_metrics: Callable[[dict[str, float]], None] = lambda metrics: None


class GenerationBackend(Protocol):
    """One async invocation at a time. A backend must return only after all work stops.

    Implementations offload blocking CUDA work to a dedicated thread, check the
    thread-safe cancellation event, and marshal stage callbacks onto the event
    loop. Cancelling an asyncio wrapper does not stop a CUDA thread. The backend
    owns encoding/validation and writes only partial_path; the manager publishes.
    """

    async def generate(self, context: GenerationContext) -> OutputInfo: ...

    async def close(self) -> None: ...

    async def initialize(self) -> None: ...

    def begin_shutdown(self) -> None: ...

    def status(self) -> BackendStatus: ...

    def validate_request(self, spec: GenerationSpec) -> None: ...


class UnavailableBackend:
    def __init__(self, settings: Settings | None = None) -> None:
        self.registry = LoraRegistry(settings)

    async def generate(self, context: GenerationContext) -> OutputInfo:
        raise ServiceError(
            ErrorCode.INFERENCE_UNAVAILABLE,
            "Inference is disabled. Enable the Linux CUDA backend to generate video.",
            503,
        )

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def begin_shutdown(self) -> None:
        pass

    def status(self) -> BackendStatus:
        return BackendStatus()

    def validate_request(self, spec: GenerationSpec) -> None:
        validate_capabilities(spec, getattr(self, "registry", None))


def validate_capabilities(spec: GenerationSpec, registry: LoraRegistry | None = None) -> None:
    (registry or LoraRegistry()).resolve(spec)
