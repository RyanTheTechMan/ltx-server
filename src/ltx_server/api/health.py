from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ltx_server import __version__
from ltx_server.api.gpu import gpu_snapshot

router = APIRouter()


class HealthInfo(BaseModel):
    status: Literal["ok", "starting", "degraded", "stopping"]
    version: str
    phase: Literal["advanced"] = "advanced"
    gpu_available: bool
    model_ready: bool = False
    inference_available: bool = False
    backend_enabled: bool = False
    model_state: str = "disabled"
    error: str | None = None


@router.get("/health", response_model=HealthInfo, tags=["diagnostics"])
async def health(request: Request) -> HealthInfo:
    """Liveness plus model readiness; missing CUDA/models never disables diagnostics."""
    info = await gpu_snapshot(request)
    backend = request.app.state.backend.status()
    status: Literal["ok", "starting", "degraded", "stopping"] = "ok"
    if not request.app.state.jobs.accepting:
        status = "stopping"
    elif backend.state == "loading_model":
        status = "starting"
    elif backend.error:
        status = "degraded"
    return HealthInfo(
        status=status,
        version=__version__,
        gpu_available=info.available,
        model_ready=backend.model_ready,
        inference_available=backend.model_ready,
        backend_enabled=backend.enabled,
        model_state=backend.state,
        error=backend.error,
    )
