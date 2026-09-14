import asyncio
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class GPUInfo(BaseModel):
    available: bool = False
    index: int
    name: str | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    vram_free_mb: int | None = None
    temperature_c: int | None = None
    utilization_percent: int | None = None
    reason: str | None = None


def read_gpu(index: int) -> GPUInfo:
    """NVML physical index; this does not initialize CUDA or prove inference readiness."""
    try:
        import pynvml
    except ImportError:
        return GPUInfo(index=index, reason="NVML Python bindings are unavailable")
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return GPUInfo(index=index, reason="NVIDIA driver/NVML is unavailable")
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        info = GPUInfo(
            index=index, available=True, name=name.decode() if isinstance(name, bytes) else name
        )
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info.vram_total_mb = memory.total // (1024 * 1024)
            info.vram_used_mb = memory.used // (1024 * 1024)
            info.vram_free_mb = memory.free // (1024 * 1024)
        except pynvml.NVMLError:
            pass
        try:
            info.temperature_c = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            pass
        try:
            info.utilization_percent = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except pynvml.NVMLError:
            pass
        return info
    except pynvml.NVMLError:
        return GPUInfo(index=index, reason="Configured NVIDIA GPU is unavailable")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


async def gpu_snapshot(request: Request) -> GPUInfo:
    # Serialize NVML init/shutdown across concurrent health/GPU requests.
    async with request.app.state.gpu_lock:
        reader: Any = request.app.state.gpu_reader
        info: GPUInfo = await asyncio.to_thread(reader, request.app.state.settings.gpu_index)
        return info


@router.get("/gpu", response_model=GPUInfo, tags=["diagnostics"])
async def gpu(request: Request) -> GPUInfo:
    """NVIDIA physical GPU metrics; unavailable metrics are null, never invented."""
    return await gpu_snapshot(request)
