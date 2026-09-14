import asyncio
import io
from contextlib import asynccontextmanager

import httpx
import pytest
from PIL import Image

from ltx_server.api.gpu import GPUInfo
from ltx_server.config import Settings
from ltx_server.inference.backend import GenerationContext, UnavailableBackend
from ltx_server.jobs.state import JobStatus, OutputInfo
from ltx_server.main import create_app


class ControlledBackend(UnavailableBackend):
    def __init__(self, *, blocked=False, fail_first=False):
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.started = asyncio.Event()
        self.order = []
        self.active = 0
        self.peak = 0
        self.closed = False
        self.fail_first = fail_first

    async def generate(self, context: GenerationContext) -> OutputInfo:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.order.append(context.id)
        self.started.set()
        try:
            context.report_stage(JobStatus.GENERATING)
            while not self.release.is_set() and not context.cancel.is_set():
                await asyncio.sleep(0.001)
            if self.fail_first and len(self.order) == 1:
                raise RuntimeError("Private backend failure /secret/model.safetensors")
            # Transport fixture, intentionally not a valid MP4 or inference simulation.
            context.partial_path.write_bytes(b"0123456789" * 100)
            spec = context.spec
            return OutputInfo(
                width=spec.width,
                height=spec.height,
                frames=spec.frames,
                fps=spec.fps,
                duration=spec.frames / spec.fps,
                has_audio=spec.generate_audio,
            )
        finally:
            self.active -= 1

    async def close(self):
        self.closed = True


async def wait_terminal(app, identifier):
    async with asyncio.timeout(3):
        while app.state.jobs.get(identifier).info.status not in {
            JobStatus.COMPLETE,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            await asyncio.sleep(0.001)
    return app.state.jobs.get(identifier)


@pytest.fixture
def png():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "navy").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def app_factory(tmp_path):
    count = 0

    @asynccontextmanager
    async def factory(backend=None, **overrides):
        nonlocal count
        count += 1
        overrides.setdefault("inference_backend", "disabled")
        settings = Settings(_env_file=None, data_dir=tmp_path / str(count), **overrides)
        app = create_app(
            settings,
            backend=backend,
            gpu_reader=lambda index: GPUInfo(index=index, reason="Test has no GPU"),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield app, client

    return factory
