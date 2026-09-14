import asyncio
import json
import struct
import threading
from pathlib import Path

import pytest

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import GenerationContext, check_cancelled
from ltx_server.inference.lifecycle import RetainedModel
from ltx_server.inference.manager import PipelineManager
from ltx_server.inference.models import CHECKPOINTS, ModelInventory, validate_safetensors
from ltx_server.jobs.state import JobStatus, OutputInfo
from ltx_server.schemas.generation import GenerationRequest, normalize_request

from .conftest import wait_terminal


class FakeOOM(Exception):
    pass


class FakeRuntime:
    instances = []

    def __init__(self, settings, inventory, stop):
        self.stop = stop
        self.loads = 0
        self.calls = []
        self.closed = False
        self.timings = {}
        self.load_thread = None
        self.block = threading.Event()
        self.block.set()
        self.started = threading.Event()
        self.fail = False
        self.instances.append(self)

    def load(self, preload=False, spec=None):
        self.loads += 1
        self.load_thread = threading.get_ident()

    def generate(self, context):
        assert threading.get_ident() == self.load_thread
        self.calls.append(context.spec)
        self.started.set()
        context.report_stage(JobStatus.ENCODING)
        while not self.block.wait(0.01):
            check_cancelled(context.cancel)
        check_cancelled(context.cancel)
        if self.fail:
            raise FakeOOM("private CUDA error")
        context.report_stage(JobStatus.GENERATING)
        context.report_stage(JobStatus.DECODING)
        context.report_stage(JobStatus.ENCODING_OUTPUT)
        context.partial_path.write_bytes(b"transport fixture")
        self.timings = {"generating": 0.01}
        return OutputInfo(
            width=context.spec.width,
            height=context.spec.height,
            frames=context.spec.frames,
            fps=context.spec.fps,
            duration=context.spec.frames / context.spec.fps,
            has_audio=context.spec.generate_audio,
        )

    def close(self):
        self.closed = True

    def is_oom(self, exc):
        return isinstance(exc, FakeOOM)


def manager(tmp_path, **settings):
    FakeRuntime.instances = []
    config = Settings(_env_file=None, data_dir=tmp_path, **settings)
    return PipelineManager(config, FakeRuntime)


async def test_cached_backend_reuses_runtime_for_resolution_changes(app_factory, tmp_path, png):
    backend = manager(tmp_path)
    async with app_factory(backend) as (app, client):
        async with asyncio.timeout(3):
            while not backend.status().model_ready:
                await asyncio.sleep(0.001)
        health = (await client.get("/v1/health")).json()
        assert health["model_ready"] and health["inference_available"]
        image = (await client.post("/v1/assets", files={"file": ("x.png", png)})).json()["id"]
        for resolution in ("540p", "720p"):
            response = await client.post(
                "/v1/generations",
                json={
                    "prompt": "animate",
                    "resolution": resolution,
                    "first_frame": image,
                },
            )
            job = await wait_terminal(app, response.json()["id"])
            assert job.info.status == JobStatus.COMPLETE
            assert job.info.metrics["generating"] == 0.01
        assert len(FakeRuntime.instances) == 1
        runtime = FakeRuntime.instances[0]
        assert runtime.loads == 1 and len(runtime.calls) == 2
        assert runtime.load_thread != threading.get_ident()
        assert [spec.width for spec in runtime.calls] == [1024, 1280]
    assert runtime.closed


async def test_oom_fails_job_and_next_job_recovers(app_factory, tmp_path):
    backend = manager(tmp_path)
    await backend.initialize()
    FakeRuntime.instances[0].fail = True
    async with app_factory(backend) as (app, client):
        first = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        job = await wait_terminal(app, first)
        assert job.info.error.code == ErrorCode.CUDA_OUT_OF_MEMORY
        assert FakeRuntime.instances[0].closed
        assert (await client.get("/v1/health")).json()["status"] == "degraded"
        second = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        job = await wait_terminal(app, second)
        assert job.info.status == JobStatus.COMPLETE
        assert len(FakeRuntime.instances) == 2


async def test_failed_preload_cleanup_keeps_api_and_worker_alive(app_factory, tmp_path):
    class FailedPreload(FakeRuntime):
        def load(self, preload=False, spec=None):
            super().load(preload)
            if preload:
                raise ServiceError(ErrorCode.MODEL_NOT_INSTALLED, "Install models", 503)

        def close(self):
            super().close()
            raise RuntimeError("simulated cleanup failure")

    backend = PipelineManager(Settings(_env_file=None, data_dir=tmp_path), FailedPreload)
    async with app_factory(backend) as (app, client):
        async with asyncio.timeout(3):
            while backend.status().state != "error":
                await asyncio.sleep(0.001)
        health = (await client.get("/v1/health")).json()
        assert health["error"] == "MODEL_NOT_INSTALLED"
        identifier = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        job = await wait_terminal(app, identifier)
        assert job.info.status == JobStatus.COMPLETE
        # Restore normal close for orderly application shutdown.
        backend._runtime.close = lambda: None


async def test_async_task_cancellation_waits_for_thread(tmp_path):
    class SlowRuntime(FakeRuntime):
        def generate(self, context):
            self.started.set()
            # Deliberately does not cooperate until released: asyncio cancellation must wait.
            assert self.block.wait(3)
            check_cancelled(context.cancel)

    backend = PipelineManager(Settings(_env_file=None, data_dir=tmp_path), SlowRuntime)
    await backend.initialize()
    runtime = backend._runtime
    runtime.block.clear()
    spec = normalize_request(GenerationRequest(prompt="x"), backend.settings, 1)
    context = GenerationContext(
        "gen_test", spec, tmp_path / "partial", {}, threading.Event(), lambda stage: None
    )
    task = asyncio.create_task(backend.generate(context))
    try:
        assert await asyncio.to_thread(runtime.started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert context.cancel.is_set()
        runtime.block.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.closed
    finally:
        runtime.block.set()
        await backend.close()


async def test_warm_loading_does_not_block_health(app_factory, tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class SlowLoad(FakeRuntime):
        def load(self, preload=False, spec=None):
            entered.set()
            assert release.wait(3)
            super().load(preload)

    backend = PipelineManager(Settings(_env_file=None, data_dir=tmp_path), SlowLoad)
    async with app_factory(backend) as (_, client):
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            async with asyncio.timeout(1):
                response = await client.get("/v1/health")
            assert response.json()["status"] == "starting"
        finally:
            release.set()


@pytest.mark.parametrize("field", ["reference_video", "loras"])
async def test_pending_capabilities_rejected_before_admission(app_factory, field):
    asset = "asset_" + "a" * 32
    values = {
        "last_frame": asset,
        "keyframes": [{"asset_id": asset, "frame": 0}],
        "audio": asset,
        "reference_video": asset,
        "loras": [{"id": "style"}],
    }
    body = {"prompt": "x", field: values[field]}
    if field == "last_frame":
        body["first_frame"] = asset
    async with app_factory() as (app, client):
        response = await client.post("/v1/generations", json=body)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == (
            "UNSUPPORTED_PIPELINE" if field == "reference_video" else "INVALID_INPUT"
        )
        assert not tuple(app.state.jobs.repository.all())


def test_retained_weights_park_without_rebuild_or_dtype_cast():
    class Model:
        def __init__(self):
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            return self

    builds = []

    def build(**kwargs):
        model = Model()
        builds.append(model)
        return model

    retained = RetainedModel(build, "cuda:0", park_on_exit=False)
    with retained.context(video_tools="small") as first:
        pass
    with retained.context(video_tools="large") as second:
        assert first is second
    retained.park()
    with retained.context() as third:
        assert third is first
    assert len(builds) == 1 and first.moves == ["cpu", "cuda:0"]
    retained.clear()
    retained.load()
    assert len(builds) == 2


def test_failed_model_call_preserves_original_error():
    class Model:
        def to(self, device):
            raise RuntimeError("A failed transfer must not mask the original OOM")

    retained = RetainedModel(lambda: Model(), "cuda:0", park_on_exit=True)
    with pytest.raises(FakeOOM):
        with retained.context():
            raise FakeOOM("original failure")
    retained.clear()


def safetensor(path: Path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)


def test_model_paths_validation_and_override(tmp_path):
    paths = {}
    for checkpoint in CHECKPOINTS:
        path = tmp_path / f"{checkpoint.component}.safetensors"
        safetensor(path)
        paths[f"ltx_{checkpoint.component}_path"] = path
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", **paths)
    inventory = ModelInventory(settings)
    assert inventory.missing() == []
    inventory.validate()
    next(iter(paths.values())).write_bytes(b"bad")
    with pytest.raises(ServiceError) as exc:
        inventory.validate()
    assert exc.value.detail.code == "MODEL_NOT_INSTALLED"


@pytest.mark.parametrize(
    "raw", [b"not a model", struct.pack("<Q", 1 << 40), struct.pack("<Q", 2) + b"{}"]
)
def test_bad_safetensors_headers(tmp_path, raw):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        validate_safetensors(path)
