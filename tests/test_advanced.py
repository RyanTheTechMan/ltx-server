import json
import sys
from threading import Event

import pytest

from ltx_server.config import Settings
from ltx_server.errors import ServiceError
from ltx_server.inference.backend import GenerationContext
from ltx_server.inference.loras import LoraRegistry
from ltx_server.inference.manager import PipelineManager
from ltx_server.inference.models import ModelInventory
from ltx_server.inference.runtime import LTXRuntime
from ltx_server.schemas.generation import GenerationRequest, normalize_request

from .conftest import wait_terminal
from .test_inference import FakeRuntime, safetensor
from .test_runtime_contract import fake_upstream as fake_upstream
from .test_runtime_contract import install_module


def manifest(tmp_path):
    for name in ("style", "guide"):
        safetensor(tmp_path / f"{name}.safetensors")
    path = tmp_path / "loras.json"
    path.write_text(
        json.dumps(
            {
                "loras": {
                    "style": {"path": "style.safetensors", "kind": "style"},
                    "guide": {"path": "guide.safetensors", "kind": "ic"},
                }
            }
        )
    )
    return path


def test_lora_registry_and_client_path_rejection(tmp_path):
    settings = Settings(_env_file=None, lora_manifest=manifest(tmp_path), default_ic_lora="guide")
    registry = LoraRegistry(settings)
    spec = normalize_request(
        GenerationRequest(prompt="x", loras=[{"id": "style", "scale": 0.5}]), settings, 1
    )
    resolved = registry.resolve(spec)
    assert resolved[0][1].path == tmp_path / "style.safetensors"
    assert resolved[0][0].scale == 0.5
    assert all("path" not in row for row in registry.public())
    for identifier in ("../secret", "/tmp/private.safetensors", "https://example.com/x"):
        with pytest.raises(ValueError):
            GenerationRequest(prompt="x", loras=[{"id": identifier}])
    for identifier in ("missing", "guide"):
        with pytest.raises(ServiceError):
            registry.resolve(
                normalize_request(
                    GenerationRequest(prompt="x", loras=[{"id": identifier}]), settings, 1
                )
            )


@pytest.mark.parametrize(
    "body",
    [
        {"loras": [{"id": "a"}, {"id": "a"}]},
        {"reference_lora": {"id": "guide"}},
        {"reference_video": "asset_" + "a" * 32, "audio": "asset_" + "b" * 32},
        {"retake": {"video": "asset_" + "a" * 32, "start": 2, "end": 1}},
        {
            "retake": {
                "video": "asset_" + "a" * 32,
                "start": 0,
                "end": 1,
                "regenerate_video": False,
                "regenerate_audio": False,
            }
        },
        {
            "retake": {"video": "asset_" + "a" * 32, "start": 0, "end": 1},
            "first_frame": "asset_" + "b" * 32,
        },
    ],
)
def test_advanced_invalid_combinations(body):
    with pytest.raises(ValueError):
        GenerationRequest(prompt="x", **body)


async def test_lora_cache_switch_drops_previous_weights(app_factory, tmp_path):
    settings = Settings(
        _env_file=None, lora_manifest=manifest(tmp_path), data_dir=tmp_path / "data"
    )
    FakeRuntime.instances = []
    backend = PipelineManager(settings, FakeRuntime)
    async with app_factory(backend) as (app, client):
        for loras in (
            [{"id": "style", "scale": 0.5}],
            [{"id": "style", "scale": 0.5}],
            [],
            [{"id": "style", "scale": 1.0}],
        ):
            response = await client.post("/v1/generations", json={"prompt": "x", "loras": loras})
            assert response.status_code == 202
            assert (await wait_terminal(app, response.json()["id"])).info.status == "complete"
        # Warm default, style 0.5, base, style 1.0. Second identical style job reuses its cache.
        assert len(FakeRuntime.instances) == 4
        assert all(item.closed for item in FakeRuntime.instances[:-1])
        assert len(FakeRuntime.instances[1].calls) == 2


@pytest.mark.parametrize("workflow", ["reference", "retake"])
def test_official_advanced_pipeline_contract(fake_upstream, tmp_path, monkeypatch, workflow):
    import ltx_server.inference.runtime as module

    settings = Settings(
        _env_file=None, data_dir=tmp_path, lora_manifest=manifest(tmp_path), default_ic_lora="guide"
    )
    base = sys.modules["ltx_pipelines.distilled"].DistilledPipeline
    forwarded = []

    class IC(base):
        def __init__(self, **kwargs):
            forwarded.append(kwargs)
            super().__init__(**kwargs)
            self.stage_1, self.stage_2 = self.stage, type(self.stage)()
            del self.stage

        def __call__(self, video_conditioning, **kwargs):
            forwarded.append(video_conditioning)
            stages = iter([self.stage_1, self.stage_2])
            self.stage = lambda **kw: next(stages)(**kw)
            return super().__call__(**kwargs)

    class Retake(base):
        def __init__(self, distilled, **kwargs):
            assert distilled
            assert "spatial_upsampler_path" not in kwargs
            forwarded.append(kwargs)
            super().__init__(spatial_upsampler_path="unused", **kwargs)

        def __call__(
            self, video_path, start_time, end_time, regenerate_video, regenerate_audio, **kwargs
        ):
            forwarded.append((video_path, start_time, end_time, regenerate_video, regenerate_audio))
            return super().__call__(
                width=1024, height=576, num_frames=25, frame_rate=24, images=[], **kwargs
            )

    install_module(monkeypatch, "ltx_pipelines.ic_lora", ICLoraPipeline=IC)
    install_module(monkeypatch, "ltx_pipelines.retake", RetakePipeline=Retake)
    install_module(
        monkeypatch,
        "ltx_core.loader",
        LTXV_LORA_COMFY_RENAMING_MAP="comfy",
        LoraPathStrengthAndSDOps=lambda *args: args,
    )
    monkeypatch.setattr(
        module,
        "prepare_reference",
        lambda source, destination, *a: destination.write_bytes(b"reference scratch"),
    )
    monkeypatch.setattr(module, "validate_retake_source", lambda *a: None)
    asset = "asset_" + "a" * 32
    fields = (
        {"reference_video": asset}
        if workflow == "reference"
        else {"retake": {"video": asset, "start": 0.2, "end": 0.8}}
    )
    spec = normalize_request(
        GenerationRequest(prompt="x", duration=1, generate_audio=False, **fields), settings, 1
    )
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load(spec=spec)
    try:
        output = runtime.generate(
            GenerationContext(
                "gen_test",
                spec,
                tmp_path / "out.partial",
                {asset: tmp_path / "source.bin"},
                Event(),
                lambda s: None,
            )
        )
        assert output.frames == 25
        if workflow == "reference":
            assert runtime.transformer is None  # distinct IC weights stay transient
            assert forwarded[0]["loras"] == [(str(tmp_path / "guide.safetensors"), 1.0, "comfy")]
            assert forwarded[1] == [(str(tmp_path / "out.partial"), 1.0)]
        else:
            assert forwarded[1] == (str(tmp_path / "source.bin"), 0.2, 0.8, True, True)
            assert runtime.transformer is not None
    finally:
        runtime.close()


def test_streaming_and_compile_keep_upstream_lifecycle(fake_upstream, tmp_path, monkeypatch):
    from enum import Enum

    class Offload(Enum):
        CPU = "cpu"

    install_module(monkeypatch, "ltx_pipelines.utils.types", OffloadMode=Offload)
    install_module(
        monkeypatch, "ltx_core.model.transformer.compiling", CompilationConfig=lambda **kw: kw
    )
    settings = Settings(
        _env_file=None, data_dir=tmp_path, offload_mode="cpu", compile_transformer=True
    )
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load(preload=True)
    try:
        assert runtime.pipeline.options == {
            "offload_mode": Offload.CPU,
            "compilation_config": {"mode": None, "capture": False},
        }
        assert runtime.transformer is None and runtime.text_encoder is None
    finally:
        runtime.close()


def test_sage_adapter_uses_nhd_and_no_mask(monkeypatch):
    from ltx_server.inference.performance import SageAttention

    captured = []

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

        def reshape(self, *shape):
            if len(shape) == 4:
                return Tensor((shape[0], self.shape[1], shape[2], shape[3]))
            return Tensor((shape[0], self.shape[1], shape[2]))

    def kernel(q, k, v, **kwargs):
        captured.append((q.shape, k.shape, v.shape, kwargs))
        return q

    install_module(monkeypatch, "sageattention", sageattn=kernel)
    attention = SageAttention()
    q, k, v = Tensor((1, 10, 128)), Tensor((1, 20, 128)), Tensor((1, 20, 128))
    assert attention(q, k, v, 2).shape == (1, 10, 128)
    assert captured == [
        (
            (1, 10, 2, 64),
            (1, 20, 2, 64),
            (1, 20, 2, 64),
            {"tensor_layout": "NHD", "is_causal": False},
        )
    ]
    with pytest.raises(ValueError):
        attention(q, k, v, 2, mask=object())


@pytest.mark.parametrize(
    "mode,source_audio,mute",
    [
        ("video", True, False),
        ("video", False, False),
        ("video", True, True),
        ("audio", True, False),
        ("audio", False, False),
    ],
)
def test_normalized_retake_preserves_unedited_stream(
    fake_upstream,
    tmp_path,
    monkeypatch,
    mode,
    source_audio,
    mute,
):
    import struct
    from types import SimpleNamespace

    import ltx_server.inference.runtime as module
    from ltx_server.media.ffmpeg import decode_source_audio

    from .test_video_inputs import make_clip

    base = sys.modules["ltx_pipelines.distilled"].DistilledPipeline
    observed = {}

    class Retake(base):
        def __init__(self, **kwargs):
            kwargs.pop("distilled")
            super().__init__(spatial_upsampler_path="unused", **kwargs)

        def __call__(
            self, video_path, start_time, end_time, regenerate_video, regenerate_audio, **kwargs
        ):
            from pathlib import Path

            prepared = Path(video_path)
            observed["prepared"] = prepared
            assert prepared != source and prepared.exists()
            if source_audio:
                observed["pcm"] = decode_source_audio(
                    prepared, duration=25 / 24, cancel=Event(), settings=settings
                )
            return super().__call__(
                width=64, height=96, num_frames=25, frame_rate=24, images=[], **kwargs
            )

    install_module(monkeypatch, "ltx_pipelines.retake", RetakePipeline=Retake)
    source = tmp_path / "source.mp4"
    make_clip(source, audio=source_audio)
    original = source.read_bytes()
    settings = Settings(_env_file=None, data_dir=tmp_path)
    asset = "asset_" + "a" * 32
    spec = normalize_request(
        GenerationRequest(
            prompt="x",
            duration=1,
            orientation="portrait",
            generate_audio=not mute,
            retake={
                "video": asset,
                "start": 0,
                "end": 1,
                "normalize_source": True,
                "regenerate_video": mode == "video",
                "regenerate_audio": mode == "audio",
            },
        ),
        settings,
        1,
    ).model_copy(update={"width": 64, "height": 96})
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load(spec=spec)

    class Wave:
        ndim = 2
        shape = (4, 2)

        def detach(self):
            return self

        float = cpu = contiguous = numpy = detach

        def astype(self, dtype):
            return self

        def tobytes(self):
            return struct.pack("<8f", *([0.1] * 8))

    if mode == "audio":
        runtime.pipeline.audio = SimpleNamespace(waveform=Wave(), sampling_rate=24000)
        monkeypatch.setattr(
            runtime.torch,
            "isfinite",
            lambda w: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
            raising=False,
        )
        real_mux = module.encode_source_video

        def mux(source_path, *args, **kwargs):
            assert source_path == observed["prepared"]
            observed["copied_video"] = True
            return real_mux(source_path, *args, **kwargs)

        monkeypatch.setattr(module, "encode_source_video", mux)
        monkeypatch.setattr(module, "encode_mp4", lambda *a, **kw: pytest.fail("Used model video"))
    else:
        runtime.pipeline.audio = None  # Must never use generated audio for video-only edits.

        def encode(chunks, path, **kwargs):
            assert list(chunks) == [b"pixels"]
            if source_audio and not mute:
                kwargs["audio"].seek(0)
                assert kwargs["audio"].read() == observed["pcm"]
            else:
                assert kwargs["audio"] is None
            path.write_bytes(b"stub")
            kwargs["on_flush"]()

        monkeypatch.setattr(module, "encode_mp4", encode)
    try:
        result = runtime.generate(
            GenerationContext(
                "gen_test", spec, tmp_path / "out.partial", {asset: source}, Event(), lambda s: None
            )
        )
        assert result.has_audio == (not mute and (source_audio or mode == "audio"))
        assert (result.width, result.height, result.frames) == (64, 96, 25)
        if mode == "audio":
            assert observed["copied_video"]
        assert not observed["prepared"].exists()
        assert source.read_bytes() == original
    finally:
        runtime.close()


@pytest.mark.parametrize("failure", ["preparation", "validation", "inference", "cancel"])
def test_retake_scratch_removed_on_failure(fake_upstream, tmp_path, monkeypatch, failure):
    from pathlib import Path

    import ltx_server.inference.runtime as module
    from ltx_server.media.video_inputs import VideoSource

    base = sys.modules["ltx_pipelines.distilled"].DistilledPipeline
    called = []

    class Retake(base):
        def __init__(self, **kwargs):
            kwargs.pop("distilled")
            super().__init__(spatial_upsampler_path="unused", **kwargs)

        def __call__(self, **kwargs):
            called.append("inference")
            assert Path(kwargs["video_path"]).exists()
            raise ServiceError("INVALID_MEDIA", "fixture failure")

    install_module(monkeypatch, "ltx_pipelines.retake", RetakePipeline=Retake)

    def prepare(source, destination, spec, settings, cancel):
        destination.write_bytes(b"partial preparation")
        if failure == "preparation":
            raise ServiceError("INVALID_MEDIA", "fixture failure")
        if failure == "cancel":
            cancel.set()

    def validate(*args):
        called.append("validation")
        if failure == "validation":
            raise ServiceError("INVALID_INPUT", "fixture failure")
        return VideoSource(False, "h264")

    monkeypatch.setattr(module, "prepare_retake", prepare)
    monkeypatch.setattr(module, "validate_retake_source", validate)
    settings = Settings(_env_file=None, data_dir=tmp_path)
    asset = "asset_" + "a" * 32
    source = tmp_path / "original.bin"
    source.write_bytes(b"original")
    spec = normalize_request(
        GenerationRequest(
            prompt="x",
            retake={
                "video": asset,
                "start": 0,
                "end": 1,
                "normalize_source": True,
            },
        ),
        settings,
        1,
    )
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load(spec=spec)
    try:
        with pytest.raises(ServiceError):
            runtime.generate(
                GenerationContext(
                    "gen_test",
                    spec,
                    tmp_path / "out.partial",
                    {asset: source},
                    Event(),
                    lambda s: None,
                )
            )
        assert not (tmp_path / "out.source.partial").exists()
        assert source.read_bytes() == b"original"
        assert ("inference" in called) == (failure == "inference")
    finally:
        runtime.close()
