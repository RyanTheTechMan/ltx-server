import json
import struct
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest

from ltx_server.config import Settings
from ltx_server.errors import ServiceError
from ltx_server.inference.backend import GenerationContext
from ltx_server.inference.models import LTX_COMMIT, ModelInventory
from ltx_server.inference.runtime import LTXRuntime, verify_upstream
from ltx_server.jobs.state import JobStatus, OutputInfo
from ltx_server.schemas.generation import GenerationRequest, normalize_request


def install_module(monkeypatch, name, **attributes):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules:
            module = ModuleType(key)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, key, module)
    for key, value in attributes.items():
        monkeypatch.setattr(sys.modules[name], key, value, raising=False)


@pytest.fixture
def fake_upstream(monkeypatch):
    import ltx_server.inference.runtime as runtime_module

    builds = []
    calls = []

    @dataclass
    class Modality:
        context: str = "prompt"
        initial_latent: object = None
        frozen: bool = False
        noise_scale: float = 1.0

    class Net:
        def __init__(self, kind):
            builds.append(kind)

        def to(self, device):
            return self

        def __call__(self):
            return None

    class Stage:
        def _build_transformer(self, **kwargs):
            return Net("transformer")

        def _transformer_ctx(self, **kwargs):
            return nullcontext(self._build_transformer())

        def with_model_wrapper(self, wrapper):
            self.wrapper = wrapper
            return self

        def __call__(self, **kwargs):
            if kwargs:
                calls[-1].setdefault("stages", []).append(kwargs)
            with self._transformer_ctx(video_tools=None) as model:
                return self.wrapper(model, None)()

    class Encoder:
        def _build_text_encoder(self):
            return Net("text")

        def _text_encoder_ctx(self):
            return nullcontext(self._build_text_encoder())

        def __call__(self):
            with self._text_encoder_ctx() as model:
                model()

    class Chunk:
        def __init__(self, height, width):
            self.shape = (1, height, width, 3)

        def movedim(self, *args):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self

        def tobytes(self):
            return b"pixels"

    class Pipeline:
        # Signatures mirror the specific official constructors/calls used by this adapter.
        def __init__(
            self,
            model_paths,
            spatial_upsampler_path,
            loras,
            device,
            quantization,
            registry,
            alloc_trim_strategy,
            **options,
        ):
            self.options = options
            assert set(model_paths) == {
                "transformer_path",
                "text_encoder_path",
                "video_vae_path",
                "audio_vae_path",
            }
            self.stage = Stage()
            self.prompt_encoder = Encoder()
            self.video_decoder = lambda height, width: iter([Chunk(height, width)])
            self.audio_decoder = lambda latent: getattr(self, "audio", None)

        def __call__(
            self,
            prompt,
            seed,
            width,
            height,
            num_frames,
            frame_rate,
            images,
            tiling_config,
            enhance_prompt,
        ):
            calls.append({"images": images, "width": width, "frames": num_frames})
            self.prompt_encoder()
            self.stage(audio=Modality(), width=width // 2)
            self.stage(audio=Modality(initial_latent="stage-one-result"), width=width)
            return SimpleNamespace(
                video=self.video_decoder(height, width), audio=self.audio_decoder("audio-latent")
            )

    install_module(
        monkeypatch,
        "torch",
        bfloat16="bf16",
        inference_mode=nullcontext,
        device=lambda value: value,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda device: None,
            get_device_capability=lambda device: (12, 0),
            reset_peak_memory_stats=lambda device: None,
            max_memory_allocated=lambda device: 1024,
            max_memory_reserved=lambda device: 2048,
            empty_cache=lambda: None,
            OutOfMemoryError=MemoryError,
        ),
    )
    install_module(
        monkeypatch,
        "ltx_core.allocator_trim_strategy",
        AllocatorTrimStrategy=SimpleNamespace(DEFER="defer"),
    )
    install_module(
        monkeypatch,
        "ltx_core.loader.registry",
        ModelRegistry=lambda **kwargs: SimpleNamespace(clear=lambda: None),
    )
    install_module(
        monkeypatch,
        "ltx_core.model.video_vae",
        DimensionSizeConfig=lambda **kw: kw,
        TileSizeConfig=lambda **kw: kw,
    )
    install_module(
        monkeypatch,
        "ltx_core.model.video_vae.model_configurator",
        is_diffusion_video_vae=lambda path: False,
    )
    install_module(monkeypatch, "ltx_core.quantization.fp8_cast", build_policy=lambda path: "fp8")
    install_module(monkeypatch, "ltx_pipelines.distilled", DistilledPipeline=Pipeline)
    install_module(
        monkeypatch,
        "ltx_pipelines.utils.model_paths",
        ModelPaths=SimpleNamespace(from_split=lambda **kwargs: kwargs),
    )
    install_module(monkeypatch, "ltx_core.color.yuv", yuv420p_bt709_converter_=lambda chunk: chunk)
    install_module(monkeypatch, "ltx_pipelines.utils.args", ImageConditioningInput=lambda **kw: kw)
    monkeypatch.setattr(runtime_module, "verify_upstream", lambda: None)
    monkeypatch.setattr(ModelInventory, "validate", lambda self: None)

    def encode(chunks, path, **kwargs):
        assert list(chunks) == [b"pixels"]
        kwargs["on_flush"]()
        path.write_bytes(b"stub")

    monkeypatch.setattr(runtime_module, "encode_mp4", encode)
    monkeypatch.setattr(
        runtime_module,
        "inspect_output",
        lambda path, settings, **kw: OutputInfo(**kw, duration=kw["frames"] / kw["fps"]),
    )
    return builds, calls


def test_official_adapter_t2v_i2v_and_cache(fake_upstream, tmp_path):
    builds, calls = fake_upstream
    settings = Settings(_env_file=None, data_dir=tmp_path)
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load(preload=True)
    assert builds == ["text", "transformer"]
    for image in (None, "asset_" + "a" * 32):
        spec = normalize_request(
            GenerationRequest(prompt="x", generate_audio=False, first_frame=image), settings, 42
        )
        stages = []
        context = GenerationContext(
            "gen_test",
            spec,
            tmp_path / "output.partial",
            {image: tmp_path / "input.bin"} if image else {},
            Event(),
            stages.append,
        )
        runtime.generate(context)
        assert stages == [
            JobStatus.ENCODING,
            JobStatus.GENERATING,
            JobStatus.DECODING,
            JobStatus.ENCODING_OUTPUT,
        ]
    assert builds == ["text", "transformer"]
    assert calls[0]["images"] == []
    assert calls[1]["images"] == [
        {"path": str(tmp_path / "input.bin"), "frame_idx": 0, "strength": 1.0}
    ]
    assert not runtime.transformer.on_gpu
    runtime.close()


def test_cancellation_at_transformer_boundary(fake_upstream, tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load()
    runtime.stop.set()
    with pytest.raises(ServiceError) as exc:
        runtime.pipeline.stage()
    assert exc.value.detail.code == "GENERATION_CANCELLED"
    runtime.close()


def test_generated_stereo_audio_is_interleaved(fake_upstream, tmp_path, monkeypatch):
    import ltx_server.inference.runtime as runtime_module

    class Wave:
        ndim = 2
        shape = (2, 3)

        def detach(self):
            return self

        float = cpu = contiguous = numpy = detach

        def transpose(self, first, second):
            assert (first, second) == (0, 1)
            self.shape = (3, 2)
            return self

        def astype(self, dtype):
            assert dtype == "<f4"
            assert self.shape == (3, 2)
            return self

        def tobytes(self):
            return struct.pack("<6f", 0.1, -0.1, 0.2, -0.2, 0.3, -0.3)

    settings = Settings(_env_file=None, data_dir=tmp_path)
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load()
    runtime.pipeline.audio = SimpleNamespace(waveform=Wave(), sampling_rate=24000)
    monkeypatch.setattr(
        runtime.torch,
        "isfinite",
        lambda waveform: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
        raising=False,
    )

    def encode(chunks, path, **kwargs):
        assert list(chunks) == [b"pixels"]
        assert kwargs["audio_rate"] == 24000
        kwargs["audio"].seek(0)
        assert struct.unpack("<6f", kwargs["audio"].read()) == pytest.approx(
            [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
        )
        kwargs["on_flush"]()

    monkeypatch.setattr(runtime_module, "encode_mp4", encode)
    spec = normalize_request(GenerationRequest(prompt="x", generate_audio=True), settings, 42)
    try:
        result = runtime.generate(
            GenerationContext(
                "gen_test", spec, tmp_path / "output.partial", {}, Event(), lambda s: None
            )
        )
        assert result.has_audio
    finally:
        runtime.close()


@pytest.mark.parametrize("commit", [LTX_COMMIT, "wrong"])
def test_exact_upstream_revision_guard(monkeypatch, commit):
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            version="1.3.0",
            read_text=lambda file: json.dumps({"vcs_info": {"commit_id": commit}}),
        ),
    )
    if commit == LTX_COMMIT:
        verify_upstream()
    else:
        with pytest.raises(ServiceError) as exc:
            verify_upstream()
        assert exc.value.detail.code == "UPSTREAM_INCOMPATIBLE"


@pytest.mark.parametrize("mode", ["endpoints", "keyframes"])
def test_all_image_targets_use_pixel_indices(fake_upstream, tmp_path, mode):
    _, calls = fake_upstream
    settings = Settings(_env_file=None, data_dir=tmp_path)
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load()
    a, b = "asset_" + "a" * 32, "asset_" + "b" * 32
    conditioning = (
        {"first_frame": a, "last_frame": b}
        if mode == "endpoints"
        else {
            "keyframes": [
                {"asset_id": b, "frame": 23, "strength": 0.4},
                {"asset_id": a, "frame": 0, "strength": 0.8},
            ]
        }
    )
    spec = normalize_request(
        GenerationRequest(prompt="x", duration=1, generate_audio=False, **conditioning), settings, 1
    )
    try:
        runtime.generate(
            GenerationContext(
                "gen_test",
                spec,
                tmp_path / "out.partial",
                {a: tmp_path / "a.bin", b: tmp_path / "b.bin"},
                Event(),
                lambda s: None,
            )
        )
        targets = calls[-1]["images"]
        assert [item["frame_idx"] for item in targets] == [0, 24 if mode == "endpoints" else 23]
        assert [item["strength"] for item in targets] == (
            [1, 1] if mode == "endpoints" else [0.8, 0.4]
        )
        assert [item["path"] for item in targets] == [
            str(tmp_path / "a.bin"),
            str(tmp_path / "b.bin"),
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize("fail", [False, True])
def test_a2v_both_stages_and_next_t2v_are_isolated(fake_upstream, tmp_path, monkeypatch, fail):
    from ltx_server.inference.conditioning import AudioConditioning

    builds, calls = fake_upstream
    settings = Settings(_env_file=None, data_dir=tmp_path)
    runtime = LTXRuntime(settings, ModelInventory(settings), Event())
    runtime.load()
    latent, source = object(), object()
    install_module(
        monkeypatch, "ltx_pipelines.utils.blocks", AudioConditioner=lambda *a, **kw: "conditioner"
    )

    def prepare(self, context, torch, conditioner, settings):
        assert context.spec.audio and conditioner == "conditioner"
        self.latent, self.source = latent, source
        if fail:
            raise ServiceError("INVALID_MEDIA", "Bad audio", 422)

    monkeypatch.setattr(AudioConditioning, "prepare", prepare)
    asset = "asset_" + "a" * 32
    spec = normalize_request(
        GenerationRequest(prompt="x", audio=asset, generate_audio=False), settings, 1
    )
    context = GenerationContext(
        "gen_test",
        spec,
        tmp_path / "out.partial",
        {asset: tmp_path / "audio.bin"},
        Event(),
        lambda s: None,
    )
    try:
        if fail:
            with pytest.raises(ServiceError):
                runtime.generate(context)
        else:
            runtime.generate(context)
            assert len(calls[-1]["stages"]) == 2
            for stage in calls[-1]["stages"]:
                assert stage["audio"].initial_latent is latent
                assert stage["audio"].frozen and stage["audio"].noise_scale == 0
        assert (
            runtime.audio_conditioning.latent is None and runtime.audio_conditioning.source is None
        )
        spec = normalize_request(GenerationRequest(prompt="x", generate_audio=False), settings, 2)
        runtime.generate(
            GenerationContext(
                "gen_next", spec, tmp_path / "next.partial", {}, Event(), lambda s: None
            )
        )
        assert all(not stage["audio"].frozen for stage in calls[-1]["stages"])
        assert builds.count("transformer") == 1
    finally:
        runtime.close()
