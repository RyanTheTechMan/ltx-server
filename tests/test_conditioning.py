import asyncio
import io
import math
import shutil
import struct
import wave
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

import pytest

from ltx_server.config import Settings
from ltx_server.errors import ServiceError
from ltx_server.inference.backend import GenerationContext
from ltx_server.inference.conditioning import (
    AudioConditioning,
    ConditionedStage,
    SourceAudioDecoder,
)
from ltx_server.media.ffmpeg import decode_source_audio
from ltx_server.schemas.generation import GenerationRequest, normalize_request

from .conftest import ControlledBackend, wait_terminal


def wav_audio(seconds=0.2, rate=16000):
    out = io.BytesIO()
    with wave.open(out, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(
            b"".join(
                struct.pack("<h", round(8000 * math.sin(2 * math.pi * 440 * n / rate)))
                for n in range(round(seconds * rate))
            )
        )
    return out.getvalue()


@pytest.mark.parametrize("seconds", [0.2, 2.0])
def test_source_audio_resampled_trimmed_padded_and_detected_by_content(tmp_path, seconds):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    path = tmp_path / "asset.bin"
    path.write_bytes(wav_audio(seconds))
    result = decode_source_audio(
        path, duration=25 / 24, cancel=Event(), settings=Settings(_env_file=None, data_dir=tmp_path)
    )
    assert len(result) == 50000 * 2 * 4
    samples = list(struct.iter_unpack("<ff", result))
    assert max(abs(left) for left, _ in samples[:4800]) > 0.1
    assert all(left == right for left, right in samples)
    # Mono -> stereo does not time stretch: verify the original 440 Hz pitch.
    crossings = sum(samples[i][0] < 0 <= samples[i + 1][0] for i in range(4799))
    assert crossings == pytest.approx(44, abs=1)
    if seconds < 1:
        assert all(left == right == 0 for left, right in samples[12000:])
    else:
        assert max(abs(left) for left, _ in samples[-4800:]) > 0.1


def test_audio_decode_cancellation_and_corruption(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    path = tmp_path / "bad.bin"
    path.write_bytes(b"not audio")
    cancel = Event()
    cancel.set()
    with pytest.raises(ServiceError) as error:
        decode_source_audio(path, duration=1, cancel=cancel, settings=settings)
    assert error.value.detail.code == "GENERATION_CANCELLED"
    if shutil.which("ffmpeg"):
        with pytest.raises(ServiceError) as error:
            decode_source_audio(path, duration=1, cancel=Event(), settings=settings)
        assert error.value.detail.code == "INVALID_MEDIA"


def test_frozen_audio_stages_and_source_decoder_reset():
    @dataclass
    class Modality:
        context: str
        frozen: bool = False
        noise_scale: float = 1.0
        initial_latent: object = None

    audio = AudioConditioning()
    audio.latent, audio.source = object(), object()
    stage = ConditionedStage(lambda **kwargs: kwargs, audio)
    decoder_calls = []
    decoder = SourceAudioDecoder(lambda latent: decoder_calls.append(latent), audio)
    original = Modality("text context")
    for scale in (1.0, 0.5):
        original.noise_scale = scale
        result = stage(audio=original, video="video", sigmas="official schedule")
        assert result["audio"].initial_latent is audio.latent
        assert result["audio"].frozen and result["audio"].noise_scale == 0
        assert result["audio"].context == original.context
        assert not original.frozen and original.initial_latent is None
        assert result["video"] == "video" and result["sigmas"] == "official schedule"
        assert decoder("model output") is audio.source
    assert not decoder_calls
    audio.clear()
    assert stage(audio=original)["audio"] is original
    decoder("generated audio")
    assert decoder_calls == ["generated audio"]


@pytest.mark.parametrize("mode", ["endpoints", "keyframes", "audio_and_images"])
async def test_conditioning_admission_asset_leases_and_cancel(app_factory, png, mode):
    backend = ControlledBackend(blocked=True)
    async with app_factory(backend) as (app, client):
        ids = [
            (await client.post("/v1/assets", files={"file": ("frame.png", png)})).json()["id"]
            for _ in range(2)
        ]
        body = {"prompt": "Move toward the destination", "duration": 1}
        if mode == "keyframes":
            body["keyframes"] = [
                {"asset_id": ids[1], "frame": 24, "strength": 0.8},
                {"asset_id": ids[0], "frame": 0},
            ]
        else:
            body.update(first_frame=ids[0], last_frame=ids[1])
        if mode == "audio_and_images":
            audio = (
                await client.post("/v1/assets", files={"file": ("audio.wav", wav_audio())})
            ).json()["id"]
            ids.append(audio)
            body["audio"] = audio
            body["generate_audio"] = False
        response = await client.post("/v1/generations", json=body)
        assert response.status_code == 202, response.text
        identifier = response.json()["id"]
        await asyncio.wait_for(backend.started.wait(), 2)
        spec = app.state.jobs.get(identifier).spec
        assert set(spec.assets()) == set(ids)
        for asset in ids:
            assert (await client.delete("/v1/assets/" + asset)).status_code == 409
        await client.delete("/v1/generations/" + identifier)
        job = await wait_terminal(app, identifier)
        assert job.info.status == "cancelled"
        for asset in ids:
            assert (await client.delete("/v1/assets/" + asset)).status_code == 204
        assert not list(app.state.storage.roots["tmp"].glob("*.partial"))


async def test_invalid_conditioning_rejected_atomically(app_factory, png):
    async with app_factory() as (app, client):
        identifier = (await client.post("/v1/assets", files={"file": ("x.png", png)})).json()["id"]
        bodies = [
            {"audio": identifier},
            {"keyframes": [{"asset_id": identifier, "frame": 25}]},
            {
                "keyframes": [
                    {"asset_id": identifier, "frame": 1},
                    {"asset_id": identifier, "frame": 1},
                ]
            },
            {"first_frame": identifier, "keyframes": [{"asset_id": identifier, "frame": 0}]},
        ]
        for conditioning in bodies:
            response = await client.post(
                "/v1/generations", json={"prompt": "x", "duration": 1, **conditioning}
            )
            assert response.status_code == 422
            assert not app.state.assets.records[identifier].leases
        assert not tuple(app.state.jobs.repository.all())
        caps = (await client.get("/v1/models")).json()["models"][0]["capabilities"]
        assert caps["first_last_frame"] and caps["multiple_keyframes"] and caps["audio_to_video"]


@pytest.mark.parametrize("latent_frames", [20, 26, 30])
def test_audio_encoder_batch_and_temporal_grid(tmp_path, monkeypatch, latent_frames):
    import ltx_server.inference.conditioning as module

    from .test_runtime_contract import install_module

    class Tensor:
        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)

        def reshape(self, *shape):
            assert shape == (-1, 2)
            return Tensor((50000, 2))

        def transpose(self, a, b):
            return Tensor(tuple(reversed(self.shape)))

        def contiguous(self):
            return self

        def unsqueeze(self, axis):
            assert axis == 0
            return Tensor((1, *self.shape))

        def __getitem__(self, slices):
            assert slices[2].stop == 26
            return Tensor((1, 8, 26, 16))

    padded = []

    def pad(tensor, sizes):
        padded.append(sizes)
        return Tensor((1, 8, 26, 16))

    torch = SimpleNamespace(
        float32="f32",
        frombuffer=lambda data, dtype: Tensor((100000,)),
        isfinite=lambda tensor: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
        nn=SimpleNamespace(functional=SimpleNamespace(pad=pad)),
    )

    def encode(audio, encoder, processor):
        assert audio.waveform.shape == (1, 2, 50000)
        assert audio.sampling_rate == 48000 and encoder == "encoder" and processor is None
        return Tensor((1, 8, latent_frames, 16))

    install_module(monkeypatch, "ltx_core.model.audio_vae", encode_audio=encode)
    install_module(
        monkeypatch,
        "ltx_core.types",
        Audio=lambda **kw: SimpleNamespace(**kw),
        AudioLatentShape=SimpleNamespace(from_duration=lambda **kw: SimpleNamespace(frames=26)),
    )
    monkeypatch.setattr(module, "decode_source_audio", lambda *a, **kw: b"pcm")
    settings = Settings(_env_file=None, data_dir=tmp_path)
    asset = "asset_" + "a" * 32
    spec = normalize_request(GenerationRequest(prompt="x", audio=asset, duration=1), settings, 1)
    context = GenerationContext(
        "gen_test",
        spec,
        tmp_path / "out.partial",
        {asset: tmp_path / "audio.bin"},
        Event(),
        lambda s: None,
    )
    audio = AudioConditioning()
    audio.prepare(context, torch, lambda fn: fn("encoder"), settings)
    assert audio.source.waveform.shape == (2, 50000)
    assert audio.latent.shape == (1, 8, 26, 16)
    assert padded == ([(0, 0, 0, 6)] if latent_frames == 20 else [])


@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_audio_decode_stops_and_reaps_running_ffmpeg(tmp_path, monkeypatch, reason):
    import ltx_server.media.ffmpeg as module

    cancel = Event()
    operations = []

    class Process:
        def __init__(self, *args, **kwargs):
            self.running = True

        def poll(self):
            if reason == "cancel":
                cancel.set()
            return None if self.running else -9

        def kill(self):
            self.running = False
            operations.append("kill")

        def wait(self):
            assert not self.running
            operations.append("wait")
            return -9

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    clock = iter([100.0, 200.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(ServiceError) as error:
        decode_source_audio(
            tmp_path / "source.bin",
            duration=1,
            cancel=cancel,
            settings=Settings(_env_file=None, data_dir=tmp_path),
        )
    assert error.value.detail.code == (
        "GENERATION_CANCELLED" if reason == "cancel" else "INVALID_MEDIA"
    )
    assert operations == ["kill", "wait"]
