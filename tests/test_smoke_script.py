import argparse
import io
import json
import sys
from types import SimpleNamespace

import pytest

from ltx_server.config import Settings
from ltx_server.inference.models import ModelInventory
from ltx_server.jobs.state import OutputInfo
from scripts import smoke_test

from .test_conditioning import wav_audio


@pytest.mark.parametrize("override", [None, "environment-key"])
def test_api_smoke_uses_dotenv_auth_with_environment_precedence(tmp_path, monkeypatch, override):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("API_KEY=dotenv-key\n")
    monkeypatch.delenv("API_KEY", raising=False)
    if override:
        monkeypatch.setenv("API_KEY", override)
    requests = []

    def urlopen(request, timeout):
        assert request.get_header("Authorization") == f"Bearer {override or 'dotenv-key'}"
        requests.append(request.full_url)
        return io.BytesIO(json.dumps({"backend_enabled": True, "model_state": "ready"}).encode())

    monkeypatch.setattr(smoke_test, "urlopen", urlopen)
    smoke_test.api_smoke("http://localhost:8000")
    assert len(requests) == 4


@pytest.mark.parametrize("invalid", [False, True])
async def test_gpu_smoke_stages_and_cleans_its_inputs(tmp_path, png, monkeypatch, invalid):
    root = tmp_path.resolve()
    image = root / "original.png"
    image.write_bytes(png)
    audio = root / "original.wav"
    audio.write_bytes(wav_audio())
    settings = Settings(_env_file=None, data_dir=root / "data")
    monkeypatch.setattr(smoke_test, "Settings", lambda **kw: settings)
    monkeypatch.setattr(ModelInventory, "validate", lambda self: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda d: "contract-test-device"
            )
        ),
    )
    calls = []

    class Backend:
        def __init__(self, settings):
            pass

        async def generate(self, context):
            assert context.spec.audio and context.spec.first_frame and context.spec.last_frame
            assert len(context.asset_paths) == 3
            assert all(path.is_file() for path in context.asset_paths.values())
            context.partial_path.write_bytes(b"transport fixture")
            calls.append("generate")
            return OutputInfo(
                width=context.spec.width,
                height=context.spec.height,
                frames=context.spec.frames,
                fps=context.spec.fps,
                duration=context.spec.frames / context.spec.fps,
                has_audio=True,
            )

        async def close(self):
            calls.append("close")

    monkeypatch.setattr(smoke_test, "PipelineManager", Backend)
    args = argparse.Namespace(
        first_frame=None if invalid else str(image),
        last_frame=str(image),
        audio=str(audio),
        keyframe=[],
        prompt="test",
        duration=1,
        no_audio=False,
        reference_video=None,
        reference_lora=None,
        lora=[],
        retake_video=None,
    )
    if invalid:
        with pytest.raises(ValueError):
            await smoke_test.gpu_smoke(args)
        assert calls == ["close"]
    else:
        await smoke_test.gpu_smoke(args)
        assert calls == ["generate", "close"]
    assert image.read_bytes() == png and audio.read_bytes() == wav_audio()
    assert not list(settings.asset_dir.glob("asset_*.bin"))
    assert not list(settings.temp_dir.glob("*.partial"))
