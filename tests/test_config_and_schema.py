from pathlib import Path

import pytest
from pydantic import ValidationError

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.schemas.generation import (
    RESOLUTIONS,
    GenerationRequest,
    frame_count,
    normalize_request,
)


def test_env_loading_and_secret_redaction(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PORT=8123\nAPI_KEY=very-secret\nDEFAULT_FPS=30\nENABLE_DOCS=false\n")
    monkeypatch.setenv("PORT", "8765")
    settings = Settings(_env_file=env, data_dir=tmp_path)
    assert settings.port == 8765
    assert settings.default_fps == 30
    assert settings.enable_docs is False
    assert settings.output_dir == tmp_path / "outputs"
    assert "very-secret" not in repr(settings)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_concurrent_generations": 2},
        {"max_queue_size": 0},
        {"output_ttl_seconds": 0},
        {"port": 99999},
        {"max_duration_seconds": 5},
        {"default_resolution": "1080p", "max_resolution": "540p"},
        {"output_dir": Path("/same"), "temp_dir": Path("/same/child")},
    ],
)
def test_invalid_settings(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": " "},
        {"prompt": "x", "duration": True},
        {"prompt": "x", "duration": "10"},
        {"prompt": "x", "duration": float("nan")},
        {"prompt": "x", "fps": 60},
        {"prompt": "x", "resolution": "2160p"},
        {"prompt": "x", "seed": -1},
        {"prompt": "x", "first_frame": "/etc/passwd"},
        {"prompt": "x", "unknown": 1},
        {"prompt": "x", "last_frame": "asset_" + "a" * 32},
        {"prompt": "x", "keyframes": [{"asset_id": "asset_" + "a" * 32, "frame": -1}]},
    ],
)
def test_strict_generation_schema(body):
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(body)


def test_normalization():
    settings = Settings(_env_file=None)
    spec = normalize_request(GenerationRequest(prompt="test", seed=0), settings, 123)
    assert spec.seed == 0
    assert (spec.width, spec.height) == (1024, 576)
    assert spec.frames == 241
    assert all(w % 64 == h % 64 == 0 for w, h in RESOLUTIONS.values())


@pytest.mark.parametrize(
    ("seconds", "fps", "expected"),
    [
        (5, 24, 121),
        (10, 24, 241),
        (10, 30, 297),
        (1, 24, 25),
        (1.5, 30, 49),
    ],
)
def test_frame_conversion(seconds, fps, expected):
    assert frame_count(seconds, fps) == expected


def test_configured_limits_and_keyframe_bounds():
    settings = Settings(_env_file=None, max_duration_seconds=10, max_resolution="540p")
    for body, code in [
        ({"duration": 11}, ErrorCode.INVALID_DURATION),
        ({"resolution": "720p"}, ErrorCode.INVALID_RESOLUTION),
        ({"keyframes": [{"asset_id": "asset_" + "a" * 32, "frame": 241}]}, ErrorCode.INVALID_INPUT),
    ]:
        with pytest.raises(ServiceError) as exc:
            normalize_request(GenerationRequest(prompt="x", **body), settings, 1)
        assert exc.value.detail.code == code


@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("orientation", ["landscape", "portrait"])
def test_orientation_preserves_grid(resolution, orientation):
    spec = normalize_request(
        GenerationRequest(prompt="test", resolution=resolution, orientation=orientation),
        Settings(_env_file=None),
        1,
    )
    expected = RESOLUTIONS[resolution]
    if orientation == "portrait":
        expected = expected[::-1]
    assert (spec.width, spec.height) == expected
    assert spec.width % 64 == spec.height % 64 == 0


def test_opt_in_retake_fps_and_strict_default():
    retake = {"video": "asset_" + "a" * 32, "start": 0, "end": 1}
    settings = Settings(_env_file=None, default_fps=30)
    strict = normalize_request(GenerationRequest(prompt="x", retake=retake), settings, 1)
    assert strict.fps == 30 and not strict.retake.normalize_source
    retake["normalize_source"] = True
    prepared = normalize_request(GenerationRequest(prompt="x", retake=retake), settings, 1)
    assert prepared.fps == 24 and prepared.frames == 241
    for fps in (25, 30):
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="x", retake=retake, fps=fps)
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="x", orientation="square")
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="x", retake={**retake, "normalize_source": "true"})
