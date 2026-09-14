import shutil
import subprocess
from threading import Event

import pytest

from ltx_server.config import Settings
from ltx_server.errors import ServiceError
from ltx_server.media.video_inputs import prepare_reference, validate_retake_source
from ltx_server.schemas.generation import GenerationRequest, normalize_request


def test_real_reference_normalization_and_retake_grid(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required")
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=96x64:r=12:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    settings = Settings(_env_file=None, data_dir=tmp_path)
    # Small dimensions exercise actual media handling without involving model inference.
    spec = normalize_request(GenerationRequest(prompt="x", duration=1), settings, 1).model_copy(
        update={"width": 64, "height": 64}
    )
    partial = tmp_path / "gen.partial"
    prepare_reference(source, partial, spec, settings, Event())
    validate_retake_source(partial, spec, settings)
    with pytest.raises(ServiceError) as error:
        validate_retake_source(source, spec, settings)
    assert error.value.detail.code == "INVALID_INPUT"
    cancel = Event()
    cancel.set()
    with pytest.raises(ServiceError) as error:
        prepare_reference(source, partial, spec, settings, cancel)
    assert error.value.detail.code == "GENERATION_CANCELLED"


@pytest.mark.parametrize(
    "values",
    [
        {"vae_tile_size": 256, "vae_tile_overlap": 256},
        {"vae_temporal_size": 16, "vae_temporal_overlap": 24},
        {"vae_tile_size": 257},
        {"offload_mode": "magic"},
        {"attention_backend": "unknown"},
    ],
)
def test_invalid_performance_settings(values):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **values)
