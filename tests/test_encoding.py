import math
import shutil
import struct
import tempfile
from threading import Event

import pytest

from ltx_server.config import Settings
from ltx_server.errors import ServiceError
from ltx_server.media.ffmpeg import encode_mp4, inspect_output


@pytest.mark.parametrize("with_audio", [False, True])
def test_real_encoded_mp4_video_and_synchronized_audio(tmp_path, with_audio):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required")
    settings = Settings(_env_file=None, data_dir=tmp_path)
    output = tmp_path / "gen.partial"
    # 25 actual YUV420p frames (a one-second-ish clip at 24 fps), no GPU stand-in.
    frame = bytes([64]) * (64 * 64) + bytes([128]) * (64 * 64 // 2)
    stages = []
    with tempfile.TemporaryFile() as audio:
        samples = [0.1 * math.sin(2 * math.pi * 440 * i / 48000) for i in range(50000)]
        for sample in samples:
            audio.write(struct.pack("<ff", sample, sample))
        encode_mp4(
            [frame] * 25,
            output,
            width=64,
            height=64,
            frames=25,
            fps=24,
            cancel=Event(),
            settings=settings,
            audio=audio if with_audio else None,
            on_flush=lambda: stages.append("flush"),
        )
    info = inspect_output(
        output, settings, width=64, height=64, frames=25, fps=24, has_audio=with_audio
    )
    assert info.has_audio is with_audio
    assert info.duration == pytest.approx(25 / 24, abs=0.001)
    assert stages == ["flush"]
    assert output.stat().st_size > 1000
    # Metadata mismatches cannot be published as a success.
    with pytest.raises(ServiceError) as exc:
        inspect_output(
            output, settings, width=128, height=64, frames=25, fps=24, has_audio=with_audio
        )
    assert exc.value.detail.code == "OUTPUT_VALIDATION_FAILED"


def test_encoder_cancellation_and_short_stream(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    settings = Settings(_env_file=None, data_dir=tmp_path)
    frame = bytes(64 * 64 * 3 // 2)
    cancel = Event()

    def chunks():
        yield frame
        cancel.set()
        yield frame

    with pytest.raises(ServiceError) as exc:
        encode_mp4(
            chunks(),
            tmp_path / "cancelled.partial",
            width=64,
            height=64,
            frames=25,
            fps=24,
            cancel=cancel,
            settings=settings,
        )
    assert exc.value.detail.code == "GENERATION_CANCELLED"
    with pytest.raises(ServiceError) as exc:
        encode_mp4(
            [frame],
            tmp_path / "short.partial",
            width=64,
            height=64,
            frames=25,
            fps=24,
            cancel=Event(),
            settings=settings,
        )
    assert exc.value.detail.code == "OUTPUT_ENCODING_FAILED"
