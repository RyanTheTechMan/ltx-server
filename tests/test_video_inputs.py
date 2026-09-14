import json
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


def make_clip(path, *, audio=False, duration=0.5, rotated=False):
    args = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=96x64:r=30:d={duration}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)
    if rotated:
        original = path.with_suffix(".unrotated.mp4")
        path.rename(original)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(original),
                "-c",
                "copy",
                "-metadata:s:v:0",
                "rotate=90",
                str(path),
            ],
            check=True,
        )
        rotation = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream_side_data=rotation",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )
        if not any(s.get("side_data_list") for s in rotation["streams"]):
            # New FFmpeg versions use the input display transform instead of the rotate tag.
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-display_rotation",
                    "90",
                    "-i",
                    str(original),
                    "-c",
                    "copy",
                    str(path),
                ],
                check=True,
            )


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("duration", [0.5, 2])
async def test_real_retake_rotation_padding_trimming_audio(tmp_path, audio, duration):
    import tempfile

    from ltx_server.media.ffmpeg import decode_source_audio, inspect_output
    from ltx_server.media.validation import validate_media
    from ltx_server.media.video_inputs import encode_source_video, prepare_retake

    source, prepared, output = [
        tmp_path / name for name in ("source.mp4", "prep.partial", "out.mp4")
    ]
    make_clip(source, audio=audio, duration=duration, rotated=True)
    original_bytes = source.read_bytes()
    settings = Settings(_env_file=None, data_dir=tmp_path)
    info = await validate_media(source, settings)
    assert info.has_audio is audio
    spec = normalize_request(
        GenerationRequest(prompt="x", duration=1, orientation="portrait"), settings, 1
    ).model_copy(update={"width": 64, "height": 96})
    with pytest.raises(ServiceError):
        validate_retake_source(source, spec, settings)
    prepare_retake(source, prepared, spec, settings, Event())
    metadata = validate_retake_source(prepared, spec, settings)
    assert metadata.has_audio is audio and metadata.codec == "h264"

    # The prepared first frame must follow display rotation, not the stored orientation.
    def frame(path, filters):
        return subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                filters,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )

    expected = frame(source, "scale=64:96")
    actual = frame(prepared, "null")
    assert sum(abs(a - b) for a, b in zip(actual, expected, strict=True)) / len(actual) < 8
    # Audio-only output uses the same compressed H.264 video packets, not VAE-decoded frames.
    with tempfile.TemporaryFile() as pcm:
        if audio:
            samples = decode_source_audio(
                prepared, duration=25 / 24, cancel=Event(), settings=settings
            )
            pcm.write(samples)
            if duration < 1:
                import struct

                tail = struct.unpack(f"<{len(samples[-4800:]) // 4}f", samples[-4800:])
                assert max(abs(value) for value in tail) < 0.001
        encode_source_video(
            prepared,
            output,
            spec,
            settings,
            Event(),
            codec="h264",
            audio=pcm if audio else None,
            audio_rate=48000,
        )
    verified = inspect_output(
        output, settings, width=64, height=96, frames=25, fps=24, has_audio=audio
    )
    assert verified.duration == pytest.approx(25 / 24, abs=0.001)

    def video_hash(path):
        return subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-f",
                "hash",
                "-",
            ]
        )

    assert video_hash(prepared) == video_hash(output)
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("reason", ["cancel", "timeout", "size"])
def test_preparation_reaps_child_process_on_limits(tmp_path, monkeypatch, reason):
    import sys
    from threading import Timer

    from ltx_server.media import processes

    event = Event()
    children = []
    popen = subprocess.Popen

    def tracked(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(processes.subprocess, "Popen", tracked)
    timer = Timer(0.1, event.set)
    if reason == "cancel":
        timer.start()
    try:
        with pytest.raises(ServiceError) as exc:
            processes.run_bounded(
                [
                    sys.executable,
                    "-c",
                    "import time; print('x'*1100000, flush=True); time.sleep(10)"
                    if reason == "size"
                    else "import time; time.sleep(10)",
                ],
                event,
                0.1 if reason == "timeout" else 5,
            )
        assert exc.value.detail.code == (
            "GENERATION_CANCELLED" if reason == "cancel" else "INVALID_MEDIA"
        )
        assert children and all(child.poll() is not None for child in children)
    finally:
        timer.cancel()


def test_center_crop_accounts_for_non_square_pixels(tmp_path):
    from ltx_server.media.video_inputs import prepare_retake

    source, prepared = tmp_path / "wide.mp4", tmp_path / "prepared.partial"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=128x64:r=24:d=1,drawbox=x=44:y=0:w=40:h=64:color=blue:t=fill,setsar=2",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    settings = Settings(_env_file=None, data_dir=tmp_path)
    spec = normalize_request(GenerationRequest(prompt="x", duration=1), settings, 1).model_copy(
        update={"width": 64, "height": 64}
    )
    prepare_retake(source, prepared, spec, settings, Event())
    pixels = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(prepared),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
    )
    assert len(pixels) == 64 * 64 * 3
    assert sum(pixels[2::3]) / (64 * 64) > 240
    assert sum(pixels[0::3]) / (64 * 64) < 10


def test_preparation_enforces_destination_size(tmp_path):
    import sys

    from ltx_server.media.processes import run_bounded

    path = tmp_path / "oversized.partial"
    with pytest.raises(ServiceError, match="size limits"):
        run_bounded(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'x'*4096)",
                str(path),
            ],
            Event(),
            5,
            destination=path,
            max_bytes=1024,
        )
