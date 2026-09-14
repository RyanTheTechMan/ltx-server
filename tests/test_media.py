import asyncio
import io
import shutil
import subprocess
import wave

import pytest
from PIL import Image

from ltx_server.media.validation import run_media_tool


@pytest.mark.parametrize(
    "content",
    [b"", b"not an image", b"#EXTM3U\nfile:///etc/passwd\n", b"RIFF\x00\x00\x00\x00WAVEjunk"],
)
async def test_invalid_uploads_leave_no_files(app_factory, content):
    async with app_factory() as (app, client):
        response = await client.post("/v1/assets", files={"file": ("x.png", content, "image/png")})
        assert response.status_code == 415
        assert not list(app.state.storage.files("tmp"))
        assert not list(app.state.storage.files("assets"))


async def test_bounded_multipart_and_unauthorized_upload(app_factory, png):
    async with app_factory(max_image_upload_mb=1, max_audio_upload_mb=1, max_video_upload_mb=1) as (
        app,
        client,
    ):

        async def body():
            yield (
                b'--boundary\r\nContent-Disposition: form-data; name="file"; '
                b'filename="x.png"\r\nContent-Type: image/png\r\n\r\n'
            )
            for _ in range(20):
                yield b"x" * (64 * 1024)
            yield b"\r\n--boundary--\r\n"

        response = await client.post(
            "/v1/assets",
            content=body(),
            headers={
                "Content-Type": "multipart/form-data; boundary=boundary",
            },
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
        response = await client.post(
            "/v1/assets",
            content=b"x",
            headers={
                "Content-Length": "9999999999",
            },
        )
        assert response.status_code == 413
        response = await client.post(
            "/v1/assets", files=[("file", ("a", png)), ("file", ("b", png))]
        )
        assert response.status_code == 400
        assert not list(app.state.storage.files("tmp"))
    async with app_factory(api_key="secret") as (_, client):
        response = await client.post("/v1/assets", content=b"not multipart")
        assert response.status_code == 401


async def test_detected_image_limit_and_pixel_limit(app_factory, png):
    async with app_factory(max_image_upload_mb=1) as (app, client):
        # PNG remains decodable with trailing bytes; type-specific size still applies.
        response = await client.post(
            "/v1/assets",
            files={
                "file": (
                    "x.mp4",
                    png + b"x" * (1024 * 1024),
                    "video/mp4",
                )
            },
        )
        assert response.status_code == 413
        assert not app.state.assets.records
    async with app_factory(max_image_pixels=100) as (_, client):
        response = await client.post("/v1/assets", files={"file": ("x.png", png)})
        assert response.status_code == 400


async def test_animated_images_rejected(app_factory):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(
        buf,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (16, 16), "blue")],
    )
    async with app_factory() as (_, client):
        response = await client.post("/v1/assets", files={"file": ("x.png", buf.getvalue())})
        assert response.status_code == 400


async def test_corrupt_image_is_client_error(app_factory, png):
    damaged = bytearray(png)
    damaged[-5] ^= 0xFF
    async with app_factory() as (app, client):
        response = await client.post("/v1/assets", files={"file": ("bad.png", bytes(damaged))})
        assert response.status_code in (400, 415)
        assert not list(app.state.storage.files("tmp"))


async def test_cleanup_preserves_in_progress_upload(app_factory, png, monkeypatch):
    import os

    from ltx_server.media import assets
    from ltx_server.media.validation import MediaInfo

    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_validate(path, settings):
        entered.set()
        await release.wait()
        return MediaInfo("image", "image/png", 16, 16)

    monkeypatch.setattr(assets, "validate_media", slow_validate)
    async with app_factory() as (app, client):
        upload = asyncio.create_task(client.post("/v1/assets", files={"file": ("x.png", png)}))
        try:
            async with asyncio.timeout(3):
                await entered.wait()
            identifier = next(iter(app.state.assets.uploading))
            path = app.state.storage.path("tmp", f"{identifier}.partial")
            os.utime(path, (0, 0))
            app.state.cleanup.sweep()
            assert path.exists()
        finally:
            release.set()
            response = await upload
        assert response.status_code == 201


def require_ffmpeg():
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg/ffprobe required for media integration tests")


async def test_real_audio_upload(app_factory):
    require_ffmpeg()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000)
    async with app_factory() as (_, client):
        response = await client.post(
            "/v1/assets",
            files={
                "file": (
                    "misleading.png",
                    buffer.getvalue(),
                    "image/png",
                )
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["type"] == "audio"
        assert response.json()["mime_type"] == "audio/wav"
        assert response.json()["duration"] == 1


async def test_real_video_upload(app_factory, tmp_path):
    require_ffmpeg()
    video = tmp_path / "source.mp4"
    await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=24",
            "-t",
            "0.25",
            "-c:v",
            "mpeg4",
            str(video),
        ],
        check=True,
        timeout=10,
    )
    async with app_factory() as (_, client):
        response = await client.post(
            "/v1/assets",
            files={
                "file": (
                    "fake.jpg",
                    video.read_bytes(),
                    "image/jpeg",
                )
            },
        )
        assert response.status_code == 201, response.text
        assert response.json()["type"] == "video"
        assert response.json()["width"] == 64


async def test_missing_media_tool_and_timeout(tmp_path):
    from ltx_server.errors import ServiceError

    with pytest.raises(ServiceError) as exc:
        await run_media_tool(str(tmp_path / "missing"), [], 1)
    assert exc.value.detail.code == "MEDIA_UNAVAILABLE"
    # A real subprocess timeout also exercises termination/reaping without GPU dependencies.
    import sys

    with pytest.raises(ServiceError) as exc:
        await run_media_tool(sys.executable, ["-c", "import time; time.sleep(30)"], 0.01)
    assert "timed out" in exc.value.detail.message
