import asyncio
from datetime import timedelta

import pytest

from ltx_server.jobs.state import JobStatus, utcnow

from .conftest import ControlledBackend, wait_terminal


async def test_disabled_backend_is_honest_and_diagnostics_work(app_factory):
    async with app_factory() as (app, client):
        health = (await client.get("/v1/health")).json()
        assert health["status"] == "ok"
        assert not health["model_ready"] and not health["gpu_available"]
        assert (await client.get("/v1/gpu")).json()["vram_used_mb"] is None
        assert (await client.get("/v1/models")).json()["active"] is None
        response = await client.post("/v1/generations", json={"prompt": "hello"})
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        identifier = response.json()["id"]
        await wait_terminal(app, identifier)
        result = (await client.get(f"/v1/generations/{identifier}")).json()
        assert result["status"] == "failed"
        assert result["error"]["code"] == "INFERENCE_UNAVAILABLE"
        assert not list(app.state.storage.files("tmp"))


@pytest.mark.parametrize("key", ["", "secret-key"])
async def test_auth(app_factory, key):
    async with app_factory(api_key=key) as (_, client):
        assert (await client.get("/v1/health")).status_code == 200
        for route in (
            "/v1/gpu",
            "/v1/models",
            "/v1/queue",
            "/v1/generations/missing",
            "/v1/assets/missing",
            "/v1/generations/missing/content",
        ):
            response = await client.get(route)
            if key:
                assert response.status_code == 401
                assert response.headers["www-authenticate"] == "Bearer"
                assert response.json()["error"]["code"] == "UNAUTHORIZED"
            else:
                assert response.status_code != 401
        assert (
            await client.get(
                "/v1/queue",
                headers={
                    "Authorization": "Bearer secret-key",
                },
            )
        ).status_code == 200
        if key:
            for method, url in (
                ("POST", "/v1/assets"),
                ("POST", "/v1/generations"),
                ("DELETE", "/v1/assets/missing"),
            ):
                response = await client.request(method, url, headers={"Authorization": "Basic x"})
                assert response.status_code == 401


async def test_private_health_and_disabled_docs(app_factory):
    async with app_factory(api_key="secret", health_requires_auth=True, enable_docs=False) as (
        _,
        client,
    ):
        assert (await client.get("/v1/health")).status_code == 401
        assert (
            await client.get(
                "/v1/health",
                headers={
                    "Authorization": "bEaReR secret",
                },
            )
        ).status_code == 200
        for route in ("/docs", "/openapi.json", "/redoc"):
            assert (await client.get(route)).status_code == 404


async def test_openapi_and_machine_readable_validation(app_factory):
    async with app_factory() as (_, client):
        schema = (await client.get("/openapi.json")).json()
        assert (
            "multipart/form-data" in schema["paths"]["/v1/assets"]["post"]["requestBody"]["content"]
        )
        assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
        response = await client.post("/v1/generations", json={"prompt": "secret", "fps": "24"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_INPUT"
        assert "secret" not in response.text


async def test_fifo_queue_full_cancel_and_shutdown(app_factory):
    backend = ControlledBackend(blocked=True)
    async with app_factory(backend, max_queue_size=1) as (app, client):
        first = (await client.post("/v1/generations", json={"prompt": "a"})).json()["id"]
        await backend.started.wait()
        second = (await client.post("/v1/generations", json={"prompt": "b"})).json()["id"]
        full = await client.post("/v1/generations", json={"prompt": "c"})
        assert full.status_code == 429
        assert full.json()["error"]["code"] == "QUEUE_FULL"
        queue = (await client.get("/v1/queue")).json()
        assert queue["running"] == first and queue["queued"] == [second]
        assert (await client.delete(f"/v1/generations/{second}")).status_code == 204
        assert app.state.jobs.get(second).info.status == JobStatus.CANCELLED
        third = (await client.post("/v1/generations", json={"prompt": "c"})).json()["id"]
        assert (await client.delete(f"/v1/generations/{first}")).status_code == 204
        assert app.state.jobs.get(first).info.cancellation_requested
        await wait_terminal(app, first)
        assert app.state.jobs.get(first).info.status == JobStatus.CANCELLED
        backend.release.set()
        await wait_terminal(app, third)
        assert backend.order == [first, third]
        assert backend.peak == 1
        for _ in range(2):
            assert (await client.delete(f"/v1/generations/{third}")).status_code == 204
        assert (await client.delete("/v1/generations/nonexistent")).status_code == 204
    assert backend.closed


async def test_worker_recovers_from_failure(app_factory):
    backend = ControlledBackend(fail_first=True)
    async with app_factory(backend) as (app, client):
        failed = (await client.post("/v1/generations", json={"prompt": "a"})).json()["id"]
        job = await wait_terminal(app, failed)
        assert job.info.error.code == "GENERATION_FAILED"
        result = await client.get(f"/v1/generations/{failed}")
        assert "/secret" not in result.text
        good = (await client.post("/v1/generations", json={"prompt": "b"})).json()["id"]
        assert (await wait_terminal(app, good)).info.status == JobStatus.COMPLETE


async def test_cancellation_wins_over_backend_error(app_factory):
    backend = ControlledBackend(blocked=True, fail_first=True)
    async with app_factory(backend) as (app, client):
        identifier = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        async with asyncio.timeout(3):
            await backend.started.wait()
        await client.delete(f"/v1/generations/{identifier}")
        job = await wait_terminal(app, identifier)
        assert job.info.status == JobStatus.CANCELLED
        assert job.info.error.code == "GENERATION_CANCELLED"
        assert not list(app.state.storage.files("outputs"))


async def test_content_ranges_grace_expiration(app_factory):
    async with app_factory(ControlledBackend()) as (app, client):
        identifier = (await client.post("/v1/generations", json={"prompt": "a"})).json()["id"]
        job = await wait_terminal(app, identifier)
        original_expiration = job.info.expires_at
        url = f"/v1/generations/{identifier}/content"
        response = await client.get(url, headers={"Range": "bytes=0-9"})
        assert response.status_code == 206 and response.content == b"0123456789"
        assert response.headers["content-range"] == "bytes 0-9/1000"
        assert job.downloaded_at is None
        assert (await client.get(url, headers={"Range": "bytes=9999-"})).status_code == 416
        assert job.readers == 0
        response = await client.get(url)
        assert response.status_code == 200 and len(response.content) == 1000
        assert job.info.expires_at < original_expiration
        assert job.info.expires_at == job.downloaded_at + timedelta(seconds=300)
        expires = job.info.expires_at
        await client.get(url)
        assert job.info.expires_at == expires
        app.state.cleanup.sweep(expires + timedelta(seconds=1))
        assert (await client.get(url)).status_code == 410
        assert not list(app.state.storage.files("outputs"))


async def test_assets_and_leases(app_factory, png):
    backend = ControlledBackend(blocked=True)
    async with app_factory(backend) as (app, client):
        upload = await client.post(
            "/v1/assets", files={"file": ("../../escape.jpg", png, "audio/mp3")}
        )
        assert upload.status_code == 201
        info = upload.json()
        assert info["type"] == "image" and info["mime_type"] == "image/png"
        assert "path" not in info
        identifier = info["id"]
        generation = (
            await client.post(
                "/v1/generations",
                json={
                    "prompt": "animate",
                    "first_frame": identifier,
                },
            )
        ).json()["id"]
        await backend.started.wait()
        assert (await client.delete(f"/v1/assets/{identifier}")).status_code == 409
        asset = app.state.assets.records[identifier]
        asset.info.expires_at = utcnow() - timedelta(seconds=1)
        app.state.cleanup.sweep()
        assert identifier in app.state.assets.records
        assert (
            await client.post(
                "/v1/generations",
                json={
                    "prompt": "animate",
                    "first_frame": identifier,
                },
            )
        ).status_code == 410
        await client.delete(f"/v1/generations/{generation}")
        await wait_terminal(app, generation)
        app.state.cleanup.sweep()
        assert identifier not in app.state.assets.records
        assert (await client.delete(f"/v1/assets/{identifier}")).status_code == 204


async def test_missing_wrong_type_and_partial_lease_rollback(app_factory, png):
    async with app_factory() as (app, client):
        identifier = (await client.post("/v1/assets", files={"file": ("x", png)})).json()["id"]
        for body, code in [
            ({"first_frame": "asset_" + "f" * 32}, 404),
            ({"audio": identifier}, 422),
            ({"first_frame": identifier, "last_frame": "asset_" + "f" * 32}, 404),
        ]:
            response = await client.post("/v1/generations", json={"prompt": "x", **body})
            assert response.status_code == code
        assert not app.state.assets.records[identifier].leases
        assert (await client.delete(f"/v1/assets/{identifier}")).status_code == 204


async def test_shutdown_cancels_running_and_waiting_jobs(app_factory):
    backend = ControlledBackend(blocked=True)
    async with app_factory(backend) as (app, client):
        first = (await client.post("/v1/generations", json={"prompt": "a"})).json()["id"]
        await backend.started.wait()
        second = (await client.post("/v1/generations", json={"prompt": "b"})).json()["id"]
    assert app.state.jobs.get(first).info.status == JobStatus.CANCELLED
    assert app.state.jobs.get(second).info.status == JobStatus.CANCELLED
    assert backend.active == 0 and backend.closed
    assert not list(app.state.storage.files("tmp"))


async def test_simultaneous_admission_does_not_overfill(app_factory):
    backend = ControlledBackend(blocked=True)
    async with app_factory(backend, max_queue_size=2) as (app, client):
        await client.post("/v1/generations", json={"prompt": "running"})
        await backend.started.wait()
        responses = await asyncio.gather(
            *(client.post("/v1/generations", json={"prompt": "queued"}) for _ in range(10))
        )
        assert sum(r.status_code == 202 for r in responses) == 2
        assert app.state.jobs.queue_info().length == 2


async def test_portrait_and_retake_capabilities(app_factory):
    from .conftest import ControlledBackend, wait_terminal

    async with app_factory(ControlledBackend()) as (app, client):
        catalog = (await client.get("/v1/models")).json()
        assert catalog["models"][0]["capabilities"]["retake_normalize_source"]
        assert catalog["orientations"] == ["landscape", "portrait"]
        assert catalog["retake_normalization_fps"] == 24
        assert catalog["resolution_presets"]["540p"] == [1024, 576]
        response = await client.post(
            "/v1/generations",
            json={
                "prompt": "portrait",
                "orientation": "portrait",
                "duration": 1,
            },
        )
        assert response.status_code == 202
        record = await wait_terminal(app, response.json()["id"])
        output = (await client.get(f"/v1/generations/{record.info.id}")).json()["output"]
        assert (output["width"], output["height"], output["frames"]) == (576, 1024, 25)
        assert output["duration"] == 25 / 24


@pytest.mark.parametrize("audio", [False, True])
async def test_uploaded_video_reports_audio(app_factory, tmp_path, audio):
    from .test_video_inputs import make_clip

    source = tmp_path / "input.mp4"
    make_clip(source, audio=audio)
    async with app_factory() as (_, client):
        result = await client.post("/v1/assets", files={"file": ("input.mp4", source.read_bytes())})
        assert result.status_code == 201
        assert result.json()["has_audio"] is audio
