import asyncio
import os
from datetime import timedelta

import pytest

from ltx_server.api.generations import OutputResponse
from ltx_server.config import Settings
from ltx_server.jobs.state import JobInfo, JobStatus, utcnow
from ltx_server.media.storage import Storage, new_id

from .conftest import ControlledBackend, wait_terminal


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", "gen_abc.mp4", "asset_a/../b.bin"])
def test_managed_paths_reject_traversal(tmp_path, name):
    storage = Storage(Settings(_env_file=None, data_dir=tmp_path))
    storage.open()
    try:
        with pytest.raises(ValueError):
            storage.path("outputs", name)
    finally:
        storage.close()


def test_storage_locks_symlinks_and_atomic_publish(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    storage = Storage(settings)
    storage.open()
    second = Storage(settings)
    try:
        with pytest.raises(RuntimeError, match="one server"):
            second.open()
        identifier = new_id("gen")
        partial = storage.create_partial(identifier)
        with pytest.raises(ValueError, match="empty"):
            storage.publish(identifier, "outputs")
        partial.write_bytes(b"completed output")
        output = storage.publish(identifier, "outputs")
        assert output.read_bytes() == b"completed output" and not partial.exists()
        outside = tmp_path / "keep.txt"
        outside.write_text("keep")
        linked = settings.asset_dir / f"{new_id('asset')}.bin"
        linked.symlink_to(outside)
        with pytest.raises(ValueError, match="Symlinks"):
            storage.path("assets", linked.name)
        assert linked not in list(storage.files("assets"))
    finally:
        storage.close()
    second.open()
    second.close()


def test_symlinked_directory_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    storage = Storage(Settings(_env_file=None, data_dir=link))
    with pytest.raises(ValueError, match="symlinks"):
        storage.open()


def test_state_transitions():
    job = JobInfo(id="gen_x", seed=1, width=1024, height=576, frames=121, fps=24)
    for status, progress in [
        (JobStatus.LOADING, 5),
        (JobStatus.GENERATING, 15),
        (JobStatus.ENCODING_OUTPUT, 95),
        (JobStatus.COMPLETE, 100),
    ]:
        job.transition(status)
        assert job.progress == progress
    assert job.started_at and job.completed_at
    with pytest.raises(ValueError):
        job.transition(JobStatus.GENERATING)
    job.transition(JobStatus.EXPIRED)
    with pytest.raises(ValueError):
        job.transition(JobStatus.COMPLETE)


async def test_cleanup_orphans_active_jobs_and_record_expiry(app_factory):
    backend = ControlledBackend(blocked=True)
    async with app_factory(backend) as (app, client):
        storage = app.state.storage
        now = utcnow()
        old = (now - timedelta(days=2)).timestamp()
        for area, prefix, suffix in [
            ("tmp", "gen", "partial"),
            ("outputs", "gen", "mp4"),
            ("assets", "asset", "bin"),
        ]:
            path = storage.path(area, f"{new_id(prefix)}.{suffix}")
            path.write_bytes(b"orphan")
            os.utime(path, (old, old))
        unknown = storage.roots["tmp"] / "do-not-delete.txt"
        unknown.write_text("user file")
        os.utime(unknown, (old, old))
        fresh = storage.create_partial(new_id("gen"))
        identifier = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        await backend.started.wait()
        active = storage.path("tmp", f"{identifier}.partial")
        os.utime(active, (old, old))
        app.state.cleanup.sweep(now)
        assert fresh.exists() and active.exists() and unknown.exists()
        assert not list(storage.files("outputs")) and not list(storage.files("assets"))
        await client.delete(f"/v1/generations/{identifier}")
        job = await wait_terminal(app, identifier)
        assert not active.exists()
        app.state.cleanup.sweep(job.info.completed_at + timedelta(days=2))
        assert (await client.get(f"/v1/generations/{identifier}")).status_code == 404


async def test_startup_cleanup(tmp_path):
    from ltx_server.main import create_app

    settings = Settings(_env_file=None, data_dir=tmp_path)
    storage = Storage(settings)
    storage.open()
    old_files = []
    old = (utcnow() - timedelta(days=2)).timestamp()
    for area, prefix, suffix in [
        ("tmp", "gen", "partial"),
        ("outputs", "gen", "mp4"),
        ("assets", "asset", "bin"),
    ]:
        path = storage.path(area, f"{new_id(prefix)}.{suffix}")
        path.write_bytes(b"orphan")
        os.utime(path, (old, old))
        old_files.append(path)
    storage.close()
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        assert not any(path.exists() for path in old_files)


async def test_active_download_pins_output_and_delete_is_deferred(app_factory):
    async with app_factory(ControlledBackend()) as (app, client):
        identifier = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        job = await wait_terminal(app, identifier)
        response = OutputResponse(app.state.jobs, identifier)
        path = app.state.storage.path("outputs", f"{identifier}.mp4")
        app.state.cleanup.sweep(job.info.expires_at + timedelta(seconds=1))
        assert path.exists() and job.readers == 1
        app.state.jobs.delete(identifier)
        assert path.exists() and job.delete_pending
        messages = []

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            messages.append(message)

        await response(
            {"type": "http", "method": "GET", "headers": [], "extensions": {}}, receive, send
        )
        assert job.readers == 0 and not path.exists()
        assert any(message["type"] == "http.response.body" for message in messages)


async def test_failed_download_does_not_start_grace(app_factory):
    async with app_factory(ControlledBackend()) as (app, client):
        identifier = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        job = await wait_terminal(app, identifier)
        response = OutputResponse(app.state.jobs, identifier)

        async def receive():
            return {"type": "http.request"}

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("Client disconnected")

        with pytest.raises(OSError):
            await response(
                {"type": "http", "method": "GET", "headers": [], "extensions": {}}, receive, send
            )
        assert job.readers == 0 and job.downloaded_at is None


async def test_delete_after_download_can_be_disabled(app_factory):
    async with app_factory(ControlledBackend(), delete_after_download=False) as (app, client):
        identifier = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        job = await wait_terminal(app, identifier)
        expiry = job.info.expires_at
        await client.get(f"/v1/generations/{identifier}/content")
        assert job.info.expires_at == expiry


async def test_periodic_cleanup_runs(app_factory):
    async with app_factory(cleanup_interval_seconds=0.01) as (app, _):
        path = app.state.storage.create_partial(new_id("gen"))
        os.utime(path, (0, 0))
        async with asyncio.timeout(2):
            while path.exists():
                await asyncio.sleep(0.001)


async def test_prepared_retake_scratch_is_protected_and_reaped(app_factory):
    backend = ControlledBackend(blocked=True)
    async with app_factory(backend) as (app, client):
        identifier = (await client.post("/v1/generations", json={"prompt": "x"})).json()["id"]
        await backend.started.wait()
        scratch = app.state.storage.path("tmp", f"{identifier}.source.partial")
        scratch.write_bytes(b"prepared source")
        os.utime(scratch, (0, 0))
        app.state.cleanup.sweep()
        assert scratch.exists()
        backend.release.set()
        await wait_terminal(app, identifier)
        app.state.cleanup.sweep()
        assert not scratch.exists()
        orphan = app.state.storage.path("tmp", f"{new_id('gen')}.source.partial")
        orphan.write_bytes(b"abandoned source")
        os.utime(orphan, (0, 0))
        app.state.cleanup.sweep()
        assert not orphan.exists()
