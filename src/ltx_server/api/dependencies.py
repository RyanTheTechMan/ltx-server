from fastapi import Request

from ltx_server.jobs.manager import JobManager
from ltx_server.media.assets import AssetManager


def jobs(request: Request) -> JobManager:
    return request.app.state.jobs  # type: ignore[no-any-return]


def assets(request: Request) -> AssetManager:
    return request.app.state.assets  # type: ignore[no-any-return]
