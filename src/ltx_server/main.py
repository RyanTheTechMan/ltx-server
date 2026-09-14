import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ltx_server import __version__
from ltx_server.api import assets, generations, gpu, health, models, queue
from ltx_server.api.auth import require_auth
from ltx_server.api.middleware import BODY_TOO_LARGE, UploadLimitMiddleware
from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ErrorDetail, ErrorResponse, ServiceError
from ltx_server.inference.backend import GenerationBackend
from ltx_server.inference.manager import create_backend
from ltx_server.jobs.cleanup import CleanupWorker
from ltx_server.jobs.manager import JobManager
from ltx_server.logging import configure_logging
from ltx_server.media.assets import AssetManager
from ltx_server.media.storage import Storage

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    backend: GenerationBackend | None = None,
    gpu_reader: Callable[[int], gpu.GPUInfo] = gpu.read_gpu,
) -> FastAPI:
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        storage = Storage(settings)
        storage.open()
        asset_manager = AssetManager(settings, storage)
        active_backend = backend if backend is not None else create_backend(settings)
        job_manager = JobManager(settings, storage, asset_manager, active_backend)
        app.state.backend = active_backend
        cleanup = CleanupWorker(settings, storage, asset_manager, job_manager)
        app.state.storage, app.state.assets = storage, asset_manager
        app.state.jobs, app.state.cleanup = job_manager, cleanup
        app.state.gpu_lock = asyncio.Lock()
        try:
            cleanup.sweep()
            await job_manager.start()
            cleanup.start()
            logger.info("server_started")
            yield
        finally:
            try:
                await job_manager.stop()
            finally:
                await cleanup.stop()
                storage.close()
                logger.info("server_stopped")

    app = FastAPI(
        title="ltx-server",
        version=__version__,
        lifespan=lifespan,
        description="Headless LTX 2.5 API: asynchronous text/image-to-video with generated audio.",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings = settings
    app.state.gpu_reader = gpu_reader
    app.add_middleware(UploadLimitMiddleware, limit=settings.upload_body_limit)
    if settings.cors_origins.strip():
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
            ],
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "Range"],
            expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"],
        )

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content=ErrorResponse(error=exc.detail).model_dump(),
            headers={"WWW-Authenticate": "Bearer"} if exc.status == 401 else None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Never reflect raw input, passwords, prompts or filesystem paths in errors.
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.INVALID_INPUT,
                    message="Invalid request; see the schema at /docs",
                )
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        if exc.detail == BODY_TOO_LARGE:
            return await service_error(
                request,
                ServiceError(
                    ErrorCode.UPLOAD_TOO_LARGE,
                    BODY_TOO_LARGE,
                    413,
                ),
            )
        return await service_error(
            request,
            ServiceError(
                ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_INPUT,
                "Endpoint not found" if exc.status_code == 404 else "Invalid HTTP request",
                exc.status_code,
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("request_failed")
        return await service_error(
            request,
            ServiceError(
                ErrorCode.INTERNAL_ERROR,
                "Internal server error",
                500,
            ),
        )

    router = APIRouter(
        prefix="/v1",
        dependencies=[Depends(require_auth)],
        responses={
            status: {"model": ErrorResponse}
            for status in (400, 401, 404, 409, 410, 413, 415, 422, 429, 503)
        },
    )
    for module in (health, gpu, models, queue, assets, generations):
        router.include_router(module.router)
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=1,
        access_log=False,
    )
