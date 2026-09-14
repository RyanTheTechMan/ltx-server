from typing import Annotated

from fastapi import APIRouter, Depends, Response
from starlette.responses import FileResponse
from starlette.types import Message, Receive, Scope, Send

from ltx_server.api.dependencies import jobs
from ltx_server.jobs.manager import JobManager
from ltx_server.jobs.state import JobInfo
from ltx_server.schemas.generation import GenerationRequest

router = APIRouter()


class OutputResponse(FileResponse):
    def __init__(self, manager: JobManager, identifier: str) -> None:
        self.manager = manager
        self.job, path = manager.begin_download(identifier)
        super().__init__(path, media_type="video/mp4", filename=f"{identifier}.mp4")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        status = 0
        complete = False

        async def track(message: Message) -> None:
            nonlocal status, complete
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True

        # Observe every byte: disable pathsend so a handed-off path is not counted as a download.
        scope = {
            **scope,
            "extensions": {
                key: value
                for key, value in scope.get("extensions", {}).items()
                if key != "http.response.pathsend"
            },
        }
        try:
            await super().__call__(scope, receive, track)
        finally:
            self.manager.end_download(
                self.job,
                successful=complete and status == 200 and scope["method"] == "GET",
            )


@router.post("/generations", response_model=JobInfo, status_code=202, tags=["generations"])
async def create_generation(
    body: GenerationRequest, manager: Annotated[JobManager, Depends(jobs)]
) -> JobInfo:
    """Enqueue text, image/keyframe or audio-conditioned generation; poll for the MP4."""
    return manager.submit(body)


@router.get("/generations/{generation_id}", response_model=JobInfo, tags=["generations"])
async def get_generation(
    generation_id: str, manager: Annotated[JobManager, Depends(jobs)]
) -> JobInfo:
    """Read generation state, stage progress and eventual output metadata."""
    return manager.get(generation_id).info


@router.get(
    "/generations/{generation_id}/content",
    response_class=FileResponse,
    responses={200: {"content": {"video/mp4": {}}}, 206: {"description": "Requested byte range"}},
    tags=["generations"],
)
async def download_generation(
    generation_id: str, manager: Annotated[JobManager, Depends(jobs)]
) -> FileResponse:
    """Stream a completed MP4 with byte ranges. Full successful GET starts download grace."""
    return OutputResponse(manager, generation_id)


@router.delete("/generations/{generation_id}", status_code=204, tags=["generations"])
async def delete_generation(
    generation_id: str, manager: Annotated[JobManager, Depends(jobs)]
) -> Response:
    """Idempotent cancellation/deletion. Active jobs retain resources until the backend stops."""
    manager.delete(generation_id)
    return Response(status_code=204)
