from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from starlette.datastructures import UploadFile

from ltx_server.api.dependencies import assets
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.media.assets import AssetManager
from ltx_server.schemas.assets import AssetInfo

router = APIRouter()


@router.post(
    "/assets",
    response_model=AssetInfo,
    status_code=201,
    tags=["assets"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                },
            },
        }
    },
)
async def upload_asset(
    request: Request, manager: Annotated[AssetManager, Depends(assets)]
) -> AssetInfo:
    """Upload one image, audio or video. Content is validated; filenames/MIME are untrusted."""
    # Parse after authentication, with one file and no extra fields. The outer body
    # limiter bounds spool usage even when Content-Length is absent or incorrect.
    async with request.form(max_files=1, max_fields=0, max_part_size=64 * 1024) as form:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise ServiceError(ErrorCode.INVALID_INPUT, "A multipart file field is required", 422)
        return await manager.upload(file)


@router.get("/assets/{asset_id}", response_model=AssetInfo, tags=["assets"])
async def get_asset(asset_id: str, manager: Annotated[AssetManager, Depends(assets)]) -> AssetInfo:
    """Read asset metadata without exposing a filesystem path."""
    return manager.get(asset_id).info


@router.delete("/assets/{asset_id}", status_code=204, tags=["assets"])
async def delete_asset(
    asset_id: str, manager: Annotated[AssetManager, Depends(assets)]
) -> Response:
    """Idempotently delete an asset; reject while queued/running jobs hold it."""
    manager.delete(asset_id)
    return Response(status_code=204)
