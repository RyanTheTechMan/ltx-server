import asyncio

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ltx_server.inference.models import LTX_COMMIT, MODEL_REVISION, ModelInventory
from ltx_server.schemas.generation import RESOLUTIONS

router = APIRouter()


class ModelCapabilities(BaseModel):
    text_to_video: bool = False
    image_to_video: bool = False
    audio_to_video: bool = False
    first_last_frame: bool = False
    multiple_keyframes: bool = False
    generated_audio: bool = False
    reference_video: bool = False
    loras: bool = False
    retake: bool = False
    retake_normalize_source: bool = False


class ModelInfo(BaseModel):
    id: str
    installed: bool
    capabilities: ModelCapabilities
    missing_components: list[str] = Field(default_factory=list)
    upstream_commit: str = LTX_COMMIT
    checkpoint_revision: str = MODEL_REVISION


class ModelCatalog(BaseModel):
    active: str | None = None
    models: list[ModelInfo] = Field(default_factory=list)
    inference_available: bool = False
    resolution_presets: dict[str, tuple[int, int]]
    orientations: list[str] = Field(default_factory=lambda: ["landscape", "portrait"])
    retake_normalization_fps: int = 24
    registered_loras: list[dict[str, str | bool]] = Field(default_factory=list)
    default_ic_lora: str | None = None
    note: str = (
        "Advanced pipelines implemented; LoRAs require operator registration. "
        "GPU benchmarks pending."
    )


@router.get("/models", response_model=ModelCatalog, tags=["diagnostics"])
async def models(request: Request) -> ModelCatalog:
    """Report configured checkpoint inventory and implemented capabilities, without paths."""
    settings = request.app.state.settings
    status = request.app.state.backend.status()
    inventory = ModelInventory(settings)
    missing = await asyncio.to_thread(inventory.missing)
    return ModelCatalog(
        active=settings.ltx_model if status.model_ready else None,
        models=[
            ModelInfo(
                id=settings.ltx_model,
                installed=not missing,
                missing_components=missing,
                capabilities=ModelCapabilities(
                    text_to_video=True,
                    image_to_video=True,
                    generated_audio=True,
                    first_last_frame=True,
                    multiple_keyframes=True,
                    audio_to_video=True,
                    reference_video=True,
                    loras=True,
                    retake=True,
                    retake_normalize_source=True,
                ),
            )
        ],
        inference_available=status.model_ready,
        registered_loras=inventory.loras.public(),
        default_ic_lora=settings.default_ic_lora,
        resolution_presets={str(key): value for key, value in RESOLUTIONS.items()},
    )
