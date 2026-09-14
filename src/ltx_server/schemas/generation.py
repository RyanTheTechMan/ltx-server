import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError

Resolution = Literal["540p", "720p", "1080p"]
Orientation = Literal["landscape", "portrait"]
AssetId = Annotated[str, Field(pattern=r"^asset_[0-9a-f]{32}$")]
# Official distilled two-stage pipeline requires both dimensions divisible by 64.
# Desktop uses the same grid. See docs/ltx-desktop-reference.md for pinned sources.
RESOLUTIONS: dict[Resolution, tuple[int, int]] = {
    "540p": (1024, 576),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
}


def frame_count(duration: float, fps: int) -> int:
    """Nearest 8k+1 frame count, ties upward; require at least one temporal block."""
    return max(1, math.floor((duration * fps - 1) / 8 + 0.5)) * 8 + 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Keyframe(StrictModel):
    asset_id: AssetId
    frame: int = Field(ge=0)
    strength: float = Field(default=1.0, ge=0, le=1)


class Lora(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    scale: float = Field(default=1.0, ge=0, le=2)


class Retake(StrictModel):
    video: AssetId
    start: float = Field(ge=0, lt=20)
    end: float = Field(gt=0, le=21)
    regenerate_video: bool = True
    regenerate_audio: bool = True
    normalize_source: bool = False

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.start >= self.end:
            raise ValueError("Retake start must precede end")
        if not self.regenerate_video and not self.regenerate_audio:
            raise ValueError("Retake must regenerate video or audio")
        return self


class GenerationRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=10000)
    duration: float | None = Field(default=None, ge=1, le=20)
    resolution: Resolution | None = None
    orientation: Orientation = "landscape"
    fps: Literal[24, 25, 30] | None = None
    generate_audio: bool | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    first_frame: AssetId | None = None
    last_frame: AssetId | None = None
    audio: AssetId | None = None
    reference_video: AssetId | None = None
    reference_lora: Lora | None = None
    reference_strength: float = Field(default=1.0, ge=0, le=1)
    retake: Retake | None = None
    keyframes: list[Keyframe] = Field(default_factory=list, max_length=32)
    loras: list[Lora] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_conditioning(self) -> Self:
        if len({lora.id for lora in self.loras}) != len(self.loras):
            raise ValueError("LoRA IDs must be unique")
        if self.reference_lora and not self.reference_video:
            raise ValueError("reference_lora requires reference_video")
        if self.reference_strength != 1.0 and not self.reference_video:
            raise ValueError("reference_strength requires reference_video")
        if self.reference_video and self.audio:
            raise ValueError("Reference video and source audio cannot be combined")
        if self.retake and (
            self.reference_video
            or self.audio
            or self.first_frame
            or self.last_frame
            or self.keyframes
        ):
            raise ValueError("Retake cannot be combined with other conditioning")
        if self.retake and self.retake.normalize_source and self.fps not in (None, 24):
            raise ValueError("Normalized retakes require 24 FPS")
        if not self.prompt.strip():
            raise ValueError("Prompt must contain non-whitespace characters")
        if self.last_frame and not self.first_frame:
            raise ValueError("last_frame requires first_frame")
        if self.keyframes and (self.first_frame or self.last_frame):
            raise ValueError("Use keyframes or first_frame/last_frame, not both")
        frames = [key.frame for key in self.keyframes]
        if len(frames) != len(set(frames)):
            raise ValueError("Keyframe positions must be unique")
        return self


class GenerationSpec(GenerationRequest):
    duration: float = Field(ge=1, le=20)
    resolution: Resolution
    fps: Literal[24, 25, 30]
    generate_audio: bool
    seed: int = Field(ge=0, le=2**32 - 1)
    width: int
    height: int
    frames: int

    def assets(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for asset in (self.first_frame, self.last_frame):
            if asset:
                result[asset] = "image"
        for key in self.keyframes:
            result[key.asset_id] = "image"
        for asset, kind in (
            (self.audio, "audio"),
            (self.reference_video, "video"),
            (self.retake.video if self.retake else None, "video"),
        ):
            if asset:
                if asset in result and result[asset] != kind:
                    raise ServiceError(ErrorCode.INVALID_INPUT, "Asset used with conflicting types")
                result[asset] = kind
        return result


def normalize_request(request: GenerationRequest, settings: Settings, seed: int) -> GenerationSpec:
    duration = (
        request.duration if request.duration is not None else settings.default_duration_seconds
    )
    resolution = request.resolution or settings.default_resolution
    fps = request.fps or settings.default_fps
    if request.retake and request.retake.normalize_source:
        fps = 24
    width, height = RESOLUTIONS[resolution]
    if request.orientation == "portrait":
        width, height = height, width
    if duration > settings.max_duration_seconds:
        raise ServiceError(ErrorCode.INVALID_DURATION, "Duration exceeds the server maximum", 422)
    if list(RESOLUTIONS).index(resolution) > list(RESOLUTIONS).index(settings.max_resolution):
        raise ServiceError(
            ErrorCode.INVALID_RESOLUTION, "Resolution exceeds the server maximum", 422
        )
    frames = frame_count(duration, fps)
    if request.retake and request.retake.end > frames / fps:
        raise ServiceError(ErrorCode.INVALID_INPUT, "Retake window extends beyond the clip", 422)
    if any(key.frame >= frames for key in request.keyframes):
        raise ServiceError(ErrorCode.INVALID_INPUT, "Keyframe lies outside the generated clip", 422)
    values = request.model_dump()
    values.update(
        duration=duration,
        resolution=resolution,
        fps=fps,
        generate_audio=(
            request.generate_audio
            if request.generate_audio is not None
            else settings.default_generate_audio
        ),
        seed=request.seed if request.seed is not None else seed,
        width=width,
        height=height,
        frames=frames,
    )
    return GenerationSpec.model_validate(values)
