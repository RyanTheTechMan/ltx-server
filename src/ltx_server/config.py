from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
        allow_inf_nan=False,
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enable_docs: bool = True
    api_key: SecretStr = SecretStr("")
    health_requires_auth: bool = False
    data_dir: Path = Path("./data")
    model_dir: Path | None = None
    output_dir: Path | None = None
    temp_dir: Path | None = None
    asset_dir: Path | None = None
    output_ttl_seconds: int = Field(default=1800, gt=0)
    download_grace_seconds: int = Field(default=300, gt=0)
    asset_ttl_seconds: int = Field(default=3600, gt=0)
    temp_ttl_seconds: int = Field(default=3600, gt=0)
    job_record_ttl_seconds: int = Field(default=86400, gt=0)
    cleanup_interval_seconds: float = Field(default=60, gt=0)
    delete_after_download: bool = True
    gpu_index: int = Field(default=0, ge=0)
    max_concurrent_generations: Literal[1] = 1
    max_queue_size: int = Field(default=20, ge=1, le=10000)
    default_resolution: Literal["540p", "720p", "1080p"] = "540p"
    default_duration_seconds: float = Field(default=10, ge=1, le=20, allow_inf_nan=False)
    default_fps: Literal[24, 25, 30] = 24
    default_generate_audio: bool = True
    max_duration_seconds: float = Field(default=20, ge=1, le=20, allow_inf_nan=False)
    max_resolution: Literal["540p", "720p", "1080p"] = "1080p"
    max_image_upload_mb: int = Field(default=50, gt=0)
    max_audio_upload_mb: int = Field(default=100, gt=0)
    max_video_upload_mb: int = Field(default=500, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    media_probe_timeout_seconds: float = Field(default=30, gt=0)
    cors_origins: str = ""
    inference_backend: Literal["auto", "ltx", "disabled"] = "auto"
    ltx_model: Literal["ltx-2.5-fast"] = "ltx-2.5-fast"
    gpu_device: str = Field(default="cuda:0", pattern=r"^cuda:[0-9]+$")
    warm_model_on_start: bool = True
    use_fp8: bool = True
    # Full weights survive between jobs on host RAM; no disk reload/cast each request.
    cache_text_encoder: bool = True
    lora_manifest: Path | None = None
    default_ic_lora: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    offload_mode: Literal["none", "cpu", "disk"] = "none"
    attention_backend: Literal["automatic", "pytorch", "sage"] = "automatic"
    compile_transformer: bool = False
    vae_tile_size: int = Field(default=512, ge=128, le=2048, multiple_of=32)
    vae_tile_overlap: int = Field(default=64, ge=0, multiple_of=32)
    vae_temporal_size: int = Field(default=80, ge=16, le=256, multiple_of=8)
    vae_temporal_overlap: int = Field(default=24, ge=0, multiple_of=8)
    synchronize_timings: bool = False
    hf_token: SecretStr = SecretStr("")
    ltx_transformer_path: Path | None = None
    ltx_text_encoder_path: Path | None = None
    ltx_video_vae_path: Path | None = None
    ltx_audio_vae_path: Path | None = None
    ltx_spatial_upsampler_path: Path | None = None
    video_crf: int = Field(default=19, ge=0, le=51)
    video_preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"] = (
        "veryfast"
    )
    ffmpeg_stall_timeout_seconds: float = Field(default=60, gt=0)

    @field_validator("default_fps", "max_concurrent_generations", mode="before")
    @classmethod
    def parse_integer_literal(cls, value: object) -> object:
        # Environment values are strings; Pydantic's integer Literal does not coerce them.
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.vae_tile_overlap >= self.vae_tile_size:
            raise ValueError("VAE_TILE_OVERLAP must be smaller than VAE_TILE_SIZE")
        if self.vae_temporal_overlap >= self.vae_temporal_size:
            raise ValueError("VAE_TEMPORAL_OVERLAP must be smaller than VAE_TEMPORAL_SIZE")
        if self.default_duration_seconds > self.max_duration_seconds:
            raise ValueError("DEFAULT_DURATION_SECONDS exceeds MAX_DURATION_SECONDS")
        order = ("540p", "720p", "1080p")
        if order.index(self.default_resolution) > order.index(self.max_resolution):
            raise ValueError("DEFAULT_RESOLUTION exceeds MAX_RESOLUTION")
        self.data_dir = self.data_dir.expanduser().absolute()
        for field, suffix in (
            ("model_dir", "models"),
            ("output_dir", "outputs"),
            ("temp_dir", "tmp"),
            ("asset_dir", "assets"),
        ):
            value = getattr(self, field) or self.data_dir / suffix
            setattr(self, field, value.expanduser().absolute())
        paths = [self.model_dir, self.output_dir, self.temp_dir, self.asset_dir]
        for i, path in enumerate(paths):
            assert path is not None
            if ".." in path.parts:
                raise ValueError("Storage paths must not contain parent traversal components")
            for other in paths[i + 1 :]:
                assert other is not None
                if path.is_relative_to(other) or other.is_relative_to(path):
                    raise ValueError("Model, output, temp and asset directories must not overlap")
        return self

    @property
    def upload_body_limit(self) -> int:
        # Bound multipart parsing, including chunked bodies, before spooling to disk.
        return (
            max(self.max_image_upload_mb, self.max_audio_upload_mb, self.max_video_upload_mb)
            * 1024
            * 1024
            + 64 * 1024
        )
