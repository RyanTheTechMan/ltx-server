"""Pinned checkpoint inventory; no torch imports, downloads or client paths."""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError

LTX_COMMIT = "a95ab856bf29407b6b066ede0abe1846050db56c"
MODEL_REPO = "Lightricks/LTX-2.5"
MODEL_REVISION = "5e6e71018ee1756ed329b697a7b4aedc934dfce9"


@dataclass(frozen=True)
class Checkpoint:
    component: str
    filename: str
    size: int
    sha256: str


CHECKPOINTS = (
    Checkpoint(
        "transformer",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        42018190584,
        "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
    ),
    Checkpoint(
        "text_encoder",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        26263858182,
        "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
    ),
    Checkpoint(
        "video_vae",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        1452269922,
        "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d",
    ),
    Checkpoint(
        "audio_vae",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        364866540,
        "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
    ),
    Checkpoint(
        "spatial_upsampler",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        995778752,
        "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
    ),
)


class ModelInventory:
    def __init__(self, settings: Settings) -> None:
        from ltx_server.inference.loras import LoraRegistry

        self.loras = LoraRegistry(settings)
        assert settings.model_dir is not None
        self.paths: dict[str, Path] = {}
        self.overrides: set[str] = set()
        for checkpoint in CHECKPOINTS:
            override = getattr(settings, f"ltx_{checkpoint.component}_path")
            if override is not None:
                self.overrides.add(checkpoint.component)
            path = override or settings.model_dir / "ltx-2.5" / checkpoint.filename
            self.paths[checkpoint.component] = path.expanduser().absolute()

    def missing(self) -> list[str]:
        missing = []
        for checkpoint in CHECKPOINTS:
            path = self.paths[checkpoint.component]
            try:
                size = path.stat().st_size
                valid = (
                    path.is_file()
                    and size > 8
                    and (checkpoint.component in self.overrides or size == checkpoint.size)
                )
            except OSError:
                valid = False
            if not valid:
                missing.append(checkpoint.component)
        return missing

    def validate(self) -> None:
        missing = self.missing()
        if missing:
            raise ServiceError(
                ErrorCode.MODEL_NOT_INSTALLED,
                "Missing or incomplete model components: " + ", ".join(missing),
                503,
            )
        for component, path in self.paths.items():
            try:
                validate_safetensors(path)
            except (OSError, ValueError, struct.error):
                raise ServiceError(
                    ErrorCode.MODEL_LOAD_FAILED, f"Invalid safetensors file for {component}", 503
                ) from None


def validate_safetensors(path: Path) -> None:
    """Bounded header/offset validation; full SHA-256 is checked by the download tool."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        raise ValueError("No tensors")
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError("Invalid tensor metadata")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(value) is int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= size - length - 8
        ):
            raise ValueError("Tensor data outside file")
