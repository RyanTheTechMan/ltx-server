"""Operator-owned LoRA allowlist. HTTP requests never supply filesystem paths."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.schemas.generation import GenerationSpec, Lora


class LoraEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    kind: Literal["style", "ic"] = "style"
    base_model: Literal["ltx-2.5"] = "ltx-2.5"


class LoraManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    loras: dict[str, LoraEntry] = Field(default_factory=dict, max_length=128)


class LoraRegistry:
    def __init__(self, settings: Settings | None = None):
        self.default_ic = settings.default_ic_lora if settings else None
        self.entries: dict[str, LoraEntry] = {}
        if settings is not None and settings.lora_manifest is not None:
            path = settings.lora_manifest.expanduser().absolute()
            try:
                if path.stat().st_size > 256 * 1024:
                    raise ValueError("Manifest is too large")
                manifest = LoraManifest.model_validate_json(path.read_text())
                for identifier, entry in manifest.loras.items():
                    Lora(id=identifier)
                    entry.path = entry.path.expanduser()
                    if not entry.path.is_absolute():
                        entry.path = path.parent / entry.path
                    self.entries[identifier] = entry
            except (OSError, ValueError, ValidationError) as exc:
                raise ValueError("Invalid LORA_MANIFEST; check its JSON, IDs and paths") from exc
        if self.default_ic is not None:
            default_entry = self.entries.get(self.default_ic)
            if default_entry is None or default_entry.kind != "ic":
                raise ValueError("DEFAULT_IC_LORA must name an IC entry in LORA_MANIFEST")

    def resolve(self, spec: GenerationSpec | None) -> list[tuple[Lora, LoraEntry]]:
        if spec is None:
            return []
        requested = [(lora, "style") for lora in spec.loras]
        if spec.reference_video:
            ic = spec.reference_lora or (Lora(id=self.default_ic) if self.default_ic else None)
            if ic is None:
                raise ServiceError(
                    ErrorCode.UNSUPPORTED_PIPELINE,
                    "Reference video requires a registered IC-LoRA",
                    422,
                )
            requested.append((ic, "ic"))
        result = []
        for lora, kind in requested:
            entry = self.entries.get(lora.id)
            if entry is None or entry.kind != kind:
                raise ServiceError(
                    ErrorCode.INVALID_INPUT, f"Unknown or incompatible LoRA ID: {lora.id}", 422
                )
            if not entry.path.is_file():
                raise ServiceError(
                    ErrorCode.MODEL_NOT_INSTALLED, f"LoRA is not installed: {lora.id}", 503
                )
            result.append((lora, entry))
        return result

    def key(self, spec: GenerationSpec | None) -> tuple[object, ...]:
        workflow = (
            "retake"
            if spec and spec.retake
            else "reference"
            if spec and spec.reference_video
            else "distilled"
        )
        return (
            workflow,
            tuple(
                (
                    lora.id,
                    lora.scale,
                    str(entry.path),
                    entry.path.stat().st_mtime_ns,
                    entry.path.stat().st_size,
                )
                for lora, entry in self.resolve(spec)
            ),
        )

    def upstream(self, spec: GenerationSpec | None) -> list[Any]:
        if not self.resolve(spec):
            return []
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

        from ltx_server.inference.models import validate_safetensors

        result = []
        for lora, entry in self.resolve(spec):
            try:
                validate_safetensors(entry.path)
            except (OSError, ValueError) as exc:
                raise ServiceError(
                    ErrorCode.MODEL_LOAD_FAILED, f"Invalid LoRA file: {lora.id}", 503
                ) from exc
            result.append(
                LoraPathStrengthAndSDOps(str(entry.path), lora.scale, LTXV_LORA_COMFY_RENAMING_MAP)
            )
        return result

    def public(self) -> list[dict[str, str | bool]]:
        return [
            {
                "id": identifier,
                "kind": entry.kind,
                "base_model": entry.base_model,
                "installed": entry.path.is_file(),
            }
            for identifier, entry in self.entries.items()
        ]
