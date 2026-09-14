"""Explicit optional performance settings; no global attention monkey-patches."""

from typing import Any

from ltx_server.config import Settings


def pipeline_options(settings: Settings) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if settings.offload_mode != "none":
        from ltx_pipelines.utils.types import OffloadMode

        options["offload_mode"] = OffloadMode(settings.offload_mode)
    if settings.compile_transformer:
        from ltx_core.model.transformer.compiling import CompilationConfig

        # No CUDA graph capture: this service moves weights and varies sequence lengths.
        options["compilation_config"] = CompilationConfig(mode=None, capture=False)
    return options


class SageAttention:
    label = "SageAttention (unmasked); upstream masked attention"

    def __init__(self) -> None:
        from sageattention import sageattn

        self.kernel = sageattn

    def __call__(self, q: Any, k: Any, v: Any, heads: int, mask: Any = None) -> Any:
        if mask is not None:
            raise ValueError("Sage adapter serves only the unmasked attention slot")
        batch, _, channels = q.shape
        head_dim = channels // heads
        query, key, value = (tensor.reshape(batch, -1, heads, head_dim) for tensor in (q, k, v))
        return self.kernel(query, key, value, tensor_layout="NHD", is_causal=False).reshape(
            batch, -1, channels
        )


def attention(settings: Settings) -> Any:
    if settings.attention_backend == "automatic":
        return None
    if settings.attention_backend == "sage":
        return SageAttention()
    from ltx_core.model.transformer.attention import AttentionFunction

    return AttentionFunction.PYTORCH
