"""Benchmark plans and summaries; importing this module never touches CUDA."""

import statistics
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "pytorch": {"attention_backend": "pytorch"},
    "sage": {"attention_backend": "sage"},
    "cpu-offload": {"offload_mode": "cpu"},
    "disk-offload": {"offload_mode": "disk"},
    "bf16-cpu": {"offload_mode": "cpu", "use_fp8": False},
    "tiles-256": {"vae_tile_size": 256},
    "tiles-768": {"vae_tile_size": 768},
    "compile": {"compile_transformer": True},
}


def profile_settings(name: str) -> dict[str, Any]:
    # Explicit shared controls prevent .env from silently changing the comparison.
    baseline = dict(
        use_fp8=True,
        offload_mode="none",
        attention_backend="automatic",
        compile_transformer=False,
        cache_text_encoder=True,
        vae_tile_size=512,
        vae_tile_overlap=64,
        vae_temporal_size=80,
        vae_temporal_overlap=24,
        warm_model_on_start=False,
        synchronize_timings=True,
    )
    return {**baseline, **PROFILES[name]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        row for row in rows if row.get("kind") == "measured" and row.get("status") == "passed"
    ]
    if not measured:
        return {"measured_runs": 0}
    return {
        "measured_runs": len(measured),
        "median_seconds": statistics.median(row["elapsed_seconds"] for row in measured),
        "min_seconds": min(row["elapsed_seconds"] for row in measured),
        "max_seconds": max(row["elapsed_seconds"] for row in measured),
        "max_allocated_vram_mb": max(
            (
                row["metrics"]["peak_vram_mb"]
                for row in measured
                if "peak_vram_mb" in row["metrics"]
            ),
            default=None,
        ),
        "max_reserved_vram_mb": max(
            (
                row["metrics"]["peak_reserved_vram_mb"]
                for row in measured
                if "peak_reserved_vram_mb" in row["metrics"]
            ),
            default=None,
        ),
    }
