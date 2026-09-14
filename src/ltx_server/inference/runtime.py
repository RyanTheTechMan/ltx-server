"""All LTX/torch imports and version-sensitive behavior live here."""

import gc
import importlib.metadata
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from typing import Any

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import GenerationContext, check_cancelled
from ltx_server.inference.conditioning import (
    AudioConditioning,
    ConditionedStage,
    SourceAudioDecoder,
    image_inputs,
)
from ltx_server.inference.lifecycle import CallAdapter, RetainedModel
from ltx_server.inference.models import LTX_COMMIT, ModelInventory
from ltx_server.inference.performance import attention, pipeline_options
from ltx_server.jobs.state import JobStatus, OutputInfo
from ltx_server.media.ffmpeg import decode_source_audio, encode_mp4, inspect_output
from ltx_server.media.video_inputs import (
    VideoSource,
    encode_source_video,
    prepare_reference,
    prepare_retake,
    validate_retake_source,
)
from ltx_server.schemas.generation import GenerationSpec

logger = logging.getLogger(__name__)


def verify_upstream() -> None:
    for name in ("ltx-core", "ltx-pipelines"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            raise ServiceError(
                ErrorCode.INFERENCE_UNAVAILABLE,
                "Install the Linux inference extra with uv sync --extra inference",
                503,
            ) from None
        try:
            source = json.loads(distribution.read_text("direct_url.json") or "{}")
            compatible = (
                distribution.version == "1.3.0"
                and source.get("vcs_info", {}).get("commit_id") == LTX_COMMIT
            )
        except (ValueError, AttributeError):
            compatible = False
        if not compatible:
            raise ServiceError(
                ErrorCode.UPSTREAM_INCOMPATIBLE,
                "LTX must be installed from the exact commit in uv.lock",
                503,
            )


class LTXRuntime:
    def __init__(self, settings: Settings, inventory: ModelInventory, stop: Event) -> None:
        self.settings, self.inventory, self.stop = settings, inventory, stop
        self.torch: Any = None
        self.pipeline: Any = None
        self.registry: Any = None
        self.transformer: RetainedModel | None = None
        self.text_encoder: RetainedModel | None = None
        self.context: GenerationContext | None = None
        self.video_iterator: Any = None
        self.audio_conditioning = AudioConditioning()
        self.audio_conditioner: Any = None
        self.timings: dict[str, float] = {}
        self._stage_started = 0.0
        self._stage_name: str | None = None

    def check(self) -> None:
        check_cancelled(self.stop)
        if self.context is not None:
            check_cancelled(self.context.cancel)

    def load(self, preload: bool = False, spec: GenerationSpec | None = None) -> None:
        self.check()
        self.inventory.validate()
        verify_upstream()
        # Set before importing torch or HF. The packed text encoder includes tokenizer assets.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
        from ltx_core.loader.registry import ModelRegistry
        from ltx_core.model.video_vae import DimensionSizeConfig, TileSizeConfig
        from ltx_core.model.video_vae.model_configurator import is_diffusion_video_vae
        from ltx_core.quantization.fp8_cast import build_policy
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths

        self.torch = torch
        if not torch.cuda.is_available():
            raise ServiceError(
                ErrorCode.CUDA_UNAVAILABLE, "A working NVIDIA CUDA runtime is required", 503
            )
        device = torch.device(self.settings.gpu_device)
        torch.cuda.set_device(device)
        if self.settings.use_fp8 and torch.cuda.get_device_capability(device) < (8, 9):
            raise ServiceError(
                ErrorCode.CUDA_UNAVAILABLE, "FP8 requires an Ada or newer CUDA GPU", 503
            )
        paths = self.inventory.paths
        # Conv VAE keeps this phase independent of NATTEN/DiffVAE kernel workarounds.
        if is_diffusion_video_vae(str(paths["video_vae"])):
            raise ServiceError(
                ErrorCode.UNSUPPORTED_PIPELINE,
                "This backend requires the LTX 2.5 convolutional video VAE",
                422,
            )
        self.registry = ModelRegistry(cache_models=True, cache_weights=False)
        self.workflow = (
            "retake"
            if spec and spec.retake
            else "reference"
            if spec and spec.reference_video
            else "distilled"
        )
        pipeline_class = DistilledPipeline
        extra = pipeline_options(self.settings)
        if self.workflow == "retake":
            from ltx_pipelines.retake import RetakePipeline

            pipeline_class = RetakePipeline
            extra["distilled"] = True
        else:
            extra["spatial_upsampler_path"] = str(paths["spatial_upsampler"])
        if self.workflow == "reference":
            from ltx_pipelines.ic_lora import ICLoraPipeline

            pipeline_class = ICLoraPipeline
        self.pipeline = pipeline_class(
            model_paths=ModelPaths.from_split(
                **{
                    f"{name}_path": str(paths[name])
                    for name in ("transformer", "text_encoder", "video_vae", "audio_vae")
                }
            ),
            loras=self.inventory.loras.upstream(spec),
            **extra,
            device=device,
            quantization=build_policy(str(paths["transformer"])) if self.settings.use_fp8 else None,
            registry=self.registry,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
        )
        self.tiling = TileSizeConfig(
            frames=DimensionSizeConfig(
                tile_size=self.settings.vae_temporal_size,
                overlap=self.settings.vae_temporal_overlap,
            ),
            height=DimensionSizeConfig(
                tile_size=self.settings.vae_tile_size, overlap=self.settings.vae_tile_overlap
            ),
            width=DimensionSizeConfig(
                tile_size=self.settings.vae_tile_size, overlap=self.settings.vae_tile_overlap
            ),
        )
        encoder = self.pipeline.prompt_encoder
        kernel = attention(self.settings)
        names = ("stage_1", "stage_2") if self.workflow == "reference" else ("stage",)
        for name in names:
            stage = getattr(self.pipeline, name)
            if kernel is not None:
                stage = stage.with_attention(kernel)
            for method in ("_build_transformer", "_transformer_ctx", "with_model_wrapper"):
                if not callable(getattr(stage, method, None)):
                    raise ServiceError(
                        ErrorCode.UPSTREAM_INCOMPATIBLE, "LTX stage API changed", 503
                    )
            if "wrapper" not in inspect.signature(stage.with_model_wrapper).parameters:
                raise ServiceError(ErrorCode.UPSTREAM_INCOMPATIBLE, "LTX wrapper API changed", 503)
            # IC stages use different fused weights; keep upstream's sequential disposal.
            # Streaming owns its own weight lifecycle and must never use our retention hook.
            if self.workflow != "reference" and self.settings.offload_mode == "none":
                self.transformer = RetainedModel(
                    stage._build_transformer, device, park_on_exit=False
                )
                stage._transformer_ctx = self.transformer.context
            stage = stage.with_model_wrapper(
                lambda model, tools: CallAdapter(model, self.check, self.check)
            )
            setattr(
                self.pipeline,
                name,
                CallAdapter(
                    ConditionedStage(stage, self.audio_conditioning),
                    lambda: self.report(JobStatus.GENERATING),
                ),
            )
        if self.settings.cache_text_encoder and self.settings.offload_mode == "none":
            if not callable(getattr(encoder, "_build_text_encoder", None)) or not callable(
                getattr(encoder, "_text_encoder_ctx", None)
            ):
                raise ServiceError(ErrorCode.UPSTREAM_INCOMPATIBLE, "LTX encoder API changed", 503)
            self.text_encoder = RetainedModel(
                encoder._build_text_encoder, device, park_on_exit=True
            )
            encoder._text_encoder_ctx = self.text_encoder.context
        self.pipeline.audio_decoder = SourceAudioDecoder(
            self.pipeline.audio_decoder, self.audio_conditioning
        )
        self.pipeline.prompt_encoder = CallAdapter(encoder, lambda: self.report(JobStatus.ENCODING))
        self.pipeline.video_decoder = CallAdapter(self.pipeline.video_decoder, self.before_decode)
        self.check()
        if preload:
            # Sequential load/park avoids simultaneously resident Gemma + transformer weights.
            with torch.inference_mode():
                if self.text_encoder:
                    with self.text_encoder.context():
                        self.check()
                if self.transformer:
                    self.transformer.load()
                    self.check()
                    self.transformer.park()

    def report(self, status: JobStatus) -> None:
        self.check()
        name = status.value
        if self._stage_name == name:
            return
        if self.settings.synchronize_timings:
            self.torch.cuda.synchronize(self.settings.gpu_device)
        now = time.monotonic()
        if self._stage_name is not None:
            self.timings[self._stage_name] = (
                self.timings.get(self._stage_name, 0) + now - self._stage_started
            )
        self._stage_name, self._stage_started = name, now
        if self.context:
            self.context.report_stage(status)

    def before_decode(self) -> None:
        self.check()
        if self.transformer:
            self.transformer.park()
        self.report(JobStatus.DECODING)

    def generate(self, context: GenerationContext) -> OutputInfo:
        from ltx_core.color.yuv import yuv420p_bt709_converter_

        self.context = context
        self.timings = {}
        self._stage_name = None
        spec = context.spec
        scratch: Path | None = None
        source: Path | None = None
        source_info: VideoSource | None = None
        try:
            self.check()
            with self.torch.inference_mode():
                self.torch.cuda.reset_peak_memory_stats(self.settings.gpu_device)
                images = image_inputs(context)
                if spec.audio:
                    from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
                    from ltx_pipelines.utils.blocks import AudioConditioner

                    self.report(JobStatus.ENCODING)
                    if self.audio_conditioner is None:
                        self.audio_conditioner = AudioConditioner(
                            str(self.inventory.paths["audio_vae"]),
                            self.torch.bfloat16,
                            self.torch.device(self.settings.gpu_device),
                            registry=self.registry,
                            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
                        )
                    self.audio_conditioning.prepare(
                        context, self.torch, self.audio_conditioner, self.settings
                    )
                    self.check()
                common = dict(
                    prompt=spec.prompt,
                    seed=spec.seed,
                    tiling_config=self.tiling,
                    enhance_prompt=False,
                )
                if spec.retake:
                    self.report(JobStatus.ENCODING)
                    source = context.asset_paths[spec.retake.video]
                    if spec.retake.normalize_source:
                        candidate = context.partial_path.with_suffix(".source.partial")
                        candidate.touch(exist_ok=False)
                        scratch = candidate
                        prepare_retake(source, scratch, spec, self.settings, context.cancel)
                        source = scratch
                    source_info = validate_retake_source(
                        source, spec, self.settings, context.cancel
                    )
                    self.check()
                    result = self.pipeline(
                        video_path=str(source),
                        start_time=spec.retake.start,
                        end_time=spec.retake.end,
                        regenerate_video=spec.retake.regenerate_video,
                        regenerate_audio=spec.retake.regenerate_audio,
                        **common,
                    )
                else:
                    video_args = {}
                    if spec.reference_video:
                        self.report(JobStatus.ENCODING)
                        # Inference consumes this scratch input before MP4 encoding overwrites it.
                        prepare_reference(
                            context.asset_paths[spec.reference_video],
                            context.partial_path,
                            spec,
                            self.settings,
                            context.cancel,
                        )
                        video_args = {
                            "video_conditioning": [
                                (str(context.partial_path), spec.reference_strength)
                            ]
                        }
                    result = self.pipeline(
                        width=spec.width,
                        height=spec.height,
                        num_frames=spec.frames,
                        frame_rate=float(spec.fps),
                        images=images,
                        **common,
                        **video_args,
                    )
                self.video_iterator = result.video

                def chunks() -> Iterator[bytes]:
                    self.check()
                    for chunk in result.video:
                        self.check()
                        if tuple(chunk.shape[1:]) != (spec.height, spec.width, 3):
                            raise ServiceError(
                                ErrorCode.OUTPUT_ENCODING_FAILED,
                                "LTX returned unexpected video dimensions",
                            )
                        pixels = yuv420p_bt709_converter_(chunk.movedim(-1, -3))
                        yield pixels.cpu().contiguous().numpy().tobytes()
                    self.check()

                # Anonymous audio storage leaves no named artifact after a crash.
                with tempfile.TemporaryFile(dir=context.partial_path.parent) as audio_file:
                    has_audio = spec.generate_audio
                    audio_rate = 48000
                    preserve_audio = spec.retake is not None and not spec.retake.regenerate_audio
                    if preserve_audio:
                        assert source is not None and source_info is not None
                        has_audio = has_audio and source_info.has_audio
                    if has_audio and preserve_audio:
                        assert source is not None
                        audio_file.write(
                            decode_source_audio(
                                source,
                                duration=spec.frames / spec.fps,
                                cancel=context.cancel,
                                settings=self.settings,
                            )
                        )
                    elif has_audio:
                        if result.audio is None:
                            raise ServiceError(
                                ErrorCode.OUTPUT_ENCODING_FAILED, "LTX returned no audio"
                            )
                        waveform = result.audio.waveform.detach().float().cpu()
                        if waveform.ndim == 2 and waveform.shape[0] == 2:
                            waveform = waveform.transpose(0, 1)
                        if waveform.ndim != 2 or waveform.shape[1] != 2 or waveform.shape[0] == 0:
                            raise ServiceError(
                                ErrorCode.OUTPUT_ENCODING_FAILED, "Invalid LTX audio shape"
                            )
                        if not self.torch.isfinite(waveform).all().item():
                            raise ServiceError(
                                ErrorCode.OUTPUT_ENCODING_FAILED, "Invalid LTX audio samples"
                            )
                        audio_file.write(waveform.contiguous().numpy().astype("<f4").tobytes())
                        audio_rate = result.audio.sampling_rate
                    if spec.retake and not spec.retake.regenerate_video:
                        assert source is not None and source_info is not None
                        self.report(JobStatus.ENCODING_OUTPUT)
                        encode_source_video(
                            source,
                            context.partial_path,
                            spec,
                            self.settings,
                            context.cancel,
                            codec=source_info.codec,
                            audio=audio_file if has_audio else None,
                            audio_rate=audio_rate,
                        )
                    else:
                        encode_mp4(
                            chunks(),
                            context.partial_path,
                            width=spec.width,
                            height=spec.height,
                            frames=spec.frames,
                            fps=spec.fps,
                            cancel=context.cancel,
                            settings=self.settings,
                            audio=audio_file if has_audio else None,
                            audio_rate=audio_rate,
                            on_flush=lambda: self.report(JobStatus.ENCODING_OUTPUT),
                        )
                self.check()
                output = inspect_output(
                    context.partial_path,
                    self.settings,
                    width=spec.width,
                    height=spec.height,
                    frames=spec.frames,
                    fps=spec.fps,
                    has_audio=has_audio,
                    cancel=context.cancel,
                )
                self.timings["peak_vram_mb"] = self.torch.cuda.max_memory_allocated(
                    self.settings.gpu_device
                ) / (1024 * 1024)
                self.timings["peak_reserved_vram_mb"] = self.torch.cuda.max_memory_reserved(
                    self.settings.gpu_device
                ) / (1024 * 1024)
                return output
        finally:
            failed = sys.exception() is not None
            if self._stage_name:
                self.timings[self._stage_name] = self.timings.get(self._stage_name, 0) + (
                    time.monotonic() - self._stage_started
                )
            try:
                if self.video_iterator is not None:
                    close = getattr(self.video_iterator, "close", None)
                    if close:
                        close()
            except Exception:
                if not failed:
                    raise
                logger.exception("video_iterator_cleanup_failed")
            finally:
                self.video_iterator = None
                self.context = None
                self.audio_conditioning.clear()
                if scratch is not None:
                    scratch.unlink(missing_ok=True)
            if self.transformer and not failed:
                self.transformer.park()

    def is_oom(self, exc: BaseException) -> bool:
        return self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError)

    def close(self) -> None:
        self.audio_conditioning.clear()
        self.audio_conditioner = None
        if self.transformer:
            self.transformer.clear()
        if self.text_encoder:
            self.text_encoder.clear()
        if self.registry:
            self.registry.clear()
        self.pipeline = None
        self.context = None
        self.video_iterator = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
