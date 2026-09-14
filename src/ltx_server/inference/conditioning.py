"""Request-local conditioning over the pinned official distilled pipeline."""

from dataclasses import replace
from typing import Any

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.inference.backend import GenerationContext
from ltx_server.media.ffmpeg import decode_source_audio


def image_inputs(context: GenerationContext) -> list[Any]:
    from ltx_pipelines.utils.args import ImageConditioningInput

    spec = context.spec
    targets = [(key.asset_id, key.frame, key.strength) for key in spec.keyframes]
    if spec.first_frame:
        targets.append((spec.first_frame, 0, 1.0))
    if spec.last_frame:
        targets.append((spec.last_frame, spec.frames - 1, 1.0))
    return [
        ImageConditioningInput(
            path=str(context.asset_paths[identifier]), frame_idx=frame, strength=strength
        )
        for identifier, frame, strength in sorted(targets, key=lambda target: target[1])
    ]


class AudioConditioning:
    """Freeze source audio in both stages; never decode it back through the lossy VAE.

    The runtime owns this state on its single CUDA thread and clears it after every
    request. No alternate transformer, checkpoint pack or global patch is needed.
    """

    def __init__(self) -> None:
        self.latent: Any = None
        self.source: Any = None

    def prepare(
        self, context: GenerationContext, torch: Any, conditioner: Any, settings: Settings
    ) -> None:
        from ltx_core.model.audio_vae import encode_audio
        from ltx_core.types import Audio, AudioLatentShape

        assert context.spec.audio is not None
        pcm = decode_source_audio(
            context.asset_paths[context.spec.audio],
            duration=context.spec.frames / context.spec.fps,
            cancel=context.cancel,
            settings=settings,
        )
        waveform = torch.frombuffer(bytearray(pcm), dtype=torch.float32).reshape(-1, 2)
        waveform = waveform.transpose(0, 1).contiguous()
        if not torch.isfinite(waveform).all().item():
            raise ServiceError(
                ErrorCode.INVALID_MEDIA, "Source audio contains invalid samples", 422
            )
        self.source = Audio(waveform=waveform, sampling_rate=48000)
        encoded = conditioner(
            lambda encoder: encode_audio(
                Audio(waveform=waveform.unsqueeze(0), sampling_rate=48000), encoder, None
            )
        )
        shape = AudioLatentShape.from_duration(
            batch=1, duration=context.spec.frames / context.spec.fps
        )
        if encoded.ndim != 4 or tuple(encoded.shape[:2]) != (1, 8) or encoded.shape[3] != 16:
            raise ServiceError(ErrorCode.UPSTREAM_INCOMPATIBLE, "Unexpected audio latent shape")
        # Account for the encoder's temporal rounding, using upstream's latent grid.
        if encoded.shape[2] < shape.frames:
            encoded = torch.nn.functional.pad(encoded, (0, 0, 0, shape.frames - encoded.shape[2]))
        self.latent = encoded[:, :, : shape.frames]

    def clear(self) -> None:
        self.latent = None
        self.source = None


class ConditionedStage:
    def __init__(self, target: Any, audio: AudioConditioning) -> None:
        self.target, self.audio = target, audio

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.audio.latent is not None:
            # ModalitySpec.frozen also forces the transformer modality sigma to zero.
            kwargs["audio"] = replace(
                kwargs["audio"], frozen=True, noise_scale=0.0, initial_latent=self.audio.latent
            )
        return self.target(*args, **kwargs)


class SourceAudioDecoder:
    def __init__(self, target: Any, audio: AudioConditioning) -> None:
        self.target, self.audio = target, audio

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.audio.source is not None:
            return self.audio.source
        return self.target(*args, **kwargs)
