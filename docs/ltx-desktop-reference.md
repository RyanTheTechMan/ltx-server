# Upstream compatibility

Source inspected on 2026-09-13 (US Eastern), with HEAD rechecked during phase 2:

| Component | Pin | CUDA runtime tested? |
| --- | --- | --- |
| [LTX-2 source](https://github.com/Lightricks/LTX-2/tree/a95ab856bf29407b6b066ede0abe1846050db56c) | `a95ab856bf29407b6b066ede0abe1846050db56c`, core/pipelines 1.3.0 | No |
| [Desktop reference](https://github.com/Lightricks/LTX-Desktop/tree/68cd86c15e5fd25f56229ea63c0dbcb0338f7812) | `68cd86c15e5fd25f56229ea63c0dbcb0338f7812` | No |
| [LTX-2.5 checkpoint pack](https://huggingface.co/Lightricks/LTX-2.5/tree/5e6e71018ee1756ed329b697a7b4aedc934dfce9) | `5e6e71018ee1756ed329b697a7b4aedc934dfce9` | No |

The source commit is installed by the optional Linux inference extra. The runtime
checks each package's version and `direct_url.json` commit identity before touching
private hooks. Checkpoint paths, sizes and SHA-256 hashes are centralized in
`inference/models.py`; the download tool verifies all five files. No weights were
downloaded or executed on the Mac.

## Verified API and server choices

The official `DistilledPipeline` constructor now takes `ModelPaths.from_split`,
a spatial upsampler path, LoRAs, quantization policy, registry and device. We pass
the distilled transformer, packed Gemma 4 encoder, convolutional video VAE and
audio VAE. Duration is explicit; the optional duration head and prompt enhancement
are unused. `PipelineOutput` has named `video`, `audio`, `num_frames`, `tiling_config`
and additional fields; the adapter uses named fields instead of tuple unpacking.

Joint video/audio modalities are still constructed in both denoising stages. The
adapter streams decoded YUV420p/BT.709 video and interleaved generated audio into
FFmpeg. `generate_audio=false` suppresses muxing, not joint-model computation.
The encoder explicitly specifies MP4 because partial filenames end in `.partial`.
The official `encode_video` implementation infers its container from the extension,
so it is not called directly. ffprobe verifies actual frame count, dimensions,
frame rate, H.264/AAC streams and duration before the atomic publish.

`ImageConditioningInput(path, frame_idx=0, strength=1.0)` uses the official image
preprocessing and first-frame latent replacement. CRF is left unset so the model's
own default applies. Resize/center-crop is upstream. Extra keyframes/last frames and A2V are implemented in phase 3. IC-LoRA and retake use the official pipelines in phase 4. Desktop's `DistilledA2VPipeline` is its own adapter,
not an identically named official class to import blindly.

Two-stage dimensions must be divisible by 64. The server follows Desktop's current
1024×576 / 1280×704 / 1920×1088 grid. Frame count follows `8k+1`; nearest rounding
with upward ties is a documented server policy.

### Dependency stack

Desktop's current backend pins LTX v1.2.0 while upstream HEAD declares 1.3.0. Desktop
patches therefore cannot be assumed compatible with this selected commit.

Upstream's root configuration prefers CUDA 13.2/torch 2.13 and contains a cuDNN 13
workaround. Inspection of that wheel index showed torchaudio 2.11.0 rather than a
matched 2.13.0 release. This server instead pins a matched **torch/torchaudio 2.10.0,
torchvision 0.25.0, CUDA 12.8** stack, consistent with Desktop's CUDA family and
within core's declared `torch~=2.7` requirement. The three Python 3.12 Linux wheels
were verified present and the entire dependency graph resolved in `uv.lock`.

Transformers 5.14.1 follows upstream's <5.15 constraint for Gemma 4. Static dependency
metadata mirrors core's exact declared requirements so its own uv index settings
do not conflict with the server's matched stack. The Mac dependency graph excludes
all inference packages. These are source/dependency checks, not CUDA execution tests.

## Retention and cancellation

`DiffusionStage` builds its transformer per invocation and disposes it on exit.
`ModelRegistry(cache_models=True, cache_weights=False)` caches structural shells;
it is not a live weight cache. Merely retaining `DistilledPipeline` would still
reload/cast the checkpoint twice per generation.

The original instance-local adapter in `inference/lifecycle.py` retains actual
transformer weights. At this exact pin, it replaces a single stage instance's
`_transformer_ctx` with a context using that instance's `_build_transformer`. It
also retains the text encoder through `_text_encoder_ctx` / `_build_text_encoder`
when configured. The transformer stays on GPU across the two stages, then parks
on CPU before video decode; Gemma parks immediately after prompt encoding. Smaller
upstream components retain their normal transient lifecycle. These hooks are private,
version checked, covered by contract tests and must be re-reviewed on a version bump.

The public `with_model_wrapper` hook installs checks before/after each transformer
forward. Stage wrappers supply honest progress; decoded chunks and nonblocking
FFmpeg writes also check cancellation. No global denoiser monkey-patch is needed.
Async cancellation signals the thread and then waits for it, preserving GPU/asset
ownership. OOM or failed/cancelled work invalidates the runtime, closes iterators and
clears caches before allowing the next job to reload. Successful jobs reuse it.

## Desktop comparison / patch audit

No Desktop patch source is copied or installed. “Used” below distinguishes original
server adaptations from imported Desktop patches. No unlinked upstream issue was
confirmed during this review.

| Desktop feature / patch | Used? | Reason / status |
| --- | --- | --- |
| `handlers/pipelines_handler.py` lifecycle | Idea, original implementation | One cached backend on one CUDA thread, with explicit cleanup and readiness. |
| `services/fast_video_pipeline/` FP8/warmup | Idea, original implementation | Official `fp8_cast.build_policy`; preload actual weights on startup. No warmup inference/compile benchmark claimed. |
| `services/ltx_pipeline_common.py` tiling | Idea, original implementation | Explicit public `TileSizeConfig` with a conv VAE. Tiling performance still unmeasured. |
| `runtime_config/` model specs/capabilities | Reference | Centralized current dimensions and split-pack components; no desktop catalog/UI copied. |
| `diffusion_stage_cache.py` | No patch | Original per-instance retention targets current 1.3.0 hooks; avoids a global patch against Desktop's 1.2.0. |
| `diffusion_interrupt.py` | No patch | Public model-wrapper hook permits cancellation between forwards at this pin. |
| `diffvae_decode_vram.py` | No | Default conv VAE plus transformer offload before decode avoids this DiffVAE-specific path. |
| `natten_libnatten_gate.py` | No | No DiffVAE/NATTEN requirement in phase 2. |
| `record_stream_fix.py` | No | Old layer-streaming hook; current upstream uses block-streaming. This adapter does not use block streaming. |
| `pinned_pool_fix.py` | No | No pinned-host block streaming. Desktop references [issue 141](https://github.com/Lightricks/LTX-Desktop/issues/141); do not apply its Windows handling blindly on Linux. |
| `safetensors_loader_fix.py` | No | Windows mmap access-violation workaround; no Linux need established. |
| `safetensors_metadata_fix.py` | No | Windows commit-charge workaround; no Linux need established. |
| `ic_lora_stage2_lora.py` | No | Current official IC pipeline intentionally uses base weights in stage two; this server preserves that behavior. |
| `diffvae_mps_tiling_budget.py` | No | Apple-only inference is outside this deployment target. |
| `mps_sdpa_torch.py` | No | Apple-only attention workaround. |
| SageAttention / compilation | Opt-in adapters | Public attention hook and official block compilation, disabled by default pending measurements. |
| `server_utils/` media handling | Reference | Original bounded FFmpeg encoder and upload validator; no Desktop filesystem logic. |
| `performance_runner/` | Reference only | No Desktop timing is represented as this server's performance. |
| Electron, credits, OAuth UI, Gemini, desktop settings | No | Outside this headless local service. |

Desktop's Apache 2.0 license and LTX's community-license routing were inspected.
See [third-party notices](../THIRD_PARTY_NOTICES.md). The optional packages and
weights retain their own terms; Desktop's Apache license does not relicense them.

Still pending on the 5090: importing the locked CUDA stack, full checkpoint loads,
output quality, I2V normalization behavior, audio synchronization with actual model
samples, peak VRAM/host RAM, cancellation latency and recovery after a real CUDA OOM.
The manual GPU smoke test is the first hardware validation step.

## Phase 3 conditioning review

The existing source/model pins remain unchanged. Reviewed implementations:

- [Official image helper](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py): `combined_image_conditionings` replaces latent zero for frame zero and uses `VideoConditionByKeyframeIndex` for subsequent pixel-frame positions. The distilled pipeline calls it at both resolutions.
- [Official A2Vid](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py): `AudioConditioner`, `encode_audio` and frozen `ModalitySpec` define the audio contract. The guided class requires a different sampling/LoRA setup; it is not directly substituted for our distilled checkpoint.
- [Desktop distilled A2V](https://github.com/Lightricks/LTX-Desktop/blob/68cd86c15e5fd25f56229ea63c0dbcb0338f7812/backend/services/a2v_pipeline/distilled_a2v_pipeline.py): demonstrates distilled denoising with frozen input audio in both stages and source-waveform output. No Desktop implementation was copied or imported.

The server's original per-instance stage adapter retains upstream distilled control
flow, including LTX 2.5's ancestral stage-one sampler. At this pin that sampler
reapplies the conditioning mask after noise injection; frozen audio has zero denoise
mask and zero modality sigma. Both stages receive the same source latent. The source
waveform bypasses the audio decoder, so audio is never regenerated for A2V.

Unlike directly decoding arbitrary-length audio onto CUDA, the server uses bounded,
cancellable FFmpeg normalization on CPU first. Source PCM is trimmed/padded to the
normalized frame duration, stereo 48 kHz. Encoder output is fitted to the official
`AudioLatentShape` grid to handle rounding. Output uses the same normalized PCM and
AAC encoding. Muting output does not disable conditioning. No extra models/downloads
are required beyond the existing five components.

Mac checks cover image index/strength mapping, source-latent injection in both stages,
request-state clearing after failure, unchanged next-job T2V behavior, mono resampling,
trim/pad policy, malformed/cancelled audio, and HTTP asset ownership. These checks
cannot validate actual model quality or fit; GPU smoke inputs are provided for the
5090 handoff.

## Phase 4 and performance integration review

The pinned `ICLoraPipeline` takes `video_conditioning=[(path, strength)]`, constructs
separate `stage_1` and `stage_2` builders and intentionally removes all LoRAs from
stage two. The adapter preserves this distinction and the public wrapper hook for
cancellation. No Desktop patch is needed to expose the official workflow. Trained
IC-LoRA files are operator-supplied and are not part of the five-file base pack.

The pinned `RetakePipeline(distilled=True)` accepts `video_path`, start/end seconds,
and independent regenerate-video/audio flags. Source metadata determines its grid;
the server checks exact agreement with the normalized HTTP request first. All source
frames pass through the video VAE, including regions outside the edit window, and
its audio decoder also reconstructs the full audio. This is documented as a model
retake rather than lossless editing. No claim of GPU-verified preservation is made.

Generic LoRAs use `LoraPathStrengthAndSDOps` with the exact
`LTXV_LORA_COMFY_RENAMING_MAP` exported by the pinned loader. Workflow/LoRA changes
rebuild a clean runtime. Official `OffloadMode.CPU`/`DISK` contexts remain untouched;
the server does not replace them with its ordinary retained-weight contexts.
`CompilationConfig(mode=None, capture=False)` avoids incompatible CUDA graph lifetime
assumptions. `DiffusionStage.with_attention` supplies the public per-builder hook
for PyTorch or optional SageAttention; masked attention retains upstream handling.

The optional Sage source-build reference is tag v2.2.0 commit
`eb615cf6cf4d221338033340ee2de1c37fbdba4a`, whose official `sageattn` API accepts
NHD tensors. Neither that extension nor any CUDA optimization was built/run here.
Phase 5 remains pending hardware measurements and default selection; see performance.md.
