# ltx-server

A lightweight headless REST API server for local LTX video generation.

**Phase 4 implements T2V, image/keyframe and audio conditioning, registered LoRAs,
IC-LoRA reference video and retake, with H.264/AAC MP4 output.** One bounded queue feeds one CUDA worker;
the HTTP request returns immediately. The server also provides validated uploads,
optional bearer authentication, progress, cancellation, diagnostics and ephemeral storage.

The deployment target is an RTX 5090 with 32 GB VRAM in a Linux LXC. Development,
contract tests and real FFmpeg encoding tests run on a Mac. **LTX/CUDA execution has
not been tested on the 5090 yet.** No minimum hardware or performance claims are made.

## Mac development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and FFmpeg
(`brew install ffmpeg`), then:

```bash
cp .env.example .env
uv sync --locked
uv run ltx-server
```

Python 3.12–3.14 is supported. Base dependencies contain no torch, LTX or CUDA
packages. `INFERENCE_BACKEND=auto` disables inference on macOS and selects LTX on
Linux x86_64. Disabled-backend requests return a queued job which fails explicitly
with `INFERENCE_UNAVAILABLE`; no placeholder video is produced.

Open [interactive docs](http://localhost:8000/docs) or
[OpenAPI JSON](http://localhost:8000/openapi.json). `ENABLE_DOCS=false` disables both.
Other entry points are `uv run python -m ltx_server` and
`uv run uvicorn ltx_server.main:app --host 0.0.0.0 --port 8000 --workers 1`.

## Linux / RTX 5090 setup

Use a dedicated non-root service account. NVIDIA devices and compatible driver
libraries must be visible inside the LXC; check `nvidia-smi` there. Install Python
3.12–3.14, uv, Git and the system media tools:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg
cp .env.example .env
uv sync --locked --extra inference
```

The inference extra pins official LTX code to commit
`a95ab856bf29407b6b066ede0abe1846050db56c` (`ltx-core`/`ltx-pipelines` 1.3.0), with
torch/torchaudio 2.10.0 and torchvision 0.25.0 from the CUDA 12.8 wheel index.
Transformers is pinned to 5.14.1. These choices are source-checked and dependency-resolved;
**they are not a claim of GPU runtime verification**. Use `--extra inference` on
subsequent `uv sync`/`uv run` commands, or uv may remove the optional packages.

Set `.env` storage paths and optionally `API_KEY`. The checkpoint pack requires
access to [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5); complete the
repository's access/license flow and set `HF_TOKEN` if required. Downloads are explicit:

```bash
uv run --extra inference python scripts/download_models.py --list
uv run --extra inference python scripts/download_models.py
uv run --extra inference python scripts/download_models.py --verify-only
```

The downloader fetches five files (about **66.2 GiB** total) at revision
`5e6e71018ee1756ed329b697a7b4aedc934dfce9`, preserving the repository's component folders
under `MODEL_DIR/ltx-2.5`. Every file is checked against its pinned byte count and SHA-256.
The pack contains the distilled transformer, packed Gemma 4 text encoder,
**convolutional** video VAE, audio VAE and 2× spatial upsampler. No extra Gemma folder,
prompt enhancer, duration head, NATTEN or SageAttention is required. The spatial
upsampler is part of distilled generation, not a separate public upscale feature.

`LTX_*_PATH` overrides support operator-managed component locations. These are
never HTTP fields. Overrides must retain compatible model components; the runtime
rejects a DiffVAE video decoder in this phase. The server performs bounded header
validation at load time; use the download tool's `--verify-only` for full hash checks.

Before starting the server, the manual GPU check runs real one-second T2V/audio
and validates dimensions, frame count, FPS, codecs and audio duration using ffprobe:

```bash
uv run --extra inference python scripts/smoke_test.py --gpu
uv run --extra inference ltx-server
```

The GPU smoke test cleans its output and refuses storage already locked by a running
server. It has been provided for the 5090, **not executed on this Mac**.

### Memory and model lifetime

FP8 cast uses the official checkpoint-aware policy. The actual transformer and
(optionally) text encoder weights remain cached in process, parked on **host RAM**
when not needed on GPU. Resolution changes reuse the same pipeline and weights.
Gemma is parked before diffusion; the transformer stays resident across both
stages and is parked before VAE decode. Smaller VAE/embedding components follow
upstream's transient lifecycle. Decode uses explicit 512×512 spatial tiles with
64-pixel overlap and 80-frame temporal tiles with 24-frame overlap.

This trades host memory and CPU↔GPU transfers for avoiding repeated large checkpoint
loads. Budget roughly 50 GiB for retained transformer/text weights alone, plus
loading buffers, activations and the OS; 32 GB of VRAM is not the host RAM budget.
`CACHE_TEXT_ENCODER=false` reduces retained host memory by reloading the text encoder
for each prompt. Host-memory requirements and peak VRAM still need measurement on
the target. Disabling FP8 is intended for larger GPUs; the BF16 transformer alone
exceeds the 5090's VRAM.

`WARM_MODEL_ON_START=true` preloads and parks the retained weights on the dedicated
thread. Health remains available during loading. This is model preloading, not a
warmup diffusion/compile benchmark. With it disabled, loading happens on the first
job. Missing models/CUDA produce a degraded health state and stable job errors;
installing models allows subsequent jobs to retry initialization.

Optional attention, streaming offload, transformer compilation and tiling controls
are available for benchmarking. Defaults remain provisional. See
[performance validation](docs/performance.md); phase 5 hardware measurements and
default selection are still pending. No Desktop-wide monkey-patches are installed.

## API

With empty `API_KEY`, authentication is disabled. Otherwise protected endpoints
require `Authorization: Bearer ...`, compared in constant time. Health remains
public unless `HEALTH_REQUIRES_AUTH=true`. Schemas/docs remain public while enabled.
CORS is disabled by default; configure comma-separated `CORS_ORIGINS` if needed.

```bash
export LTX_URL=http://localhost:8000
export API_KEY=your-configured-key

curl "$LTX_URL/v1/health"
curl -H "Authorization: Bearer $API_KEY" "$LTX_URL/v1/gpu"
curl -H "Authorization: Bearer $API_KEY" "$LTX_URL/v1/models"
curl -H "Authorization: Bearer $API_KEY" "$LTX_URL/v1/queue"

curl -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"prompt":"A husky runs through snow, paws crunching softly","duration":10,"resolution":"540p","fps":24,"generate_audio":true}' \
  "$LTX_URL/v1/generations"
```

Admission returns HTTP 202 with an ID and `status: queued`. Poll that ID:

```bash
curl -H "Authorization: Bearer $API_KEY" "$LTX_URL/v1/generations/gen_REPLACE_WITH_ID"
curl -H "Authorization: Bearer $API_KEY" -o output.mp4 \
  "$LTX_URL/v1/generations/gen_REPLACE_WITH_ID/content"
curl -X DELETE -H "Authorization: Bearer $API_KEY" \
  "$LTX_URL/v1/generations/gen_REPLACE_WITH_ID"
```

For image-to-video, upload a first frame and pass its returned asset ID:

```bash
curl -H "Authorization: Bearer $API_KEY" -F 'file=@start.png' "$LTX_URL/v1/assets"
curl -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"prompt":"The subject turns toward the camera","first_frame":"asset_REPLACE_WITH_ID","duration":5}' \
  "$LTX_URL/v1/generations"
```

For a starting and destination image, provide both `first_frame` and `last_frame`.
The destination maps to the **last normalized pixel frame**, `frames - 1`. For
multiple targets, use `keyframes` instead of the first/last fields:

```json
{
  "prompt": "The camera moves around the sculpture",
  "duration": 10,
  "fps": 24,
  "keyframes": [
    {"asset_id": "asset_REPLACE_START", "frame": 0, "strength": 1.0},
    {"asset_id": "asset_REPLACE_MIDDLE", "frame": 120, "strength": 0.8},
    {"asset_id": "asset_REPLACE_END", "frame": 240, "strength": 1.0}
  ]
}
```

Positions are zero-based **pixel-frame indices**, not VAE latent indices; arbitrary
in-range positions are accepted. Up to 32 unique positions are allowed, with strengths
from 0 to 1. Targets are sorted before inference. Reusing the same image at different
positions is allowed. First-frame conditioning uses latent replacement; later targets
use official keyframe conditioning at both resolutions. Conditioning guides the model;
it does not guarantee pixel-identical endpoint images. Image preprocessing uses
upstream's aspect-preserving center crop. No external image interpolation is performed.

Upload audio through `/v1/assets` and pass its ID as `audio`:

```json
{
  "prompt": "A drummer performing energetically under stage lights",
  "audio": "asset_REPLACE_AUDIO",
  "duration": 10,
  "generate_audio": true
}
```

Source audio may be combined with first/last frames or keyframes. Duration follows
the same request/server default as T2V, then rounds to the video frame grid; it is
**not inferred from the audio file**. Audio starts at zero, is trimmed if longer,
and padded with silence if shorter. It is decoded to stereo 48 kHz PCM without time
stretching, then encoded through the official audio VAE for conditioning. Those
latents are frozen in both distilled stages. The output uses the source PCM rather
than VAE-decoded/generated audio, with AAC encoding; this preserves content, not
bit-identical compressed bytes. Mono is duplicated and multichannel inputs are
remixed to stereo by FFmpeg.

`generate_audio=false` mutes the output while retaining source-audio conditioning.
With no source audio, it omits the generated track; the joint model still computes
audio internally. Advanced workflows are described below.

| Preset | Actual dimensions |
| --- | --- |
| 540p | 1024 × 576 |
| 720p | 1280 × 704 |
| 1080p | 1920 × 1088 |

FPS accepts 24, 25 or 30. Requested duration accepts 1–20 seconds, subject to server
limits. Frames round to the nearest `8k+1`, ties upward: 10 seconds at 24 fps gives
241 frames (about 10.042 seconds). Returned metadata reports real generated/encoded
values. A preset being accepted does not imply every duration fits 32 GB.

Progress reports stages without invented per-step precision. Cancellation is checked
before/after transformer forwards, between decoded chunks and during FFmpeg writes.
A CUDA kernel, checkpoint load or VAE tile already executing must finish before
cancellation can be observed. The worker retains its slot and asset leases until
that thread has stopped. CUDA OOM fails the job, clears cached runtime state and
allows the next job to reload; no automatic lower-resolution retry occurs.

Machine-readable errors use `{ "error": { "code", "message" } }`; failed jobs put
the same code/message under their `error` field. Queue capacity counts waiting jobs,
excluding the running job. Full queues return 429 / `QUEUE_FULL`. DELETE is idempotent.

## Storage and uploads

Default retention: outputs 30 minutes; the first successful full GET shortens this
to at most five minutes after download. Retries do not extend retention. Byte-range
requests support seeking but do not count as a full download. Active responses pin
files until completion/disconnect. Outputs publish by atomic rename from `.partial`
only after FFmpeg exits and ffprobe validates the result.

Assets expire after one hour; queued/running jobs hold leases. Expired assets cannot
be used by new jobs. Job records remain one day after their latest terminal state.
Expired output returns 410 while its record remains, then 404 after record cleanup.

Metadata is **in memory**: restart invalidates previous IDs and does not resume jobs.
Startup/periodic sweeps remove recognizable orphan files according to their mtime
and TTL. Download grace does not survive restart. Unknown files/symlinks are not swept.

Uploads support static PNG/JPEG/WebP images, WAV/MP3/FLAC/Ogg audio and MP4/M4A/WebM
media. Content, not filename/MIME, determines type. Pillow fully decodes images;
ffprobe checks streams and FFmpeg decodes a bounded sample. Upload limits default to
50/100/500 MiB and include a multipart-body cap for chunked requests. Audio uploads support conditioning; video uploads can drive IC-LoRA or retake.

Storage paths derive from `DATA_DIR`; optional overrides are in [.env.example](.env.example).
Mutable directories must be separate, service-owned and on the same filesystem.
Symlinked storage directories are rejected. Run **one process and one Uvicorn worker**;
per-directory locks prevent accidental concurrent instances using the same storage.

## Deployment and verification

The [systemd example](deploy/systemd/ltx-server.service) expects `/opt/ltx-server`
and `DATA_DIR=/var/lib/ltx-server`. Install the checkout/virtualenv under `/opt`,
create the `ltx-server` account, and restrict `.env` permissions to that account.
Systemd creates the state directory. GPU devices remain visible. SIGTERM stops
admission, cancels queued work and waits for active work before releasing resources.
The unit's shutdown timeout should be reviewed after measuring cancellation latency.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/smoke_test.py   # API-only smoke against a running server
```

Tests mock LTX/CUDA while exercising the actual adapter, lifecycle and HTTP contracts.
Separate tests encode real H.264 video and AAC audio using system FFmpeg. They do not
prove the model runs or fits on a 5090. See [architecture](docs/architecture.md) and
[upstream compatibility notes](docs/ltx-desktop-reference.md).

Phase 4 APIs and phase 5 benchmark tooling are implemented. Actual 5090
measurements, quality checks and measured default selection remain pending.

After deploying to the GPU machine, stop the server and exercise phase 3 inputs:

```bash
uv run --extra inference python scripts/smoke_test.py --gpu --first-frame start.png --last-frame end.png
uv run --extra inference python scripts/smoke_test.py --gpu --keyframe 0 start.png 1 --keyframe 24 end.png 0.8
uv run --extra inference python scripts/smoke_test.py --gpu --audio speech.wav --first-frame portrait.png --duration 5
```

The tool validates and stages local inputs, checks output media, and removes only
its managed copies/output. Add `--no-audio` for a muted A2V result. Actual conditioning
quality, speech synchronization and VRAM usage remain unverified on the Mac.

## LoRAs, reference video and retake

Configure `LORA_MANIFEST` with an operator-managed JSON file using the structure in
[examples/loras.json](examples/loras.json). The example paths are placeholders, not
bundled or downloaded models. Use LoRAs compatible with this pinned LTX 2.5 distilled
checkpoint and the official Comfy key mapping. A manifest entry declares `path`,
`kind` (`style` or `ic`) and `base_model: "ltx-2.5"`. Relative paths resolve against
the manifest's directory. Restart after changing the manifest. Set `DEFAULT_IC_LORA`
to a registered IC entry if reference requests will omit `reference_lora`.

`GET /v1/models` lists registered IDs, kinds and installation status without exposing
paths. The API accepts IDs only; arbitrary paths, URLs and uploaded LoRA weights are
not accepted. Unknown IDs, incorrect kinds, duplicate IDs and missing files fail
explicitly. Structural safetensors validation occurs before weights are loaded;
compatibility of third-party LoRA tensors still requires a GPU smoke test.

```json
{
  "prompt": "A cinematic tracking shot of a runner",
  "loras": [{"id": "cinematic", "scale": 0.8}],
  "duration": 5
}
```

Style LoRAs apply to both distilled stages, including image/A2V jobs, or the retake
stage. The worker retains one active workflow/LoRA combination. Identical requests
reuse weights across resolutions; changing workflow, LoRA IDs/scales or file stats
unloads the previous runtime before building the next. Changes never accumulate on
already-fused weights. Switching variants can be expensive and remains serial.

For reference video, upload a video asset and select an IC-LoRA:

```json
{
  "prompt": "Recreate this shot at night with cinematic blue lighting",
  "reference_video": "asset_REPLACE_VIDEO",
  "reference_lora": {"id": "reference", "scale": 1.0},
  "reference_strength": 1.0,
  "duration": 5
}
```

Use a reference representation appropriate for the selected IC-LoRA (for example,
a trained control representation). The server does not invent depth, pose or edge
preprocessing. Reference video is resampled to requested FPS, aspect-preserving
scaled/center-cropped to requested dimensions, trimmed or extended with its final
frame to the normalized length. Source audio is discarded; output audio is generated.
Image/keyframe targets can accompany a reference, but separate source-audio
conditioning is rejected. The official IC pipeline applies all selected LoRAs and
reference conditioning in stage one, then uses the base distilled model in stage
two. Additional style LoRAs therefore affect stage one only for this workflow.
The two differently fused transformers retain upstream's sequential transient
lifecycle rather than caching two full sets of weights. This can load checkpoints
more often than ordinary T2V. No old Desktop stage-two patch is applied.

Retake regenerates a time window of an uploaded video:

```json
{
  "prompt": "The subject smiles while a bird chirps",
  "duration": 5,
  "resolution": "540p",
  "fps": 24,
  "retake": {
    "video": "asset_REPLACE_VIDEO",
    "start": 1.0,
    "end": 3.0,
    "regenerate_video": true,
    "regenerate_audio": true
  }
}
```

The uploaded source must match the request's **normalized frame count, FPS and
actual dimensions**. Retake performs no implicit resizing or duration changes;
server-generated videos are suitable inputs when the same request grid is used.
The interval is measured in seconds from source start, must be nonempty and inside
the clip, and must regenerate at least one modality. Style LoRAs are allowed;
other conditioning fields cannot be combined with retake. `generate_audio=false`
only mutes the final output. The official retake pipeline encodes/decodes the source
through its VAEs: unchanged regions/modalities are model-conditioned, not guaranteed
pixel-identical or bit-identical copies of the original. Preservation quality and
window boundaries need visual/audio review on the GPU host.

Hardware smoke examples (stop the server first):

```bash
uv run --extra inference python scripts/smoke_test.py --gpu --lora cinematic 0.8
uv run --extra inference python scripts/smoke_test.py --gpu --reference-video control.mp4 --reference-lora reference
uv run --extra inference python scripts/smoke_test.py --gpu --retake-video source.mp4 --duration 5 --start 1 --end 3
uv run --extra inference python scripts/benchmark.py --all --dry-run
```

See [performance validation](docs/performance.md) for the isolated benchmark matrix,
optional SageAttention setup, measurements, retained samples and remaining GPU checks.
