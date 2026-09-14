# Advanced server architecture

One FastAPI process owns a bounded FIFO, one asynchronous worker, an in-memory job
repository, an asset registry and a cleanup task. The FastAPI lifespan starts and
stops these resources. No globals hold jobs between application instances. Merely
importing the application does not create directories or initialize GPU resources.

```text
HTTP request -> authentication -> schema/type checks -> asset leases -> bounded FIFO
                                                                         |
                                                                   single worker
                                                                         |
                                                       GenerationBackend (async protocol)
                                                                         |
                                                       temp file -> atomic publish
```

## Boundaries

- `api/`: typed endpoints, bearer auth, upload body limit, file responses and diagnostics.
- `schemas/`: strict HTTP contract, normalized request and one resolution/frame policy.
- `jobs/`: explicit state, repository, scheduling, cancellation, download leases and cleanup.
- `media/`: flat managed storage, asset ownership, Pillow, streaming FFmpeg encoding
  and bounded media validation.
- `inference/`: backend protocol, pinned LTX adapter, checkpoint inventory and cached
  model lifecycle. Optional LTX/PyTorch imports are isolated here and loaded only
  when the Linux backend initializes.

`INFERENCE_BACKEND=auto` selects LTX on Linux x86_64 and disables inference on Mac.
The enabled backend wraps the official two-stage `DistilledPipeline` for T2V and
image/keyframe conditioning and A2V. Tests inject controlled backends and mock upstream objects to test
transport, lifecycle and adapter contracts without CUDA. Separate media tests
encode and inspect real H.264/AAC MP4s. Model execution requires the NVIDIA host.

## Scheduling and ownership

Admission and state mutation run synchronously on the event loop with no awaits
between capacity checks, all-or-nothing asset acquisition, repository insertion
and FIFO insertion. This makes concurrent request admission atomic in the single
process. The capacity excludes the running job. The response snapshots admission
state before the worker can mutate it.

Blocking image decode, file chunk writes and NVML operations run off the event
loop; media subprocesses use asyncio and are killed/reaped on timeout. Small
filesystem metadata operations and sweeps are synchronous, suitable for this
single-worker internal service. Extremely large directories or slow network
filesystems are outside the intended deployment; use local storage.

Inputs remain leased while queued or running. Expiration prevents new admissions
but does not remove an existing job's files. Completion, failure and cancellation
release leases. A request referencing several assets either acquires all or none.

The `GenerationBackend` receives a thread-safe cancellation event, normalized
parameters, trusted asset paths, a managed partial path and an event-loop stage
callback. `PipelineManager` runs synchronous LTX work in a dedicated one-thread
executor and marshals progress and metrics back onto the event loop. It shields
the executor future and waits for thread completion even if the async caller is
cancelled. Cancelling an asyncio wrapper alone cannot release a CUDA worker slot.

Cancellation is cooperative at stage boundaries, before/after transformer forwards,
between decoded chunks and during FFmpeg writes/finalization. A running CUDA kernel,
checkpoint load or VAE tile cannot be interrupted. ffprobe has a bounded timeout.

## Model lifetime and media

One cached pipeline serves all accepted resolutions. Instance-local adapters retain
actual transformer and optional text encoder weights; upstream's default model
registry alone retains meta-device structure and does not prevent disk reloads.
The exact upstream commit and lifecycle hooks are checked before use. No global
Desktop patches are applied.

Text encoder weights move back to CPU after prompt encoding. The FP8 transformer
stays on GPU for both diffusion stages, then moves to CPU before tiled VAE decode.
CPU parking preserves its dtype. Retained weights require substantial host RAM;
this is separate from the 32 GB GPU budget. Transient embedding/VAE components keep
their upstream lifecycle. Startup preloading runs in the same dedicated thread;
disabling it defers initialization until the first job. Health stays responsive.

`inference/conditioning.py` maps images to sorted pixel-frame inputs and keeps
source audio state local to the active request. A bounded cancellable FFmpeg decode
normalizes input audio to stereo 48 kHz and the requested video duration. The official
`AudioConditioner`/`encode_audio` creates its latents. `ConditionedStage` replaces only
the audio `ModalitySpec` with frozen, zero-noise source latents in both calls of the
existing distilled stage. `SourceAudioDecoder` bypasses audio VAE decoding for A2V
and returns the source waveform. T2V retains its normal audio decoder. This original
adapter follows official A2Vid's frozen-modality contract and Desktop's distilled
A2V design, while keeping the pinned distilled sampler, image helpers and weight
cache. No second transformer is constructed. The runtime clears waveform/latent
references on success, failure and cancellation, so they cannot leak into later jobs.

First/last and keyframe fields are mutually exclusive. Positions must be unique and
inside the normalized clip; the last-frame alias resolves to `frames - 1`. Asset
admission validates all image/audio types before acquiring leases. Input audio does
not change duration policy and `generate_audio` controls output muxing only.

The model pack and source revisions are pinned. Startup checks checkpoint sizes and
bounded safetensors headers; the explicit downloader additionally verifies SHA-256.
Model initialization never downloads weights. Missing dependencies, models or CUDA
leave diagnostics available and subsequent jobs can retry initialization. Generation
failures, including OOM and cancellation, invalidate the cached runtime. Cleanup
errors are logged without hiding the original job error or terminating the queue.

Video chunks use upstream's BT.709 limited-range YUV420 conversion. FFmpeg receives
bounded chunks through stdin and optional stereo float PCM through an anonymous
temporary-file descriptor. It writes H.264/AAC with an explicit MP4 container so
the managed `.partial` extension works. ffprobe checks dimensions, exact frame count,
FPS, codecs and audio/video duration before publication. Audio is trimmed/padded to
the generated frame duration. Disabling output audio does not skip joint-model
audio computation.

Job metrics report stage wall time, pipeline initialization time, total time and
peak allocated VRAM. Lazy weight loads are included in their corresponding stage;
pipeline initialization time is not a standalone checkpoint-loading benchmark.
These are diagnostics, not measured 5090 performance guarantees.

## State and publication

Active stages are `queued`, `loading`, `encoding`, `generating`, `decoding`,
`encoding_output`; successful work ends at `complete`. Stages can skip forward
when a backend lacks callbacks. Progress is a stage estimate, not fabricated
denoising precision. Terminal failure/cancellation ends work. Complete media can
transition to `expired` or `cancelled` after cleanup/DELETE.

The backend validates/encodes the output in `tmp/gen_<uuid>.partial`. The manager
checks cancellation before publication and atomically renames it to
`outputs/gen_<uuid>.mp4`. Same-filesystem temp/output/asset roots are enforced at
startup. Failed/cancelled jobs remove partials and never publish late results.

DELETE removes a waiting job immediately. Running DELETE sets cancellation intent;
status remains active until the backend returns, retaining its assets and worker
slot. Repeated DELETE succeeds. DELETE of complete content removes it; downloads
already in progress defer physical deletion. Unknown IDs also return 204.

## Expiration and responses

Output deadline starts at successful publication. The first full GET whose final
body message is successfully sent can shorten it to the earlier of the original
deadline and download time plus grace. This is transport completion, not proof a
client saved the file. Partial range requests and failed downloads do not start
grace; later retries do not extend it. `FileResponse` supplies byte-range support.

Each response pins its file until the send finishes or raises. Cleanup cannot
remove a pinned file, and new downloads after the deadline receive 410. Explicit
DELETE changes metadata immediately but waits for readers before unlinking. The
response releases its lease in `finally`, including invalid range and disconnect paths.

Assets retain their original TTL during leases; cleanup resumes after release,
and may immediately delete an asset whose deadline passed while in use. Job record
retention starts at the latest terminal transition, including output expiration.

Metadata is intentionally volatile. Restart invalidates all previous IDs and does
not resume jobs. Startup and periodic sweeps remove only old regular files matching
the server naming pattern. Their mtime is reset at publication so interrupted
encoding time does not consume completed artifact retention. No download grace
survives restart; orphan outputs use the standard TTL. A future persistent
repository must restore asset leases, terminal timestamps and download deadlines,
and reconcile interrupted jobs before accepting requests.

## Process lifecycle and deployment

Storage directories must not overlap or contain symlinks. Advisory locks in each
mutable directory prevent accidental multi-process use. Only the service account
should write these directories; this is not protection against hostile code with
the same account privileges. Client-supplied names are never storage paths.

Shutdown stops admission, cancels waiting work, signals active work, waits for the
backend, closes it, then stops cleanup and releases storage locks. The systemd unit's
120-second timeout is an outer operator limit to review after measuring real CUDA
cancellation latency on the target. Forced process
termination leaves only managed partials for startup cleanup.

NVML diagnostics are optional and read-only. NVIDIA availability is distinct from
CUDA readiness and model readiness. The Mac's lack of NVML is expected and does not
prevent the API from running. GPU errors are not converted into fake zero metrics.
FP8 and explicit VAE tiling are provisional memory settings; phase 5 will measure
them and evaluate attention kernels, offload, warmup and compilation on the 5090.

## Advanced workflows and performance experiments

LoRA manifests are operator-owned allowlists. A normalized workflow/LoRA signature
(including file stats) keys the single active runtime; switching signatures disposes
the previous runtime before loading another. HTTP admission resolves IDs and kinds
before acquiring asset leases. Upstream applies fresh LoRA weights at build time.
There is no in-place cumulative merge or multi-variant GPU cache.

IC-LoRA uses the official two-stage pipeline, preserving its stage-one LoRA and
reference conditioning and its base-model stage two. Those distinct transformer
builds keep upstream's transient disposal. Reference normalization uses the job's
existing managed partial as scratch; the upstream call consumes it fully before
output encoding overwrites it. Failure/cancellation deletes the partial, and only
the validated final generation can be published. Retake uses the official distilled
pipeline and checks the complete source video grid before inference. No arbitrary
source-video resizing/trimming is hidden in retake.

Official CPU/disk streaming owns model lifetime, so the server disables its retained
transformer/text hooks in those modes. Attention is changed per builder through the
public hook, never globally. Compilation uses upstream block compilation without
CUDA graph capture, which would conflict with CPU parking and shape changes.
Spatial/temporal tiling values are configuration-validated. Benchmark timing can
synchronize stage boundaries; it is off in ordinary serving.

The benchmark supervisor starts one worker process per profile, uses the same
storage locks, separates cold/warmup/measured calls, preserves incremental reports
and a sample MP4, and reaps timed-out process groups. It never alters deployment
settings. See performance.md for metrics, limits and the still-pending 5090 review.
