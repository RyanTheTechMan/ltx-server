# RTX 5090 performance validation

**Status: tooling implemented; hardware measurements and default selection pending.**
No CUDA generation or benchmark was run on the development Mac. Existing FP8,
512-pixel tiling and CPU parking defaults are provisional, not tuned 5090 results.
A successful CPU test suite does not establish GPU fit, output quality or speed.

## Run on the GPU host

Install the locked inference environment, verify the five model hashes, stop the
server, and use a dedicated idle 5090. The benchmark takes the same storage locks
as the server. It runs one fresh process per profile, never simultaneous inference.
Keep host RAM, driver, power settings and background workloads consistent.

```bash
uv sync --locked --extra inference
uv run --extra inference python scripts/download_models.py --verify-only
uv run --extra inference python scripts/smoke_test.py --gpu
uv run --extra inference python scripts/benchmark.py --all --dry-run
uv run --extra inference python scripts/benchmark.py --all --duration 5 --resolution 540p --warmup 1 --repeats 3
```

`--dry-run` works on the Mac: no CUDA imports, subprocesses or files. The default
without `--all` measures only `baseline`. Choose a subset with
`--profiles baseline cpu-offload bf16-cpu`. Each profile reports one cold call,
then the requested warmup calls, then measured calls. Only successful **measured**
calls contribute to the summary. A warmup failure fails that profile. Missing
optional SageAttention is reported as skipped; incompatible kernels are failures.

The runner creates a new timestamped directory under `benchmark-results`, or a new
`--output DIRECTORY`. Existing directories are refused. `plan.json` records the
comparison, per-profile JSON/logs preserve partial progress, and `results.json`
collects outcomes. A validated MP4 from each profile's first measured run is retained
for quality comparison; `--no-keep-videos` disables this. Managed temporary outputs
are removed after each call. A per-profile timeout (default 3600 seconds) terminates
and reaps the worker process group, including its FFmpeg child. Failed profiles do
not prevent subsequent profiles from running; the overall command exits nonzero
if any profile fails. Forced termination may leave managed partials for server cleanup.

## Comparison profiles

Every profile explicitly fixes common controls, overriding corresponding `.env`
values so the comparison cannot silently change. Model paths and device selection
still come from the operator configuration. All runs use the same prompt, seed,
resolution, duration, FPS and generated-audio setting. Use `--prompt` to change the
shared prompt. The current benchmark workload is T2V; repeat hardware smoke tests
for conditioned workloads before deploying any selected settings.

| Profile | Change from baseline |
| --- | --- |
| baseline | FP8, automatic upstream attention, CPU parking between jobs, 512×512 / 80-frame tiles, eager transformer |
| pytorch | Explicit upstream PyTorch SDPA |
| sage | Optional SageAttention in unmasked transformer attention; upstream masked attention remains |
| cpu-offload | Official CPU block streaming; server weight-retention hooks disabled |
| disk-offload | Official disk streaming; server weight-retention hooks disabled |
| bf16-cpu | CPU streaming with FP8 disabled; compare directly with cpu-offload |
| tiles-256 | 256-pixel spatial tiles |
| tiles-768 | 768-pixel spatial tiles |
| compile | Official transformer block compilation, without CUDA graphs/capture |

BF16 with full GPU residency is deliberately not a standard profile: the transformer
weights alone exceed 32 GB. BF16/FP8 comparison therefore uses the same CPU-streaming
mode. `CACHE_TEXT_ENCODER` retention applies only without streaming. IC-LoRA stages
use distinct fused weights and keep upstream's transient transformer lifecycle.

Stage timing synchronization is enabled for benchmarks; ordinary serving leaves it
off. JSON includes end-to-end wall times, stage times, pipeline construction time,
peak torch-allocated/reserved VRAM, process peak RSS, GPU identity/capability, CUDA,
Python/PyTorch versions and pinned LTX/model revisions. CUDA timing is synchronized
at reported stage boundaries; final video conversion also transfers to CPU before
FFmpeg validation. Peak RSS is per worker, not per individual job. Torch allocator
metrics exclude allocations outside torch and other processes; they are not total
NVML GPU usage. Pipeline construction time excludes lazy checkpoint loads, which
appear in stage times. Compilation overhead belongs in cold/warmup measurements.

## Optional SageAttention

The original adapter uses the public `sageattn` API with `tensor_layout="NHD"`
and installs it through upstream `DiffusionStage.with_attention`. No torch-global
patch or silent kernel fallback is installed. The default remains upstream automatic
attention. See the [official SageAttention source](https://github.com/thu-ml/SageAttention/tree/eb615cf6cf4d221338033340ee2de1c37fbdba4a).

SageAttention is intentionally outside the base lock: its compiled extension must
match the host's CUDA/torch environment. On Linux, install the CUDA toolkit with
`nvcc` and build tools first. After the locked inference environment is installed,
an optional source build pinned to the inspected v2.2.0 tag is:

```bash
uv pip install setuptools wheel ninja
uv pip install --no-build-isolation 'git+https://github.com/thu-ml/SageAttention.git@eb615cf6cf4d221338033340ee2de1c37fbdba4a'
.venv/bin/python scripts/benchmark.py --profiles baseline sage --duration 5
```

The upstream instructions require CUDA 12.8 or later for Blackwell. This source build
has not been tested here. Use the virtualenv executable directly after installing
optional kernels so a dependency sync cannot remove them. Record the build/toolkit
and extension version with results. The runner records the installed SageAttention
version for its profile. Benchmark the kernel before enabling `ATTENTION_BACKEND=sage`.

## Choosing deployment defaults

1. Run short smoke tests for T2V, image/keyframes, A2V, installed LoRAs, reference
   video and retake. Check cancellation and recovery after a real OOM.
2. Compare the profiles at 540p; repeat promising ones at 720p/1080p and the longest
   required duration. Start conservatively; acceptance of a request is not a fit guarantee.
3. Inspect retained samples for image detail, temporal consistency, artifacts and
   synchronized audio. Fixed seeds make comparisons repeatable but do not guarantee
   identical pixels between kernels, quantization or tiling settings.
4. Compare warm medians, cold latency, peak VRAM, host RAM and stability. Repeat
   candidates after a restart and during a realistic serial workload.
5. Explicitly set `.env` defaults only after those measurements and quality checks.

Reports always contain `recommended_defaults: null` and `quality_review_required: true`.
The tool never treats synthetic tests, a single fast run or an unreviewed MP4 as a
recommendation, and never edits server configuration. Phase 5 is complete only when
the 5090 results have been reviewed and actual deployment defaults selected.
