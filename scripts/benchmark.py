"""Compare opt-in inference profiles in separate processes on the NVIDIA host."""

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

from ltx_server import __version__
from ltx_server.benchmarking import PROFILES, profile_settings, summarize
from ltx_server.config import Settings
from ltx_server.errors import ServiceError
from ltx_server.inference.backend import GenerationContext
from ltx_server.inference.manager import PipelineManager
from ltx_server.inference.models import LTX_COMMIT, MODEL_REVISION
from ltx_server.media.storage import Storage, new_id
from ltx_server.schemas.generation import GenerationRequest, normalize_request


def write_json(path, data):
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def error_detail(exc):
    if isinstance(exc, ServiceError):
        return exc.detail.model_dump(mode="json")
    return {"code": type(exc).__name__, "message": str(exc)}


async def worker(args):
    output = Path(args.output)
    report = {
        "profile": args.worker,
        "settings": profile_settings(args.worker),
        "server_version": __version__,
        "upstream_commit": LTX_COMMIT,
        "checkpoint_revision": MODEL_REVISION,
        "status": "failed",
        "runs": [],
        "quality_review_required": True,
        "recommended_defaults": None,
    }
    backend = storage = None
    try:
        settings = Settings(**profile_settings(args.worker))
        storage = Storage(settings)
        storage.open()
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; benchmarks require the NVIDIA host")
        torch.cuda.set_device(settings.gpu_device)
        properties = torch.cuda.get_device_properties(settings.gpu_device)
        report["hardware"] = {
            "gpu": properties.name,
            "vram_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(settings.gpu_device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "target_5090": "5090" in properties.name,
        }
        report["packages"] = {}
        for package in ("ltx-core", "ltx-pipelines", "transformers", "sageattention"):
            try:
                distribution = importlib.metadata.distribution(package)
                provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
                report["packages"][package] = {
                    "version": distribution.version,
                    "source_commit": provenance.get("vcs_info", {}).get("commit_id"),
                }
            except importlib.metadata.PackageNotFoundError:
                report["packages"][package] = None
        if args.worker == "sage":
            try:
                report["sageattention_version"] = importlib.metadata.version("sageattention")
            except importlib.metadata.PackageNotFoundError:
                report["status"] = "skipped"
                report["reason"] = "Optional SageAttention is not installed"
                return
        backend = PipelineManager(settings)
        request = GenerationRequest(
            prompt=args.prompt,
            duration=args.duration,
            resolution=args.resolution,
            fps=24,
            seed=42,
            generate_audio=True,
        )
        spec = normalize_request(request, settings, 42)
        report["request"] = spec.model_dump(mode="json")
        # First call is always reported separately, then explicit unmeasured warmups.
        kinds = ["cold"] + ["warmup"] * args.warmup + ["measured"] * args.repeats
        saved = False
        for index, kind in enumerate(kinds):
            identifier = new_id("gen")
            partial = storage.create_partial(identifier)
            metrics = {}
            row = {"kind": kind, "index": index, "status": "failed", "metrics": metrics}
            started = time.perf_counter()
            try:
                result = await backend.generate(
                    GenerationContext(
                        identifier, spec, partial, {}, Event(), lambda stage: None, metrics.update
                    )
                )
                row.update(
                    status="passed",
                    output=result.model_dump(),
                    elapsed_seconds=time.perf_counter() - started,
                )
                if args.keep_videos and kind == "measured" and not saved:
                    destination = output.parent / f"{args.worker}.mp4"
                    await asyncio.to_thread(shutil.copyfile, partial, destination)
                    report["sample_video"] = destination.name
                    saved = True
            except Exception as exc:
                row.update(error=error_detail(exc), elapsed_seconds=time.perf_counter() - started)
                raise
            finally:
                report["runs"].append(row)
                storage.remove("tmp", f"{identifier}.partial")
                report["summary"] = summarize(report["runs"])
                write_json(output, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = error_detail(exc)
    finally:
        try:
            if backend is not None:
                await backend.close()
        except Exception as exc:
            report["cleanup_error"] = str(exc)
            report["status"] = "failed"
        finally:
            if storage is not None:
                storage.close()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["process_peak_rss_mb"] = rss / (
                1024 * 1024 if sys.platform == "darwin" else 1024
            )
            report["summary"] = summarize(report["runs"])
            write_json(output, report)


def run_child(command, log, timeout):
    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=["baseline"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan; no files, subprocesses or CUDA"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--resolution", choices=["540p", "720p", "1080p"], default="540p")
    parser.add_argument(
        "--prompt", default="A red balloon floats over a quiet meadow, birds chirping."
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--timeout", type=float, default=3600, help="Per-profile process timeout in seconds"
    )
    parser.add_argument("--keep-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker", choices=list(PROFILES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not 1 <= args.duration <= 20
        or not 0 <= args.warmup <= 20
        or not 1 <= args.repeats <= 100
        or not args.timeout > 0
    ):
        parser.error("Use duration 1–20, warmup 0–20, repeats 1–100 and a positive timeout")
    if args.worker:
        if args.output is None:
            parser.error("Worker requires output path")
        asyncio.run(worker(args))
        return
    profiles = list(PROFILES) if args.all else list(dict.fromkeys(args.profiles))
    plan = {
        "profiles": {name: profile_settings(name) for name in profiles},
        "duration": args.duration,
        "resolution": args.resolution,
        "cold_runs": 1,
        "warmup_runs": args.warmup,
        "measured_runs": args.repeats,
        "quality_review_required": True,
        "recommended_defaults": None,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output = Path(
        args.output or ("benchmark-results/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    results = []
    for name in profiles:
        target = output / f"{name}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--output",
            str(target),
            "--duration",
            str(args.duration),
            "--resolution",
            args.resolution,
            "--prompt",
            args.prompt,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        if not args.keep_videos:
            command.append("--no-keep-videos")
        print(f"Benchmarking {name}", flush=True)
        try:
            with (output / f"{name}.log").open("w") as log:
                code = run_child(command, log, args.timeout)
            if not target.is_file() or code:
                raise RuntimeError(f"Worker exited with code {code}")
            result = json.loads(target.read_text())
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            result = {"profile": name, "status": "failed", "error": str(exc)}
        results.append(result)
        write_json(
            output / "results.json",
            {"profiles": results, "quality_review_required": True, "recommended_defaults": None},
        )
        print(f"{name}: {result['status']}", flush=True)
    print(f"Results: {output / 'results.json'}")
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
