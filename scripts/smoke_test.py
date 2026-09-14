"""CPU API smoke by default; --gpu validates real T2V or conditioned generation."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from threading import Event
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import ValidationError
from starlette.datastructures import UploadFile

from ltx_server.config import Settings
from ltx_server.errors import ServiceError
from ltx_server.inference.backend import GenerationContext
from ltx_server.inference.manager import PipelineManager
from ltx_server.inference.models import ModelInventory
from ltx_server.media.assets import AssetManager
from ltx_server.media.storage import Storage, new_id
from ltx_server.schemas.generation import GenerationRequest, normalize_request


def api_smoke(url: str) -> None:
    key = Settings().api_key.get_secret_value()

    def request(path: str, method: str = "GET", body: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        with urlopen(
            Request(url.rstrip("/") + path, data=data, headers=headers, method=method), timeout=10
        ) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}

    health = request("/v1/health")
    request("/v1/gpu")
    request("/v1/models")
    request("/v1/queue")
    if health.get("backend_enabled"):
        print(
            json.dumps(
                {
                    "api": "passed",
                    "model_state": health["model_state"],
                    "inference_tested": False,
                    "note": "Use --gpu for a real generation",
                }
            )
        )
        return
    job = request("/v1/generations", "POST", {"prompt": "API smoke test", "duration": 1})
    identifier = job["id"]
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = request(f"/v1/generations/{identifier}")
            if job["status"] == "failed":
                break
            time.sleep(0.05)
        if (job.get("error") or {}).get("code") != "INFERENCE_UNAVAILABLE":
            raise SystemExit(f"Unexpected disabled-backend state: {job['status']}")
        print(json.dumps({"api": "passed", "inference_tested": False}, indent=2))
    finally:
        request(f"/v1/generations/{identifier}", "DELETE")


async def gpu_smoke(args: argparse.Namespace) -> None:
    settings = Settings(warm_model_on_start=False)
    storage = Storage(settings)
    storage.open()  # Refuse to compete with a server using this data directory.
    backend = PipelineManager(settings)
    assets = AssetManager(settings, storage)
    identifier = new_id("gen")
    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; this mode requires the Linux NVIDIA machine")
        print(f"CUDA device: {torch.cuda.get_device_name(settings.gpu_device)}")
        ModelInventory(settings).validate()

        async def upload(path: str | None) -> str | None:
            if path is None:
                return None
            source = await asyncio.to_thread(Path(path).expanduser)
            with await asyncio.to_thread(source.open, "rb") as stream:
                asset = await assets.upload(UploadFile(file=stream, filename=source.name))
            return asset.id

        inputs = {
            "first_frame": await upload(args.first_frame),
            "last_frame": await upload(args.last_frame),
            "audio": await upload(args.audio),
            "reference_video": await upload(args.reference_video),
            "reference_lora": {"id": args.reference_lora} if args.reference_lora else None,
            "loras": [{"id": identifier, "scale": float(scale)} for identifier, scale in args.lora],
            "retake": {
                "video": await upload(args.retake_video),
                "start": args.start,
                "end": args.end,
            }
            if args.retake_video
            else None,
            "keyframes": [
                {"frame": int(frame), "asset_id": await upload(file), "strength": float(strength)}
                for frame, file, strength in args.keyframe
            ],
        }
        path = storage.create_partial(identifier)
        spec = normalize_request(
            GenerationRequest(
                prompt=args.prompt,
                duration=args.duration,
                resolution="540p",
                fps=24,
                seed=42,
                generate_audio=not args.no_audio,
                **inputs,
            ),
            settings,
            42,
        )
        assets.acquire(identifier, spec.assets())
        metrics = {}
        result = await backend.generate(
            GenerationContext(
                identifier,
                spec,
                path,
                {asset: storage.path("assets", f"{asset}.bin") for asset in spec.assets()},
                Event(),
                lambda stage: print(f"Stage: {stage.value}"),
                metrics.update,
            )
        )
        print(
            json.dumps(
                {"gpu_smoke": "passed", "output": result.model_dump(), "metrics": metrics}, indent=2
            )
        )
    finally:
        try:
            await backend.close()
        finally:
            assets.release(identifier)
            for asset in list(assets.records):
                assets.delete(asset)
            storage.remove("tmp", f"{identifier}.partial")
            storage.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--first-frame", help="Local image for frame zero (GPU mode)")
    parser.add_argument("--last-frame", help="Local image for final normalized frame (GPU mode)")
    parser.add_argument("--audio", help="Local source audio (GPU mode)")
    parser.add_argument("--reference-video", help="Local IC-LoRA source (GPU mode)")
    parser.add_argument("--reference-lora", help="Registered IC-LoRA ID (GPU mode)")
    parser.add_argument("--lora", nargs=2, action="append", default=[], metavar=("ID", "SCALE"))
    parser.add_argument("--retake-video", help="Local video matching the requested output grid")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, default=1)
    parser.add_argument(
        "--keyframe",
        nargs=3,
        action="append",
        default=[],
        metavar=("FRAME", "FILE", "STRENGTH"),
        help="Repeat for image targets (GPU mode)",
    )
    parser.add_argument("--duration", type=float, default=1, help="Requested seconds (GPU mode)")
    parser.add_argument(
        "--no-audio", action="store_true", help="Mute output; retain audio conditioning"
    )
    parser.add_argument(
        "--prompt", default="A red balloon floats gently above a quiet meadow, birds chirping."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run CUDA + model + T2V/audio/MP4 checks; stop the server first",
    )
    args = parser.parse_args()
    if args.gpu:
        asyncio.run(gpu_smoke(args))
    else:
        if (
            args.first_frame
            or args.last_frame
            or args.audio
            or args.keyframe
            or args.no_audio
            or args.reference_video
            or args.reference_lora
            or args.lora
            or args.retake_video
        ):
            parser.error("Conditioning options require --gpu")
        api_smoke(args.url)


if __name__ == "__main__":
    try:
        main()
    except ServiceError as exc:
        raise SystemExit(f"{exc.detail.code}: {exc.detail.message}") from None
    except ModuleNotFoundError:
        raise SystemExit(
            "Missing inference dependencies; install the Linux inference extra"
        ) from None
    except HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}; verify server URL and API_KEY") from None
    except (OSError, ValidationError, ValueError) as exc:
        raise SystemExit(f"Invalid smoke-test input: {exc}") from None
