import argparse
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ltx_server.benchmarking import profile_settings, summarize
from ltx_server.config import Settings
from ltx_server.jobs.state import OutputInfo
from scripts import benchmark


def test_profiles_and_summary_exclude_cold_warmup_and_failures():
    assert profile_settings("bf16-cpu")["use_fp8"] is False
    assert profile_settings("cpu-offload")["use_fp8"] is True
    assert profile_settings("bf16-cpu")["offload_mode"] == "cpu"
    for profile in ("baseline", "cpu-offload", "compile"):
        assert profile_settings(profile)["synchronize_timings"]
    rows = [
        dict(kind=kind, status=status, elapsed_seconds=seconds, metrics={"peak_vram_mb": 10})
        for kind, status, seconds in [
            ("cold", "passed", 100),
            ("warmup", "passed", 30),
            ("measured", "passed", 10),
            ("measured", "passed", 20),
            ("measured", "failed", 1000),
        ]
    ]
    assert summarize(rows)["median_seconds"] == 15
    assert summarize(rows)["measured_runs"] == 2
    assert summarize([]) == {"measured_runs": 0}


@pytest.mark.parametrize("failure", [False, True])
async def test_benchmark_worker_report_and_cleanup(tmp_path, monkeypatch, failure):
    settings = Settings(_env_file=None, data_dir=tmp_path.resolve() / "data")
    monkeypatch.setattr(benchmark, "Settings", lambda **kw: settings)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda="test"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                set_device=lambda d: None,
                get_device_properties=lambda d: SimpleNamespace(
                    name="Test GPU", total_memory=32 * 1024**3
                ),
                get_device_capability=lambda d: (12, 0),
            ),
        ),
    )
    events = []

    class Backend:
        def __init__(self, settings):
            pass

        async def generate(self, context):
            events.append("generate")
            context.partial_path.write_bytes(b"transport fixture")
            if failure:
                raise RuntimeError("model failure")
            context.report_metrics({"peak_vram_mb": 100, "peak_reserved_vram_mb": 150})
            return OutputInfo(
                width=context.spec.width,
                height=context.spec.height,
                frames=context.spec.frames,
                fps=context.spec.fps,
                duration=context.spec.frames / context.spec.fps,
                has_audio=True,
            )

        async def close(self):
            events.append("close")

    monkeypatch.setattr(benchmark, "PipelineManager", Backend)
    path = tmp_path / "baseline.json"
    args = argparse.Namespace(
        worker="baseline",
        output=str(path),
        duration=1,
        resolution="540p",
        prompt="test",
        warmup=1,
        repeats=2,
        keep_videos=True,
    )
    await benchmark.worker(args)
    report = json.loads(path.read_text())
    assert report["recommended_defaults"] is None and report["quality_review_required"]
    assert report["hardware"]["target_5090"] is False
    assert report["status"] == ("failed" if failure else "passed")
    assert events[-1] == "close"
    assert not list(settings.temp_dir.glob("*.partial"))
    if not failure:
        assert [row["kind"] for row in report["runs"]] == ["cold", "warmup", "measured", "measured"]
        assert report["summary"]["measured_runs"] == 2
        assert (tmp_path / "baseline.mp4").is_file()


def test_benchmark_dry_run_requires_no_gpu_or_files(tmp_path):
    result = subprocess.run(
        [sys.executable, str(benchmark.__file__), "--all", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert "sage" in report["profiles"] and "compile" in report["profiles"]
    assert list(tmp_path.iterdir()) == []


def test_benchmark_timeout_reaps_child(tmp_path):
    with (tmp_path / "child.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            benchmark.run_child([sys.executable, "-c", "import time; time.sleep(30)"], log, 0.05)
