import sys
from types import SimpleNamespace

from ltx_server.api.gpu import read_gpu


class NVMLError(Exception):
    pass


def test_nvml_absent_driver(monkeypatch):
    def fail():
        raise NVMLError("private driver details")

    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(NVMLError=NVMLError, nvmlInit=fail))
    info = read_gpu(0)
    assert not info.available and "private" not in info.reason


def test_nvml_partial_metrics_and_shutdown(monkeypatch):
    shutdown = []

    def unsupported(*args):
        raise NVMLError()

    module = SimpleNamespace(
        NVMLError=NVMLError,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdown.append(True),
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda handle: b"RTX 5090",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(
            total=32 * 1024**3,
            used=1024**3,
            free=31 * 1024**3,
        ),
        nvmlDeviceGetTemperature=unsupported,
        NVML_TEMPERATURE_GPU=0,
        nvmlDeviceGetUtilizationRates=unsupported,
    )
    monkeypatch.setitem(sys.modules, "pynvml", module)
    info = read_gpu(0)
    assert info.available and info.vram_total_mb == 32768 and info.vram_used_mb == 1024
    assert info.temperature_c is None and info.utilization_percent is None
    assert shutdown == [True]
