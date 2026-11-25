"""`python -m minesweepml report`: result charts and the README results section.

Everything is read from a bot's metrics database and the saved benchmark
results, so the README only ever shows numbers that a real run produced.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


def machine_info() -> dict:
    """Hardware and software of the current machine."""
    import psutil
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": "none (CPU only)" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
    }
