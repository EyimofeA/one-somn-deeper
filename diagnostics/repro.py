"""Reproducibility metadata every training script should save alongside its
run_config: git commit, exact command, environment info, wall-clock, param
count. Call capture() once at the start of a run and dump the result (plus
whatever the caller adds -- checkpoint path, param count, etc.) into
run_config.json / eval_report.json.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return None


def capture() -> dict:
    import torch

    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in (None, ""),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "command": " ".join(sys.argv),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
