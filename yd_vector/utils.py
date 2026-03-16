from __future__ import annotations

from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))