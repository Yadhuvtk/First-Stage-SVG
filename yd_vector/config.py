from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    return data