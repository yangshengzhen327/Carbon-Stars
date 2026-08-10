# -*- coding: utf-8 -*-
"""Recall entry point for the CSpec-DB-Net workspace package.

The original recall implementation remains at the workspace root as
``recall.py``. This module exposes it from inside ``cspec_db_net`` without
changing the original script's path-dependent behavior.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


_MODULE: ModuleType | None = None


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_recall_module() -> ModuleType:
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    recall_path = _workspace_root() / "recall.py"
    spec = importlib.util.spec_from_file_location("_cspec_workspace_recall", recall_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load recall module from {recall_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def main() -> None:
    """Run the original recall pipeline."""

    _load_recall_module().main()


def __getattr__(name: str) -> Any:
    return getattr(_load_recall_module(), name)


if __name__ == "__main__":
    main()
