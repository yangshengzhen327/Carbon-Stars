# -*- coding: utf-8 -*-
"""Import-friendly wrappers for interpretability analysis experiments.

The experiment scripts live in the folder named
``interpretability analysis experiments``. This module provides valid Python
function names for importing those entry points.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent / "interpretability analysis experiments"


def _load_script(module_key: str, filename: str) -> ModuleType:
    script_path = EXPERIMENT_DIR / filename
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))

    module_name = f"_cspec_{module_key}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_band_occlusion_experiment(argv=None) -> None:
    module = _load_script("band_occlusion_experiment", "run_snr_band_occlusion_experiment.py")
    return module.main(argv)


def run_shap_experiment(argv=None) -> None:
    module = _load_script("shap_experiment", "run_snr_shap_attribution_experiment.py")
    return module.main(argv)


__all__ = [
    "run_band_occlusion_experiment",
    "run_shap_experiment",
]
