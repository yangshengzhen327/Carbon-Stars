# -*- coding: utf-8 -*-
"""Experiment entry points for SHAP and band occlusion analyses."""

__all__ = [
    "run_shap_experiment",
    "run_band_occlusion_experiment",
]


def run_shap_experiment(argv=None):
    from .run_snr_shap_attribution_experiment import main

    return main(argv)


def run_band_occlusion_experiment(argv=None):
    from .run_snr_band_occlusion_experiment import main

    return main(argv)
