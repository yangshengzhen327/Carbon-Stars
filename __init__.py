# -*- coding: utf-8 -*-
"""Modular CSpec-DB-Net package."""

from .models.cspec_db_net import CSpecDBNet, WDTransformerCarbonNet
from .models.registry import create_model

__all__ = [
    "CSpecDBNet",
    "WDTransformerCarbonNet",
    "create_model",
    "LAMOSTDataset",
    "build_samples_from_root",
    "main",
    "predict_probs",
    "train_one_epoch",
]


def __getattr__(name):
    if name in {"LAMOSTDataset", "build_samples_from_root"}:
        from . import preprocessing

        return getattr(preprocessing, name)
    if name == "main":
        from .main import main

        return main
    if name in {"predict_probs", "train_one_epoch"}:
        from . import training

        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
