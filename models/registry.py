# -*- coding: utf-8 -*-
"""Unified model registry for CSpec-DB-Net and comparison baselines."""

import json

import torch.nn as nn

from .convnext_1d import CONVNEXT1D
from .cspec_db_net import CSpecDBNet
from .efficientnetv2_1d import EfficientNetV2_1D
from .rac_net import RACNET
from .sscnn import SSCNN
from .wdcnn1d import WDCNN1D


MODEL_NAMES = ("cspec_db_net", "convnext1d", "efficientnetv2_1d", "rac_net", "sscnn", "wdcnn1d")
ALIASES = {
    "cspec": "cspec_db_net",
    "convnext": "convnext1d",
    "efficientnetv2": "efficientnetv2_1d",
    "racnet": "rac_net",
    "1d_sscnn": "sscnn",
    "wdcnn": "wdcnn1d",
}


def normalize_model_name(name):
    normalized = str(name).strip().lower().replace("-", "_")
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
    return normalized


def parse_model_kwargs(text):
    if not text:
        return {}
    if isinstance(text, dict):
        return dict(text)
    text = str(text).strip()
    if text.startswith("{"):
        return json.loads(text)
    result = {}
    for part in text.split(","):
        key, value = part.split("=", 1)
        result[key.strip()] = json.loads(value.strip())
    return result


class UnifiedBinaryModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_idx=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return {"bin_logits": self.model(x)}


def create_baseline_model(name, signal_len=4096, **kwargs):
    name = normalize_model_name(name)
    constructors = {
        "convnext1d": CONVNEXT1D,
        "efficientnetv2_1d": EfficientNetV2_1D,
        "rac_net": RACNET,
        "sscnn": SSCNN,
        "wdcnn1d": WDCNN1D,
    }
    if name == "cspec_db_net":
        raise ValueError("create_baseline_model does not construct CSpecDBNet")
    model = constructors[name](in_channel=1, out_channel=2, spectrum_size=signal_len, **kwargs)
    return UnifiedBinaryModel(model)


def create_model(name="cspec_db_net", signal_len=4096, **kwargs):
    name = normalize_model_name(name)
    if name == "cspec_db_net":
        return CSpecDBNet(signal_len=signal_len, **kwargs)
    return create_baseline_model(name, signal_len=signal_len, **kwargs)


__all__ = [
    "MODEL_NAMES",
    "UnifiedBinaryModel",
    "create_baseline_model",
    "create_model",
    "normalize_model_name",
    "parse_model_kwargs",
]
