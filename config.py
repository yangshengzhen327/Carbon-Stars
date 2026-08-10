# -*- coding: utf-8 -*-
"""Shared constants and argument helpers for CSpec-DB-Net."""

import numpy as np

WAVE_START = 3986.0
WAVE_END = 9100.0
N_PIX = 4096
TARGET_WAVE = np.linspace(WAVE_START, WAVE_END, N_PIX).astype(np.float32)

CARBON_KEYWORD = "carbon"
CARBON_BINARY_LABEL = 0
NON_CARBON_BINARY_LABEL = 1

DEFAULT_TRAINVAL_ROOT = r"D:\deeplearning study\model\dataset_7classes_train"
DEFAULT_TEST_ROOT = r"D:\deeplearning study\model\dataset_7classes_test"
DEFAULT_EXPERIMENT_NAME = "wdcnn_transformer_idx"
DEFAULT_EPOCHS = 42
DEFAULT_BATCH_SIZE = 48
DEFAULT_LR = 6e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42
DEFAULT_CACHE_DATA = True
DEFAULT_PREPROCESS_MODE = "dual_branch"
DEFAULT_THRESHOLD_MODE = "benchmark"
DEFAULT_SELECT_BY = "test_benchmark"
DEFAULT_VAL_RATIO = 0.2
DEFAULT_EARLY_STOP_PATIENCE = 14
DEFAULT_WARMUP_EPOCHS = 4
DEFAULT_MIN_LR_RATIO = 0.02
DEFAULT_EMA_DECAY = 0.9985
DEFAULT_EMA_START_EPOCH = 2
DEFAULT_USE_AMP = True

INDEX_BAND_KEYS = ["c2_4737", "c2_5165", "c2_5635", "cn_7065", "cn_7820"]
INDEX_CONTROL_KEYS = ["ctrl_1", "ctrl_2", "ctrl_3", "ctrl_4"]
INDEX_FEATURE_MODE = "mode3"

def require_index_feature_mode(mode):
    if mode != INDEX_FEATURE_MODE:
        raise ValueError(f"Unsupported index feature mode: {mode}. This script keeps only {INDEX_FEATURE_MODE}.")


def build_index_feature_names(mode=INDEX_FEATURE_MODE):
    require_index_feature_mode(mode)
    names = []
    for prefix in ["band_mean", "band_min", "band_std", "depth", "pseudo_ew", "continuum_balance"]:
        for band in INDEX_BAND_KEYS:
            names.append(f"{prefix}_{band}")

    names.extend(
        [
            "ratio_depth_c2_5165_over_c2_5635",
            "ratio_depth_cn_7065_over_cn_7820",
            "ratio_depth_c2_mean_over_cn_mean",
            "ratio_pseudo_ew_c2_5165_over_c2_5635",
            "ratio_pseudo_ew_c2_sum_over_cn_sum",
            "ratio_depth_c2_max_over_min",
        ]
    )

    for control in INDEX_CONTROL_KEYS:
        names.append(f"{control}_mean")
    for control in INDEX_CONTROL_KEYS:
        names.append(f"{control}_std")

    for band in INDEX_BAND_KEYS:
        names.append(f"prior_band_mean_{band}")
    names.extend(
        [
            "prior_coarse_4100_5200_mean",
            "prior_coarse_5200_7000_mean",
            "prior_coarse_7000_8600_mean",
            "prior_red_minus_blue_slope",
            "prior_global_spread",
            "prior_band_span",
            "prior_band_strength",
        ]
    )

    names.extend(
        [
            "global_mean",
            "global_std",
            "global_max",
            "global_min",
            "global_end_to_end_slope",
            "global_abs_mean",
        ]
    )

    for prefix in ["d1_mean", "d1_abs_mean", "d2_mean", "d2_abs_mean"]:
        for band in INDEX_BAND_KEYS:
            names.append(f"{prefix}_{band}")
    return names

DEFAULT_THRESHOLD_STEPS = 999
DEFAULT_TARGET_METRICS_TEXT = "accuracy:0.971526,precision:0.945767,recall:0.867718,f1:0.905063"

MODEL_CARBON_BANDS = [
    (4620.0, 4742.0),
    (4980.0, 5170.0),
    (5350.0, 5640.0),
    (7065.0, 7190.0),
    (7820.0, 8000.0),
]

AUGMENT_CARBON_BANDS = [
    (4620, 4812),
    (4980, 5235),
    (5350, 5700),
    (7025, 7230),
    (7790, 8040),
    (7680, 7795),
    (8880, 9020),
]

def parse_int_tuple(text, expected_len=None):
    if isinstance(text, (tuple, list)):
        values = tuple(int(v) for v in text)
    else:
        values = tuple(int(v.strip()) for v in str(text).split(",") if v.strip())
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} integers, got {values}")
    return values


def parse_float_tuple(text, expected_len=None):
    if isinstance(text, (tuple, list)):
        values = tuple(float(v) for v in text)
    else:
        values = tuple(float(v.strip()) for v in str(text).split(",") if v.strip())
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {values}")
    return values


def parse_metric_targets(text):
    targets = {}
    if not text:
        return targets
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        key = key.strip().lower()
        if key in {"accuracy", "precision", "recall", "f1", "f2"}:
            targets[key] = float(value.strip())
    return targets

def wave_to_index(a, b):
    idx_a = int(round((a - WAVE_START) / (WAVE_END - WAVE_START) * (N_PIX - 1)))
    idx_b = int(round((b - WAVE_START) / (WAVE_END - WAVE_START) * (N_PIX - 1)))
    idx_a = max(0, min(N_PIX - 1, idx_a))
    idx_b = max(0, min(N_PIX, idx_b))
    if idx_b <= idx_a:
        idx_b = min(N_PIX, idx_a + 1)
    return idx_a, idx_b


CARBON_BAND_IDXS = [wave_to_index(a, b) for a, b in AUGMENT_CARBON_BANDS]
