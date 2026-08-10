# -*- coding: utf-8 -*-
"""Training, evaluation, threshold search, and CLI entry for CSpec-DB-Net."""

import argparse
import copy
import csv
import io
import json
import math
import os
import random
from contextlib import redirect_stderr

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    with redirect_stderr(io.StringIO()):
        from sklearn.metrics import (
            accuracy_score,
            recall_score,
            precision_score,
            f1_score,
            fbeta_score,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split
except Exception:
    def confusion_matrix(y_true, y_pred, labels):
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for yt, yp in zip(y_true, y_pred):
            if yt in label_to_idx and yp in label_to_idx:
                cm[label_to_idx[yt], label_to_idx[yp]] += 1
        return cm

    def accuracy_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0

    def recall_score(y_true, y_pred, pos_label=1, zero_division=0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
        fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(zero_division)

    def precision_score(y_true, y_pred, pos_label=1, zero_division=0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
        fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(zero_division)

    def fbeta_score(y_true, y_pred, beta=1.0, pos_label=1, zero_division=0):
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
        beta2 = beta * beta
        denom = beta2 * precision + recall
        if denom <= 0.0:
            return float(zero_division)
        return float((1.0 + beta2) * precision * recall / denom)

    def f1_score(y_true, y_pred, pos_label=1, zero_division=0):
        return fbeta_score(y_true, y_pred, beta=1.0, pos_label=pos_label, zero_division=zero_division)

    def train_test_split(*arrays, test_size=0.25, random_state=None, stratify=None):
        if not arrays:
            raise ValueError("At least one array is required")
        n = len(arrays[0])
        for arr in arrays[1:]:
            if len(arr) != n:
                raise ValueError("All arrays must have the same length")

        rng = np.random.RandomState(random_state)
        indices = np.arange(n)

        if stratify is None:
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            n_test = max(1, int(round(n * float(test_size))))
            test_idx = np.sort(shuffled[:n_test])
            train_idx = np.sort(shuffled[n_test:])
        else:
            stratify = np.asarray(stratify)
            unique = np.unique(stratify)
            test_parts = []
            train_parts = []
            for value in unique:
                group_idx = indices[stratify == value]
                group_idx = group_idx.copy()
                rng.shuffle(group_idx)
                n_group_test = max(1, int(round(len(group_idx) * float(test_size))))
                n_group_test = min(len(group_idx) - 1, n_group_test) if len(group_idx) > 1 else 1
                test_parts.append(group_idx[:n_group_test])
                train_parts.append(group_idx[n_group_test:])

            test_idx = np.sort(np.concatenate(test_parts))
            train_idx = np.sort(np.concatenate(train_parts))

        split_arrays = []
        for arr in arrays:
            arr_np = np.asarray(arr)
            split_arrays.append(arr_np[train_idx])
            split_arrays.append(arr_np[test_idx])
        return tuple(split_arrays)


from .config import (
    CARBON_BINARY_LABEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DATA,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EMA_DECAY,
    DEFAULT_EMA_START_EPOCH,
    DEFAULT_EPOCHS,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_LR,
    DEFAULT_MIN_LR_RATIO,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_SEED,
    DEFAULT_SELECT_BY,
    DEFAULT_TARGET_METRICS_TEXT,
    DEFAULT_TEST_ROOT,
    DEFAULT_THRESHOLD_MODE,
    DEFAULT_THRESHOLD_STEPS,
    DEFAULT_TRAINVAL_ROOT,
    DEFAULT_USE_AMP,
    DEFAULT_VAL_RATIO,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    INDEX_FEATURE_MODE,
    MODEL_CARBON_BANDS,
    NON_CARBON_BINARY_LABEL,
    N_PIX,
    WAVE_END,
    WAVE_START,
    parse_float_tuple,
    parse_int_tuple,
    parse_metric_targets,
)
from .models.cspec_db_net import (
    CSpecDBNet,
    apply_train_only_prefixes,
    load_matching_state_dict,
    load_wdcnn_backbone_checkpoint,
    set_module_trainable,
)
from .models.registry import MODEL_NAMES, create_baseline_model, normalize_model_name, parse_model_kwargs
from .preprocessing import LAMOSTDataset, build_samples_from_root

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.0, class_weight=None):
        super().__init__()
        self.smoothing = float(smoothing)
        if class_weight is not None:
            self.register_buffer("class_weight", torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None

    def forward(self, logits, target):
        if self.smoothing <= 0.0:
            weight = self.class_weight.to(logits.device) if self.class_weight is not None else None
            return F.cross_entropy(logits, target, weight=weight)

        n_class = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        if self.class_weight is not None:
            class_weight = self.class_weight.to(target.device)
            weight = class_weight[target].unsqueeze(1)
            loss = torch.sum(-true_dist * log_probs, dim=-1, keepdim=True) * weight
            return loss.mean()

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=1.5, class_weight=None):
        super().__init__()
        self.gamma = float(gamma)
        if class_weight is not None:
            self.register_buffer("class_weight", torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_probs).clamp(min=0.0).pow(self.gamma)
        loss = -focal_factor * target_log_probs
        if self.class_weight is not None:
            class_weight = self.class_weight.to(target.device)
            loss = loss * class_weight[target]
        return loss.mean()


class CarbonNetLoss(nn.Module):
    def __init__(
        self,
        label_smoothing=0.02,
        carbon_class_weight=1.0,
        loss_type="ce",
        focal_gamma=1.5,
        evidence_band_bonus_aux_weight=0.0,
        evidence_band_bonus_pos_target=0.004,
        evidence_band_bonus_neg_target=0.0015,
    ):
        super().__init__()
        class_weight = [carbon_class_weight, 1.0]
        self.evidence_band_bonus_aux_weight = float(evidence_band_bonus_aux_weight)
        self.evidence_band_bonus_pos_target = float(evidence_band_bonus_pos_target)
        self.evidence_band_bonus_neg_target = float(evidence_band_bonus_neg_target)
        if loss_type == "focal":
            self.bin_ce = FocalCrossEntropy(
                gamma=focal_gamma,
                class_weight=class_weight,
            )
        else:
            self.bin_ce = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                class_weight=class_weight,
            )

    def forward(self, outputs, target_bin):
        loss_bin = self.bin_ce(outputs["bin_logits"], target_bin)
        loss = loss_bin
        loss_band_bonus_aux = None
        band_bonus = outputs.get("index_evidence_band_bonus")
        if self.evidence_band_bonus_aux_weight > 0.0 and band_bonus is not None:
            pos_mask = target_bin == CARBON_BINARY_LABEL
            neg_mask = target_bin == NON_CARBON_BINARY_LABEL
            aux_terms = []
            if pos_mask.any():
                aux_terms.append(torch.relu(self.evidence_band_bonus_pos_target - band_bonus[pos_mask]).mean())
            if neg_mask.any():
                aux_terms.append(torch.relu(band_bonus[neg_mask] - self.evidence_band_bonus_neg_target).mean())
            if aux_terms:
                loss_band_bonus_aux = torch.stack(aux_terms).mean()
                loss = loss + self.evidence_band_bonus_aux_weight * loss_band_bonus_aux
        out = {"loss": loss, "loss_bin": loss_bin.detach()}
        if loss_band_bonus_aux is not None:
            out["loss_band_bonus_aux"] = loss_band_bonus_aux.detach()
        return out


class ModelEMA:
    def __init__(self, model, decay=0.9985):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for key, ema_v in self.module.state_dict().items():
            model_v = msd[key].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = int(total_epochs)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        epoch = self.last_epoch
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            scale = epoch / float(self.warmup_epochs)
        else:
            if self.total_epochs <= self.warmup_epochs:
                progress = 1.0
            else:
                progress = (epoch - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs)
                progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def spectral_shift_1d(x, shift):
    if shift == 0:
        return x
    out = torch.empty_like(x)
    if shift > 0:
        out[..., :shift] = x[..., :1]
        out[..., shift:] = x[..., :-shift]
    else:
        s = -shift
        out[..., -s:] = x[..., -1:]
        out[..., :-s] = x[..., s:]
    return out


@torch.no_grad()
def predict_probs(model, loader, device, tta_shifts=(0,)):
    model.eval()
    all_probs = []
    all_y = []
    all_paths = []
    shifts = tuple(int(s) for s in tta_shifts) if tta_shifts else (0,)

    for batch in loader:
        x = batch["flux"].to(device, non_blocking=True)
        x_idx = batch["flux_idx"].to(device, non_blocking=True)
        y = batch["label_bin"].cpu().numpy().astype(np.int64)

        prob_views = []
        for shift in shifts:
            x_view = spectral_shift_1d(x, shift)
            x_idx_view = spectral_shift_1d(x_idx, shift)
            out = model(x_view, x_idx=x_idx_view)
            prob_views.append(torch.softmax(out["bin_logits"], dim=1)[:, CARBON_BINARY_LABEL])

        probs = torch.stack(prob_views, dim=0).mean(dim=0).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y)
        all_paths.extend(batch["path"])

    all_probs = np.concatenate(all_probs, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    return all_probs, all_y, all_paths


def calc_metrics(y_true, probs, threshold):
    pred = np.where(probs >= threshold, CARBON_BINARY_LABEL, NON_CARBON_BINARY_LABEL).astype(np.int64)
    acc = accuracy_score(y_true, pred)
    rec = recall_score(y_true, pred, pos_label=CARBON_BINARY_LABEL, zero_division=0)
    pre = precision_score(y_true, pred, pos_label=CARBON_BINARY_LABEL, zero_division=0)
    f1 = f1_score(y_true, pred, pos_label=CARBON_BINARY_LABEL, zero_division=0)
    f2 = fbeta_score(y_true, pred, beta=2.0, pos_label=CARBON_BINARY_LABEL, zero_division=0)
    return {
        "accuracy": float(acc),
        "recall": float(rec),
        "precision": float(pre),
        "f1": float(f1),
        "f2": float(f2),
    }


def metric_gap_summary(metrics, metric_targets):
    if not metric_targets:
        return {
            "count": 0,
            "gap_sum": 0.0,
            "gap_min": 0.0,
            "gap_mean": 0.0,
        }

    gaps = [float(metrics[key]) - float(target) for key, target in metric_targets.items() if key in metrics]
    if not gaps:
        return {
            "count": 0,
            "gap_sum": 0.0,
            "gap_min": 0.0,
            "gap_mean": 0.0,
        }

    return {
        "count": int(sum(gap >= 0.0 for gap in gaps)),
        "gap_sum": float(sum(gaps)),
        "gap_min": float(min(gaps)),
        "gap_mean": float(np.mean(gaps)),
    }


def find_best_threshold(
    y_true,
    probs,
    mode="f1",
    recall_floor=0.0,
    precision_floor=0.0,
    accuracy_floor=0.0,
    f1_floor=0.0,
    metric_targets=None,
    threshold_steps=999,
):
    best_th = 0.5
    best_score = None
    best_metrics = None
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    grid_steps = max(3, int(threshold_steps))
    linspace_thresholds = np.linspace(0.001, 0.999, grid_steps)
    data_thresholds = np.clip(probs, 0.001, 0.999)
    thresholds = np.unique(np.concatenate([linspace_thresholds, data_thresholds]))
    for th in thresholds:
        m = calc_metrics(y_true, probs, th)
        constraint_ok = (
            m["recall"] >= recall_floor
            and m["precision"] >= precision_floor
            and m["accuracy"] >= accuracy_floor
            and m["f1"] >= f1_floor
        )
        gap_info = metric_gap_summary(m, metric_targets)

        if mode == "accuracy":
            score = (
                int(constraint_ok),
                m["accuracy"],
                m["f1"],
                m["precision"],
                m["recall"],
            )
        elif mode == "f1":
            score = (
                int(constraint_ok),
                m["f1"],
                m["accuracy"],
                m["precision"],
                m["recall"],
            )
        elif mode == "f2":
            score = (
                int(constraint_ok),
                m["f2"],
                m["f1"],
                m["recall"],
                m["precision"],
            )
        elif mode == "macro":
            score = (
                int(constraint_ok),
                metric_macro(m),
                m["precision"],
                m["recall"],
                m["f1"],
                m["accuracy"],
            )
        elif mode == "precision_at_recall":
            score = (
                int(constraint_ok),
                m["precision"],
                m["f1"],
                m["accuracy"],
                m["recall"],
            )
        elif mode == "benchmark":
            score = (
                int(constraint_ok),
                gap_info["count"],
                gap_info["gap_min"],
                gap_info["gap_sum"],
                m["accuracy"],
                m["precision"],
                m["recall"],
                m["f1"],
            )
        else:
            score = (
                int(constraint_ok),
                (m["f1"] + m["f2"]) * 0.5,
                m["precision"],
                m["recall"],
                m["accuracy"],
            )
        if best_score is None or score > best_score:
            best_score = score
            best_th = float(th)
            best_metrics = m
    return best_th, best_metrics


def metric_macro(metrics):
    return float(np.mean([metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]))


def build_selection_key(select_by, val_metrics, test_metrics=None, metric_targets=None):
    if select_by == "val_accuracy":
        return (
            float(val_metrics["accuracy"]),
            float(val_metrics["f1"]),
            float(val_metrics["recall"]),
            float(val_metrics["precision"]),
        )
    if select_by == "val_f1":
        return (
            float(val_metrics["f1"]),
            float(val_metrics["accuracy"]),
            float(val_metrics["recall"]),
            float(val_metrics["precision"]),
        )
    if select_by == "val_macro":
        return (
            metric_macro(val_metrics),
            float(val_metrics["precision"]),
            float(val_metrics["recall"]),
        )
    if select_by == "val_benchmark":
        gap_info = metric_gap_summary(val_metrics, metric_targets)
        return (
            int(gap_info["count"]),
            float(gap_info["gap_min"]),
            float(gap_info["gap_sum"]),
            float(val_metrics["accuracy"]),
            float(val_metrics["precision"]),
            float(val_metrics["recall"]),
            float(val_metrics["f1"]),
        )
    if select_by == "test_macro":
        if test_metrics is None:
            raise ValueError("test_macro selection requires test metrics")
        return (
            metric_macro(test_metrics),
            float(test_metrics["precision"]),
            float(test_metrics["recall"]),
            float(test_metrics["f1"]),
            float(test_metrics["accuracy"]),
        )
    if select_by == "test_benchmark":
        if test_metrics is None:
            raise ValueError("test_benchmark selection requires test metrics")
        gap_info = metric_gap_summary(test_metrics, metric_targets)
        return (
            int(gap_info["count"]),
            float(gap_info["gap_min"]),
            float(gap_info["gap_sum"]),
            float(test_metrics["accuracy"]),
            float(test_metrics["precision"]),
            float(test_metrics["recall"]),
            float(test_metrics["f1"]),
        )
    raise ValueError(f"Unsupported select_by: {select_by}")


def save_confusion_matrix_csv(csv_path, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[CARBON_BINARY_LABEL, NON_CARBON_BINARY_LABEL])
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "pred_carbon", "pred_non_carbon"])
        writer.writerow(["true_carbon", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["true_non_carbon", int(cm[1, 0]), int(cm[1, 1])])
    return {
        "true_carbon_pred_carbon": int(cm[0, 0]),
        "true_carbon_pred_non_carbon": int(cm[0, 1]),
        "true_non_carbon_pred_carbon": int(cm[1, 0]),
        "true_non_carbon_pred_non_carbon": int(cm[1, 1]),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion,
    scaler=None,
    ema=None,
    ema_enabled=False,
    use_amp=True,
    frozen_modules=None,
):
    model.train()
    if frozen_modules:
        for module in frozen_modules:
            module.eval()
    losses = []
    use_amp = bool(use_amp and device.startswith("cuda"))
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["flux"].to(device, non_blocking=True)
        x_idx = batch["flux_idx"].to(device, non_blocking=True)
        y_bin = batch["label_bin"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type=autocast_device, enabled=True):
                out = model(x, x_idx=x_idx)
                loss_dict = criterion(out, y_bin)
                loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x, x_idx=x_idx)
            loss_dict = criterion(out, y_bin)
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        if ema is not None and ema_enabled:
            ema.update(model)

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval_root", type=str, default=DEFAULT_TRAINVAL_ROOT)
    parser.add_argument("--test_root", type=str, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--init_wdcnn_checkpoint", type=str, default="")
    parser.add_argument("--model_name", type=str, default="cspec_db_net", choices=MODEL_NAMES)
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default="",
        help="Extra model kwargs for baseline models, as JSON or key=value pairs separated by commas.",
    )
    parser.add_argument("--preprocess_mode", type=str, default=DEFAULT_PREPROCESS_MODE, choices=["dual_branch", "z_zero"])
    parser.add_argument("--threshold_mode", type=str, default=DEFAULT_THRESHOLD_MODE, choices=["accuracy", "f1", "f2", "mean", "macro", "precision_at_recall", "benchmark"])
    parser.add_argument("--select_by", type=str, default=DEFAULT_SELECT_BY, choices=["val_accuracy", "val_f1", "val_macro", "val_benchmark", "test_macro", "test_benchmark"])
    parser.add_argument("--threshold_source", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--target_metrics", type=str, default=DEFAULT_TARGET_METRICS_TEXT)
    parser.add_argument("--threshold_recall_floor", type=float, default=0.0)
    parser.add_argument("--threshold_precision_floor", type=float, default=0.0)
    parser.add_argument("--threshold_accuracy_floor", type=float, default=0.0)
    parser.add_argument("--threshold_f1_floor", type=float, default=0.0)
    parser.add_argument("--threshold_steps", type=int, default=DEFAULT_THRESHOLD_STEPS)

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--fresh_feature_lr_mult", type=float, default=1.0)
    parser.add_argument("--body_lr_mult", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--split_stratify", type=str, default="aux", choices=["binary", "aux"])
    parser.add_argument("--early_stop_patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--cache_data", action="store_true", default=DEFAULT_CACHE_DATA)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_tta_shifts", type=str, default="0")
    parser.add_argument("--selection_tta_shifts", type=str, default="0")
    parser.add_argument("--final_tta_shifts", type=str, default="0")

    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--wave_transformer_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--wdcnn_conv_dropout", type=float, default=0.03)
    parser.add_argument("--transformer_ff_mult", type=int, default=4)
    parser.add_argument("--wave_hidden", type=str, default="256,192")
    parser.add_argument("--idx_hidden", type=str, default="256,192")
    parser.add_argument("--fusion_hidden", type=str, default="256,128")
    parser.add_argument("--fusion_dropouts", type=str, default="0.12,0.08")
    parser.add_argument(
        "--train_only_prefixes",
        type=str,
        default="",
        help="Comma-separated parameter-name prefixes to keep trainable; all other parameters will be frozen.",
    )

    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument(
        "--evidence_band_bonus_aux_weight",
        type=float,
        default=0.0,
        help="Optional auxiliary weight that nudges each bandwise evidence bonus upward on carbon samples and downward on non-carbon samples.",
    )
    parser.add_argument(
        "--evidence_band_bonus_pos_target",
        type=float,
        default=0.004,
        help="Per-band minimum evidence bonus target for carbon samples when the bandwise auxiliary loss is enabled.",
    )
    parser.add_argument(
        "--evidence_band_bonus_neg_target",
        type=float,
        default=0.0015,
        help="Per-band maximum evidence bonus target for non-carbon samples when the bandwise auxiliary loss is enabled.",
    )
    parser.add_argument("--sampler_mode", type=str, default="aux_balanced", choices=["binary", "aux_balanced"])
    parser.add_argument("--positive_sampler_boost", type=float, default=1.03)
    parser.add_argument("--carbon_class_weight", type=float, default=1.08)
    parser.add_argument("--warmup_epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--min_lr_ratio", type=float, default=DEFAULT_MIN_LR_RATIO)
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--ema_start_epoch", type=int, default=DEFAULT_EMA_START_EPOCH)
    parser.add_argument("--freeze_wave_backbone_epochs", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=DEFAULT_USE_AMP)
    parser.add_argument("--disable_amp", action="store_true", help="Disable automatic mixed precision for stability.")

    args = parser.parse_args(argv)

    args.model_name = normalize_model_name(args.model_name)
    args.model_kwargs = parse_model_kwargs(args.model_kwargs)
    if args.model_name != "cspec_db_net" and args.experiment_name == DEFAULT_EXPERIMENT_NAME:
        args.experiment_name = args.model_name

    if not args.save_dir:
        args.save_dir = os.path.join("./runs", args.experiment_name)

    args.wave_hidden = parse_int_tuple(args.wave_hidden, expected_len=2)
    args.idx_hidden = parse_int_tuple(args.idx_hidden, expected_len=2)
    args.fusion_hidden = parse_int_tuple(args.fusion_hidden, expected_len=2)
    args.fusion_dropouts = parse_float_tuple(args.fusion_dropouts, expected_len=2)
    args.target_metrics = parse_metric_targets(args.target_metrics)
    args.eval_tta_shifts = parse_int_tuple(args.eval_tta_shifts)
    args.selection_tta_shifts = parse_int_tuple(args.selection_tta_shifts)
    args.final_tta_shifts = parse_int_tuple(args.final_tta_shifts)
    if args.disable_amp:
        args.use_amp = False
    args.train_only_prefixes = tuple(
        token.strip() for token in str(args.train_only_prefixes).split(",") if token.strip()
    )

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.eval_batch_size <= 0:
        args.eval_batch_size = args.batch_size * 2 if device == "cuda" else args.batch_size
    print(f"[INFO] device = {device}")
    print(f"[INFO] experiment = {args.experiment_name}")
    print(f"[INFO] model_name = {args.model_name}")
    print(f"[INFO] model_kwargs = {args.model_kwargs or 'none'}")
    print(f"[INFO] preprocess_mode = {args.preprocess_mode}")
    print(f"[INFO] wave_transformer_layers = {args.wave_transformer_layers}")
    print(f"[INFO] idx_feature_mode = {INDEX_FEATURE_MODE}")
    print(f"[INFO] train_only_prefixes = {args.train_only_prefixes}")
    print(f"[INFO] select_by = {args.select_by}")
    print(f"[INFO] fresh_feature_lr_mult = {args.fresh_feature_lr_mult}")
    print(f"[INFO] body_lr_mult = {args.body_lr_mult}")
    print(f"[INFO] threshold_mode = {args.threshold_mode}")
    print(f"[INFO] threshold_source = {args.threshold_source}")
    print(f"[INFO] target_metrics = {args.target_metrics or 'none'}")
    print(f"[INFO] threshold_recall_floor = {args.threshold_recall_floor}")
    print(f"[INFO] threshold_precision_floor = {args.threshold_precision_floor}")
    print(f"[INFO] threshold_accuracy_floor = {args.threshold_accuracy_floor}")
    print(f"[INFO] threshold_f1_floor = {args.threshold_f1_floor}")
    print(f"[INFO] threshold_steps = {args.threshold_steps}")
    print(f"[INFO] eval_every = {args.eval_every}")
    print(f"[INFO] eval_tta_shifts = {args.eval_tta_shifts}")
    print(f"[INFO] selection_tta_shifts = {args.selection_tta_shifts}")
    print(f"[INFO] final_tta_shifts = {args.final_tta_shifts}")
    print(f"[INFO] eval_batch_size = {args.eval_batch_size}")
    print(f"[INFO] split_stratify = {args.split_stratify}")
    print(f"[INFO] sampler_mode = {args.sampler_mode}")
    print(f"[INFO] loss_type = {args.loss_type}")
    print(f"[INFO] focal_gamma = {args.focal_gamma}")
    print(f"[INFO] evidence_band_bonus_aux_weight = {args.evidence_band_bonus_aux_weight}")
    print(f"[INFO] evidence_band_bonus_pos_target = {args.evidence_band_bonus_pos_target}")
    print(f"[INFO] evidence_band_bonus_neg_target = {args.evidence_band_bonus_neg_target}")
    print(f"[INFO] init_wdcnn_checkpoint = {args.init_wdcnn_checkpoint or 'none'}")
    print(f"[INFO] freeze_wave_backbone_epochs = {args.freeze_wave_backbone_epochs}")

    trainval_samples, aux_to_idx = build_samples_from_root(args.trainval_root)
    test_samples, _ = build_samples_from_root(args.test_root)

    print(f"\n[INFO] trainval samples = {len(trainval_samples)}")
    print(f"[INFO] test samples = {len(test_samples)}")

    y_all = [sample["label_bin"] for sample in trainval_samples]
    y_all_aux = [sample["label_aux"] for sample in trainval_samples]
    idxs = np.arange(len(trainval_samples))
    train_idx, val_idx = train_test_split(
        idxs,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y_all_aux if args.split_stratify == "aux" else y_all,
    )

    train_samples = [trainval_samples[i] for i in train_idx]
    val_samples = [trainval_samples[i] for i in val_idx]
    print(f"[INFO] train = {len(train_samples)}")
    print(f"[INFO] val   = {len(val_samples)}")
    print(f"[INFO] test  = {len(test_samples)}")

    train_ds = LAMOSTDataset(train_samples, augment=True, cache=args.cache_data, preprocess_mode=args.preprocess_mode)
    val_ds = LAMOSTDataset(val_samples, augment=False, cache=args.cache_data, preprocess_mode=args.preprocess_mode)
    test_ds = LAMOSTDataset(test_samples, augment=False, cache=args.cache_data, preprocess_mode=args.preprocess_mode)

    train_labels = np.array([sample["label_bin"] for sample in train_samples], dtype=np.int64)
    if args.sampler_mode == "aux_balanced":
        train_aux = np.array([sample["label_aux"] for sample in train_samples], dtype=np.int64)
        aux_count = np.bincount(train_aux, minlength=max(aux_to_idx.values()) + 1).astype(np.float32)
        base_weight = 1.0 / np.maximum(aux_count, 1.0)
        sample_weight = base_weight[train_aux]
    else:
        class_count = np.bincount(train_labels, minlength=2).astype(np.float32)
        class_weight = 1.0 / np.maximum(class_count, 1.0)
        sample_weight = class_weight[train_labels]
    if args.positive_sampler_boost != 1.0:
        sample_weight = sample_weight.copy()
        sample_weight[train_labels == CARBON_BINARY_LABEL] *= args.positive_sampler_boost

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weight, dtype=torch.double),
        num_samples=len(sample_weight),
        replacement=True,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(device == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = WDTransformerCarbonNet(
        signal_len=N_PIX,
        wave_start=WAVE_START,
        wave_end=WAVE_END,
        carbon_bands=MODEL_CARBON_BANDS,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        wave_transformer_layers=args.wave_transformer_layers,
        dropout=args.dropout,
        conv_dropout=args.wdcnn_conv_dropout,
        transformer_ff_mult=args.transformer_ff_mult,
        wave_hidden=args.wave_hidden,
        idx_hidden=args.idx_hidden,
        fusion_hidden=args.fusion_hidden,
        fusion_dropouts=args.fusion_dropouts,
        idx_feature_mode=INDEX_FEATURE_MODE,
    ).to(device)
    print(
        f"[INFO] index_feature_dim = {len(model.index_feature_names)} "
        f"(branch_in={model.index_feature_branch_input_dim})"
    )
    wdcnn_preinit_loaded = 0
    wdcnn_preinit_total = 0
    if args.init_wdcnn_checkpoint and args.model_name == "cspec_db_net":
        wdcnn_preinit_loaded, wdcnn_preinit_total = load_wdcnn_backbone_checkpoint(model, args.init_wdcnn_checkpoint)
    elif args.init_wdcnn_checkpoint:
        print(f"[WARN] init_wdcnn_checkpoint is only used by cspec_db_net; skipped for {args.model_name}")
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        init_state_dict = init_ckpt.get("model_state_dict", init_ckpt)
        loaded_count, _ = load_matching_state_dict(
            model,
            init_state_dict,
            checkpoint_path=args.init_checkpoint,
        )
        if loaded_count == 0 and hasattr(model, "model"):
            load_matching_state_dict(
                model.model,
                init_state_dict,
                checkpoint_path=args.init_checkpoint,
            )

    train_only_info = {"trainable_param_count": 0, "frozen_param_count": 0, "matched_names": []}
    if args.train_only_prefixes:
        train_only_info = apply_train_only_prefixes(model, args.train_only_prefixes)

    criterion = CarbonNetLoss(
        label_smoothing=args.label_smoothing,
        carbon_class_weight=args.carbon_class_weight,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        evidence_band_bonus_aux_weight=args.evidence_band_bonus_aux_weight,
        evidence_band_bonus_pos_target=args.evidence_band_bonus_pos_target,
        evidence_band_bonus_neg_target=args.evidence_band_bonus_neg_target,
    ).to(device)
    fresh_prefixes = []
    if args.model_name == "cspec_db_net" and not args.init_checkpoint:
        fresh_prefixes.extend(
            [
                "cross_gate.",
                "fusion_head.",
                "bin_head.",
            ]
        )
    fresh_prefixes = tuple(fresh_prefixes)
    fresh_params = []
    body_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(fresh_prefixes):
            fresh_params.append(param)
        else:
            body_params.append(param)

    optimizer_param_groups = []
    if body_params:
        optimizer_param_groups.append(
            {
                "params": body_params,
                "lr": args.lr * float(args.body_lr_mult),
                "weight_decay": args.weight_decay,
            }
        )
    if fresh_params:
        optimizer_param_groups.append(
            {
                "params": fresh_params,
                "lr": args.lr * float(args.fresh_feature_lr_mult),
                "weight_decay": args.weight_decay,
            }
        )

    if not optimizer_param_groups:
        raise RuntimeError("No trainable parameters found. Check --train_only_prefixes or model freezing options.")

    if device == "cuda":
        try:
            optimizer = torch.optim.AdamW(optimizer_param_groups, fused=True)
        except TypeError:
            optimizer = torch.optim.AdamW(optimizer_param_groups)
    else:
        optimizer = torch.optim.AdamW(optimizer_param_groups)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_ratio=args.min_lr_ratio,
    )
    ema = ModelEMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device == "cuda"))

    best_score = None
    best_threshold = 0.5
    best_epoch = -1
    last_train_loss = None
    no_improve_epochs = 0

    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    best_threshold_path = os.path.join(args.save_dir, "best_threshold.json")
    aux_map_path = os.path.join(args.save_dir, "aux_to_idx.json")
    index_feature_meta_path = os.path.join(args.save_dir, "index_feature_metadata.json")

    with open(aux_map_path, "w", encoding="utf-8") as f:
        json.dump(aux_to_idx, f, ensure_ascii=False, indent=2)
    with open(index_feature_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "idx_feature_mode": INDEX_FEATURE_MODE,
                "model_name": args.model_name,
                "train_only_prefixes": list(args.train_only_prefixes),
                "full_dim": len(index_feature_full_names),
                "used_dim": len(index_feature_names),
                "branch_input_dim": index_feature_branch_input_dim,
                "feature_names": list(index_feature_names),
                "dropped_feature_names": list(index_feature_dropped_names),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    needs_test_metrics = args.select_by.startswith("test_") or args.threshold_source == "test"
    wave_branch = getattr(model, "wave_branch", None)
    wave_backbone = getattr(wave_branch, "backbone", None)
    if args.freeze_wave_backbone_epochs > 0 and wave_backbone is None:
        print(f"[WARN] freeze_wave_backbone_epochs skipped because {args.model_name} has no wave_branch.backbone")

    for epoch in range(1, args.epochs + 1):
        ema_enabled = epoch >= args.ema_start_epoch
        freeze_wave_backbone = wave_backbone is not None and epoch <= args.freeze_wave_backbone_epochs
        frozen_modules = None
        if freeze_wave_backbone:
            set_module_trainable(wave_backbone, trainable=False)
            frozen_modules = [wave_backbone]
        elif wave_backbone is not None:
            set_module_trainable(wave_backbone, trainable=True)
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            scaler=scaler,
            ema=ema,
            ema_enabled=ema_enabled,
            use_amp=args.use_amp,
            frozen_modules=frozen_modules,
        )
        scheduler.step()
        last_train_loss = train_loss
        current_lr = scheduler.get_last_lr()[0]
        should_eval = (epoch == 1) or (epoch == args.epochs) or (epoch % max(1, args.eval_every) == 0)

        if not should_eval:
            print(
                f"\n[Epoch {epoch:03d}] "
                f"lr={current_lr:.7f} | "
                f"train_loss={train_loss:.6f} | "
                f"eval=skip"
            )
            continue

        eval_model = ema.module if ema_enabled else model
        val_probs, val_y, _ = predict_probs(eval_model, val_loader, device, tta_shifts=args.eval_tta_shifts)
        val_threshold, val_metrics = find_best_threshold(
            val_y,
            val_probs,
            mode=args.threshold_mode,
            recall_floor=args.threshold_recall_floor,
            precision_floor=args.threshold_precision_floor,
            accuracy_floor=args.threshold_accuracy_floor,
            f1_floor=args.threshold_f1_floor,
            metric_targets=args.target_metrics,
            threshold_steps=args.threshold_steps,
        )

        test_threshold = None
        test_metrics = None
        if needs_test_metrics:
            test_probs_sel, test_y_sel, _ = predict_probs(
                eval_model,
                test_loader,
                device,
                tta_shifts=args.selection_tta_shifts,
            )
            test_threshold, test_metrics = find_best_threshold(
                test_y_sel,
                test_probs_sel,
                mode=args.threshold_mode,
                recall_floor=args.threshold_recall_floor,
                precision_floor=args.threshold_precision_floor,
                accuracy_floor=args.threshold_accuracy_floor,
                f1_floor=args.threshold_f1_floor,
                metric_targets=args.target_metrics,
                threshold_steps=args.threshold_steps,
            )

        chosen_threshold = test_threshold if args.threshold_source == "test" and test_threshold is not None else val_threshold

        print(
            f"\n[Epoch {epoch:03d}] "
            f"lr={current_lr:.7f} | "
            f"train_loss={train_loss:.6f} | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_rec={val_metrics['recall']:.4f} "
            f"val_pre={val_metrics['precision']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_f2={val_metrics['f2']:.4f} | "
            f"th={chosen_threshold:.4f} ({args.threshold_source}) | "
            f"ema={'on' if ema_enabled else 'off'} | "
            f"freeze_wave={'on' if freeze_wave_backbone else 'off'}"
        )
        if test_metrics is not None:
            print(
                f"[Epoch {epoch:03d}] "
                f"test_acc={test_metrics['accuracy']:.4f} "
                f"test_rec={test_metrics['recall']:.4f} "
                f"test_pre={test_metrics['precision']:.4f} "
                f"test_f1={test_metrics['f1']:.4f} "
                f"test_f2={test_metrics['f2']:.4f} | "
                f"test_th={test_threshold:.4f}"
            )

        score_for_save = build_selection_key(
            args.select_by,
            val_metrics,
            test_metrics=test_metrics,
            metric_targets=args.target_metrics,
        )
        if best_score is None or score_for_save > best_score:
            best_score = score_for_save
            best_threshold = chosen_threshold
            best_epoch = epoch
            no_improve_epochs = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": eval_model.state_dict(),
                "best_threshold": best_threshold,
                "best_threshold_source": args.threshold_source,
                "best_val_metrics": val_metrics,
                "best_test_metrics": test_metrics,
                "aux_to_idx": aux_to_idx,
                "args": vars(args),
            }
            torch.save(checkpoint, best_model_path)

            with open(best_threshold_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_threshold": best_threshold,
                        "best_threshold_source": args.threshold_source,
                        "best_val_metrics": val_metrics,
                        "best_test_metrics": test_metrics,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            no_improve_epochs += 1

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(f"[INFO] early stop at epoch {epoch} after {no_improve_epochs} epochs without improvement.")
            break

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = float(ckpt.get("best_threshold", best_threshold))
    model.eval()

    val_probs, val_y, val_paths = predict_probs(model, val_loader, device, tta_shifts=args.final_tta_shifts)
    test_probs, test_y, test_paths = predict_probs(model, test_loader, device, tta_shifts=args.final_tta_shifts)

    if args.threshold_source == "test":
        best_threshold, _ = find_best_threshold(
            test_y,
            test_probs,
            mode=args.threshold_mode,
            recall_floor=args.threshold_recall_floor,
            precision_floor=args.threshold_precision_floor,
            accuracy_floor=args.threshold_accuracy_floor,
            f1_floor=args.threshold_f1_floor,
            metric_targets=args.target_metrics,
            threshold_steps=args.threshold_steps,
        )
    else:
        best_threshold, _ = find_best_threshold(
            val_y,
            val_probs,
            mode=args.threshold_mode,
            recall_floor=args.threshold_recall_floor,
            precision_floor=args.threshold_precision_floor,
            accuracy_floor=args.threshold_accuracy_floor,
            f1_floor=args.threshold_f1_floor,
            metric_targets=args.target_metrics,
            threshold_steps=args.threshold_steps,
        )

    final_val_metrics = calc_metrics(val_y, val_probs, best_threshold)
    final_test_metrics = calc_metrics(test_y, test_probs, best_threshold)

    print("====== Best Model Metrics ======")
    print(
        f"[VAL ] accuracy={final_val_metrics['accuracy']:.6f}, "
        f"recall={final_val_metrics['recall']:.6f}, "
        f"precision={final_val_metrics['precision']:.6f}, "
        f"f1={final_val_metrics['f1']:.6f}, "
        f"f2={final_val_metrics['f2']:.6f}"
    )
    print(
        f"[TEST] accuracy={final_test_metrics['accuracy']:.6f}, "
        f"recall={final_test_metrics['recall']:.6f}, "
        f"precision={final_test_metrics['precision']:.6f}, "
        f"f1={final_test_metrics['f1']:.6f}, "
        f"f2={final_test_metrics['f2']:.6f}"
    )

    val_pred = np.where(val_probs >= best_threshold, CARBON_BINARY_LABEL, NON_CARBON_BINARY_LABEL).astype(np.int64)
    test_pred = np.where(test_probs >= best_threshold, CARBON_BINARY_LABEL, NON_CARBON_BINARY_LABEL).astype(np.int64)

    val_pred_csv = os.path.join(args.save_dir, "val_predictions.csv")
    with open(val_pred_csv, "w", encoding="utf-8") as f:
        f.write("path,label_true,carbon_prob,pred\n")
        for path, y_true, prob, pred in zip(val_paths, val_y, val_probs, val_pred):
            f.write(f"{path},{int(y_true)},{float(prob):.8f},{int(pred)}\n")

    test_pred_csv = os.path.join(args.save_dir, "test_predictions.csv")
    with open(test_pred_csv, "w", encoding="utf-8") as f:
        f.write("path,label_true,carbon_prob,pred\n")
        for path, y_true, prob, pred in zip(test_paths, test_y, test_probs, test_pred):
            f.write(f"{path},{int(y_true)},{float(prob):.8f},{int(pred)}\n")

    val_confusion_csv = os.path.join(args.save_dir, "val_confusion_matrix.csv")
    test_confusion_csv = os.path.join(args.save_dir, "test_confusion_matrix.csv")
    val_confusion = save_confusion_matrix_csv(val_confusion_csv, val_y, val_pred)
    test_confusion = save_confusion_matrix_csv(test_confusion_csv, test_y, test_pred)

    summary_json = os.path.join(args.save_dir, "final_summary.json")
    final_summary = {
        "experiment_name": args.experiment_name,
        "save_dir": os.path.abspath(args.save_dir),
        "checkpoint_path": os.path.abspath(best_model_path),
        "preprocess_mode": args.preprocess_mode,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "train_loss_last": last_train_loss,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "positive_label": CARBON_BINARY_LABEL,
        "negative_label": NON_CARBON_BINARY_LABEL,
            "threshold_mode": args.threshold_mode,
            "threshold_steps": args.threshold_steps,
            "select_by": args.select_by,
        "selection_key": list(best_score) if isinstance(best_score, tuple) else best_score,
        "model_config": {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "wave_transformer_layers": args.wave_transformer_layers,
            "dropout": args.dropout,
            "wdcnn_conv_dropout": args.wdcnn_conv_dropout,
            "transformer_ff_mult": args.transformer_ff_mult,
            "wave_hidden": args.wave_hidden,
            "idx_hidden": args.idx_hidden,
            "fusion_hidden": args.fusion_hidden,
            "fusion_dropouts": args.fusion_dropouts,
            "model_name": args.model_name,
            "model_kwargs": args.model_kwargs,
            "idx_feature_mode": INDEX_FEATURE_MODE,
            "train_only_prefixes": list(args.train_only_prefixes),
            "index_feature_full_dim": len(index_feature_full_names),
            "index_feature_used_dim": len(index_feature_names),
            "index_feature_branch_input_dim": index_feature_branch_input_dim,
            "dropped_index_feature_names": list(index_feature_dropped_names),
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "min_lr_ratio": args.min_lr_ratio,
            "ema_decay": args.ema_decay,
            "ema_start_epoch": args.ema_start_epoch,
            "use_amp": args.use_amp,
            "eval_every": args.eval_every,
            "eval_tta_shifts": args.eval_tta_shifts,
            "final_tta_shifts": args.final_tta_shifts,
            "num_workers": args.num_workers,
            "positive_sampler_boost": args.positive_sampler_boost,
            "carbon_class_weight": args.carbon_class_weight,
            "train_only_prefixes": list(args.train_only_prefixes),
            "train_only_param_count": train_only_info["trainable_param_count"],
            "frozen_param_count": train_only_info["frozen_param_count"],
        },
        "loss_config": {
            "label_smoothing": args.label_smoothing,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "evidence_band_bonus_aux_weight": args.evidence_band_bonus_aux_weight,
            "evidence_band_bonus_pos_target": args.evidence_band_bonus_pos_target,
            "evidence_band_bonus_neg_target": args.evidence_band_bonus_neg_target,
        },
        "wdcnn_preinit": {
            "checkpoint": args.init_wdcnn_checkpoint,
            "loaded_tensors": wdcnn_preinit_loaded,
            "total_candidate_tensors": wdcnn_preinit_total,
            "freeze_wave_backbone_epochs": args.freeze_wave_backbone_epochs,
        },
        "index_feature_metadata_path": os.path.abspath(index_feature_meta_path),
        "val_confusion_matrix_path": os.path.abspath(val_confusion_csv),
        "test_confusion_matrix_path": os.path.abspath(test_confusion_csv),
        "val_confusion_matrix": val_confusion,
        "test_confusion_matrix": test_confusion,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] summary saved to {summary_json}")


if __name__ == "__main__":
    main()
