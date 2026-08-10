from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fif_test.band_learning_analysis import SampleRecord, build_occluded_flux
from fif_test import train_wdcnn_transformer_idx_candidate20_backup_20260507 as wd_checkpoint
from fif_test.wdcnn_transformer_idx_band_learning_analysis import (
    build_region_defs_local,
    predict_prob_numpy_batch,
    safe_empty_cuda_cache,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "wdcnn_transformer_idx_mode11_inject_candidate20_gpu" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "snr_gi_band_occlusion_candidate20_20260516"
DEFAULT_POSITIVE_ROOTS = (
    Path(r"D:\deeplearning study\model\dataset_7classes_train\carbon_new_fits"),
    Path(r"D:\deeplearning study\model\dataset_7classes_test\carbon_new_fits"),
)

SNR_GROUP_KEYS = ("SNRG", "SNRI")
SNR_AUX_KEYS = ("SNRU", "SNRG", "SNRR", "SNRI", "SNRZ")

GROUP_ORDER = ("snr_lt10", "snr_10_20", "snr_20_50", "snr_gt50")
GROUP_LABELS = {
    "snr_lt10": "g & i S/N < 10",
    "snr_10_20": "10 <= g & i S/N < 20",
    "snr_20_50": "20 < g & i S/N < 50",
    "snr_gt50": "g & i S/N > 50",
}

SCENARIO_ORDER = (
    "C2_4737",
    "C2_5165",
    "C2_5635",
    "CN_7065",
    "CN_7820",
    "OTHER_1",
    "OTHER_2",
    "OTHER_3",
    "OTHER_4",
    "OTHER_5",
    "ALL_FEATURE_BANDS",
    "ALL_REFERENCE_BANDS",
    "ALL_TEN_BANDS",
)
SCENARIO_LABELS = {
    "C2_4737": "C2 4737",
    "C2_5165": "C2 5165",
    "C2_5635": "C2 5635",
    "CN_7065": "CN 7065",
    "CN_7820": "CN 7820",
    "OTHER_1": "Ref. 1",
    "OTHER_2": "Ref. 2",
    "OTHER_3": "Ref. 3",
    "OTHER_4": "Ref. 4",
    "OTHER_5": "Ref. 5",
    "ALL_FEATURE_BANDS": "All feature",
    "ALL_REFERENCE_BANDS": "All reference",
    "ALL_TEN_BANDS": "All ten",
}

PLOT_SCENARIO_LABELS = {
    "C2_4737": r"C$_2$ 4737",
    "C2_5165": r"C$_2$ 5165",
    "C2_5635": r"C$_2$ 5635",
    "CN_7065": "CN 7065",
    "CN_7820": "CN 7820",
    "OTHER_1": "Ref. 1",
    "OTHER_2": "Ref. 2",
    "OTHER_3": "Ref. 3",
    "OTHER_4": "Ref. 4",
    "OTHER_5": "Ref. 5",
    "ALL_FEATURE_BANDS": "All feature",
    "ALL_REFERENCE_BANDS": "All reference",
    "ALL_TEN_BANDS": "All ten",
}

HEATMAP_SCENARIO_LABELS = {
    "C2_4737": r"C$_2$" + "\n4737",
    "C2_5165": r"C$_2$" + "\n5165",
    "C2_5635": r"C$_2$" + "\n5635",
    "CN_7065": "CN\n7065",
    "CN_7820": "CN\n7820",
    "OTHER_1": "Ref.\n1",
    "OTHER_2": "Ref.\n2",
    "OTHER_3": "Ref.\n3",
    "OTHER_4": "Ref.\n4",
    "OTHER_5": "Ref.\n5",
    "ALL_FEATURE_BANDS": "All\nfeature",
    "ALL_REFERENCE_BANDS": "All\nreference",
    "ALL_TEN_BANDS": "All\nten-band",
}

PALETTE = {
    "feature": "#B84E5A",
    "reference": "#3E78A8",
    "feature_set": "#7D1F2A",
    "reference_set": "#214F77",
    "all_set": "#30343B",
    "text": "#1F2933",
    "muted": "#667085",
    "grid": "#D8DEE8",
}

PUBLICATION_COLORS = {
    "feature": "#A85E55",
    "reference": "#5E8CA6",
    "feature_set": "#596372",
    "reference_set": "#9AA3AE",
    "all_set": "#2F3846",
    "baseline": "#56616F",
    "all_feature": "#A24B62",
    "all_reference": "#3B7896",
    "available": "#AAB2BE",
    "selected": "#202833",
    "grid": "#E2E7EE",
    "axis": "#1F2933",
    "text": "#17212B",
    "muted": "#6B7280",
    "feature_bg": "#FCF4F2",
    "reference_bg": "#F4F8FA",
    "set_bg": "#F6F7F9",
}

SHORT_GROUP_LABELS = {
    "snr_lt10": "g,i < 10",
    "snr_10_20": "10 <= g,i < 20",
    "snr_20_50": "20 < g,i < 50",
    "snr_gt50": "g,i > 50",
}

APJS_GROUP_LABELS = {
    "snr_lt10": r"$g,i<10$",
    "snr_10_20": r"$10\leq g,i<20$",
    "snr_20_50": r"$20<g,i<50$",
    "snr_gt50": r"$g,i>50$",
}

SNR_GROUP_COLORS = {
    "snr_lt10": "#222222",
    "snr_10_20": "#0072B2",
    "snr_20_50": "#D55E00",
    "snr_gt50": "#009E73",
}

SNR_GROUP_MARKERS = {
    "snr_lt10": "o",
    "snr_10_20": "s",
    "snr_20_50": "^",
    "snr_gt50": "D",
}

PROBABILITY_SERIES_COLORS = {
    "baseline": "#56616F",
    "feature occlusion": "#A24B62",
    "reference occlusion": "#3B7896",
    "ten-band occlusion": "#2F3846",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.size": 8.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "axes.edgecolor": "#26313D",
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["text"],
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_int_tuple(value, expected_len: int | None = None) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        result = tuple(int(v) for v in value)
    else:
        result = tuple(int(v.strip()) for v in str(value).split(",") if v.strip())
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"Expected {expected_len} integers, got {result}")
    return result


def ensure_float_tuple(value, expected_len: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (tuple, list)):
        result = tuple(float(v) for v in value)
    else:
        result = tuple(float(v.strip()) for v in str(value).split(",") if v.strip())
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {result}")
    return result


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, float, int, str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    args = dict(ckpt.get("args", {}) or {})

    model_kwargs = dict(
        signal_len=wd_checkpoint.N_PIX,
        wave_start=wd_checkpoint.WAVE_START,
        wave_end=wd_checkpoint.WAVE_END,
        carbon_bands=wd_checkpoint.MODEL_CARBON_BANDS,
        embed_dim=int(args.get("embed_dim", 192)),
        num_heads=int(args.get("num_heads", 6)),
        wave_transformer_layers=int(args.get("wave_transformer_layers", 3)),
        dropout=float(args.get("dropout", 0.08)),
        conv_dropout=float(args.get("wdcnn_conv_dropout", 0.03)),
        transformer_ff_mult=int(args.get("transformer_ff_mult", 4)),
        wave_hidden=ensure_int_tuple(args.get("wave_hidden", (256, 192)), expected_len=2),
        idx_hidden=ensure_int_tuple(args.get("idx_hidden", (256, 192)), expected_len=2),
        fusion_hidden=ensure_int_tuple(args.get("fusion_hidden", (256, 128)), expected_len=2),
        fusion_dropouts=ensure_float_tuple(args.get("fusion_dropouts", (0.12, 0.08)), expected_len=2),
        idx_feature_mode=str(args.get("idx_feature_mode", "mode3")),
        idx_branch_style=str(args.get("idx_branch_style", "structured")),
        dedup_index_features=bool(args.get("dedup_index_features", False)),
        use_index_feature_lift=bool(args.get("use_index_feature_lift", False)),
        learnable_feature_scale=bool(args.get("learnable_feature_scale", False)),
        feature_scale_init=float(args.get("feature_scale_init", 1.0)),
        use_index_evidence_bonus=bool(args.get("use_index_evidence_bonus", False)),
        use_index_evidence_feature_injection=bool(
            args.get(
                "use_index_evidence_feature_injection",
                not bool(args.get("disable_index_evidence_feature_injection", False)),
            )
        ),
        use_index_calibration_penalty=bool(args.get("use_index_calibration_penalty", False)),
        index_evidence_bonus_style=str(args.get("index_evidence_bonus_style", "shared")),
        index_evidence_gain_init=float(args.get("index_evidence_gain_init", -1.5)),
        index_evidence_gain_mult=float(args.get("index_evidence_gain_mult", 1.0)),
        index_evidence_feature_scale_init=float(args.get("index_evidence_feature_scale_init", -2.2)),
        index_calibration_penalty_scale_init=float(args.get("index_calibration_penalty_scale_init", -3.0)),
    )
    model = wd_checkpoint.WDTransformerCarbonNet(**model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    model = model.to(device)
    model.eval()

    threshold = float(ckpt.get("best_threshold", 0.5))
    carbon_binary_label = int(ckpt.get("positive_label", wd_checkpoint.CARBON_BINARY_LABEL))
    preprocess_mode = str(args.get("preprocess_mode", "dual_branch"))
    return model, threshold, carbon_binary_label, preprocess_mode


def preprocess_flux_pair_checkpoint(flux_resampled: np.ndarray, preprocess_mode: str) -> tuple[np.ndarray, np.ndarray]:
    flux_resampled = np.asarray(flux_resampled, dtype=np.float32)
    if preprocess_mode == "dual_branch":
        x_main = wd_checkpoint.normalize_flux(flux_resampled)
        x_idx = wd_checkpoint.normalize_flux_for_indices(flux_resampled)
    elif preprocess_mode == "z_zero":
        x_main = wd_checkpoint.z_zero(flux_resampled)
        x_idx = wd_checkpoint.normalize_flux_for_indices(flux_resampled)
    else:
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")
    return x_main.astype(np.float32, copy=False), x_idx.astype(np.float32, copy=False)


def prepare_sample_checkpoint(
    path: str,
    sample_group: str,
    sample_index: int,
    max_raw_points: int,
    preprocess_mode: str,
) -> SampleRecord | None:
    wave, flux = wd_checkpoint.read_lamost_fits(path)
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    flux = np.asarray(flux, dtype=np.float32).reshape(-1)

    if wave.size == 0 or flux.size == 0 or wave.size != flux.size:
        return None
    if wave.size > max_raw_points or flux.size > max_raw_points:
        return None

    valid = np.isfinite(wave) & np.isfinite(flux)
    if int(valid.sum()) < 100:
        return None
    wave = wave[valid]
    flux = flux[valid]
    if wave.size < 100 or flux.size < 100:
        return None
    if wave[0] > wave[-1]:
        wave = wave[::-1]
        flux = flux[::-1]

    flux_resampled = wd_checkpoint.resample_to_target_grid(wave, flux, wd_checkpoint.TARGET_WAVE)
    if flux_resampled.size != wd_checkpoint.N_PIX:
        return None
    if not np.isfinite(flux_resampled).all():
        flux_resampled = wd_checkpoint.robust_fix(flux_resampled)

    norm_wave, norm_idx = preprocess_flux_pair_checkpoint(flux_resampled, preprocess_mode=preprocess_mode)
    if not np.isfinite(norm_wave).all() or not np.isfinite(norm_idx).all():
        return None

    return SampleRecord(
        sample_index=int(sample_index),
        source_path=path,
        source_name=Path(path).name,
        source_class=Path(path).parent.name,
        sample_group=sample_group,
        object_id=Path(path).stem,
        raw_wave=wave.astype(np.float32, copy=False),
        raw_flux=flux.astype(np.float32, copy=False),
        flux_resampled=flux_resampled.astype(np.float32, copy=False),
        norm_wave=norm_wave,
        norm_idx=norm_idx,
        ew_by_band={},
    )


def list_fits(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    suffixes = (".fits", ".fit", ".fits.gz", ".fz")
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.name.lower().endswith(suffixes))


def read_snr_header(path: Path) -> dict[str, float]:
    header = fits.getheader(path, 0)
    row: dict[str, float] = {}
    for key in SNR_AUX_KEYS:
        try:
            value = float(header.get(key, float("nan")))
        except Exception:
            value = float("nan")
        row[key] = value if math.isfinite(value) else float("nan")
    return row


def snr_group(snr_g: float, snr_i: float) -> str | None:
    if not (math.isfinite(snr_g) and math.isfinite(snr_i)):
        return None
    if snr_g < 10.0 and snr_i < 10.0:
        return "snr_lt10"
    if 10.0 <= snr_g < 20.0 and 10.0 <= snr_i < 20.0:
        return "snr_10_20"
    if 20.0 < snr_g < 50.0 and 20.0 < snr_i < 50.0:
        return "snr_20_50"
    if snr_g > 50.0 and snr_i > 50.0:
        return "snr_gt50"
    return "boundary_or_mixed"


def build_candidate_manifest(positive_roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in positive_roots:
        split_name = root.parent.name
        for path in list_fits(root):
            snr = read_snr_header(path)
            group = snr_group(float(snr.get("SNRG", float("nan"))), float(snr.get("SNRI", float("nan"))))
            rows.append(
                {
                    "path": str(path),
                    "split": split_name,
                    "source_class": root.name,
                    "object_id": path.stem,
                    "snr_group_pool": group or "nan",
                    **{key.lower(): snr[key] for key in SNR_AUX_KEYS},
                }
            )
    rows.sort(key=lambda row: (row["split"], row["path"]))
    return rows


def select_manifest_rows(manifest: list[dict], max_per_group: int, seed: int) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = {
        "snr_lt10": [],
        "snr_10_20": [],
        "snr_20_50": [],
        "snr_gt50": [],
        "boundary_or_mixed": [],
        "nan": [],
    }
    for row in manifest:
        groups.setdefault(str(row["snr_group_pool"]), []).append(row)

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for group_name in GROUP_ORDER:
        pool = groups[group_name]
        take = min(int(max_per_group), len(pool))
        if take < len(pool):
            chosen_idx = np.sort(rng.choice(len(pool), size=take, replace=False))
            chosen = [pool[int(i)] for i in chosen_idx]
        else:
            chosen = pool
        for row in chosen:
            new_row = dict(row)
            new_row["sample_group"] = group_name
            selected.append(new_row)

    selected.sort(key=lambda row: (GROUP_ORDER.index(row["sample_group"]), row["split"], row["path"]))

    counts = {
        "available": {
            "snr_lt10": len(groups["snr_lt10"]),
            "snr_10_20": len(groups["snr_10_20"]),
            "snr_20_50": len(groups["snr_20_50"]),
            "snr_gt50": len(groups["snr_gt50"]),
            "boundary_or_mixed": len(groups["boundary_or_mixed"]),
            "nan": len(groups["nan"]),
            "total_positive": len(manifest),
        },
        "max_per_group": int(max_per_group),
        "selected": {group: sum(1 for row in selected if row["sample_group"] == group) for group in GROUP_ORDER},
    }
    return selected, counts


def build_scenarios() -> list[tuple[str, str, str, list]]:
    region_defs = build_region_defs_local()
    feature_regions = [region for region in region_defs if region.region_group == "feature"]
    reference_regions = [region for region in region_defs if region.region_group == "other"]

    scenarios: list[tuple[str, str, str, list]] = []
    for region in feature_regions:
        scenarios.append(("feature", region.region_name, SCENARIO_LABELS[region.region_name], [region]))
    for region in reference_regions:
        scenarios.append(("reference", region.region_name, SCENARIO_LABELS[region.region_name], [region]))
    scenarios.append(("feature_set", "ALL_FEATURE_BANDS", SCENARIO_LABELS["ALL_FEATURE_BANDS"], feature_regions))
    scenarios.append(("reference_set", "ALL_REFERENCE_BANDS", SCENARIO_LABELS["ALL_REFERENCE_BANDS"], reference_regions))
    scenarios.append(("all_set", "ALL_TEN_BANDS", SCENARIO_LABELS["ALL_TEN_BANDS"], feature_regions + reference_regions))
    return scenarios


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_detail_rows(detail_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], dict] = {}
    for row in detail_rows:
        key = (row["sample_group"], row["occlusion_group"], row["occlusion_name"], row["display_name"])
        item = grouped.get(key)
        if item is None:
            item = {
                "sample_group": row["sample_group"],
                "occlusion_group": row["occlusion_group"],
                "occlusion_name": row["occlusion_name"],
                "display_name": row["display_name"],
                "baseline": [],
                "occluded": [],
                "drop": [],
                "relative_drop": [],
                "flip": 0,
                "baseline_positive": 0,
                "occluded_positive": 0,
            }
            grouped[key] = item
        item["baseline"].append(float(row["baseline_pred_carbon_prob"]))
        item["occluded"].append(float(row["occluded_pred_carbon_prob"]))
        item["drop"].append(float(row["prob_drop"]))
        item["relative_drop"].append(float(row["relative_prob_drop_pct"]))
        item["flip"] += int(row["decision_flipped"])
        item["baseline_positive"] += int(row["baseline_is_positive"])
        item["occluded_positive"] += int(row["occluded_is_positive"])

    rows: list[dict] = []
    for item in grouped.values():
        drop = np.asarray(item["drop"], dtype=np.float64)
        rel_drop = np.asarray(item["relative_drop"], dtype=np.float64)
        baseline = np.asarray(item["baseline"], dtype=np.float64)
        occluded = np.asarray(item["occluded"], dtype=np.float64)
        n = int(drop.size)
        std = float(np.nanstd(drop, ddof=1)) if n > 1 else 0.0
        ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
        rows.append(
            {
                "sample_group": item["sample_group"],
                "sample_group_label": GROUP_LABELS[item["sample_group"]],
                "occlusion_group": item["occlusion_group"],
                "occlusion_name": item["occlusion_name"],
                "display_name": item["display_name"],
                "sample_count": n,
                "mean_baseline_pred_carbon_prob": float(np.nanmean(baseline)),
                "mean_occluded_pred_carbon_prob": float(np.nanmean(occluded)),
                "mean_prob_drop": float(np.nanmean(drop)),
                "median_prob_drop": float(np.nanmedian(drop)),
                "std_prob_drop": std,
                "ci95_prob_drop": ci95,
                "q25_prob_drop": float(np.nanpercentile(drop, 25)),
                "q75_prob_drop": float(np.nanpercentile(drop, 75)),
                "mean_relative_prob_drop_pct": float(np.nanmean(rel_drop)),
                "decision_flip_rate": float(item["flip"] / max(n, 1)),
                "baseline_positive_rate": float(item["baseline_positive"] / max(n, 1)),
                "occluded_positive_rate": float(item["occluded_positive"] / max(n, 1)),
            }
        )
    rows.sort(key=lambda row: (GROUP_ORDER.index(row["sample_group"]), SCENARIO_ORDER.index(row["occlusion_name"])))
    return rows


def scenario_color(row: dict) -> str:
    group = row["occlusion_group"]
    return PALETTE.get(group, PALETTE["all_set"])


def save_figure(fig: mpl.figure.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(outbase.with_suffix(suffix))
    plt.close(fig)


def save_figure_fixed_canvas(fig: mpl.figure.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(outbase.with_suffix(suffix), bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=PUBLICATION_COLORS["axis"],
    )


def snr_plot_label(group: str, counts: dict | None = None) -> str:
    label = SHORT_GROUP_LABELS[group]
    if counts is None:
        return label
    return f"{label}\n(n={counts['selected'][group]:,})"


def snr_plot_label_apjs(group: str, counts: dict | None = None) -> str:
    label = APJS_GROUP_LABELS[group]
    if counts is None:
        return label
    return f"{label}\n$n={counts['selected'][group]:,}$"


def snr_legend_label_apjs(group: str, counts: dict) -> str:
    return f"{APJS_GROUP_LABELS[group]} ($n={counts['selected'][group]:,}$)"


def occlusion_lookup(summary_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["sample_group"], row["occlusion_name"]): row for row in summary_rows}


def occlusion_matrix(summary_rows: list[dict]) -> np.ndarray:
    lookup = occlusion_lookup(summary_rows)
    return np.asarray(
        [
            [float(lookup[(group, name)]["mean_prob_drop"]) for name in SCENARIO_ORDER]
            for group in GROUP_ORDER
        ],
        dtype=np.float64,
    )


def add_scenario_background(ax: mpl.axes.Axes) -> None:
    ax.axhspan(-0.5, 4.5, color=PUBLICATION_COLORS["feature_bg"], zorder=0)
    ax.axhspan(4.5, 9.5, color=PUBLICATION_COLORS["reference_bg"], zorder=0)
    ax.axhspan(9.5, 12.5, color=PUBLICATION_COLORS["set_bg"], zorder=0)
    ax.axhline(4.5, color="#FFFFFF", linewidth=1.0, zorder=1)
    ax.axhline(9.5, color="#FFFFFF", linewidth=1.0, zorder=1)


def plot_snr_count_panel(ax: mpl.axes.Axes, counts: dict) -> None:
    y = np.arange(len(GROUP_ORDER), dtype=np.float64)
    available = np.asarray([counts["available"][group] for group in GROUP_ORDER], dtype=np.float64)
    selected = np.asarray([counts["selected"][group] for group in GROUP_ORDER], dtype=np.float64)

    ax.scatter(
        available,
        y + 0.11,
        s=26,
        facecolors="#FFFFFF",
        edgecolors=PUBLICATION_COLORS["available"],
        linewidths=1.1,
        label="available",
        zorder=3,
    )
    ax.scatter(
        selected,
        y - 0.11,
        s=28,
        color=PUBLICATION_COLORS["selected"],
        marker="s",
        label="used",
        zorder=4,
    )
    for yi, avail, used in zip(y, available, selected):
        ax.plot([min(avail, used), max(avail, used)], [yi, yi], color=PUBLICATION_COLORS["grid"], linewidth=0.9, zorder=2)
        ax.text(avail * 1.06, yi + 0.11, f"{int(avail):,}", ha="left", va="center", fontsize=6.8, color=PUBLICATION_COLORS["muted"])
        ax.text(used * 1.06, yi - 0.11, f"{int(used):,}", ha="left", va="center", fontsize=6.8, color=PUBLICATION_COLORS["text"])

    ax.set_xscale("log")
    ax.set_xlim(25, max(available.max(), selected.max()) * 1.75)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_GROUP_LABELS[group] for group in GROUP_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("positive spectra")
    ax.grid(axis="x", color=PUBLICATION_COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=6.9, handletextpad=0.4, borderaxespad=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)


def collect_combined_probability(detail_rows: list[dict]) -> list[tuple[str, dict[str, list[float]], str, str]]:
    grouped = {
        "baseline": {group: [] for group in GROUP_ORDER},
        "all feature occluded": {group: [] for group in GROUP_ORDER},
        "all reference occluded": {group: [] for group in GROUP_ORDER},
        "all ten occluded": {group: [] for group in GROUP_ORDER},
    }
    seen: set[tuple[str, str]] = set()
    for row in detail_rows:
        group = row["sample_group"]
        key = (group, row["source_path"])
        if key not in seen:
            grouped["baseline"][group].append(float(row["baseline_pred_carbon_prob"]))
            seen.add(key)
        if row["occlusion_name"] == "ALL_FEATURE_BANDS":
            grouped["all feature occluded"][group].append(float(row["occluded_pred_carbon_prob"]))
        elif row["occlusion_name"] == "ALL_REFERENCE_BANDS":
            grouped["all reference occluded"][group].append(float(row["occluded_pred_carbon_prob"]))
        elif row["occlusion_name"] == "ALL_TEN_BANDS":
            grouped["all ten occluded"][group].append(float(row["occluded_pred_carbon_prob"]))

    return [
        ("baseline", grouped["baseline"], PUBLICATION_COLORS["baseline"], "o"),
        ("feature occlusion", grouped["all feature occluded"], PUBLICATION_COLORS["all_feature"], "s"),
        ("reference occlusion", grouped["all reference occluded"], PUBLICATION_COLORS["all_reference"], "^"),
        ("ten-band occlusion", grouped["all ten occluded"], PUBLICATION_COLORS["all_set"], "D"),
    ]


def plot_combined_probability_panel(
    ax: mpl.axes.Axes,
    detail_rows: list[dict],
    counts: dict,
    *,
    direct_labels: bool,
) -> None:
    x = np.arange(len(GROUP_ORDER), dtype=np.float64)
    line_end_labels = []
    for label, grouped, color, marker in collect_combined_probability(detail_rows):
        means = []
        errs = []
        for group in GROUP_ORDER:
            mean, err = mean_ci95(grouped[group])
            means.append(mean)
            errs.append(err)
        ax.errorbar(
            x,
            means,
            yerr=errs,
            color=color,
            marker=marker,
            markersize=3.8,
            linewidth=1.25,
            elinewidth=0.75,
            capsize=1.9,
            label=label,
            zorder=3,
        )
        if direct_labels:
            line_end_labels.append((label, float(means[-1]), color))

    if direct_labels:
        sorted_labels = sorted(line_end_labels, key=lambda item: item[1], reverse=True)
        placed: list[tuple[str, float, str]] = []
        min_gap = 0.045
        last_y = 1.04
        for label, y_value, color in sorted_labels:
            y_position = min(y_value, last_y - min_gap)
            y_position = max(0.06, min(1.02, y_position))
            placed.append((label, y_position, color))
            last_y = y_position
        for label, y_position, color in placed:
            ax.text(
                x[-1] + 0.08,
                y_position,
                label,
                ha="left",
                va="center",
                fontsize=6.8,
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([snr_plot_label(group, counts) for group in GROUP_ORDER])
    ax.set_ylabel(r"carbon-star probability, $P_{\rm C}$")
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlim(-0.25, 3.82 if direct_labels else 3.25)
    ax.grid(axis="y", color=PUBLICATION_COLORS["grid"], linewidth=0.65)
    if not direct_labels:
        ax.legend(frameon=False, fontsize=6.9, loc="lower right", borderaxespad=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)


def plot_occlusion_heatmap_panel(
    ax: mpl.axes.Axes,
    summary_rows: list[dict],
    counts: dict,
    *,
    colorbar_ax: mpl.axes.Axes | None,
    annotate: bool = True,
) -> mpl.image.AxesImage:
    matrix = occlusion_matrix(summary_rows)
    bound = max(0.03, float(np.nanmax(np.abs(matrix))) * 1.04)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("occlusion_diverging", ["#2166AC", "#F8F8F5", "#B2182B"], N=256)
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(SCENARIO_ORDER)))
    ax.set_xticklabels([HEATMAP_SCENARIO_LABELS[name] for name in SCENARIO_ORDER], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(GROUP_ORDER)))
    ax.set_yticklabels([snr_plot_label(group, counts) for group in GROUP_ORDER])
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, len(SCENARIO_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(GROUP_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.8)
    ax.axvline(4.5, color="#FFFFFF", linewidth=1.35)
    ax.axvline(9.5, color="#FFFFFF", linewidth=1.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text_color = "#FFFFFF" if abs(value) > 0.58 * bound else PUBLICATION_COLORS["text"]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.2, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    if colorbar_ax is not None:
        cbar = ax.figure.colorbar(image, cax=colorbar_ax)
        cbar.set_label(r"$\Delta P_{\rm C}$", fontsize=8.0)
        cbar.ax.tick_params(labelsize=7.1, length=2.4, width=0.65)
    return image


def snr_legend_handles(counts: dict) -> list[mpl.patches.Patch]:
    return [
        mpl.patches.Patch(
            facecolor=SNR_GROUP_COLORS[group],
            edgecolor="none",
            label=f"{SHORT_GROUP_LABELS[group]} (n={counts['selected'][group]:,})",
        )
        for group in GROUP_ORDER
    ]


def plot_horizontal_occlusion_bars(
    ax: mpl.axes.Axes,
    summary_rows: list[dict],
    counts: dict,
    scenarios: tuple[str, ...],
    title: str,
    xlim: tuple[float, float],
) -> None:
    lookup = occlusion_lookup(summary_rows)
    y = np.arange(len(scenarios), dtype=np.float64)
    bar_height = 0.145
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_height

    for offset, group in zip(offsets, GROUP_ORDER):
        means = np.asarray([float(lookup[(group, name)]["mean_prob_drop"]) for name in scenarios], dtype=np.float64)
        ax.barh(
            y + offset,
            means,
            height=bar_height * 0.92,
            color=SNR_GROUP_COLORS[group],
            edgecolor="#FFFFFF",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
        )

    ax.axvline(0.0, color=PUBLICATION_COLORS["axis"], linewidth=0.75, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_yticks(y)
    ax.set_yticklabels([PLOT_SCENARIO_LABELS[name] for name in scenarios])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", pad=3)
    ax.grid(axis="x", color=PUBLICATION_COLORS["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.5, width=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def plot_sensitivity_profile_panel(
    ax: mpl.axes.Axes,
    summary_rows: list[dict],
    counts: dict,
    scenarios: tuple[str, ...],
    title: str,
    ylim: tuple[float, float],
) -> None:
    lookup = occlusion_lookup(summary_rows)
    x = np.arange(len(scenarios), dtype=np.float64)
    for group in GROUP_ORDER:
        values = np.asarray([float(lookup[(group, name)]["mean_prob_drop"]) for name in scenarios], dtype=np.float64)
        ax.plot(
            x,
            values,
            color=SNR_GROUP_COLORS[group],
            marker=SNR_GROUP_MARKERS[group],
            markersize=4.3,
            markerfacecolor="#FFFFFF",
            markeredgecolor=SNR_GROUP_COLORS[group],
            markeredgewidth=1.15,
            linewidth=1.25,
            label=f"{SHORT_GROUP_LABELS[group]} (n={counts['selected'][group]:,})",
            zorder=3,
        )
    ax.axhline(0.0, color=PUBLICATION_COLORS["axis"], linewidth=0.7, zorder=2)
    ax.set_xlim(-0.18, len(scenarios) - 0.82)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([HEATMAP_SCENARIO_LABELS[name] for name in scenarios])
    ax.set_title(title, loc="left", pad=3)
    ax.grid(axis="y", color=PUBLICATION_COLORS["grid"], linewidth=0.62, zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def collect_combined_probability_stats(detail_rows: list[dict]) -> list[tuple[str, dict[str, list[float]], str]]:
    return [(label, grouped, color) for label, grouped, color, _marker in collect_combined_probability(detail_rows)]


def plot_bar_panels(summary_rows: list[dict], counts: dict, outbase: Path) -> None:
    lookup = {(row["sample_group"], row["occlusion_name"]): row for row in summary_rows}
    values = [abs(float(row["mean_prob_drop"])) + float(row["ci95_prob_drop"]) for row in summary_rows]
    x_abs = max(values) if values else 0.1
    x_abs = max(0.02, x_abs * 1.18)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0), sharex=True)
    axes = axes.ravel()
    y = np.arange(len(SCENARIO_ORDER))
    for ax, group in zip(axes, GROUP_ORDER):
        rows = [lookup[(group, name)] for name in SCENARIO_ORDER]
        means = np.asarray([float(row["mean_prob_drop"]) for row in rows], dtype=np.float64)
        errs = np.asarray([float(row["ci95_prob_drop"]) for row in rows], dtype=np.float64)
        colors = [scenario_color(row) for row in rows]
        ax.barh(y, means, xerr=errs, color=colors, edgecolor="#26313D", linewidth=0.55, height=0.72)
        ax.axvline(0.0, color="#26313D", linewidth=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([SCENARIO_LABELS[name] for name in SCENARIO_ORDER], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="-", alpha=0.8)
        ax.set_xlim(-x_abs * 0.18, x_abs)
        n = counts["selected"][group]
        ax.set_title(f"{GROUP_LABELS[group]} (n={n:,})", loc="left", fontsize=11.5, fontweight="bold", pad=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles = [
        mpl.patches.Patch(color=PALETTE["feature"], label="Feature band"),
        mpl.patches.Patch(color=PALETTE["reference"], label="Reference band"),
        mpl.patches.Patch(color=PALETTE["feature_set"], label="Combined feature"),
        mpl.patches.Patch(color=PALETTE["reference_set"], label="Combined reference"),
        mpl.patches.Patch(color=PALETTE["all_set"], label="All ten bands"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=9.5)
    fig.supxlabel("Mean decrease in carbon-star probability after occlusion", fontsize=11)
    fig.suptitle(
        "Band-occlusion sensitivity across g- and i-band S/N-stratified positive samples",
        y=1.055,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_figure(fig, outbase)


def plot_heatmap(summary_rows: list[dict], counts: dict, outbase: Path) -> None:
    lookup = {(row["sample_group"], row["occlusion_name"]): row for row in summary_rows}
    matrix = np.zeros((len(GROUP_ORDER), len(SCENARIO_ORDER)), dtype=np.float64)
    for i, group in enumerate(GROUP_ORDER):
        for j, name in enumerate(SCENARIO_ORDER):
            matrix[i, j] = float(lookup[(group, name)]["mean_prob_drop"])

    bound = float(np.nanmax(np.abs(matrix)))
    if not math.isfinite(bound) or bound <= 0.0:
        bound = 1.0

    fig, ax = plt.subplots(figsize=(13.5, 4.7))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(np.arange(len(SCENARIO_ORDER)))
    ax.set_xticklabels([SCENARIO_LABELS[name] for name in SCENARIO_ORDER], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(GROUP_ORDER)))
    ax.set_yticklabels([f"{GROUP_LABELS[group]}\n(n={counts['selected'][group]:,})" for group in GROUP_ORDER], fontsize=9.5)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "#FFFFFF" if abs(matrix[i, j]) > 0.55 * bound else "#17212B"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label("Mean probability drop", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Mean carbon-probability drop by joint g/i S/N group and occluded region",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    save_figure(fig, outbase)


def plot_probability_distributions(detail_rows: list[dict], manifest_rows: list[dict], outbase: Path) -> None:
    baseline_by_group: dict[str, list[float]] = {group: [] for group in GROUP_ORDER}
    all_feature_by_group: dict[str, list[float]] = {group: [] for group in GROUP_ORDER}
    all_reference_by_group: dict[str, list[float]] = {group: [] for group in GROUP_ORDER}

    seen_baseline: set[tuple[str, str]] = set()
    for row in detail_rows:
        group = row["sample_group"]
        key = (group, row["source_path"])
        if key not in seen_baseline:
            baseline_by_group[group].append(float(row["baseline_pred_carbon_prob"]))
            seen_baseline.add(key)
        if row["occlusion_name"] == "ALL_FEATURE_BANDS":
            all_feature_by_group[group].append(float(row["occluded_pred_carbon_prob"]))
        elif row["occlusion_name"] == "ALL_REFERENCE_BANDS":
            all_reference_by_group[group].append(float(row["occluded_pred_carbon_prob"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), gridspec_kw={"width_ratios": [0.85, 1.45]})

    counts = {group: sum(1 for row in manifest_rows if row["sample_group"] == group) for group in GROUP_ORDER}
    bar_colors = ["#8E6C8A", "#5D8A86", "#B84E5A", "#3E78A8"]
    x = np.arange(len(GROUP_ORDER))
    axes[0].bar(x, [counts[group] for group in GROUP_ORDER], color=bar_colors, edgecolor="#26313D", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER], rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Selected positive spectra", fontsize=10)
    axes[0].set_title("A. Sample counts", loc="left", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y")
    for xi, group in enumerate(GROUP_ORDER):
        axes[0].text(xi, counts[group], f"{counts[group]:,}", ha="center", va="bottom", fontsize=9)
    for spine in ("top", "right"):
        axes[0].spines[spine].set_visible(False)

    positions = []
    data = []
    colors = []
    labels = []
    width = 0.22
    for i, group in enumerate(GROUP_ORDER):
        for offset, source, color in (
            (-width, baseline_by_group[group], "#30343B"),
            (0.0, all_feature_by_group[group], PALETTE["feature_set"]),
            (width, all_reference_by_group[group], PALETTE["reference_set"]),
        ):
            positions.append(i + offset)
            data.append(source)
            colors.append(color)
        labels.append(GROUP_LABELS[group])

    parts = axes[1].violinplot(data, positions=positions, widths=0.18, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#26313D")
        body.set_alpha(0.76)
        body.set_linewidth(0.55)
    parts["cmedians"].set_color("#FFFFFF")
    parts["cmedians"].set_linewidth(1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Carbon-star probability", fontsize=10)
    axes[1].set_title("B. Probability distributions", loc="left", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y")
    legend_handles = [
        mpl.patches.Patch(color="#30343B", label="Baseline"),
        mpl.patches.Patch(color=PALETTE["feature_set"], label="All feature bands occluded"),
        mpl.patches.Patch(color=PALETTE["reference_set"], label="All reference bands occluded"),
    ]
    axes[1].legend(handles=legend_handles, frameon=False, loc="lower left", fontsize=8.8)
    for spine in ("top", "right"):
        axes[1].spines[spine].set_visible(False)

    fig.suptitle(
        "Joint g/i S/N-stratified positive samples and probability response",
        y=1.03,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, outbase)


def publication_scenario_color(occlusion_group: str) -> str:
    return PUBLICATION_COLORS.get(occlusion_group, PUBLICATION_COLORS["all_set"])


def mean_ci95(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(arr, ddof=1) / math.sqrt(arr.size))


def plot_publication_main_plate(summary_rows: list[dict], detail_rows: list[dict], counts: dict, outbase: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, height_ratios=(0.86, 1.28), width_ratios=(0.88, 1.34), hspace=0.48, wspace=0.38)

    ax_counts = fig.add_subplot(gs[0, 0])
    plot_snr_count_panel(ax_counts, counts)
    ax_counts.set_title("Joint g/i S/N strata", loc="left", pad=3)
    add_panel_label(ax_counts, "A", x=-0.19, y=1.16)

    ax_response = fig.add_subplot(gs[0, 1])
    plot_combined_probability_panel(ax_response, detail_rows, counts, direct_labels=False)
    ax_response.set_title("Combined-region response", loc="left", pad=3)
    add_panel_label(ax_response, "B", x=-0.13, y=1.16)

    heat_gs = gs[1, :].subgridspec(1, 2, width_ratios=(36, 1.15), wspace=0.035)
    ax_heat = fig.add_subplot(heat_gs[0, 0])
    cax = fig.add_subplot(heat_gs[0, 1])
    plot_occlusion_heatmap_panel(ax_heat, summary_rows, counts, colorbar_ax=cax, annotate=True)
    ax_heat.set_xlabel("occluded spectral region")
    ax_heat.set_ylabel("S/N stratum")
    ax_heat.set_title(r"Mean probability decrease after occlusion, $\Delta P_{\rm C}$", loc="left", pad=4)
    add_panel_label(ax_heat, "C", x=-0.075, y=1.17)

    save_figure(fig, outbase)


def plot_publication_effects(summary_rows: list[dict], counts: dict, outbase: Path) -> None:
    lookup = occlusion_lookup(summary_rows)
    row_groups = (
        ("Feature bands", ("C2_4737", "C2_5165", "C2_5635", "CN_7065", "CN_7820")),
        ("Reference bands", ("OTHER_1", "OTHER_2", "OTHER_3", "OTHER_4", "OTHER_5")),
        ("Combined masks", ("ALL_FEATURE_BANDS", "ALL_REFERENCE_BANDS", "ALL_TEN_BANDS")),
    )
    row_names = tuple(name for _, names in row_groups for name in names)
    matrix = np.asarray(
        [
            [float(lookup[(group, name)]["mean_prob_drop"]) for group in GROUP_ORDER]
            for name in row_names
        ],
        dtype=np.float64,
    )
    positive_bound = max(0.08, float(np.nanmax(matrix)) * 1.04)
    negative_bound = max(0.015, abs(float(np.nanmin(matrix))) * 1.20)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "apjs_occlusion_effect",
        ["#2F6F96", "#F8FAFC", "#F1C27D", "#C85F3E", "#6B2D2A"],
        N=256,
    )

    fig = plt.figure(figsize=(6.75, 3.95))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.035), wspace=0.035)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=mpl.colors.TwoSlopeNorm(vmin=-negative_bound, vcenter=0.0, vmax=positive_bound),
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([snr_plot_label_apjs(group, counts) for group in GROUP_ORDER])
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels([PLOT_SCENARIO_LABELS[name] for name in row_names])
    ax.set_xlabel("g/i signal-to-noise ratio", labelpad=7)
    ax.set_ylabel("occluded region", labelpad=7)

    ax.set_xticks(np.arange(-0.5, len(GROUP_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_names), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.axhline(4.5, color="#FFFFFF", linewidth=2.1)
    ax.axhline(9.5, color="#FFFFFF", linewidth=2.1)
    ax.axhline(4.5, color="#B8C0CA", linewidth=0.55)
    ax.axhline(9.5, color="#B8C0CA", linewidth=0.55)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "#FFFFFF" if value > 0.58 * positive_bound or value < -0.58 * negative_bound else PUBLICATION_COLORS["text"]
            label = "0.000" if abs(value) < 0.0005 else f"{value:.3f}"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=5.75, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$\Delta P_{\rm C}$", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.5, length=2.2, width=0.6)
    cbar.outline.set_linewidth(0.55)

    fig.subplots_adjust(left=0.20, right=0.94, top=0.955, bottom=0.16)
    save_figure_fixed_canvas(fig, outbase)


def plot_publication_heatmap(summary_rows: list[dict], counts: dict, outbase: Path) -> None:
    matrix = occlusion_matrix(summary_rows)
    bound = max(0.03, float(np.nanmax(np.abs(matrix))) * 1.04)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "publication_heatmap_diverging",
        ["#2F78B7", "#F4F4F1", "#BF1F38"],
        N=256,
    )

    fig = plt.figure(figsize=(9.25, 3.55))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.035), wspace=0.035)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(SCENARIO_ORDER)))
    ax.set_xticklabels([HEATMAP_SCENARIO_LABELS[name] for name in SCENARIO_ORDER], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(GROUP_ORDER)))
    ax.set_yticklabels([snr_plot_label(group, counts) for group in GROUP_ORDER])
    ax.set_xlabel("occluded spectral region", labelpad=5)
    ax.set_ylabel("joint g/i S/N stratum", labelpad=8)
    ax.set_title(r"Mean probability decrease after occlusion, $\Delta P_{\rm C}$", loc="left", pad=5, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(SCENARIO_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(GROUP_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.82)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.axvline(4.5, color="#FFFFFF", linewidth=1.65)
    ax.axvline(9.5, color="#FFFFFF", linewidth=1.65)
    ax.axvline(4.5, color="#B8C0CA", linewidth=0.55)
    ax.axvline(9.5, color="#B8C0CA", linewidth=0.55)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "#FFFFFF" if abs(value) > 0.58 * bound else PUBLICATION_COLORS["text"]
            ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=6.2, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$\Delta P_{\rm C}$", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.7, length=2.2, width=0.6)
    cbar.outline.set_linewidth(0.55)

    fig.subplots_adjust(left=0.125, right=0.94, top=0.885, bottom=0.215)
    save_figure_fixed_canvas(fig, outbase)


def plot_publication_bar_heatmap_combo(summary_rows: list[dict], counts: dict, outbase: Path) -> None:
    feature_scenarios = ("C2_4737", "C2_5165", "C2_5635", "CN_7065", "CN_7820")
    reference_scenarios = ("OTHER_1", "OTHER_2", "OTHER_3", "OTHER_4", "OTHER_5")
    combined_scenarios = ("ALL_FEATURE_BANDS", "ALL_REFERENCE_BANDS", "ALL_TEN_BANDS")
    lookup = occlusion_lookup(summary_rows)

    def panel_xlim(scenarios: tuple[str, ...]) -> tuple[float, float]:
        values = [float(lookup[(group, name)]["mean_prob_drop"]) for group in GROUP_ORDER for name in scenarios]
        return (min(-0.0045, min(values) * 1.20), max(0.018, max(values) * 1.16))

    fig = plt.figure(figsize=(12.8, 7.2))
    outer = fig.add_gridspec(
        3,
        3,
        width_ratios=(0.95, 1.75, 0.045),
        height_ratios=(1.0, 1.0, 0.82),
        hspace=0.45,
        wspace=0.30,
    )
    bar_axes = [fig.add_subplot(outer[i, 0]) for i in range(3)]

    plot_horizontal_occlusion_bars(
        bar_axes[0],
        summary_rows,
        counts,
        feature_scenarios,
        r"Feature molecular bands",
        panel_xlim(feature_scenarios),
    )
    plot_horizontal_occlusion_bars(
        bar_axes[1],
        summary_rows,
        counts,
        reference_scenarios,
        "Reference bands",
        panel_xlim(reference_scenarios),
    )
    plot_horizontal_occlusion_bars(
        bar_axes[2],
        summary_rows,
        counts,
        combined_scenarios,
        "Combined occlusion masks",
        panel_xlim(combined_scenarios),
    )
    for label, ax in zip(("A", "B", "C"), bar_axes):
        add_panel_label(ax, label, x=-0.105, y=1.12)
    bar_axes[2].set_xlabel(r"mean decrease, $\Delta P_{\rm C}$", labelpad=5)

    ax_heat = fig.add_subplot(outer[:, 1])
    cax = fig.add_subplot(outer[:, 2])
    plot_occlusion_heatmap_panel(ax_heat, summary_rows, counts, colorbar_ax=cax, annotate=True)
    ax_heat.set_xlabel("occluded spectral region", labelpad=5)
    ax_heat.set_ylabel("joint g/i S/N stratum")
    ax_heat.set_title(r"Full occlusion-response matrix", loc="left", pad=4)
    ax_heat.tick_params(axis="x", labelsize=7.0)
    add_panel_label(ax_heat, "D", x=-0.065, y=1.04)

    fig.legend(
        handles=snr_legend_handles(counts),
        frameon=False,
        ncol=4,
        loc="upper center",
        fontsize=7.0,
        handlelength=1.15,
        columnspacing=1.0,
        bbox_to_anchor=(0.51, 0.982),
    )
    fig.subplots_adjust(left=0.065, right=0.935, top=0.895, bottom=0.10)
    save_figure_fixed_canvas(fig, outbase)


def plot_publication_probability_response(detail_rows: list[dict], counts: dict, outbase: Path) -> None:
    series = collect_combined_probability(detail_rows)
    display_names = {
        "baseline": "Baseline",
        "feature occlusion": "Feature occlusion",
        "reference occlusion": "Reference occlusion",
        "ten-band occlusion": "All-band occlusion",
    }
    line_styles = {
        "baseline": "-",
        "feature occlusion": "-",
        "reference occlusion": "-",
        "ten-band occlusion": "-",
    }
    x = np.arange(len(GROUP_ORDER), dtype=np.float64)
    pooled_probabilities = {
        label: [value for group in GROUP_ORDER for value in grouped[group]]
        for label, grouped, _color, _marker in series
    }
    drop_specs = (
        ("ALL_FEATURE_BANDS", "feature occlusion", "Feature drop"),
        ("ALL_REFERENCE_BANDS", "reference occlusion", "Reference drop"),
        ("ALL_TEN_BANDS", "ten-band occlusion", "All-band drop"),
    )
    drop_values = {name: {group: [] for group in GROUP_ORDER} for name, _label, _display in drop_specs}
    for row in detail_rows:
        occlusion_name = row["occlusion_name"]
        if occlusion_name in drop_values:
            drop_values[occlusion_name][row["sample_group"]].append(float(row["prob_drop"]))

    fig = plt.figure(figsize=(8.45, 4.55))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.34),
        height_ratios=(1.0, 0.58),
        wspace=0.19,
        hspace=0.18,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_drop = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_pool = fig.add_subplot(gs[:, 1])
    legend_handles: list[mpl.lines.Line2D] = []
    legend_labels: list[str] = []

    for index, xpos in enumerate(x):
        if index % 2 == 0:
            ax.axvspan(xpos - 0.5, xpos + 0.5, color="#F7F8FA", zorder=0)
            ax_drop.axvspan(xpos - 0.5, xpos + 0.5, color="#F7F8FA", zorder=0)

    for label, grouped, _color, marker in series:
        means: list[float] = []
        errs: list[float] = []
        for group in GROUP_ORDER:
            mean, err = mean_ci95(grouped[group])
            means.append(mean)
            errs.append(err)
        means_array = np.asarray(means, dtype=np.float64)
        errs_array = np.asarray(errs, dtype=np.float64)
        ax.fill_between(
            x,
            np.clip(means_array - errs_array, 0.0, 1.05),
            np.clip(means_array + errs_array, 0.0, 1.05),
            color=PROBABILITY_SERIES_COLORS[label],
            alpha=0.075,
            linewidth=0.0,
            zorder=1,
        )
        ax.errorbar(
            x,
            means_array,
            yerr=errs_array,
            color=PROBABILITY_SERIES_COLORS[label],
            linestyle=line_styles[label],
            marker=marker,
            markersize=4.25,
            markerfacecolor="#FFFFFF",
            markeredgecolor=PROBABILITY_SERIES_COLORS[label],
            markeredgewidth=1.0,
            linewidth=1.35,
            elinewidth=0.58,
            capsize=1.8,
            capthick=0.58,
            label=display_names[label],
            zorder=3,
        )
        legend_handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                color=PROBABILITY_SERIES_COLORS[label],
                linestyle=line_styles[label],
                marker=marker,
                markersize=4.0,
                markerfacecolor="#FFFFFF",
                markeredgecolor=PROBABILITY_SERIES_COLORS[label],
                markeredgewidth=0.9,
                linewidth=1.2,
            )
        )
        legend_labels.append(display_names[label])

    ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
        fontsize=6.45,
        handlelength=1.55,
        handletextpad=0.55,
        columnspacing=0.95,
        labelspacing=0.55,
        borderaxespad=0.0,
    )

    bar_offsets = (-0.22, 0.0, 0.22)
    bar_width = 0.17
    max_drop = 0.0
    bar_handles: list[mpl.patches.Patch] = []
    bar_labels: list[str] = []
    for offset, (occlusion_name, label, _display) in zip(bar_offsets, drop_specs):
        means: list[float] = []
        errs: list[float] = []
        for group in GROUP_ORDER:
            mean, err = mean_ci95(drop_values[occlusion_name][group])
            means.append(mean)
            errs.append(err)
        means_array = np.asarray(means, dtype=np.float64)
        errs_array = np.asarray(errs, dtype=np.float64)
        max_drop = max(max_drop, float(np.nanmax(means_array + errs_array)))
        ax_drop.bar(
            x + offset,
            means_array,
            width=bar_width,
            color=PROBABILITY_SERIES_COLORS[label],
            alpha=0.88,
            edgecolor="#FFFFFF",
            linewidth=0.35,
            zorder=3,
        )
        bar_handles.append(
            mpl.patches.Patch(
                facecolor=PROBABILITY_SERIES_COLORS[label],
                edgecolor="none",
                alpha=0.88,
            )
        )
        bar_labels.append(_display)
        ax_drop.errorbar(
            x + offset,
            means_array,
            yerr=errs_array,
            fmt="none",
            ecolor=PROBABILITY_SERIES_COLORS[label],
            elinewidth=0.55,
            capsize=1.4,
            capthick=0.55,
            zorder=4,
        )

    ax_pool_order = (
        ("baseline", "Baseline"),
        ("feature occlusion", "Feature"),
        ("reference occlusion", "Reference"),
        ("ten-band occlusion", "All-band"),
    )
    pooled_stats = {
        label: mean_ci95(pooled_probabilities[label])
        for label, _display in ax_pool_order
    }
    baseline_mean = pooled_stats["baseline"][0]
    y_positions = np.arange(len(ax_pool_order) - 1, -1, -1, dtype=np.float64)
    pool_x_min = 0.55
    for y_position, (label, display) in zip(y_positions, ax_pool_order):
        mean, err = pooled_stats[label]
        color = PROBABILITY_SERIES_COLORS[label]
        marker = next(marker for series_label, _grouped, _color, marker in series if series_label == label)
        ax_pool.barh(
            y_position,
            max(mean - pool_x_min, 0.0),
            left=pool_x_min,
            height=0.46,
            color=color,
            alpha=0.13 if label == "baseline" else 0.18,
            edgecolor="none",
            zorder=1,
        )
        ax_pool.errorbar(
            mean,
            y_position,
            xerr=err,
            fmt=marker,
            color=color,
            markerfacecolor="#FFFFFF",
            markeredgecolor=color,
            markeredgewidth=1.05,
            markersize=4.4,
            elinewidth=0.65,
            capsize=1.8,
            capthick=0.65,
            zorder=3,
        )
        row_y = (y_position + 0.55) / len(ax_pool_order)
        ax_pool.text(
            1.04,
            row_y,
            f"{mean:.2f}",
            transform=ax_pool.transAxes,
            ha="left",
            va="center",
            fontsize=5.9,
            color=PUBLICATION_COLORS["text"],
            clip_on=False,
        )
    ax_pool.axvline(baseline_mean, color=PUBLICATION_COLORS["muted"], linestyle=":", linewidth=0.85, zorder=0)

    ax.set_title("A  Stratified probability response", loc="left", pad=4, fontweight="bold", fontsize=8.4)
    ax.set_ylabel(r"carbon-star probability, $P_{\rm C}$")
    ax.set_ylim(0.0, 1.06)
    ax.set_xlim(-0.45, 3.45)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.grid(axis="y", color=PUBLICATION_COLORS["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False)

    ax_drop.axhline(0.0, color=PUBLICATION_COLORS["axis"], linewidth=0.65, zorder=2)
    ax_drop.set_xticks(x)
    ax_drop.set_xticklabels([snr_plot_label_apjs(group, counts) for group in GROUP_ORDER])
    ax_drop.set_title("B  Mean probability decrease", loc="left", pad=3, fontweight="bold", fontsize=8.4)
    ax_drop.set_ylabel(r"mean decrease, $\Delta P_{\rm C}$")
    ax_drop.set_xlabel("g/i signal-to-noise ratio")
    ax_drop.set_ylim(-0.01, max(0.12, max_drop * 1.16))
    ax_drop.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax_drop.grid(axis="y", color=PUBLICATION_COLORS["grid"], linewidth=0.55, zorder=0)
    ax_drop.set_axisbelow(True)
    ax_drop.legend(
        bar_handles,
        bar_labels,
        frameon=False,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=5.8,
        handlelength=1.1,
        handletextpad=0.45,
        columnspacing=0.85,
        borderaxespad=0.0,
    )

    ax_pool.set_title("C  Pooled response\n(all S/N)", loc="left", pad=4, fontweight="bold", fontsize=8.4)
    ax_pool.set_yticks(y_positions)
    ax_pool.set_yticklabels(
        [
            f"{display}\nref." if label == "baseline" else f"{display}\n$\\Delta={pooled_stats[label][0] - baseline_mean:+.2f}$"
            for label, display in ax_pool_order
        ]
    )
    ax_pool.set_xlim(0.55, 0.95)
    ax_pool.set_ylim(-0.55, len(ax_pool_order) - 0.45)
    ax_pool.set_xlabel(r"mean $P_{\rm C}$ after occlusion")
    ax_pool.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.10))
    ax_pool.grid(axis="x", color=PUBLICATION_COLORS["grid"], linewidth=0.55, zorder=0)
    ax_pool.set_axisbelow(True)
    ax_pool.text(
        1.04,
        1.015,
        r"$P_{\rm C}$",
        transform=ax_pool.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=PUBLICATION_COLORS["muted"],
        clip_on=False,
    )

    for active_ax in (ax, ax_drop, ax_pool):
        for spine in ("top", "right"):
            active_ax.spines[spine].set_visible(False)
        active_ax.tick_params(axis="both", length=2.3, width=0.65)
    ax_pool.tick_params(axis="y", labelsize=6.8)
    ax_pool.tick_params(axis="x", labelsize=6.6)
    fig.subplots_adjust(left=0.090, right=0.970, top=0.925, bottom=0.17)
    save_figure_fixed_canvas(fig, outbase)


def run_experiment(args: argparse.Namespace) -> None:
    configure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_roots = [Path(path) for path in args.positive_roots]
    manifest = build_candidate_manifest(positive_roots)
    selected_manifest, counts = select_manifest_rows(
        manifest,
        max_per_group=int(args.max_per_group),
        seed=int(args.seed),
    )

    write_csv(
        output_dir / "snr_positive_pool_manifest.csv",
        ["path", "split", "source_class", "object_id", "snr_group_pool", "snru", "snrg", "snrr", "snri", "snrz"],
        manifest,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, carbon_binary_label, preprocess_mode = load_checkpoint_model(Path(args.checkpoint), device=device)
    scenarios = build_scenarios()

    samples = []
    valid_manifest_rows = []
    skipped_rows = []
    for idx, row in enumerate(selected_manifest, start=1):
        try:
            sample = prepare_sample_checkpoint(
                path=str(row["path"]),
                sample_group=str(row["sample_group"]),
                sample_index=idx,
                max_raw_points=int(args.max_raw_points),
                preprocess_mode=preprocess_mode,
            )
        except Exception as exc:
            sample = None
            row = dict(row)
            row["skip_reason"] = repr(exc)
        if sample is None:
            skipped_rows.append(row)
            continue
        samples.append(sample)
        valid_manifest_rows.append(row)
        if idx % 100 == 0 or idx == len(selected_manifest):
            print(f"[INFO] prepared positive samples: {idx}/{len(selected_manifest)}")

    if not samples:
        raise RuntimeError("No valid positive samples were prepared.")

    path_meta = {str(row["path"]): row for row in valid_manifest_rows}
    batch_size = max(1, int(args.predict_batch_size))
    detail_rows: list[dict] = []
    baseline_prob_by_path: dict[str, float] = {}
    total = len(samples)
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_samples = samples[batch_start:batch_end]
        baseline_wave = np.stack([sample.norm_wave for sample in batch_samples]).astype(np.float32, copy=False)
        baseline_idx = np.stack([sample.norm_idx for sample in batch_samples]).astype(np.float32, copy=False)
        baseline_pred = predict_prob_numpy_batch(
            models=[model],
            wave_batch=baseline_wave,
            idx_batch=baseline_idx,
            carbon_binary_label=carbon_binary_label,
            device=device,
        )

        for sample, prob in zip(batch_samples, baseline_pred):
            baseline_prob_by_path[sample.source_path] = float(prob)

        for occlusion_group, occlusion_name, display_name, scenario_regions in scenarios:
            occluded_flux_batch = np.stack(
                [build_occluded_flux(sample.flux_resampled, scenario_regions) for sample in batch_samples]
            ).astype(np.float32, copy=False)
            occluded_wave_batch = []
            occluded_idx_batch = []
            for flux in occluded_flux_batch:
                norm_wave, norm_idx = preprocess_flux_pair_checkpoint(flux, preprocess_mode=preprocess_mode)
                occluded_wave_batch.append(norm_wave)
                occluded_idx_batch.append(norm_idx)
            occluded_wave = np.stack(occluded_wave_batch).astype(np.float32, copy=False)
            occluded_idx = np.stack(occluded_idx_batch).astype(np.float32, copy=False)
            occluded_pred = predict_prob_numpy_batch(
                models=[model],
                wave_batch=occluded_wave,
                idx_batch=occluded_idx,
                carbon_binary_label=carbon_binary_label,
                device=device,
            )

            for inner_idx, sample in enumerate(batch_samples):
                meta = path_meta.get(sample.source_path, {})
                base_prob = float(baseline_pred[inner_idx])
                occ_prob = float(occluded_pred[inner_idx])
                prob_drop = base_prob - occ_prob
                rel_drop = float(prob_drop / base_prob * 100.0) if base_prob > 1e-8 else float("nan")
                baseline_is_positive = bool(base_prob >= threshold)
                occluded_is_positive = bool(occ_prob >= threshold)
                detail_rows.append(
                    {
                        "sample_index": int(sample.sample_index),
                        "sample_group": sample.sample_group,
                        "sample_group_label": GROUP_LABELS[sample.sample_group],
                        "split": meta.get("split", ""),
                        "source_class": sample.source_class,
                        "source_name": sample.source_name,
                        "source_path": sample.source_path,
                        "object_id": sample.object_id,
                        "snru": meta.get("snru", ""),
                        "snrg": meta.get("snrg", ""),
                        "snrr": meta.get("snrr", ""),
                        "snri": meta.get("snri", ""),
                        "snrz": meta.get("snrz", ""),
                        "threshold": float(threshold),
                        "occlusion_group": occlusion_group,
                        "occlusion_name": occlusion_name,
                        "display_name": display_name,
                        "baseline_pred_carbon_prob": base_prob,
                        "occluded_pred_carbon_prob": occ_prob,
                        "prob_drop": prob_drop,
                        "relative_prob_drop_pct": rel_drop,
                        "baseline_is_positive": int(baseline_is_positive),
                        "occluded_is_positive": int(occluded_is_positive),
                        "decision_flipped": int(baseline_is_positive != occluded_is_positive),
                    }
                )

        elapsed = time.time() - start_time
        print(f"[INFO] occlusion progress: {batch_end}/{total} samples, elapsed={elapsed:.1f}s")
        safe_empty_cuda_cache()

    summary_rows = summarize_detail_rows(detail_rows)
    valid_manifest_out = []
    for idx, row in enumerate(valid_manifest_rows, start=1):
        out = dict(row)
        out["sample_index"] = idx
        out["baseline_pred_carbon_prob"] = baseline_prob_by_path.get(str(row["path"]), float("nan"))
        valid_manifest_out.append(out)

    detail_fields = [
        "sample_index",
        "sample_group",
        "sample_group_label",
        "split",
        "source_class",
        "source_name",
        "source_path",
        "object_id",
        "snru",
        "snrg",
        "snrr",
        "snri",
        "snrz",
        "threshold",
        "occlusion_group",
        "occlusion_name",
        "display_name",
        "baseline_pred_carbon_prob",
        "occluded_pred_carbon_prob",
        "prob_drop",
        "relative_prob_drop_pct",
        "baseline_is_positive",
        "occluded_is_positive",
        "decision_flipped",
    ]
    summary_fields = [
        "sample_group",
        "sample_group_label",
        "occlusion_group",
        "occlusion_name",
        "display_name",
        "sample_count",
        "mean_baseline_pred_carbon_prob",
        "mean_occluded_pred_carbon_prob",
        "mean_prob_drop",
        "median_prob_drop",
        "std_prob_drop",
        "ci95_prob_drop",
        "q25_prob_drop",
        "q75_prob_drop",
        "mean_relative_prob_drop_pct",
        "decision_flip_rate",
        "baseline_positive_rate",
        "occluded_positive_rate",
    ]
    manifest_fields = [
        "sample_index",
        "sample_group",
        "path",
        "split",
        "source_class",
        "object_id",
        "snr_group_pool",
        "snru",
        "snrg",
        "snrr",
        "snri",
        "snrz",
        "baseline_pred_carbon_prob",
    ]

    write_csv(output_dir / "snr_band_occlusion_detail.csv", detail_fields, detail_rows)
    write_csv(output_dir / "snr_band_occlusion_summary.csv", summary_fields, summary_rows)
    write_csv(output_dir / "snr_selected_positive_manifest.csv", manifest_fields, valid_manifest_out)
    if skipped_rows:
        write_csv(output_dir / "snr_skipped_positive_samples.csv", list(skipped_rows[0].keys()), skipped_rows)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "threshold": float(threshold),
        "carbon_binary_label": int(carbon_binary_label),
        "preprocess_mode": preprocess_mode,
        "snr_keys": list(SNR_GROUP_KEYS),
        "snr_group_definitions": {
            "snr_lt10": "SNRG < 10 and SNRI < 10",
            "snr_10_20": "10 <= SNRG < 20 and 10 <= SNRI < 20",
            "snr_20_50": "20 < SNRG < 50 and 20 < SNRI < 50",
            "snr_gt50": "SNRG > 50 and SNRI > 50",
            "boundary_or_mixed": "SNRG and SNRI do not jointly fall into the same strict group",
        },
        "positive_roots": [str(path) for path in positive_roots],
        "seed": int(args.seed),
        "max_per_group": int(args.max_per_group),
        "predict_batch_size": batch_size,
        "max_raw_points": int(args.max_raw_points),
        "counts": counts,
        "valid_selected_count": len(samples),
        "skipped_selected_count": len(skipped_rows),
        "scenario_order": list(SCENARIO_ORDER),
        "elapsed_seconds": float(time.time() - start_time),
        "outputs": {
            "pool_manifest_csv": str(output_dir / "snr_positive_pool_manifest.csv"),
            "selected_manifest_csv": str(output_dir / "snr_selected_positive_manifest.csv"),
            "detail_csv": str(output_dir / "snr_band_occlusion_detail.csv"),
            "summary_csv": str(output_dir / "snr_band_occlusion_summary.csv"),
            "bar_figure": str(output_dir / "figures" / "snr_band_occlusion_bar_panels.png"),
            "heatmap_figure": str(output_dir / "figures" / "snr_band_occlusion_heatmap.png"),
            "distribution_figure": str(output_dir / "figures" / "snr_probability_distributions.png"),
            "publication_main_plate_figure": str(output_dir / "figures_publication" / "fig0_snr_gi_occlusion_main_plate.png"),
            "publication_effects_figure": str(output_dir / "figures_publication" / "fig1_snr_gi_occlusion_effects.png"),
            "publication_heatmap_figure": str(output_dir / "figures_publication" / "fig2_snr_gi_occlusion_heatmap.png"),
            "publication_probability_figure": str(output_dir / "figures_publication" / "fig3_snr_gi_probability_response.png"),
            "publication_bar_heatmap_combo_figure": str(output_dir / "figures_publication" / "fig4_snr_gi_occlusion_bar_heatmap_combo.png"),
        },
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    figures_dir = output_dir / "figures"
    plot_bar_panels(summary_rows, counts, figures_dir / "snr_band_occlusion_bar_panels")
    plot_heatmap(summary_rows, counts, figures_dir / "snr_band_occlusion_heatmap")
    plot_probability_distributions(detail_rows, valid_manifest_rows, figures_dir / "snr_probability_distributions")
    publication_dir = output_dir / "figures_publication"
    plot_publication_main_plate(summary_rows, detail_rows, counts, publication_dir / "fig0_snr_gi_occlusion_main_plate")
    plot_publication_effects(summary_rows, counts, publication_dir / "fig1_snr_gi_occlusion_effects")
    plot_publication_heatmap(summary_rows, counts, publication_dir / "fig2_snr_gi_occlusion_heatmap")
    plot_publication_probability_response(detail_rows, counts, publication_dir / "fig3_snr_gi_probability_response")
    plot_publication_bar_heatmap_combo(summary_rows, counts, publication_dir / "fig4_snr_gi_occlusion_bar_heatmap_combo")

    print(f"[INFO] valid selected samples = {len(samples)}")
    print(f"[INFO] skipped selected samples = {len(skipped_rows)}")
    print(f"[INFO] outputs written to = {output_dir}")


def render_existing_outputs(output_dir: Path) -> None:
    configure_style()
    summary_rows = read_csv_rows(output_dir / "snr_band_occlusion_summary.csv")
    detail_rows = read_csv_rows(output_dir / "snr_band_occlusion_detail.csv")
    metadata_path = output_dir / "analysis_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    counts = metadata["counts"]

    publication_dir = output_dir / "figures_publication"
    plot_publication_main_plate(summary_rows, detail_rows, counts, publication_dir / "fig0_snr_gi_occlusion_main_plate")
    plot_publication_effects(summary_rows, counts, publication_dir / "fig1_snr_gi_occlusion_effects")
    plot_publication_heatmap(summary_rows, counts, publication_dir / "fig2_snr_gi_occlusion_heatmap")
    plot_publication_probability_response(detail_rows, counts, publication_dir / "fig3_snr_gi_probability_response")
    plot_publication_bar_heatmap_combo(summary_rows, counts, publication_dir / "fig4_snr_gi_occlusion_bar_heatmap_combo")

    metadata.setdefault("outputs", {})["publication_main_plate_figure"] = str(
        publication_dir / "fig0_snr_gi_occlusion_main_plate.png"
    )
    metadata["outputs"]["publication_effects_figure"] = str(publication_dir / "fig1_snr_gi_occlusion_effects.png")
    metadata["outputs"]["publication_heatmap_figure"] = str(publication_dir / "fig2_snr_gi_occlusion_heatmap.png")
    metadata["outputs"]["publication_probability_figure"] = str(publication_dir / "fig3_snr_gi_probability_response.png")
    metadata["outputs"]["publication_bar_heatmap_combo_figure"] = str(
        publication_dir / "fig4_snr_gi_occlusion_bar_heatmap_combo.png"
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"[INFO] publication figures refreshed in = {publication_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SNR-stratified positive-sample band occlusion for the candidate20 WDTransformer checkpoint."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--positive-roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_POSITIVE_ROOTS],
        help="Positive carbon-star FITS folders to scan.",
    )
    parser.add_argument("--max-per-group", type=int, default=512, help="Maximum selected samples in each joint g/i S/N group.")
    parser.add_argument("--seed", type=int, default=20260516, help="Random seed for per-group sampling.")
    parser.add_argument("--max-raw-points", type=int, default=20000, help="Maximum raw wavelength points.")
    parser.add_argument("--predict-batch-size", type=int, default=96, help="Prediction batch size.")
    parser.add_argument("--render-only", action="store_true", help="Only regenerate figures from existing CSV outputs.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    parsed_args = parse_args(argv)
    if parsed_args.render_only:
        render_existing_outputs(Path(parsed_args.output_dir))
    else:
        run_experiment(parsed_args)


if __name__ == "__main__":
    main()
