from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (EXPERIMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fif_test.paper_style_shap import compute_dual_input_gradient_shap_batch
from run_snr_band_occlusion_experiment import (
    APJS_GROUP_LABELS,
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR as DEFAULT_OCCLUSION_DIR,
    GROUP_LABELS,
    GROUP_ORDER,
    HEATMAP_SCENARIO_LABELS,
    SCENARIO_ORDER,
    SHORT_GROUP_LABELS,
    SNR_GROUP_COLORS,
    build_region_defs_local,
    load_checkpoint_model,
    prepare_sample_checkpoint,
    safe_empty_cuda_cache,
    wd_checkpoint,
)


DEFAULT_MANIFEST = DEFAULT_OCCLUSION_DIR / "snr_selected_positive_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "snr_gi_shap_attribution_candidate20_20260519"

INDIVIDUAL_REGION_ORDER = tuple(SCENARIO_ORDER[:10])
SET_REGION_ORDER = ("ALL_FEATURE_BANDS", "ALL_REFERENCE_BANDS", "ALL_TEN_BANDS")
REGION_ORDER = INDIVIDUAL_REGION_ORDER + SET_REGION_ORDER
SUMMARY_GROUP_ORDER = ("all",) + tuple(GROUP_ORDER)
SUMMARY_GROUP_LABELS = {
    "all": "All selected",
    **GROUP_LABELS,
}
SUMMARY_SHORT_LABELS = {
    "all": "All",
    **SHORT_GROUP_LABELS,
}
SUMMARY_APJS_LABELS = {
    "all": "All",
    **APJS_GROUP_LABELS,
}
SUMMARY_GROUP_COLORS = {
    "all": "#242933",
    **SNR_GROUP_COLORS,
}

REGION_LABELS = {
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
    "ALL_TEN_BANDS": "All ten-band",
}

REGION_LABELS_TWO_LINE = {
    key: HEATMAP_SCENARIO_LABELS.get(key, REGION_LABELS[key].replace(" ", "\n"))
    for key in REGION_ORDER
}

PALETTE = {
    "axis": "#1F2933",
    "muted": "#667085",
    "grid": "#D8DEE8",
    "violin_face": "#EEF2F6",
    "violin_edge": "#8792A1",
    "median": "#111827",
    "feature_bg": "#F9F1EF",
    "reference_bg": "#F1F6F8",
    "set_bg": "#F5F6F8",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "font.size": 7.2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "axes.edgecolor": PALETTE["axis"],
            "axes.linewidth": 0.72,
            "axes.labelcolor": PALETTE["axis"],
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "xtick.color": PALETTE["axis"],
            "ytick.color": PALETTE["axis"],
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.48,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_manifest(path: Path, max_samples: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("sample_group") not in GROUP_ORDER:
                continue
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise RuntimeError(f"No SNR-selected rows were read from {path}")
    return rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_figure(fig: mpl.figure.Figure, outbase: Path) -> list[str]:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = outbase.with_suffix(suffix)
        fig.savefig(path)
        saved.append(str(path))
    plt.close(fig)
    return saved


def parse_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def stable_jitter(keys: list[str], scale: float = 0.18) -> np.ndarray:
    values = np.empty(len(keys), dtype=np.float64)
    for idx, key in enumerate(keys):
        acc = 2166136261
        for ch in key:
            acc ^= ord(ch)
            acc = (acc * 16777619) & 0xFFFFFFFF
        values[idx] = ((acc / 0xFFFFFFFF) - 0.5) * 2.0 * scale
    return values


def robust_signed_limit(values: list[float] | np.ndarray, percentile: float = 99.2) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(arr), percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    return max(limit, 1e-6)


def finite_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "ci95": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "q25": float(np.nanpercentile(arr, 25)),
        "q75": float(np.nanpercentile(arr, 75)),
    }


def build_sample_records(
    manifest_rows: list[dict],
    max_raw_points: int,
    preprocess_mode: str,
) -> tuple[list, list[dict], list[dict]]:
    samples = []
    sample_meta: list[dict] = []
    skipped: list[dict] = []
    total = len(manifest_rows)
    next_sample_index = 1
    for row_index, row in enumerate(manifest_rows, start=1):
        path = str(row["path"])
        try:
            sample = prepare_sample_checkpoint(
                path=path,
                sample_group=str(row["sample_group"]),
                sample_index=next_sample_index,
                max_raw_points=int(max_raw_points),
                preprocess_mode=preprocess_mode,
            )
        except Exception as exc:
            sample = None
            skipped.append({"path": path, "reason": repr(exc)})
        if sample is None:
            if not skipped or skipped[-1].get("path") != path:
                skipped.append({"path": path, "reason": "prepare_sample_checkpoint returned None"})
            continue

        samples.append(sample)
        sample_meta.append(
            {
                "sample_index": int(next_sample_index),
                "manifest_sample_index": int(row.get("sample_index", next_sample_index)),
                "snr_group": str(row["sample_group"]),
                "snr_group_label": GROUP_LABELS[str(row["sample_group"])],
                "path": path,
                "split": str(row.get("split", "")),
                "manifest_source_class": str(row.get("source_class", "")),
                "object_id": str(row.get("object_id", Path(path).stem)),
                "snru": parse_float(row.get("snru")),
                "snrg": parse_float(row.get("snrg")),
                "snrr": parse_float(row.get("snrr")),
                "snri": parse_float(row.get("snri")),
                "snrz": parse_float(row.get("snrz")),
                "manifest_baseline_pred_carbon_prob": parse_float(row.get("baseline_pred_carbon_prob")),
            }
        )
        next_sample_index += 1
        if row_index % 100 == 0 or row_index == total:
            print(f"[INFO] prepared SNR manifest spectra: {row_index}/{total}")
    return samples, sample_meta, skipped


def build_region_lookup() -> tuple[list, dict[str, object], dict[str, list[str]], dict[str, str], dict[str, int]]:
    region_defs = build_region_defs_local()
    region_by_name = {region.region_name: region for region in region_defs}
    set_members = {
        "ALL_FEATURE_BANDS": [name for name in INDIVIDUAL_REGION_ORDER if region_by_name[name].region_group == "feature"],
        "ALL_REFERENCE_BANDS": [name for name in INDIVIDUAL_REGION_ORDER if region_by_name[name].region_group == "other"],
        "ALL_TEN_BANDS": list(INDIVIDUAL_REGION_ORDER),
    }
    region_group_by_name = {region.region_name: region.region_group for region in region_defs}
    region_group_by_name.update(
        {
            "ALL_FEATURE_BANDS": "feature_set",
            "ALL_REFERENCE_BANDS": "reference_set",
            "ALL_TEN_BANDS": "all_set",
        }
    )
    pixel_count_by_name = {region.region_name: int(region.pixel_count) for region in region_defs}
    for set_name, members in set_members.items():
        pixel_count_by_name[set_name] = int(sum(pixel_count_by_name[name] for name in members))
    return region_defs, region_by_name, set_members, region_group_by_name, pixel_count_by_name


def compute_snr_shap(
    models: list[torch.nn.Module],
    samples: list,
    sample_meta: list[dict],
    carbon_binary_label: int,
    device: torch.device,
    gradient_samples: int,
    shap_batch_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    region_defs, _, set_members, region_group_by_name, pixel_count_by_name = build_region_lookup()

    rng = np.random.default_rng(seed)
    background_wave = np.stack([sample.norm_wave for sample in samples]).astype(np.float32, copy=False)
    background_idx = np.stack([sample.norm_idx for sample in samples]).astype(np.float32, copy=False)

    detail_rows: list[dict] = []
    sample_rows: list[dict] = []
    current_batch_size = max(1, int(shap_batch_size))
    batch_start = 0
    total = len(samples)
    start_time = time.time()

    while batch_start < total:
        batch_end = min(batch_start + current_batch_size, total)
        batch_samples = samples[batch_start:batch_end]
        batch_meta = sample_meta[batch_start:batch_end]
        batch_wave = np.stack([sample.norm_wave for sample in batch_samples]).astype(np.float32, copy=False)
        batch_idx = np.stack([sample.norm_idx for sample in batch_samples]).astype(np.float32, copy=False)

        try:
            batch_shap_wave, batch_shap_idx, batch_pred_prob = compute_dual_input_gradient_shap_batch(
                models=models,
                sample_wave_batch=batch_wave,
                sample_idx_batch=batch_idx,
                background_wave=background_wave,
                background_idx=background_idx,
                carbon_binary_label=carbon_binary_label,
                gradient_samples=int(gradient_samples),
                device=device,
                rng=rng,
            )
        except RuntimeError as exc:
            text = str(exc).lower()
            if ("out of memory" in text or "cublas_status_alloc_failed" in text) and current_batch_size > 1:
                new_batch_size = max(1, current_batch_size // 2)
                print(f"[WARN] OOM in SHAP batch, reduce batch size from {current_batch_size} to {new_batch_size}")
                current_batch_size = new_batch_size
                gc.collect()
                safe_empty_cuda_cache()
                continue
            raise

        for inner_idx, sample in enumerate(batch_samples):
            meta = batch_meta[inner_idx]
            combined_signed = (
                batch_shap_wave[inner_idx].astype(np.float64, copy=False)
                + batch_shap_idx[inner_idx].astype(np.float64, copy=False)
            )
            combined_abs = (
                np.abs(batch_shap_wave[inner_idx].astype(np.float64, copy=False))
                + np.abs(batch_shap_idx[inner_idx].astype(np.float64, copy=False))
            )
            pred_prob = float(batch_pred_prob[inner_idx])

            signed_by_region: dict[str, float] = {}
            abs_by_region: dict[str, float] = {}

            for region in region_defs:
                signed_values = combined_signed[region.mask]
                abs_values = combined_abs[region.mask]
                signed_sum = float(signed_values.sum())
                abs_sum = float(abs_values.sum())
                signed_by_region[region.region_name] = signed_sum
                abs_by_region[region.region_name] = abs_sum
                detail_rows.append(
                    {
                        **meta,
                        "sample_group": meta["snr_group"],
                        "source_name": sample.source_name,
                        "source_class": sample.source_class,
                        "pred_carbon_prob": pred_prob,
                        "region_group": region.region_group,
                        "region_name": region.region_name,
                        "display_name": REGION_LABELS[region.region_name],
                        "wave_start": float(region.wave_start),
                        "wave_end": float(region.wave_end),
                        "pixel_count": int(region.pixel_count),
                        "combined_shap_sum": signed_sum,
                        "combined_abs_shap_sum": abs_sum,
                        "combined_shap_mean": float(signed_values.mean()),
                        "combined_abs_shap_mean": float(abs_values.mean()),
                    }
                )

            for set_name, members in set_members.items():
                signed_sum = float(sum(signed_by_region[name] for name in members))
                abs_sum = float(sum(abs_by_region[name] for name in members))
                pixel_count = int(pixel_count_by_name[set_name])
                detail_rows.append(
                    {
                        **meta,
                        "sample_group": meta["snr_group"],
                        "source_name": sample.source_name,
                        "source_class": sample.source_class,
                        "pred_carbon_prob": pred_prob,
                        "region_group": region_group_by_name[set_name],
                        "region_name": set_name,
                        "display_name": REGION_LABELS[set_name],
                        "wave_start": "",
                        "wave_end": "",
                        "pixel_count": pixel_count,
                        "combined_shap_sum": signed_sum,
                        "combined_abs_shap_sum": abs_sum,
                        "combined_shap_mean": float(signed_sum / pixel_count) if pixel_count > 0 else float("nan"),
                        "combined_abs_shap_mean": float(abs_sum / pixel_count) if pixel_count > 0 else float("nan"),
                    }
                )

            sample_rows.append(
                {
                    **meta,
                    "sample_group": meta["snr_group"],
                    "source_name": sample.source_name,
                    "source_class": sample.source_class,
                    "pred_carbon_prob": pred_prob,
                    "all_feature_shap_sum": float(sum(signed_by_region[name] for name in set_members["ALL_FEATURE_BANDS"])),
                    "all_reference_shap_sum": float(sum(signed_by_region[name] for name in set_members["ALL_REFERENCE_BANDS"])),
                    "all_ten_band_shap_sum": float(sum(signed_by_region[name] for name in set_members["ALL_TEN_BANDS"])),
                }
            )

        batch_start = batch_end
        elapsed = time.time() - start_time
        rate = batch_start / elapsed if elapsed > 0 else 0.0
        print(f"[INFO] SHAP progress: {batch_start}/{total} (batch_size={batch_end - (batch_start - len(batch_samples))}, rate={rate:.1f} spectra/sec)")
        gc.collect()
        safe_empty_cuda_cache()

    return detail_rows, sample_rows


def summarize_detail_rows(detail_rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []

    def rows_for(group: str, region: str) -> list[dict]:
        if group == "all":
            return [row for row in detail_rows if row["region_name"] == region]
        return [row for row in detail_rows if row["snr_group"] == group and row["region_name"] == region]

    mean_abs_by_group_region: dict[tuple[str, str], float] = {}
    for group in SUMMARY_GROUP_ORDER:
        for region in REGION_ORDER:
            subset = rows_for(group, region)
            abs_stats = finite_stats([parse_float(row["combined_abs_shap_sum"]) for row in subset])
            mean_abs_by_group_region[(group, region)] = abs_stats["mean"]

    for group in SUMMARY_GROUP_ORDER:
        all_ten_abs = mean_abs_by_group_region.get((group, "ALL_TEN_BANDS"), float("nan"))
        for region in REGION_ORDER:
            subset = rows_for(group, region)
            if not subset:
                continue
            signed_stats = finite_stats([parse_float(row["combined_shap_sum"]) for row in subset])
            abs_stats = finite_stats([parse_float(row["combined_abs_shap_sum"]) for row in subset])
            pred_stats = finite_stats([parse_float(row["pred_carbon_prob"]) for row in subset])
            first = subset[0]
            mean_abs = abs_stats["mean"]
            abs_share = float(mean_abs / all_ten_abs * 100.0) if np.isfinite(mean_abs) and np.isfinite(all_ten_abs) and all_ten_abs > 0 else float("nan")
            summary_rows.append(
                {
                    "sample_group": group,
                    "sample_group_label": SUMMARY_GROUP_LABELS[group],
                    "region_group": first["region_group"],
                    "region_name": region,
                    "display_name": REGION_LABELS[region],
                    "wave_start": first["wave_start"],
                    "wave_end": first["wave_end"],
                    "pixel_count": first["pixel_count"],
                    "sample_count": int(signed_stats["count"]),
                    "mean_pred_carbon_prob": pred_stats["mean"],
                    "mean_shap_sum": signed_stats["mean"],
                    "median_shap_sum": signed_stats["median"],
                    "std_shap_sum": signed_stats["std"],
                    "ci95_shap_sum": signed_stats["ci95"],
                    "q25_shap_sum": signed_stats["q25"],
                    "q75_shap_sum": signed_stats["q75"],
                    "mean_abs_shap_sum": mean_abs,
                    "median_abs_shap_sum": abs_stats["median"],
                    "mean_abs_share_pct_among_ten_windows": abs_share,
                }
            )
    return summary_rows


def summary_lookup(summary_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["sample_group"], row["region_name"]): row for row in summary_rows}


def draw_heatmap_panel(
    ax: plt.Axes,
    summary_rows: list[dict],
    panel_label: str = "",
    show_xlabel: bool = True,
) -> mpl.image.AxesImage:
    lookup = summary_lookup(summary_rows)
    matrix = np.array(
        [
            [parse_float(lookup[(group, region)]["mean_shap_sum"]) for region in REGION_ORDER]
            for group in SUMMARY_GROUP_ORDER
        ],
        dtype=np.float64,
    )
    limit = robust_signed_limit(matrix, percentile=99.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")

    counts = {
        group: int(lookup[(group, REGION_ORDER[0])]["sample_count"])
        for group in SUMMARY_GROUP_ORDER
    }
    y_labels = [f"{SUMMARY_APJS_LABELS[group]}\n(n={counts[group]:,})" for group in SUMMARY_GROUP_ORDER]
    ax.set_yticks(np.arange(len(SUMMARY_GROUP_ORDER)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_xticklabels([REGION_LABELS_TWO_LINE[name] for name in REGION_ORDER], rotation=0)
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel("g,i S/N stratum")
    if show_xlabel:
        ax.set_xlabel("Spectral region")

    for idx in range(len(SUMMARY_GROUP_ORDER) + 1):
        ax.axhline(idx - 0.5, color="#FFFFFF", lw=0.75, zorder=3)
    for idx in range(len(REGION_ORDER) + 1):
        ax.axvline(idx - 0.5, color="#FFFFFF", lw=0.75, zorder=3)
    for sep in (4.5, 9.5):
        ax.axvline(sep, color="#AEB6C2", lw=1.0, zorder=4)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "#FFFFFF" if abs(value) > 0.62 * limit else PALETTE["axis"]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.25, color=color)

    if panel_label:
        ax.set_title(panel_label, loc="left", pad=4.5)
    return im


def group_region_values(
    detail_rows: list[dict],
    group: str,
    region: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    rows = [
        row
        for row in detail_rows
        if row["region_name"] == region and (group == "all" or row["snr_group"] == group)
    ]
    values = np.asarray([parse_float(row["combined_shap_sum"]) for row in rows], dtype=np.float64)
    point_groups = [str(row["snr_group"]) for row in rows]
    keys = [f"{row['sample_index']}|{row['region_name']}|{row['snr_group']}" for row in rows]
    return values, point_groups, keys


def draw_distribution_panel(
    ax: plt.Axes,
    detail_rows: list[dict],
    group: str,
    xlim: tuple[float, float],
    panel_label: str,
    show_y_labels: bool = True,
    point_alpha: float = 0.28,
    point_size: float = 5.5,
) -> None:
    regions = list(reversed(INDIVIDUAL_REGION_ORDER))
    positions = np.arange(len(regions), dtype=np.float64)
    violin_data = []
    for region in regions:
        values, _, _ = group_region_values(detail_rows, group, region)
        values = values[np.isfinite(values)]
        violin_data.append(values if values.size >= 2 else np.array([0.0, 0.0]))

    parts = ax.violinplot(
        violin_data,
        positions=positions,
        vert=False,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE["violin_face"])
        body.set_edgecolor(PALETTE["violin_edge"])
        body.set_alpha(0.78)
        body.set_linewidth(0.65)

    for pos, region in zip(positions, regions):
        values, point_groups, keys = group_region_values(detail_rows, group, region)
        valid = np.isfinite(values)
        values = values[valid]
        point_groups = [g for g, keep in zip(point_groups, valid) if keep]
        keys = [k for k, keep in zip(keys, valid) if keep]
        if values.size == 0:
            continue
        jitter = stable_jitter(keys, scale=0.20 if group == "all" else 0.17)
        for snr_group in GROUP_ORDER:
            mask = np.asarray([g == snr_group for g in point_groups], dtype=bool)
            if not np.any(mask):
                continue
            ax.scatter(
                values[mask],
                np.full(int(mask.sum()), pos) + jitter[mask],
                s=point_size,
                color=SNR_GROUP_COLORS[snr_group],
                alpha=point_alpha,
                edgecolors="none",
                rasterized=True,
                label=SHORT_GROUP_LABELS[snr_group] if region == regions[0] and group == "all" else None,
                zorder=3,
            )
        q25, med, q75 = np.nanpercentile(values, [25, 50, 75])
        ax.plot([q25, q75], [pos, pos], color=PALETTE["median"], lw=1.2, solid_capstyle="round", zorder=4)
        ax.scatter([med], [pos], marker="D", s=19, color=PALETTE["median"], edgecolors="#FFFFFF", linewidths=0.45, zorder=5)

    ax.axvline(0.0, color=PALETTE["axis"], lw=0.9, alpha=0.85, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.65, len(regions) - 0.35)
    ax.grid(axis="x", zorder=0)
    ax.set_yticks(positions)
    if show_y_labels:
        ax.set_yticklabels([REGION_LABELS[name] for name in regions])
        ax.tick_params(axis="y", labelleft=True)
        ax.set_ylabel("Spectral window")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(r"Signed SHAP sum per spectrum-window, $\phi_C$")
    ax.set_title(panel_label, loc="left", pad=4.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_summary_heatmap(summary_rows: list[dict], outbase: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.25, 3.35))
    im = draw_heatmap_panel(ax, summary_rows, panel_label="(a) Mean signed SHAP by S/N stratum", show_xlabel=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.018)
    cbar.set_label(r"Mean signed SHAP, $\phi_C$")
    fig.subplots_adjust(left=0.104, right=0.956, bottom=0.205, top=0.93)
    return save_figure(fig, outbase)


def draw_summary_distribution(detail_rows: list[dict], outbase: Path, xlim: tuple[float, float]) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.25, 4.95))
    draw_distribution_panel(
        ax,
        detail_rows,
        group="all",
        xlim=xlim,
        panel_label="(b) All selected spectra, colored by S/N stratum",
        show_y_labels=True,
        point_alpha=0.24,
        point_size=5.0,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", ncol=2, handletextpad=0.35, columnspacing=0.8)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.115, top=0.925)
    return save_figure(fig, outbase)


def draw_stratified_distribution(detail_rows: list[dict], summary_rows: list[dict], outbase: Path, xlim: tuple[float, float]) -> list[str]:
    lookup = summary_lookup(summary_rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.35), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    for idx, (ax, group) in enumerate(zip(axes, GROUP_ORDER)):
        n = int(lookup[(group, REGION_ORDER[0])]["sample_count"])
        draw_distribution_panel(
            ax,
            detail_rows,
            group=group,
            xlim=xlim,
            panel_label=f"({chr(ord('a') + idx)}) {SUMMARY_APJS_LABELS[group]} (n={n:,})",
            show_y_labels=idx in (0, 2),
            point_alpha=0.34 if n >= 100 else 0.58,
            point_size=5.2 if n >= 100 else 8.0,
        )
        if idx in (0, 2):
            regions = list(reversed(INDIVIDUAL_REGION_ORDER))
            ax.set_yticks(np.arange(len(regions), dtype=np.float64))
            ax.set_yticklabels([REGION_LABELS[name] for name in regions])
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)
        if idx < 2:
            ax.set_xlabel("")
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.085, top=0.945, hspace=0.20, wspace=0.08)
    return save_figure(fig, outbase)


def draw_main_plate(summary_rows: list[dict], detail_rows: list[dict], outbase: Path, xlim: tuple[float, float]) -> list[str]:
    fig = plt.figure(figsize=(7.25, 8.05))
    gs = GridSpec(2, 1, height_ratios=[0.82, 1.18], hspace=0.27, figure=fig)
    ax_heat = fig.add_subplot(gs[0, 0])
    im = draw_heatmap_panel(ax_heat, summary_rows, panel_label="(a) Mean signed SHAP by S/N stratum", show_xlabel=False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.034, pad=0.018)
    cbar.set_label(r"Mean signed SHAP, $\phi_C$")

    ax_dist = fig.add_subplot(gs[1, 0])
    draw_distribution_panel(
        ax_dist,
        detail_rows,
        group="all",
        xlim=xlim,
        panel_label="(b) Spectrum-window SHAP distributions",
        show_y_labels=True,
        point_alpha=0.22,
        point_size=4.8,
    )
    handles, labels = ax_dist.get_legend_handles_labels()
    if handles:
        ax_dist.legend(handles, labels, loc="upper right", ncol=2, handletextpad=0.35, columnspacing=0.8)
    fig.subplots_adjust(left=0.13, right=0.968, bottom=0.064, top=0.972)
    return save_figure(fig, outbase)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SNR-stratified Gradient SHAP attribution on the same selected spectra used by the SNR g/i occlusion experiment."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Selected positive manifest from the SNR occlusion experiment.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Candidate20 checkpoint used for SHAP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV and publication figures.")
    parser.add_argument("--gradient-samples", type=int, default=16, help="Monte Carlo samples per spectrum for Gradient SHAP.")
    parser.add_argument("--shap-batch-size", type=int, default=8, help="Initial SHAP batch size.")
    parser.add_argument("--seed", type=int, default=20260519, help="Random seed for Gradient SHAP background sampling.")
    parser.add_argument("--max-raw-points", type=int, default=20000, help="Maximum raw spectral points before a FITS file is skipped.")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug option: only process the first N manifest rows. 0 means all.")
    parser.add_argument("--skip-figures", action="store_true", help="Only write tables and metadata.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    configure_style()
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures_publication"

    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    manifest_rows = read_manifest(manifest_path, max_samples=int(args.max_samples))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] ===== SNR-stratified SHAP Config =====")
    print(f"[INFO] device = {device}")
    print(f"[INFO] checkpoint = {checkpoint_path}")
    print(f"[INFO] manifest = {manifest_path}")
    print(f"[INFO] manifest rows = {len(manifest_rows)}")
    print(f"[INFO] output dir = {output_dir}")
    print(f"[INFO] gradient samples = {args.gradient_samples}")
    print(f"[INFO] initial SHAP batch size = {args.shap_batch_size}")

    models: list[torch.nn.Module] = []
    total_start = time.time()
    try:
        model, threshold, carbon_binary_label, preprocess_mode = load_checkpoint_model(checkpoint_path, device)
        models.append(model)
        print(f"[INFO] threshold = {threshold:.6f}")
        print(f"[INFO] carbon label = {carbon_binary_label}")
        print(f"[INFO] preprocess mode = {preprocess_mode}")

        prep_start = time.time()
        samples, sample_meta, skipped_rows = build_sample_records(
            manifest_rows=manifest_rows,
            max_raw_points=int(args.max_raw_points),
            preprocess_mode=preprocess_mode,
        )
        if not samples:
            raise RuntimeError("No valid spectra were prepared from the selected manifest")
        prep_seconds = time.time() - prep_start
        print(f"[INFO] valid spectra = {len(samples)}")
        print(f"[INFO] skipped spectra = {len(skipped_rows)}")
        print(f"[INFO] preparation finished in {prep_seconds:.1f} sec")

        shap_start = time.time()
        detail_rows, sample_rows = compute_snr_shap(
            models=models,
            samples=samples,
            sample_meta=sample_meta,
            carbon_binary_label=int(carbon_binary_label),
            device=device,
            gradient_samples=int(args.gradient_samples),
            shap_batch_size=int(args.shap_batch_size),
            seed=int(args.seed),
        )
        shap_seconds = time.time() - shap_start
        summary_rows = summarize_detail_rows(detail_rows)
        print(f"[INFO] SHAP finished in {shap_seconds:.1f} sec")

        detail_csv = output_dir / "snr_gi_shap_detail.csv"
        sample_csv = output_dir / "snr_gi_shap_sample_predictions.csv"
        summary_csv = output_dir / "snr_gi_shap_summary.csv"
        skipped_csv = output_dir / "snr_gi_shap_skipped.csv"
        metadata_json = output_dir / "analysis_metadata.json"

        common_fields = [
            "sample_index",
            "manifest_sample_index",
            "sample_group",
            "snr_group",
            "snr_group_label",
            "path",
            "split",
            "manifest_source_class",
            "source_class",
            "source_name",
            "object_id",
            "snru",
            "snrg",
            "snrr",
            "snri",
            "snrz",
            "manifest_baseline_pred_carbon_prob",
            "pred_carbon_prob",
        ]
        write_csv_rows(
            detail_csv,
            common_fields
            + [
                "region_group",
                "region_name",
                "display_name",
                "wave_start",
                "wave_end",
                "pixel_count",
                "combined_shap_sum",
                "combined_abs_shap_sum",
                "combined_shap_mean",
                "combined_abs_shap_mean",
            ],
            detail_rows,
        )
        write_csv_rows(
            sample_csv,
            common_fields + ["all_feature_shap_sum", "all_reference_shap_sum", "all_ten_band_shap_sum"],
            sample_rows,
        )
        write_csv_rows(
            summary_csv,
            [
                "sample_group",
                "sample_group_label",
                "region_group",
                "region_name",
                "display_name",
                "wave_start",
                "wave_end",
                "pixel_count",
                "sample_count",
                "mean_pred_carbon_prob",
                "mean_shap_sum",
                "median_shap_sum",
                "std_shap_sum",
                "ci95_shap_sum",
                "q25_shap_sum",
                "q75_shap_sum",
                "mean_abs_shap_sum",
                "median_abs_shap_sum",
                "mean_abs_share_pct_among_ten_windows",
            ],
            summary_rows,
        )
        write_csv_rows(skipped_csv, ["path", "reason"], skipped_rows)

        all_distribution_values = [
            parse_float(row["combined_shap_sum"])
            for row in detail_rows
            if row["region_name"] in INDIVIDUAL_REGION_ORDER
        ]
        x_limit = robust_signed_limit(all_distribution_values, percentile=99.35)
        xlim = (-1.10 * x_limit, 1.10 * x_limit)

        figure_outputs: dict[str, list[str]] = {}
        if not args.skip_figures:
            figure_outputs["main_plate"] = draw_main_plate(
                summary_rows,
                detail_rows,
                figures_dir / "fig0_snr_gi_shap_attribution_main_plate",
                xlim,
            )
            figure_outputs["heatmap"] = draw_summary_heatmap(
                summary_rows,
                figures_dir / "fig1_snr_gi_shap_mean_heatmap",
            )
            figure_outputs["summary_distribution"] = draw_summary_distribution(
                detail_rows,
                figures_dir / "fig2_snr_gi_shap_signed_distribution_summary",
                xlim,
            )
            figure_outputs["stratified_distribution"] = draw_stratified_distribution(
                detail_rows,
                summary_rows,
                figures_dir / "fig3_snr_gi_shap_signed_distribution_by_snr",
                xlim,
            )

        counts = {group: 0 for group in GROUP_ORDER}
        for meta in sample_meta:
            counts[str(meta["snr_group"])] += 1
        metadata = {
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
            "device": str(device),
            "threshold": float(threshold),
            "carbon_binary_label": int(carbon_binary_label),
            "preprocess_mode": preprocess_mode,
            "gradient_samples": int(args.gradient_samples),
            "shap_batch_size_initial": int(args.shap_batch_size),
            "seed": int(args.seed),
            "max_raw_points": int(args.max_raw_points),
            "max_samples": int(args.max_samples),
            "sample_counts": {
                **counts,
                "all": int(len(samples)),
                "skipped": int(len(skipped_rows)),
            },
            "snr_group_definitions": {
                "snr_lt10": "SNRG < 10 and SNRI < 10",
                "snr_10_20": "10 <= SNRG < 20 and 10 <= SNRI < 20",
                "snr_20_50": "20 < SNRG < 50 and 20 < SNRI < 50",
                "snr_gt50": "SNRG > 50 and SNRI > 50",
            },
            "region_order": list(REGION_ORDER),
            "merge_rule": "combined_point_shap = wave_branch_shap + index_branch_shap; window SHAP is the sum over TARGET_WAVE pixels in the window",
            "background_pool": "all valid spectra from the selected SNR occlusion manifest",
            "timing_seconds": {
                "prepare": float(prep_seconds),
                "shap": float(shap_seconds),
                "total": float(time.time() - total_start),
            },
            "outputs": {
                "detail_csv": str(detail_csv),
                "sample_prediction_csv": str(sample_csv),
                "summary_csv": str(summary_csv),
                "skipped_csv": str(skipped_csv),
                "figures": figure_outputs,
            },
        }
        save_json(metadata_json, metadata)

        print(f"[DONE] detail csv = {detail_csv}")
        print(f"[DONE] summary csv = {summary_csv}")
        print(f"[DONE] figures = {figures_dir}")
        print(f"[DONE] metadata = {metadata_json}")
    finally:
        for model in models:
            try:
                model.cpu()
            except Exception:
                pass
        del models
        gc.collect()
        safe_empty_cuda_cache()


if __name__ == "__main__":
    main()
