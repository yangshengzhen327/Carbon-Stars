# -*- coding: utf-8 -*-
"""DESI recall/inference entry point for the current CSpec-DB-Net model."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cspec_db_net.config import CARBON_BINARY_LABEL, N_PIX, TARGET_WAVE, WAVE_END, WAVE_START
    from cspec_db_net.models.cspec_db_net import CSpecDBNet
    from cspec_db_net.preprocessing import normalize_flux, normalize_flux_for_indices, resample_to_target_grid
else:
    from .config import CARBON_BINARY_LABEL, N_PIX, TARGET_WAVE, WAVE_END, WAVE_START
    from .models.cspec_db_net import CSpecDBNet
    from .preprocessing import normalize_flux, normalize_flux_for_indices, resample_to_target_grid


DEFAULT_INPUT_DIR = r"D:\deeplearning study\desi_recall\temp_result"
DEFAULT_SOURCE_RECALL = r"D:\deeplearning study\desi_recall\recall.py"


def build_parser():
    parser = argparse.ArgumentParser(description="Recall carbon-star candidates from DESI pickle files.")
    parser.add_argument("--checkpoint", required=True, help="CSpec-DB-Net best_model.pt produced by training.py")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="DESI pickle file or directory")
    parser.add_argument(
        "--source_recall_script",
        default=DEFAULT_SOURCE_RECALL,
        help="Original DESI script providing the memory-safe pickle row iterator.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint best_threshold")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_records", type=int, default=0, help="Stop after N valid records; 0 means unlimited")
    parser.add_argument("--max_files", type=int, default=0, help="Process at most N pickle files; 0 means unlimited")
    parser.add_argument("--dry_run", action="store_true", help="Run inference and report counts without saving candidates")
    return parser


def _load_memory_safe_row_iterator(script_path):
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Source recall script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("_desi_recall_source", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source recall script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    iterator = getattr(module, "iter_rows_from_pickle", None)
    if iterator is None:
        raise AttributeError(f"{script_path} does not define iter_rows_from_pickle")
    return iterator


def _model_kwargs_from_checkpoint(ckpt):
    args = ckpt.get("args", {}) or {}
    return {
        "signal_len": N_PIX,
        "wave_start": WAVE_START,
        "wave_end": WAVE_END,
        "embed_dim": int(args.get("embed_dim", 192)),
        "num_heads": int(args.get("num_heads", 6)),
        "wave_transformer_layers": int(args.get("wave_transformer_layers", 3)),
        "dropout": float(args.get("dropout", 0.08)),
        "conv_dropout": float(args.get("wdcnn_conv_dropout", 0.03)),
        "transformer_ff_mult": int(args.get("transformer_ff_mult", 4)),
        "wave_hidden": tuple(args.get("wave_hidden", (256, 192))),
        "idx_hidden": tuple(args.get("idx_hidden", (256, 192))),
        "fusion_hidden": tuple(args.get("fusion_hidden", (256, 128))),
        "fusion_dropouts": tuple(args.get("fusion_dropouts", (0.12, 0.08))),
    }


def load_model(checkpoint, device, threshold_override=None):
    ckpt = torch.load(checkpoint, map_location=device)
    model = CSpecDBNet(**_model_kwargs_from_checkpoint(ckpt)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    threshold = threshold_override
    if threshold is None:
        if "best_threshold" not in ckpt:
            raise KeyError("Checkpoint has no best_threshold; pass --threshold explicitly")
        threshold = float(ckpt["best_threshold"])
    return model, float(threshold)


def _input_files(input_path, max_files=0):
    input_path = Path(input_path)
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.pkl"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No pickle files found under: {input_path}")
    return files


@torch.inference_mode()
def recall(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold = load_model(args.checkpoint, device, args.threshold)
    iter_rows = _load_memory_safe_row_iterator(args.source_recall_script)
    files = _input_files(args.input, args.max_files)

    processed = 0
    rejected = 0
    recalled = 0
    flux_batch = []
    index_batch = []

    def flush():
        nonlocal recalled
        if not flux_batch:
            return
        x = torch.from_numpy(np.stack(flux_batch)).to(device)
        x_idx = torch.from_numpy(np.stack(index_batch)).to(device)
        probs = torch.softmax(model(x, x_idx=x_idx)["bin_logits"], dim=1)[:, CARBON_BINARY_LABEL]
        recalled += int((probs >= threshold).sum().item())
        flux_batch.clear()
        index_batch.clear()

    stop = False
    for pkl_file in files:
        for row in iter_rows(str(pkl_file)):
            if args.max_records > 0 and processed >= args.max_records:
                stop = True
                break
            try:
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    raise ValueError("row has no wavelength/flux columns")
                wave = np.asarray(row[4], dtype=np.float32).reshape(-1)
                flux = np.asarray(row[5], dtype=np.float32).reshape(-1)
                if len(wave) != len(flux) or len(wave) < 10:
                    raise ValueError("invalid wavelength/flux arrays")
                resampled = resample_to_target_grid(wave, flux, TARGET_WAVE)
                flux_batch.append(normalize_flux(resampled))
                index_batch.append(normalize_flux_for_indices(resampled))
                processed += 1
                if len(flux_batch) >= args.batch_size:
                    flush()
            except Exception:
                rejected += 1
        if stop:
            break
    flush()

    result = {
        "device": str(device),
        "threshold": threshold,
        "files": len(files),
        "processed": processed,
        "rejected": rejected,
        "recalled": recalled,
        "saved": False,
    }
    print("[INFO] recall summary:", result)
    if not args.dry_run:
        print("[WARN] Candidate persistence is intentionally disabled; use --dry_run for validation runs.")
    return result


def main(argv=None):
    return recall(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
