# -*- coding: utf-8 -*-
"""Self-contained DESI recall/inference entry point for CSpec-DB-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cspec_db_net.config import CARBON_BINARY_LABEL, N_PIX, TARGET_WAVE, WAVE_END, WAVE_START
    from cspec_db_net.desi_io import PickleStreamWriter, iter_rows_from_pickle
    from cspec_db_net.models.cspec_db_net import CSpecDBNet
    from cspec_db_net.preprocessing import normalize_flux, normalize_flux_for_indices, resample_to_target_grid
else:
    from .config import CARBON_BINARY_LABEL, N_PIX, TARGET_WAVE, WAVE_END, WAVE_START
    from .desi_io import PickleStreamWriter, iter_rows_from_pickle
    from .models.cspec_db_net import CSpecDBNet
    from .preprocessing import normalize_flux, normalize_flux_for_indices, resample_to_target_grid


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PACKAGE_ROOT / "data" / "desi"
DEFAULT_OUTPUT_PATH = PACKAGE_ROOT / "outputs" / "recall" / "candidates.pkl"


def build_parser():
    parser = argparse.ArgumentParser(description="Recall carbon-star candidates from DESI pickle files.")
    parser.add_argument("--checkpoint", required=True, help="CSpec-DB-Net best_model.pt produced by training.py")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="DESI pickle file or directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.97,
        help="DESI carbon-star recall threshold (default: 0.97).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_chunk_size", type=int, default=1024)
    parser.add_argument("--max_raw_points", type=int, default=20000)
    parser.add_argument("--max_records", type=int, default=0, help="Stop after N valid records; 0 means unlimited")
    parser.add_argument("--max_files", type=int, default=0, help="Process at most N pickle files; 0 means unlimited")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output pickle for recalled source rows")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output pickle")
    parser.add_argument("--dry_run", action="store_true", help="Run inference and report counts without saving candidates")
    return parser


def _as_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return tuple(value)


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
        "wave_hidden": tuple(int(item) for item in _as_tuple(args.get("wave_hidden"), (256, 192))),
        "idx_hidden": tuple(int(item) for item in _as_tuple(args.get("idx_hidden"), (256, 192))),
        "fusion_hidden": tuple(int(item) for item in _as_tuple(args.get("fusion_hidden"), (256, 128))),
        "fusion_dropouts": tuple(float(item) for item in _as_tuple(args.get("fusion_dropouts"), (0.12, 0.08))),
    }


def load_model(checkpoint, device, threshold_override=None):
    ckpt = torch.load(checkpoint, map_location=device)
    model = CSpecDBNet(**_model_kwargs_from_checkpoint(ckpt)).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if threshold_override is None:
        if "best_threshold" not in ckpt:
            raise KeyError("Checkpoint has no best_threshold; pass --threshold explicitly")
        threshold = float(ckpt["best_threshold"])
    else:
        threshold = float(threshold_override)
    return model, threshold


def _input_files(input_path, max_files=0):
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.rglob("*.pkl"))
    else:
        raise FileNotFoundError(f"DESI input does not exist: {input_path}")
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No pickle files found under: {input_path}")
    return files


@torch.inference_mode()
def recall(args):
    if args.batch_size <= 0 or args.save_chunk_size <= 0:
        raise ValueError("batch_size and save_chunk_size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold = load_model(args.checkpoint, device, args.threshold)
    files = _input_files(args.input, args.max_files)
    writer = None if args.dry_run else PickleStreamWriter(args.output, overwrite=args.overwrite)

    processed = 0
    rejected = 0
    recalled = 0
    flux_batch: list[np.ndarray] = []
    index_batch: list[np.ndarray] = []
    row_batch: list[object] = []
    candidate_buffer: list[object] = []

    def persist_candidates(force=False):
        if writer is None or not candidate_buffer:
            return
        if not force and len(candidate_buffer) < args.save_chunk_size:
            return
        writer.write_rows(candidate_buffer)
        candidate_buffer.clear()

    def flush():
        nonlocal recalled
        if not flux_batch:
            return
        x = torch.from_numpy(np.stack(flux_batch)).to(device)
        x_idx = torch.from_numpy(np.stack(index_batch)).to(device)
        probs = torch.softmax(model(x, x_idx=x_idx)["bin_logits"], dim=1)[:, CARBON_BINARY_LABEL]
        positive_indices = torch.nonzero(probs >= threshold, as_tuple=False).flatten().cpu().tolist()
        recalled += len(positive_indices)
        if writer is not None:
            candidate_buffer.extend(row_batch[index] for index in positive_indices)
            persist_candidates()
        flux_batch.clear()
        index_batch.clear()
        row_batch.clear()

    stop = False
    for pkl_file in files:
        for row in iter_rows_from_pickle(pkl_file):
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
                if len(wave) > args.max_raw_points:
                    raise ValueError("raw spectrum exceeds max_raw_points")
                valid = np.isfinite(wave) & np.isfinite(flux)
                if int(valid.sum()) < 10:
                    raise ValueError("too few finite wavelength/flux samples")
                resampled = resample_to_target_grid(wave[valid], flux[valid], TARGET_WAVE)
                flux_batch.append(normalize_flux(resampled))
                index_batch.append(normalize_flux_for_indices(resampled))
                row_batch.append(row)
                processed += 1
                if len(flux_batch) >= args.batch_size:
                    flush()
            except Exception:
                rejected += 1
        if stop:
            break

    flush()
    persist_candidates(force=True)
    result = {
        "device": str(device),
        "threshold": threshold,
        "files": len(files),
        "processed": processed,
        "rejected": rejected,
        "recalled": recalled,
        "saved": bool(writer is not None and writer.rows_written > 0),
        "output": str(Path(args.output).resolve()) if writer is not None else None,
    }
    print("[INFO] recall summary:", result)
    return result


def main(argv=None):
    return recall(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
