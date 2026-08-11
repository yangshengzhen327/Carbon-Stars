# -*- coding: utf-8 -*-
"""Data discovery and spectrum preprocessing for CSpec-DB-Net."""

import io
import os
import random
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
from astropy.io import fits

import torch
from torch.utils.data import Dataset

try:
    with redirect_stderr(io.StringIO()):
        from scipy.ndimage import gaussian_filter1d as scipy_gaussian_filter1d
except Exception:
    scipy_gaussian_filter1d = None

from .config import (
    CARBON_BAND_IDXS,
    CARBON_BINARY_LABEL,
    CARBON_KEYWORD,
    NON_CARBON_BINARY_LABEL,
    TARGET_WAVE,
    WAVE_END,
    WAVE_START,
)

def read_lamost_fits(path):
    with fits.open(path, memmap=False) as hdul:
        if len(hdul) < 2:
            raise RuntimeError(f"{path} missing hdul[1]")

        data = hdul[1].data
        header = hdul[1].header
        if data is None:
            raise RuntimeError(f"{path} missing hdul[1].data")

        wave, flux = None, None
        names = getattr(data, "names", None)
        if names:
            upper_to_name = {name.upper(): name for name in names}
            if "FLUX" in upper_to_name and len(data) > 0:
                row0 = data[0]
                flux = np.asarray(row0[upper_to_name["FLUX"]], dtype=np.float32)
                if "WAVELENGTH" in upper_to_name:
                    wave = np.asarray(row0[upper_to_name["WAVELENGTH"]], dtype=np.float32)

        arr = np.asarray(data)
        if flux is None and arr.ndim == 2:
            if arr.shape[0] == 5:
                flux = np.asarray(arr[0], dtype=np.float32)
                wave = np.asarray(arr[2], dtype=np.float32)
            elif arr.shape[1] == 5:
                flux = np.asarray(arr[:, 0], dtype=np.float32)
                wave = np.asarray(arr[:, 2], dtype=np.float32)
            else:
                flux = np.asarray(arr[0] if arr.shape[0] < arr.shape[1] else arr[:, 0], dtype=np.float32)
        elif flux is None and arr.ndim == 1:
            flux = np.asarray(arr, dtype=np.float32)
        elif flux is None:
            raise RuntimeError(f"{path} invalid data shape: {arr.shape}")

        flux = np.asarray(flux, dtype=np.float32).reshape(-1)
        if wave is not None:
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            if len(wave) != len(flux):
                wave = None

        if wave is None:
            if "COEFF0" in header and "COEFF1" in header:
                coeff0 = header["COEFF0"]
                coeff1 = header["COEFF1"]
                pix = np.arange(len(flux), dtype=np.float32)
                wave = (10 ** (coeff0 + coeff1 * pix)).astype(np.float32)
            else:
                wave = np.linspace(WAVE_START, WAVE_END, len(flux), dtype=np.float32)

    return wave, flux


def robust_fix(x):
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    if np.isnan(x).all():
        return np.zeros_like(x, dtype=np.float32)
    med = np.nanmedian(x)
    x = np.where(np.isnan(x), med, x)
    return x.astype(np.float32)


def resample_to_target_grid(wave, flux, target_wave=TARGET_WAVE):
    wave = robust_fix(wave)
    flux = robust_fix(flux)

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    uniq_wave, uniq_idx = np.unique(wave, return_index=True)
    wave = uniq_wave
    flux = flux[uniq_idx]

    valid = (wave >= target_wave[0] - 50) & (wave <= target_wave[-1] + 50)
    if valid.sum() < 10:
        return np.zeros_like(target_wave, dtype=np.float32)

    wave = wave[valid]
    flux = flux[valid]
    return np.interp(target_wave, wave, flux, left=flux[0], right=flux[-1]).astype(np.float32)


def normalize_flux(flux):
    flux = robust_fix(flux)
    med = np.median(flux)
    if abs(med) < 1e-8:
        med = 1.0
    flux = flux / med
    p1, p99 = np.percentile(flux, [1, 99])
    flux = np.clip(flux, p1, p99)
    mu = flux.mean()
    std = flux.std()
    if std < 1e-8:
        std = 1.0
    flux = (flux - mu) / std
    return flux.astype(np.float32)


def z_zero(flux):
    return normalize_flux(flux)


def normalize_flux_for_indices(flux, clip_percentiles=(0.5, 99.5)):
    flux = robust_fix(flux)
    med = np.median(flux)
    if abs(med) < 1e-8:
        med = 1.0
    flux = flux / med
    if clip_percentiles is not None:
        lo, hi = np.percentile(flux, clip_percentiles)
        flux = np.clip(flux, lo, hi)
    return flux.astype(np.float32)

def random_response_curve(x):
    t = np.linspace(-1, 1, len(x), dtype=np.float32)
    a = np.random.uniform(-0.08, 0.08)
    b = np.random.uniform(-0.06, 0.06)
    return x * (1.0 + a * t + b * (t ** 2))


def random_noise(x):
    sigma = np.random.uniform(0.002, 0.02)
    return x + np.random.randn(len(x)).astype(np.float32) * sigma


def random_blur(x):
    sigma = np.random.uniform(0.0, 1.2)
    if sigma > 1e-6:
        if scipy_gaussian_filter1d is not None:
            return scipy_gaussian_filter1d(x, sigma=sigma).astype(np.float32)
        radius = max(1, int(round(3.0 * sigma)))
        grid = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(grid ** 2) / (2.0 * sigma * sigma + 1e-6))
        kernel /= kernel.sum()
        return np.convolve(x, kernel, mode="same").astype(np.float32)
    return x


def random_shift(x):
    shift = np.random.uniform(-2.0, 2.0)
    src = np.arange(len(x), dtype=np.float32)
    dst = src + shift
    return np.interp(src, dst, x, left=x[0], right=x[-1]).astype(np.float32)


def feature_aware_mask(x):
    x = x.copy()
    r = np.random.rand()

    key_mask = np.zeros(len(x), dtype=np.bool_)
    for a, b in CARBON_BAND_IDXS:
        key_mask[a:b] = True

    if r < 0.80:
        n_seg = np.random.randint(1, 4)
        for _ in range(n_seg):
            seg_len = np.random.randint(16, 65)
            tries = 0
            while True:
                s = np.random.randint(0, len(x) - seg_len)
                e = s + seg_len
                if (~key_mask[s:e]).mean() > 0.8 or tries > 20:
                    x[s:e] = 0.0
                    break
                tries += 1
    elif r < 0.97:
        a, b = random.choice(CARBON_BAND_IDXS)
        x[a:b] *= np.random.uniform(0.65, 0.9)
    else:
        idx = np.random.randint(0, len(CARBON_BAND_IDXS))
        a, b = CARBON_BAND_IDXS[idx]
        x[a:b] *= np.random.uniform(0.25, 0.5)

    return x.astype(np.float32)


def augment_spectrum(x):
    if np.random.rand() < 0.40:
        x = random_response_curve(x)
    if np.random.rand() < 0.35:
        x = random_noise(x)
    if np.random.rand() < 0.20:
        x = random_blur(x)
    if np.random.rand() < 0.20:
        x = random_shift(x)
    if np.random.rand() < 0.08:
        x = feature_aware_mask(x)
    mu = x.mean()
    std = x.std()
    if std < 1e-8:
        std = 1.0
    return ((x - mu) / std).astype(np.float32)


def augment_dual_branch_flux(flux):
    flux = robust_fix(flux)
    med = np.median(flux)
    if abs(med) < 1e-8:
        med = 1.0
    x = (flux / med).astype(np.float32)
    if np.random.rand() < 0.40:
        x = random_response_curve(x)
    if np.random.rand() < 0.35:
        x = random_noise(x)
    if np.random.rand() < 0.20:
        x = random_blur(x)
    if np.random.rand() < 0.20:
        x = random_shift(x)
    if np.random.rand() < 0.08:
        x = feature_aware_mask(x)
    return x.astype(np.float32)


def recursive_list_fits(folder):
    out = []
    for root, _, files in os.walk(folder):
        for fn in files:
            low = fn.lower()
            if low.endswith(".fits") or low.endswith(".fit") or low.endswith(".fits.gz") or low.endswith(".fz"):
                out.append(os.path.join(root, fn))
    return sorted(out)


def is_carbon_name(text):
    return CARBON_KEYWORD in str(text).lower()


OBJECT_ID_HEADER_KEYS = (
    "GAIA_SOURCE_ID",
    "GAIAEDR3_SOURCE_ID",
    "GAIADR3_SOURCE_ID",
    "SOURCE_ID",
    "TARGETID",
    "OBJID",
    "DESIG",
    "DESIGNATION",
)


def read_object_id(path):
    """Return a stable astronomical-source key from FITS metadata.

    Catalog/source identifiers are preferred. If none is present, the target
    coordinates are used; repeated observations of the same coordinates then
    remain in the same data partition. A filename is deliberately not accepted
    because LAMOST filenames normally identify observations, not unique objects.
    """
    with fits.open(path, memmap=False) as hdul:
        headers = [hdu.header for hdu in hdul]
        for key in OBJECT_ID_HEADER_KEYS:
            for header in headers:
                value = header.get(key)
                if value is not None and str(value).strip() not in {"", "-", "None", "nan"}:
                    return f"{key}:{str(value).strip()}"

        for ra_key, dec_key in (("RA", "DEC"), ("OBJRA", "OBJDEC"), ("PLUG_RA", "PLUG_DEC")):
            for header in headers:
                if ra_key not in header or dec_key not in header:
                    continue
                try:
                    ra = float(header[ra_key]) % 360.0
                    dec = float(header[dec_key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(ra) and np.isfinite(dec) and -90.0 <= dec <= 90.0:
                    return f"SKYCOORD:{ra:.6f}:{dec:.6f}"

    raise RuntimeError(
        f"Cannot determine a real astronomical object ID for {path}. "
        f"Expected one of {OBJECT_ID_HEADER_KEYS} or a valid RA/DEC pair."
    )


def build_samples_from_root(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise RuntimeError(f"{root_dir} does not exist")
    if not root.is_dir():
        raise RuntimeError(f"{root_dir} is not a directory")

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name.lower())
    carbon_subdirs = [d for d in subdirs if is_carbon_name(d.name)]

    if not carbon_subdirs:
        raise RuntimeError(f"{root_dir} missing a folder containing '{CARBON_KEYWORD}'")
    if len(subdirs) < 2:
        raise RuntimeError(f"{root_dir} must contain at least 2 subfolders")

    print(f"\n[INFO] Subfolder order in {root_dir}:")
    for i, d in enumerate(subdirs):
        tag = "positive(carbon)" if is_carbon_name(d.name) else "negative"
        print(f"  {i + 1}. {d.name} -> {tag}")

    non_carbon_names = [d.name for d in subdirs if not is_carbon_name(d.name)]
    aux_to_idx = {name: idx + 1 for idx, name in enumerate(non_carbon_names)}
    for d in carbon_subdirs:
        aux_to_idx[d.name] = 0

    samples = []
    for sub in subdirs:
        files = recursive_list_fits(str(sub))
        class_name = sub.name
        label_aux = aux_to_idx[class_name]
        label_bin = CARBON_BINARY_LABEL if is_carbon_name(sub.name) else NON_CARBON_BINARY_LABEL
        for fp in files:
            object_id = read_object_id(fp)
            samples.append(
                {
                    "path": fp,
                    "label_bin": label_bin,
                    "label_aux": label_aux,
                    "label_aux_name": class_name,
                    "object_id": object_id,
                }
            )
    return samples, aux_to_idx


class LAMOSTDataset(Dataset):
    def __init__(self, samples, augment=False, cache=False, preprocess_mode="dual_branch"):
        self.samples = samples
        self.augment = augment
        self.cache = cache
        self.preprocess_mode = preprocess_mode
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def load_one(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]
        wave, flux = read_lamost_fits(sample["path"])
        flux_resampled = resample_to_target_grid(wave, flux, TARGET_WAVE)
        if self.cache:
            self._cache[idx] = flux_resampled
        return flux_resampled

    def __getitem__(self, idx):
        sample = self.samples[idx]
        flux_resampled = self.load_one(idx)

        if self.preprocess_mode == "dual_branch":
            if self.augment:
                flux_resampled = augment_dual_branch_flux(flux_resampled)
            x_main = normalize_flux(flux_resampled)
            x_idx = normalize_flux_for_indices(flux_resampled)
        elif self.preprocess_mode == "z_zero":
            x_main = z_zero(flux_resampled)
            if self.augment:
                x_main = augment_spectrum(x_main)
            x_idx = normalize_flux_for_indices(flux_resampled)
        else:
            raise ValueError(f"Unsupported preprocess_mode: {self.preprocess_mode}")

        return {
            "flux": torch.tensor(x_main, dtype=torch.float32),
            "flux_idx": torch.tensor(x_idx, dtype=torch.float32),
            "label_bin": torch.tensor(sample["label_bin"], dtype=torch.long),
            "label_aux": torch.tensor(sample["label_aux"], dtype=torch.long),
            "path": sample["path"],
        }
