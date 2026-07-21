#!/usr/bin/env python3
"""Energy-Binned Weak Signal Evaluation (EB-WSE) metrics.

Divides the reference data into energy quantile bins and computes per-bin
recovery quality: NE (Normalized Error) and SNR. Designed to reveal
weak-signal loss that global SNR/MSE may hide.

Reference: idea/energy_binned_weak_signal_evaluation.md
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


def _uniform_filter_numpy(arr: np.ndarray, size: int) -> np.ndarray:
    """Pure-numpy N-D box filter with 'same' output and edge padding."""
    if size <= 1:
        return arr.astype(np.float64, copy=False)

    result = arr.astype(np.float64, copy=False)
    half = size // 2
    for axis in range(arr.ndim):
        moved = np.moveaxis(result, axis, -1)
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(half, half)]
        padded = np.pad(moved, pad_width, mode="edge")
        cum = np.cumsum(padded, axis=-1)
        filtered = (cum[..., size:] - cum[..., :-size]) / size
        result = np.moveaxis(filtered, -1, axis)
    return result


def _smooth_energy(reference: np.ndarray, sigma: float) -> np.ndarray:
    """Compute smoothed energy map E = sqrt(gaussian_filter(S^2, sigma)).

    Falls back to a simple uniform average if ``scipy`` is unavailable.
    """
    ref_sq = reference.astype(np.float64) ** 2
    try:
        from scipy.ndimage import gaussian_filter

        return np.sqrt(gaussian_filter(ref_sq, sigma=sigma))
    except ImportError:
        # scipy not installed: use a numpy-only uniform box filter.
        if sigma <= 0.0:
            return np.sqrt(ref_sq)
        half = max(1, int(round(sigma)))
        return np.sqrt(_uniform_filter_numpy(ref_sq, size=2 * half + 1))


def energy_binned_weak_signal_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    bins: Sequence[Tuple[int, int]] = ((5, 20), (20, 40), (40, 70), (70, 100)),
    smooth_sigma: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Any]]:
    """Compute energy-binned weak-signal evaluation metrics.

    Parameters
    ----------
    reference    : reference clean data, any shape (broadcasted to common size).
    prediction   : model output, same shape as ``reference``.
    bins         : sequence of (low_percentile, high_percentile) tuples.
                   Default covers very-weak, weak, medium, and strong bands.
    smooth_sigma : Gaussian smoothing sigma for the energy map. ``0.0`` disables.
    eps          : small constant to avoid division by zero.

    Returns
    -------
    result       : ``{bin_key: {"NE", "SNR", "num_samples", "energy_percentile_range", "energy_mean", "ratio_to_total"}}``.
                 ``bin_key`` is ``very_weak_5_20`` etc. for the default bins,
                 or ``bin_low_high`` for custom bins.

    Metrics meaning
    ---------------
    - NE (Normalized Error):  RMS(prediction - reference) / (RMS(reference) + eps).
      Smaller is better.  >1 means the error is larger than the signal itself.
    - SNR (Signal-to-Noise Ratio): 10*log10(sum(reference^2) / sum((prediction-reference)^2)).
      Higher is better.
    """
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs prediction {prediction.shape}."
        )

    ref = reference.astype(np.float64)
    pred = prediction.astype(np.float64)

    # Step 1: smoothed energy map on reference
    energy = _smooth_energy(ref, smooth_sigma)

    # Flatten for easier masking (shape doesn't matter for these scalar metrics)
    ref_flat = ref.ravel()
    pred_flat = pred.ravel()
    energy_flat = energy.ravel()
    total_samples = ref_flat.size

    # Exclude points where reference is strictly zero (no signal to evaluate)
    valid_mask = ref_flat != 0.0
    valid_indices = np.where(valid_mask)[0]
    n_valid = valid_indices.size

    # Sort valid points by energy to create rank-based bins
    valid_energy = energy_flat[valid_indices]
    valid_sorted = np.argsort(valid_energy, kind="mergesort")
    sorted_valid_indices = valid_indices[valid_sorted]

    result: Dict[str, Dict[str, Any]] = {}

    for low, high in bins:
        # Bin key naming
        if low == 5 and high == 20:
            key = "very_weak_5_20"
        elif low == 20 and high == 40:
            key = "weak_20_40"
        elif low == 40 and high == 70:
            key = "medium_40_70"
        elif low == 70 and high == 100:
            key = "strong_70_100"
        else:
            key = f"bin_{low}_{high}"

        # Rank-based indices on valid (non-zero) points guarantee exact proportions
        idx_low = int(n_valid * low / 100.0)
        idx_high = int(n_valid * high / 100.0)
        if high == 100:
            idx_high = n_valid

        mask_indices = sorted_valid_indices[idx_low:idx_high]
        mask = np.zeros_like(energy_flat, dtype=bool)
        mask[mask_indices] = True

        num_samples = int(mask.sum())
        ratio = num_samples / n_valid if n_valid > 0 else 0.0
        ratio_to_total = num_samples / total_samples if total_samples > 0 else 0.0

        if num_samples == 0:
            result[key] = {
                "NE": np.nan,
                "SNR": np.nan,
                "num_samples": 0,
                "ratio_to_total": ratio_to_total,
                "energy_mean": np.nan,
                "mean_ref_sq": np.nan,
                "mean_pred_sq": np.nan,
                "mean_err_sq": np.nan,
                "energy_percentile_range": (low, high),
            }
            continue

        r_m = ref_flat[mask]
        p_m = pred_flat[mask]
        e_m = energy_flat[mask]

        # Energy statistics for diagnostics
        energy_mean = float(np.mean(e_m))
        mean_ref_sq = float(np.mean(r_m ** 2))
        mean_pred_sq = float(np.mean(p_m ** 2))
        mean_err_sq = float(np.mean((p_m - r_m) ** 2))

        # Normalized Error (mean-based, sample-count invariant)
        # NE = RMS(prediction - reference) / (RMS(reference) + eps)
        ne_num = np.sqrt(mean_err_sq)
        ne_den = np.sqrt(mean_ref_sq) + eps
        ne = float(ne_num / ne_den)

        # Per-bin SNR (dB) — same formula as utils.metrics._snr_numpy
        signal = float(np.sum(r_m ** 2))
        noise = float(np.sum((p_m - r_m) ** 2))
        if noise == 0.0:
            snr = float("inf") if signal > 0.0 else float("nan")
        elif signal == 0.0:
            snr = float("-inf")
        else:
            snr = float(10.0 * np.log10(signal / noise))

        result[key] = {
            "NE": ne,
            "SNR": snr,
            "num_samples": num_samples,
            "ratio_to_total": ratio_to_total,
            "energy_mean": energy_mean,
            "mean_ref_sq": mean_ref_sq,
            "mean_pred_sq": mean_pred_sq,
            "mean_err_sq": mean_err_sq,
            "energy_percentile_range": (low, high),
        }

    return result


def _demo() -> None:
    """Simple sanity-check with random data."""
    np.random.seed(0)
    ref = np.random.randn(256, 64).astype(np.float32)
    pred = ref + np.random.randn(256, 64).astype(np.float32) * 0.1

    result = energy_binned_weak_signal_metrics(ref, pred)
    print("EB-WSE demo result:")
    for k, v in result.items():
        print(f"  {k}: NE={v['NE']:.4f}, SNR={v['SNR']:.2f}dB, "
              f"samples={v['num_samples']}, ratio={v['ratio_to_total']:.3%}, E_mean={v['energy_mean']:.2e}")


if __name__ == "__main__":
    _demo()
