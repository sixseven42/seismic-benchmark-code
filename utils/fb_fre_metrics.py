#!/usr/bin/env python3
"""Frequency-Binned Fidelity and Recovery Evaluation (FB-FRE) metrics.

Divides the reference and prediction into frequency bands derived from the
reference amplitude spectrum and computes per-band recovery quality: BNE (Band
Normalized Error), BER (Band Energy Ratio), and BCC (Band Correlation
Coefficient).

Reference: idea/frequency_binned_fidelity_recovery_evaluation.md
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


_DEFAULT_BAND_RATIOS: Tuple[float, float, float, float] = (0.20, 0.30, 0.30, 0.20)
_DEFAULT_BAND_NAMES: Tuple[str, str, str, str] = ("low", "mid", "high", "very_high")


def _time_axis_index(data: np.ndarray, axis: int) -> int:
    """Return the normalized time axis index."""
    return axis % data.ndim


def _average_along_non_time_axes(arr: np.ndarray, axis: int) -> np.ndarray:
    """Average ``arr`` over all axes except ``axis``."""
    time_axis = _time_axis_index(arr, axis)
    axes_to_avg = tuple(i for i in range(arr.ndim) if i != time_axis)
    if not axes_to_avg:
        return arr
    return np.mean(arr, axis=axes_to_avg)


def compute_average_amplitude_spectrum(
    reference: np.ndarray,
    dt: float,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean amplitude spectrum of ``reference`` and corresponding frequencies.

    Parameters
    ----------
    reference : reference clean data, any shape.
    dt : time sampling interval in seconds.
    axis : time axis along which FFT is performed. Default ``-1``.

    Returns
    -------
    freqs : 1-D array of frequencies in Hz.
    mean_amp : 1-D array of mean amplitudes across all non-time dimensions.
    """
    ref = reference.astype(np.float64)
    n_time = ref.shape[_time_axis_index(ref, axis)]
    freqs = np.fft.rfftfreq(n_time, d=dt)
    spectrum = np.fft.rfft(ref, axis=axis)
    mean_amp = _average_along_non_time_axes(np.abs(spectrum), axis=axis)
    return freqs, mean_amp


def estimate_effective_band(
    reference: np.ndarray,
    dt: float,
    axis: int = -1,
    method: str = "threshold",
    rel_threshold: float = 0.001,
    cumulative_ratio: float = 0.95,
) -> Tuple[float, float]:
    """Estimate the effective frequency band of ``reference``.

    Parameters
    ----------
    reference : reference clean data, any shape.
    dt : time sampling interval in seconds.
    axis : time axis along which FFT is performed. Default ``-1``.
    method : ``"threshold"`` (relative to peak power) or ``"cumulative"``
             (central energy ratio).
    rel_threshold : fraction of peak power used in ``"threshold"`` mode.
    cumulative_ratio : central energy fraction used in ``"cumulative"`` mode.

    Returns
    -------
    f_min, f_max : estimated effective frequency band in Hz.
    """
    ref = reference.astype(np.float64)
    n_time = ref.shape[_time_axis_index(ref, axis)]
    freqs = np.fft.rfftfreq(n_time, d=dt)
    spectrum = np.fft.rfft(ref, axis=axis)
    mean_power = _average_along_non_time_axes(np.abs(spectrum) ** 2, axis=axis)

    if method == "threshold":
        peak = float(np.max(mean_power))
        threshold = peak * rel_threshold
        valid = freqs[mean_power >= threshold]
        if valid.size == 0:
            return float(freqs[0]), float(freqs[-1])
        f_min = float(valid[0])
        f_max = float(valid[-1])
    elif method == "cumulative":
        total = float(np.sum(mean_power))
        cumsum = np.cumsum(mean_power)
        lower = total * (1.0 - cumulative_ratio) / 2.0
        upper = total * (1.0 + cumulative_ratio) / 2.0
        f_min = float(freqs[np.searchsorted(cumsum, lower, side="left")])
        f_max = float(freqs[np.searchsorted(cumsum, upper, side="right")])
    else:
        raise ValueError(f"Unknown effective-band method: {method}")

    f_min = max(float(freqs[0]), f_min)
    f_max = min(float(freqs[-1]), f_max)
    return f_min, f_max


def build_auto_bands(
    f_min: float,
    f_max: float,
    ratios: Sequence[float] = _DEFAULT_BAND_RATIOS,
    names: Sequence[str] = _DEFAULT_BAND_NAMES,
) -> Tuple[Tuple[str, Tuple[float, float]], ...]:
    """Build contiguous frequency bands from ``f_min`` to ``f_max`` using ``ratios``.

    Parameters
    ----------
    f_min : lower bound of the effective band (Hz).
    f_max : upper bound of the effective band (Hz).
    ratios : relative widths of each band. Must sum to 1.0.
    names : name for each band.

    Returns
    -------
    bands : tuple of ``(name, (fmin, fmax))`` in ascending frequency order.
    """
    if len(ratios) != len(names):
        raise ValueError("ratios and names must have the same length")
    if len(ratios) == 0:
        raise ValueError("at least one band is required")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    if f_max <= f_min:
        raise ValueError(f"f_max must be greater than f_min: {f_min}, {f_max}")

    width = f_max - f_min
    bands = []
    cursor = f_min
    cumulative = 0.0
    for i, (name, ratio) in enumerate(zip(names, ratios)):
        cumulative += ratio
        next_cursor = f_min + width * cumulative
        if i == len(ratios) - 1:
            next_cursor = f_max
        bands.append((name, (float(cursor), float(next_cursor))))
        cursor = next_cursor
    return tuple(bands)


def _cosine_taper_mask(
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    taper_width: float,
) -> np.ndarray:
    """Build a band-pass mask with cosine taper edges."""
    mask = np.zeros_like(freqs, dtype=np.float64)

    flat_low = fmin + taper_width
    flat_high = fmax - taper_width
    if flat_low < flat_high:
        mask[(freqs >= flat_low) & (freqs <= flat_high)] = 1.0

    # Rising taper
    rise = (freqs >= fmin) & (freqs < flat_low)
    if np.any(rise):
        t = (freqs[rise] - fmin) / max(taper_width, 1e-12)
        mask[rise] = 0.5 * (1.0 - np.cos(np.pi * t))

    # Falling taper
    fall = (freqs > flat_high) & (freqs <= fmax)
    if np.any(fall):
        t = (freqs[fall] - flat_high) / max(taper_width, 1e-12)
        mask[fall] = 0.5 * (1.0 + np.cos(np.pi * t))

    return mask


def _band_mask(
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    taper_width: float,
) -> np.ndarray:
    """Build band-pass mask (rectangular or cosine-tapered)."""
    if taper_width <= 0.0:
        return ((freqs >= fmin) & (freqs <= fmax)).astype(np.float64)
    return _cosine_taper_mask(freqs, fmin, fmax, taper_width)


def _bandpass_filter(
    data: np.ndarray,
    dt: float,
    fmin: float,
    fmax: float,
    axis: int = -1,
    taper_width: float = 0.0,
) -> np.ndarray:
    """Apply band-pass filter along ``axis`` via FFT."""
    spectrum = np.fft.rfft(data, axis=axis)
    n_time = data.shape[axis]
    freqs = np.fft.rfftfreq(n_time, d=dt)

    mask = _band_mask(freqs, fmin, fmax, taper_width)

    # Broadcast mask to spectrum shape
    reshape_shape = [1] * spectrum.ndim
    reshape_shape[axis] = -1
    mask_broadcast = mask.reshape(reshape_shape)

    filtered_spectrum = spectrum * mask_broadcast

    # Inverse FFT back to real domain
    filtered = np.fft.irfft(filtered_spectrum, n=n_time, axis=axis)
    return filtered.astype(np.float64, copy=False)


def frequency_binned_fidelity_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    dt: float,
    bands: Sequence[Tuple[str, Tuple[float, float]]] | str | None = None,
    axis: int = -1,
    taper_width: float = 2.0,
    rel_threshold: float = 0.001,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Any]]:
    """Compute frequency-binned fidelity and recovery evaluation metrics.

    Parameters
    ----------
    reference : reference clean data, any shape.
    prediction : model output, same shape as ``reference``.
    dt : time sampling interval in seconds (e.g. 0.004).
    bands : sequence of ``(name, (fmin, fmax))`` tuples, ``"auto"``, or ``None``.
            ``"auto"`` / ``None`` estimates the effective band from ``reference``
            and splits it into low/mid/high/very_high bins using
            ``_DEFAULT_BAND_RATIOS``.
    axis : time axis along which FFT is performed. Default ``-1``.
    taper_width : cosine taper width in Hz at band edges. ``0.0`` = rectangular
        passband, which causes Gibbs ringing in the time domain. A positive
        value (default ``2.0`` Hz) suppresses ringing by smoothing the cutoff.
    rel_threshold : relative power threshold used when ``bands`` is ``"auto"``.
    eps : small constant to avoid division by zero.

    Returns
    -------
    result : dict mapping band name to dict with ``BNE``, ``BER``, ``BCC``,
             ``ref_band_energy``, ``pred_band_energy``, ``valid``,
             ``frequency_range``, ``nyquist``, ``effective_band``,
             ``auto_band_ratios``.
    """
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs prediction {prediction.shape}."
        )

    ref = reference.astype(np.float64)
    pred = prediction.astype(np.float64)

    nyquist = 1.0 / (2.0 * dt)

    effective_band = None
    auto_band_ratios = None
    if bands is None or bands == "auto":
        effective_band = estimate_effective_band(
            ref, dt, axis=axis, method="threshold", rel_threshold=rel_threshold
        )
        bands = build_auto_bands(*effective_band)
        auto_band_ratios = _DEFAULT_BAND_RATIOS

    result: Dict[str, Dict[str, Any]] = {}

    for band_name, (fmin, fmax) in bands:
        # Validate against Nyquist
        if fmin >= nyquist or fmax > nyquist:
            result[band_name] = {
                "BNE": np.nan,
                "BER": np.nan,
                "BCC": np.nan,
                "ref_band_energy": np.nan,
                "pred_band_energy": np.nan,
                "valid": False,
                "frequency_range": (fmin, fmax),
                "nyquist": nyquist,
                "truncated": False,
                "effective_band": effective_band,
                "auto_band_ratios": auto_band_ratios,
            }
            continue

        # Apply identical band-pass filter to both
        ref_band = _bandpass_filter(ref, dt, fmin, fmax, axis=axis, taper_width=taper_width)
        pred_band = _bandpass_filter(pred, dt, fmin, fmax, axis=axis, taper_width=taper_width)

        r_b = ref_band.ravel()
        p_b = pred_band.ravel()

        # Band energies (sum of squares)
        ref_energy = float(np.sum(r_b ** 2))
        pred_energy = float(np.sum(p_b ** 2))

        # BNE — Band Normalized Error
        err_sq = float(np.sum((p_b - r_b) ** 2))
        bne = float(np.sqrt(err_sq) / (np.sqrt(ref_energy) + eps))

        # BER — Band Energy Ratio
        ber = float(pred_energy / (ref_energy + eps))

        # BCC — Band Correlation Coefficient
        cc_num = float(np.sum(r_b * p_b))
        cc_den = float(np.sqrt(ref_energy) * np.sqrt(pred_energy) + eps)
        bcc = float(cc_num / cc_den)

        result[band_name] = {
            "BNE": bne,
            "BER": ber,
            "BCC": bcc,
            "ref_band_energy": ref_energy,
            "pred_band_energy": pred_energy,
            "valid": True,
            "frequency_range": (fmin, fmax),
            "nyquist": nyquist,
            "truncated": False,
            "effective_band": effective_band,
            "auto_band_ratios": auto_band_ratios,
        }

    return result


def _demo() -> None:
    """Simple sanity check with synthetic multi-frequency data."""
    np.random.seed(0)
    dt = 0.004
    t = np.arange(0, 1.0, dt)
    nx = 64

    # Reference: 10 Hz + 25 Hz + 45 Hz
    ref = np.zeros((len(t), nx), dtype=np.float32)
    for x in range(nx):
        ref[:, x] = (
            np.sin(2 * np.pi * 10 * t)
            + np.sin(2 * np.pi * 25 * t)
            + np.sin(2 * np.pi * 45 * t)
        )

    # Prediction: high-frequency suppressed (over-smoothing)
    pred = np.zeros_like(ref)
    for x in range(nx):
        pred[:, x] = (
            np.sin(2 * np.pi * 10 * t)
            + np.sin(2 * np.pi * 25 * t)
            + 0.3 * np.sin(2 * np.pi * 45 * t)
        )

    result = frequency_binned_fidelity_metrics(ref, pred, dt=dt)
    print("FB-FRE demo result:")
    for k, v in result.items():
        print(f"  {k}: BNE={v['BNE']:.4f}, BER={v['BER']:.4f}, BCC={v['BCC']:.4f}, valid={v['valid']}")

    # The 45 Hz component is suppressed; at least one upper band should show BER < 1.
    upper_bers = [
        v["BER"] for k, v in result.items()
        if k in ("high", "very_high") and v.get("valid")
    ]
    assert any(ber < 1.0 for ber in upper_bers), (
        "High-frequency bands should show BER < 1 when high freq is suppressed."
    )
    print("Sanity check passed: at least one upper-frequency band has BER < 1.0.")


if __name__ == "__main__":
    _demo()
