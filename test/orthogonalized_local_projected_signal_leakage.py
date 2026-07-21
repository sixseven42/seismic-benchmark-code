#!/usr/bin/env python3
"""Orthogonalized Local Projected Signal Leakage (O-LPSL) metric.

Quantifies how much valid signal a model accidentally removes in denoising /
coherent-noise attenuation tasks. The metric orthogonalizes both the extra
removal component and the signal label against the true noise before measuring
their local projection, so coherent noise that naturally correlates with signal
does not inflate the score.

Reference: idea/orthogonalized_local_projected_signal_leakage.md
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _local_sum(arr: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """Local window sum via scipy's uniform_filter."""
    try:
        from scipy.ndimage import uniform_filter
    except ImportError as exc:
        raise ImportError("O-LPSL requires scipy for uniform_filter.") from exc
    area = window[0] * window[1]
    return uniform_filter(arr, size=window, mode="reflect") * area


def orthogonalized_local_projected_signal_leakage(
    input_data: np.ndarray,
    signal_label: np.ndarray,
    prediction: np.ndarray,
    prediction_type: str = "signal",
    noise_label: np.ndarray | None = None,
    window: Tuple[int, int] = (31, 31),
    eps: float = 1e-8,
    separability_threshold: float = 0.05,
    return_maps: bool = False,
) -> Dict[str, Any]:
    """Compute O-LPSL for a single 2D shot gather.

    Parameters
    ----------
    input_data : noisy input ``X``.
    signal_label : clean signal ``S``.
    prediction : model output. Interpreted as ``S_hat`` if ``prediction_type``
                 is ``"signal"``, or ``N_hat`` if ``"noise"``.
    prediction_type : ``"signal"`` or ``"noise"``.
    noise_label : true noise ``N``. If ``None``, computed as ``X - S``.
    window : local window size ``(nt, nx)``.
    eps : small constant for division stability.
    separability_threshold : minimum ``rho_i`` for a window to contribute.
    return_maps : if True, return intermediate maps for diagnostics.

    Returns
    -------
    result : dict with ``O_LPSL``, ``valid_ratio``, sample counts, params,
             and optionally diagnostic maps.
    """
    if input_data.shape != signal_label.shape or input_data.shape != prediction.shape:
        raise ValueError("Shape mismatch among input_data, signal_label, and prediction.")
    if noise_label is not None and noise_label.shape != input_data.shape:
        raise ValueError("Shape mismatch: noise_label must match input_data.")

    X = input_data.astype(np.float64)
    S = signal_label.astype(np.float64)
    P = prediction.astype(np.float64)
    N = noise_label.astype(np.float64) if noise_label is not None else (X - S)

    # Step 1: model removal component R
    if prediction_type == "signal":
        R = X - P
    elif prediction_type == "noise":
        R = P
    else:
        raise ValueError(f"prediction_type must be 'signal' or 'noise', got {prediction_type}")

    # Step 2: extra removal D = R - N
    D = R - N

    # Step 3: local noise energy
    sum_N2 = _local_sum(N ** 2, window)

    # Step 4: orthogonalize D and S against N
    beta_D = _local_sum(D * N, window) / (sum_N2 + eps)
    D_perp = D - beta_D * N

    beta_S = _local_sum(S * N, window) / (sum_N2 + eps)
    S_perp = S - beta_S * N

    # Step 5: local leakage ratio alpha
    sum_Sperp2 = _local_sum(S_perp ** 2, window)
    sum_Dperp_Sperp = _local_sum(D_perp * S_perp, window)
    alpha = sum_Dperp_Sperp / (sum_Sperp2 + eps)
    alpha_positive = np.maximum(alpha, 0.0)

    # Step 6: separability and valid mask
    sum_S2 = _local_sum(S ** 2, window)
    separability = sum_Sperp2 / (sum_S2 + eps)
    valid_mask = separability > separability_threshold

    # Step 7: weighted O-LPSL
    weights = sum_Sperp2 * valid_mask.astype(np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= eps:
        olpsl = float("nan")
    else:
        olpsl = float(np.sqrt(np.sum(weights * alpha_positive ** 2) / weight_sum))

    num_total = int(input_data.size)
    num_valid = int(valid_mask.sum())

    result: Dict[str, Any] = {
        "O_LPSL": olpsl,
        "valid_ratio": num_valid / num_total if num_total > 0 else 0.0,
        "num_valid_samples": num_valid,
        "num_total_samples": num_total,
        "window": window,
        "separability_threshold": separability_threshold,
        "prediction_type": prediction_type,
    }

    if return_maps:
        result.update({
            "alpha_map": alpha,
            "alpha_positive_map": alpha_positive,
            "separability_map": separability,
            "valid_mask": valid_mask,
            "D_perp": D_perp,
            "S_perp": S_perp,
            "beta_D_map": beta_D,
            "beta_S_map": beta_S,
            "sum_Sperp2_map": sum_Sperp2,
            "weight_map": weights,
            "weight_sum": weight_sum,
        })

    return result


def _demo() -> None:
    """Sanity-check with synthetic 2D data."""
    np.random.seed(0)
    nt, nx = 256, 64
    t = np.arange(nt)[:, None]
    x = np.arange(nx)[None, :]
    dt = 0.004

    S = np.sin(2 * np.pi * 20 * t * dt) * np.ones((nt, nx))
    N = 0.5 * np.sin(2 * np.pi * 8 * t * dt + 0.1 * x)
    X = S + N

    # Case 1: perfect noise prediction
    r1 = orthogonalized_local_projected_signal_leakage(
        X, S, prediction=N, prediction_type="noise"
    )
    print(f"Perfect noise: O-LPSL={r1['O_LPSL']:.4f}")

    # Case 2: under-predicted noise, no signal leakage
    r2 = orthogonalized_local_projected_signal_leakage(
        X, S, prediction=0.8 * N, prediction_type="noise"
    )
    print(f"Under-predicted noise: O-LPSL={r2['O_LPSL']:.4f}")

    # Case 3: noise prediction mixed with signal
    r3 = orthogonalized_local_projected_signal_leakage(
        X, S, prediction=N + 0.2 * S, prediction_type="noise"
    )
    print(f"Signal leakage: O-LPSL={r3['O_LPSL']:.4f}")

    assert r1["O_LPSL"] < r3["O_LPSL"], (
        "Perfect noise should have lower O-LPSL than signal leakage."
    )
    assert r2["O_LPSL"] < r3["O_LPSL"], (
        "Under-predicted noise should have lower O-LPSL than signal leakage."
    )
    print("Sanity checks passed.")


if __name__ == "__main__":
    _demo()
