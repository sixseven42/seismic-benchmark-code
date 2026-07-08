"""Per shot-gather preprocessing primitives (pure numpy).

Inputs accept ``(n_shots, n_traces, n_time)`` or ``(n_traces, n_time)``;
outputs have the same shape. SNR uses ``SNR_dB = 10*log10(var_signal/var_noise)``.
See ``tools/README.md`` for the per-function summary.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from ._array_utils import (
    RNGLike,
    as_3d as _as_3d,
    as_generator as _as_generator,
    restore as _restore,
)

_EPS: float = 1e-8

NoiseKind = Literal["gaussian", "poisson"]
MaskMode = Literal["uniform", "random", "continuous"]
NormMode = Literal["minmax", "max_abs", "mean_std"]
NormScope = Literal["shot", "trace", "global"]


# ----------------------------------------------------------------------
# Small internal helpers (kept private by leading underscore)
# ----------------------------------------------------------------------

def _float(x: np.ndarray) -> np.ndarray:
    """Promote integer arrays to float32; keep any floating dtype as-is."""
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float32)


_REDUCE_AXES: Dict[str, Tuple[int, ...]] = {
    "shot":   (-2, -1),
    "trace":  (-1,),
    "global": (0, 1, 2),
}


# ----------------------------------------------------------------------
# 1. Noise (Gaussian / Poisson, SNR-controlled per shot)
# ----------------------------------------------------------------------

def add_noise(
    shots: np.ndarray,
    kind: NoiseKind = "gaussian",
    snr_db: float = 20.0,
    rng: RNGLike = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Inject IID per-shot SNR-controlled noise.

    Parameters
    ----------
    kind   : ``"gaussian"`` (additive ``N(0, sigma_i)``) or ``"poisson"``
             (scaled Poisson on the per-shot non-negative shift).
    snr_db : target SNR in dB; larger = cleaner.
    rng    : seed / :class:`numpy.random.Generator`; ``None`` = fresh.

    Returns
    -------
    noisy : same shape / float dtype as ``shots``.
    info  : ``{"sigma": (n_shots,)}`` for gaussian, ``{"scale": (n_shots,)}`` for poisson.
    """
    gen = _as_generator(rng)
    x, was_2d = _as_3d(_float(shots))

    var_s = x.var(axis=(-2, -1), keepdims=True)                  # (n_shots, 1, 1)
    var_n = np.maximum(var_s / (10.0 ** (snr_db / 10.0)), _EPS)

    if kind == "gaussian":
        sigma = np.sqrt(var_n)
        noisy = x + gen.standard_normal(x.shape).astype(x.dtype) * sigma
        info = {"sigma": sigma.reshape(-1)}
    elif kind == "poisson":
        # Shift per shot so Poisson is well-defined, then match target variance.
        x_min = x.min(axis=(-2, -1), keepdims=True)
        x_pos = x - x_min                                        # >= 0
        mean_pos = np.maximum(x_pos.mean(axis=(-2, -1), keepdims=True), _EPS)
        # Poisson(lam)/k has variance lam/k^2 ~ x_pos/k per pixel;
        # pixel-averaged variance = mean(x_pos) / k. Pick k so this equals var_n.
        scale = mean_pos / var_n
        sampled = gen.poisson(scale * x_pos).astype(x.dtype) / scale
        noisy = x + (sampled - x_pos)
        info = {"scale": scale.reshape(-1)}
    else:
        raise ValueError(f"Unknown noise kind: {kind!r}.")

    return _restore(noisy.astype(x.dtype, copy=False), was_2d), info


# ----------------------------------------------------------------------
# 1b. Coherent linear-moveout noise (PLACEHOLDER, not implemented yet)
# ----------------------------------------------------------------------

def add_linear_noise(
    shots: np.ndarray,
    dt: float,
    *,
    rng: RNGLike = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Inject coherent linear-moveout events (ground-roll-like). NOT IMPLEMENTED YET.

    Parameters
    ----------
    dt  : time-axis sampling interval (s).
    rng : seed / :class:`numpy.random.Generator`.

    Returns
    -------
    noisy : same shape / dtype as ``shots``.
    info  : per-shot event parameters (velocity, t0, amplitude, ...).
    """
    raise NotImplementedError(
        "add_linear_noise is a placeholder; finalise the signature when the "
        "first experiment needs structured noise."
    )


# ----------------------------------------------------------------------
# 1c. Coherent hyperbolic-moveout noise (PLACEHOLDER, not implemented yet)
# ----------------------------------------------------------------------

def add_hyperbolic_noise(
    shots: np.ndarray,
    dt: float,
    *,
    rng: RNGLike = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Inject coherent hyperbolic-moveout events (multiples-like). NOT IMPLEMENTED YET.

    Parameters
    ----------
    dt  : time-axis sampling interval (s).
    rng : seed / :class:`numpy.random.Generator`.

    Returns
    -------
    noisy : same shape / dtype as ``shots``.
    info  : per-shot event parameters (t0, x0, velocity, amplitude, ...).
    """
    raise NotImplementedError(
        "add_hyperbolic_noise is a placeholder; finalise the signature when "
        "the first experiment needs structured noise."
    )


# ----------------------------------------------------------------------
# 2. Trace masking (uniform / random / continuous)
# ----------------------------------------------------------------------

def mask_traces(
    shots: np.ndarray,
    mode: MaskMode = "random",
    ratio: float = 0.5,
    *,
    uniform_stride: Optional[int] = None,
    rng: RNGLike = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero out traces along the trace axis and return the boolean mask.

    Parameters
    ----------
    mode           : ``"uniform"`` keeps every ``uniform_stride``-th trace
                     (fallback stride = ``max(2, round(1/(1-ratio)))`` if
                     ``uniform_stride`` is omitted); ``"random"`` masks
                     ``round(n_traces*ratio)`` positions per shot; ``"continuous"``
                     masks one contiguous block of the same length per shot.
    ratio          : missing fraction in ``(0, 1)``.
    uniform_stride : keyword-only; integer >= 2; only valid for ``mode="uniform"``.
    rng            : seed / :class:`numpy.random.Generator`.

    Returns
    -------
    masked : same shape as ``shots``; masked traces are zeroed.
    mask   : bool of shape ``(n_shots, n_traces)`` (or ``(n_traces,)`` for 2D input);
             ``True`` = missing.
    """
    if mode != "uniform" and uniform_stride is not None:
        raise ValueError(
            f"uniform_stride only applies when mode='uniform', got mode={mode!r}."
        )
    gen = _as_generator(rng)
    x, was_2d = _as_3d(shots)
    n_shots, n_traces, _ = x.shape

    mask = np.zeros((n_shots, n_traces), dtype=bool)

    if mode == "uniform":
        if uniform_stride is None:
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"ratio must be in (0, 1), got {ratio}.")
            uniform_stride = max(2, int(round(1.0 / max(1.0 - ratio, _EPS))))
        if not isinstance(uniform_stride, (int, np.integer)) or uniform_stride < 2:
            raise ValueError(
                f"uniform_stride must be an integer >= 2, got {uniform_stride!r}."
            )
        keep_idx = np.arange(0, n_traces, int(uniform_stride))
        mask[:] = True
        mask[:, keep_idx] = False
    else:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}.")
        n_missing = max(1, min(int(round(n_traces * ratio)), n_traces - 1))
        rows = np.arange(n_shots)[:, None]
        if mode == "random":
            keys = gen.random((n_shots, n_traces))
            idx = np.argpartition(keys, n_missing - 1, axis=1)[:, :n_missing]
            mask[rows, idx] = True
        elif mode == "continuous":
            starts = gen.integers(0, n_traces - n_missing + 1, size=n_shots)
            cols = starts[:, None] + np.arange(n_missing)[None, :]
            mask[rows, cols] = True
        else:
            raise ValueError(f"Unknown mask mode: {mode!r}.")

    masked = x * (~mask)[..., None]
    return _restore(masked, was_2d), _restore(mask, was_2d)


# ----------------------------------------------------------------------
# 3. Spherical-divergence compensation
# ----------------------------------------------------------------------

def spherical_divergence_correction(
    shots: np.ndarray,
    dt: float,
    t0: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    """Apply spherical-divergence gain ``(t + t0) ** power`` along the time axis.

    Parameters
    ----------
    dt    : sampling interval (s); must be > 0.
    t0    : time of the first sample (s); clamped to ``dt`` to avoid zero-gain.
    power : exponent. Default ``1.0`` for 3D amplitude compensation
            (Yilmaz 2001). Use ``2.0`` only when compensating energy decay.

    Returns
    -------
    corrected : same shape / float dtype as ``shots``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    x, was_2d = _as_3d(_float(shots))
    n_time = x.shape[-1]
    t = np.arange(n_time, dtype=x.dtype) * dt + t0
    t = np.maximum(t, dt)
    gain = t ** power
    return _restore(x * gain, was_2d)


# ----------------------------------------------------------------------
# 4. Normalization (min-max / max-abs / mean-std, optional abs-percentile clip)
# ----------------------------------------------------------------------

_REQUIRED_STATS_KEYS: Dict[str, Tuple[str, ...]] = {
    "minmax":   ("min", "max"),
    "max_abs":  ("max_abs",),
    "mean_std": ("mean", "std"),
}


def normalize(
    shots: np.ndarray,
    mode: NormMode = "max_abs",
    clip_percentile: Optional[float] = None,
    per: NormScope = "shot",
    override_stats: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize with optional absolute-percentile clipping.

    Parameters
    ----------
    mode            : ``"minmax"`` -> [0, 1]; ``"max_abs"`` -> [-1, 1];
                      ``"mean_std"`` -> zero mean, unit std.
    clip_percentile : percentile of ``|x|`` used to symmetrically clip the
                      input **before** computing stats; ``None`` disables.
    per             : reduction scope — ``"shot"`` (default), ``"trace"``, or ``"global"``.
    override_stats  : pre-computed scalars that bypass on-the-fly reduction
                      (train-set stats at inference, physical bounds, ...).
                      Required keys: ``{"min","max"}`` / ``{"max_abs"}`` /
                      ``{"mean","std"}``. Shape must broadcast to the reduction
                      axes implied by ``per``; mutually exclusive with
                      ``clip_percentile``.

    Returns
    -------
    normalized : same shape as ``shots``.
    stats      : reducers actually used (echoes ``override_stats`` when given);
                 includes ``"clip_threshold"`` when ``clip_percentile`` is set.
    """
    if per not in _REDUCE_AXES:
        raise ValueError(f"per must be one of {list(_REDUCE_AXES)}, got {per!r}.")
    if mode not in _REQUIRED_STATS_KEYS:
        raise ValueError(f"Unknown normalization mode: {mode!r}.")
    if override_stats is not None:
        if clip_percentile is not None:
            raise ValueError(
                "override_stats and clip_percentile are mutually exclusive: "
                "user-supplied stats are authoritative; clip the input "
                "yourself before calling normalize() if needed."
            )
        missing = [k for k in _REQUIRED_STATS_KEYS[mode] if k not in override_stats]
        if missing:
            raise ValueError(
                f"override_stats is missing required keys for mode={mode!r}: "
                f"{missing}. Expected: {list(_REQUIRED_STATS_KEYS[mode])}."
            )

    x, was_2d = _as_3d(_float(shots))
    axes = _REDUCE_AXES[per]

    stats: Dict[str, Any] = {}

    if override_stats is not None and "clip_threshold" in override_stats:
        thresh = np.asarray(override_stats["clip_threshold"], dtype=x.dtype)
        if per == "shot":
            thresh = thresh.reshape(-1, 1, 1)
        elif per == "trace":
            n_shots, n_traces, _ = x.shape
            thresh = thresh.reshape(n_shots, n_traces, 1)
        x = np.clip(x, -thresh, thresh)
        stats["clip_threshold"] = np.asarray(thresh).squeeze()

    if clip_percentile is not None:
        if not 0.0 < clip_percentile <= 100.0:
            raise ValueError(
                f"clip_percentile must be in (0, 100], got {clip_percentile}."
            )
        thresh = np.quantile(
            np.abs(x), clip_percentile / 100.0, axis=axes, keepdims=True,
        )
        x = np.clip(x, -thresh, thresh)
        stats["clip_threshold"] = np.asarray(thresh).squeeze()

    if mode == "minmax":
        if override_stats is not None:
            xmin = np.asarray(override_stats["min"], dtype=x.dtype)
            xmax = np.asarray(override_stats["max"], dtype=x.dtype)
            if per == "shot":
                xmin = xmin.reshape(-1, 1, 1)
                xmax = xmax.reshape(-1, 1, 1)
            elif per == "trace":
                xmin = xmin.reshape(n_shots, n_traces, 1)
                xmax = xmax.reshape(n_shots, n_traces, 1)
        else:
            xmin = x.min(axis=axes, keepdims=True)
            xmax = x.max(axis=axes, keepdims=True)
        out = (x - xmin) / np.maximum(xmax - xmin, _EPS)
        stats["min"] = np.asarray(xmin).squeeze()
        stats["max"] = np.asarray(xmax).squeeze()
    elif mode == "max_abs":
        if override_stats is not None:
            m = np.asarray(override_stats["max_abs"], dtype=x.dtype)
            if per == "shot":
                m = m.reshape(-1, 1, 1)
            elif per == "trace":
                m = m.reshape(n_shots, n_traces, 1)
        else:
            m = np.abs(x).max(axis=axes, keepdims=True)
        out = x / np.maximum(m, _EPS)
        stats["max_abs"] = np.asarray(m).squeeze()
    else:  # mean_std
        if override_stats is not None:
            mu = np.asarray(override_stats["mean"], dtype=x.dtype)
            sd = np.asarray(override_stats["std"], dtype=x.dtype)
            if per == "shot":
                mu = mu.reshape(-1, 1, 1)
                sd = sd.reshape(-1, 1, 1)
            elif per == "trace":
                mu = mu.reshape(n_shots, n_traces, 1)
                sd = sd.reshape(n_shots, n_traces, 1)
        else:
            mu = x.mean(axis=axes, keepdims=True)
            sd = x.std(axis=axes, keepdims=True)
        out = (x - mu) / np.maximum(sd, _EPS)
        stats["mean"] = np.asarray(mu).squeeze()
        stats["std"] = np.asarray(sd).squeeze()

    return _restore(out.astype(x.dtype, copy=False), was_2d), stats


def denormalize(
    normalized: np.ndarray,
    stats: Dict[str, Any],
    mode: NormMode = "max_abs",
    per: NormScope = "shot",
) -> np.ndarray:
    """Inverse of ``normalize``: restore the original scale from ``stats``.

    Parameters
    ----------
    normalized : array normalized by a previous call to ``normalize``.
    stats      : dict returned by that ``normalize`` call (``min``/``max``,
                 ``max_abs``, or ``mean``/``std``).
    mode       : must match the mode used during normalization.
    per        : must match the scope used during normalization.

    Returns
    -------
    restored : same shape / dtype as ``normalized``.
    """
    if per not in _REDUCE_AXES:
        raise ValueError(f"per must be one of {list(_REDUCE_AXES)}, got {per!r}.")
    if mode not in _REQUIRED_STATS_KEYS:
        raise ValueError(f"Unknown normalization mode: {mode!r}.")

    x, was_2d = _as_3d(_float(normalized))
    n_shots, n_traces, _ = x.shape

    if per == "global":
        shape = (1, 1, 1)
    elif per == "shot":
        shape = (-1, 1, 1)
    else:  # trace
        shape = (n_shots, n_traces, 1)

    if mode == "minmax":
        xmin = np.asarray(stats["min"], dtype=x.dtype).reshape(shape)
        xmax = np.asarray(stats["max"], dtype=x.dtype).reshape(shape)
        out = x * np.maximum(xmax - xmin, _EPS) + xmin
    elif mode == "max_abs":
        m = np.asarray(stats["max_abs"], dtype=x.dtype).reshape(shape)
        out = x * np.maximum(m, _EPS)
    else:  # mean_std
        mu = np.asarray(stats["mean"], dtype=x.dtype).reshape(shape)
        sd = np.asarray(stats["std"], dtype=x.dtype).reshape(shape)
        out = x * np.maximum(sd, _EPS) + mu

    return _restore(out.astype(x.dtype, copy=False), was_2d)


def inverse_spherical_divergence_correction(
    corrected: np.ndarray,
    dt: float,
    t0: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    """Undo ``spherical_divergence_correction`` by dividing by ``(t + t0) ** power``.

    Parameters
    ----------
    dt    : sampling interval (s); must be > 0.
    t0    : time of the first sample (s); clamped to ``dt`` to avoid division by zero.
    power : exponent used during forward correction. Default ``1.0``.

    Returns
    -------
    restored : same shape / float dtype as ``corrected``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}.")
    x, was_2d = _as_3d(_float(corrected))
    n_time = x.shape[-1]
    t = np.arange(n_time, dtype=x.dtype) * dt + t0
    t = np.maximum(t, dt)
    gain = t ** power
    return _restore(x / gain, was_2d)


def first_pick_from_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
    valid_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract first-arrival sample indices from a binary probability mask.

    Parameters
    ----------
    mask       : 2D numpy array of shape ``(traces, time)`` with values in [0, 1].
    threshold  : probability threshold applied to binarise the mask.
    valid_mask : optional boolean array same shape as ``mask``; where ``False``,
                 the pick is forced to be invalid regardless of the mask value.

    Returns
    -------
    pick : ``(traces,)`` float32 array; the time index of the first ``True``
           pixel per trace (0 when no pick).
    valid : ``(traces,)`` bool array; whether any pick was found.
    """
    binary = np.asarray(mask) > float(threshold)
    if valid_mask is not None:
        binary = binary & np.asarray(valid_mask)
    valid = binary.any(axis=1)
    pick = np.argmax(binary, axis=1)
    return pick.astype(np.float32), valid
