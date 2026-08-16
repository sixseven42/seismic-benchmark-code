"""Metric registry + factory, plus reconstruction metrics for denoising / restoration.

Implemented: ``mse``, ``rmse``, ``mae``, ``snr``, ``psnr``, ``ssim``.
See ``utils/README.md`` for the registry workflow (how to add a new metric).
"""

from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F

Reduction = Literal["global", "per_sample"]
_REDUCTIONS: Tuple[str, ...] = ("global", "per_sample")

METRIC_REGISTRY: Dict[str, Type["BaseMetric"]] = {}


def register_metric(name: str) -> Callable[[Type["BaseMetric"]], Type["BaseMetric"]]:
    """Class decorator that registers a metric under ``name``."""

    def _decorator(cls: Type["BaseMetric"]) -> Type["BaseMetric"]:
        if name in METRIC_REGISTRY:
            raise KeyError(f"Metric '{name}' already registered.")
        METRIC_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseMetric:
    """Common metric interface; ``__call__(pred, target) -> float`` (or 0-d Tensor)."""

    higher_is_better: bool = True

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

_EPS: float = 1e-12


def _prepare(
    pred: torch.Tensor, target: torch.Tensor, name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape-check, detach from autograd, and cast to float."""
    if pred.shape != target.shape:
        raise ValueError(
            f"{name}: pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}."
        )
    pred = pred.detach()
    target = target.detach()
    if not pred.is_floating_point():
        pred = pred.float()
    if not target.is_floating_point():
        target = target.float()
    return pred, target


def _check_reduction(name: str, reduction: str) -> None:
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"{name}: reduction must be one of {_REDUCTIONS}, got {reduction!r}."
        )


def _flatten_per_sample(t: torch.Tensor) -> torch.Tensor:
    """Reshape to ``(B, N)`` with B = first dim. ``ndim < 2`` -> single sample."""
    if t.dim() < 2:
        return t.reshape(1, -1)
    return t.reshape(t.size(0), -1)


# ----------------------------------------------------------------------
# Numpy core implementations (shared by torch metrics + inference)
# ----------------------------------------------------------------------

def _reduce_axes(t: np.ndarray) -> Tuple[int, ...]:
    """Axes to reduce for per-sample statistics (all except leading batch dim)."""
    return tuple(range(1, t.ndim))


def _mse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).mean())


def _mse_per_sample_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    axes = _reduce_axes(pred)
    return ((pred - target) ** 2).mean(axis=axes)


def _mae_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def _mae_per_sample_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    axes = _reduce_axes(pred)
    return np.abs(pred - target).mean(axis=axes)


def _rmse_per_sample_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(_mse_per_sample_numpy(pred, target))


def _snr_numpy(pred: np.ndarray, target: np.ndarray, eps: float = _EPS) -> float:
    signal = float((target ** 2).sum())
    noise = float(((pred - target) ** 2).sum())
    if noise == 0.0:
        return float("inf") if signal > 0.0 else float("nan")
    if signal == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal / noise))


def _snr_per_sample_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = _EPS,
    min_signal_energy: float = 0.0,
) -> np.ndarray:
    axes = _reduce_axes(target)
    signal = (target ** 2).sum(axis=axes)
    noise = ((pred - target) ** 2).sum(axis=axes)
    ratio = signal / np.maximum(noise, eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10.0 * np.log10(ratio)
    # signal=0, noise=0 -> nan (0/0 undefined)
    snr = np.where((signal == 0.0) & (noise == 0.0), np.nan, snr)
    if min_signal_energy > 0.0:
        snr = np.where(signal < min_signal_energy, np.nan, snr)
    return snr


def _psnr_numpy(pred: np.ndarray, target: np.ndarray, peak: float, eps: float = _EPS) -> float:
    mse = _mse_numpy(pred, target)
    return float(10.0 * np.log10((peak ** 2) / max(mse, eps)))


def _psnr_per_sample_numpy(
    pred: np.ndarray, target: np.ndarray, peak: float, eps: float = _EPS
) -> np.ndarray:
    mse = _mse_per_sample_numpy(pred, target)
    return 10.0 * np.log10((peak ** 2) / np.maximum(mse, eps))


# ----------------------------------------------------------------------
# Regression-style metrics (MSE / RMSE / MAE)
# ----------------------------------------------------------------------
#
# Reduction conventions
# ---------------------
# * ``MSE`` and ``MAE`` use a single global mean over every element. Because
#   ``mean`` is linear, the global mean equals the average of per-sample means,
#   so a separate ``reduction`` knob would be redundant.
# * ``RMSE`` / ``SNR`` / ``PSNR`` involve a non-linear step (``sqrt`` or
#   ``log10``) AFTER reduction. They expose ``reduction``:
#       - ``"per_sample"`` (default): compute the metric for each sample in the
#         leading batch dimension, then average across the batch. This matches
#         the classical seismic / image-quality reporting convention ("mean
#         per-shot SNR / PSNR").
#       - ``"global"``: pool every element first, then apply the non-linear
#         step. This preserves the textbook identities
#         ``RMSE == sqrt(MSE)`` and ``PSNR == 10*log10(data_range**2 / MSE)``,
#         which are useful for unit tests / debugging.
#
# In every case the return value is a plain python ``float``.

@register_metric("mse")
class MSE(BaseMetric):
    """Mean squared error, reduced globally.

    Formula
    -------
    MSE = mean_all( (pred - target) ** 2 )

    Inputs / Returns
    ----------------
    pred, target : same-shape tensors (any ndim).
    returns      : python ``float``; lower is better.
    """

    higher_is_better = False

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred, target = _prepare(pred, target, "MSE")
        return _mse_numpy(pred.cpu().numpy(), target.cpu().numpy())


@register_metric("rmse")
class RMSE(BaseMetric):
    """Root mean squared error.

    Formulas
    --------
    per_sample : RMSE = mean_b( sqrt( mean_i( (pred_b - target_b) ** 2 ) ) )
    global     : RMSE = sqrt( mean_all( (pred - target) ** 2 ) )

    Parameters
    ----------
    reduction : ``"per_sample"`` (default) or ``"global"``.

    Notes
    -----
    Only ``reduction="global"`` satisfies ``RMSE == sqrt(MSE)`` exactly.
    """

    higher_is_better = False

    def __init__(self, reduction: Reduction = "per_sample") -> None:
        _check_reduction("RMSE", reduction)
        self.reduction = reduction

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred, target = _prepare(pred, target, "RMSE")
        if self.reduction == "global":
            return float(np.sqrt(_mse_numpy(pred.cpu().numpy(), target.cpu().numpy())))
        return float(_rmse_per_sample_numpy(pred.cpu().numpy(), target.cpu().numpy()).mean())


@register_metric("mae")
class MAE(BaseMetric):
    """Mean absolute error, reduced globally.

    Formula
    -------
    MAE = mean_all( | pred - target | )
    """

    higher_is_better = False

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred, target = _prepare(pred, target, "MAE")
        return _mae_numpy(pred.cpu().numpy(), target.cpu().numpy())


# ----------------------------------------------------------------------
# Signal-to-noise metrics (SNR / PSNR)
# ----------------------------------------------------------------------

@register_metric("snr")
class SNR(BaseMetric):
    """Reconstruction signal-to-noise ratio, in dB.

    Formulas
    --------
    per_sample : SNR = mean_b( 10 * log10( sum_i target_b**2 / sum_i (pred_b - target_b)**2 ) )
    global     : SNR = 10 * log10( sum_all(target**2) / sum_all((pred-target)**2) )

    Powers are clamped to ``eps`` before ``log10`` to avoid ``-inf``.

    Parameters
    ----------
    reduction         : ``"per_sample"`` (default) or ``"global"``.
    eps               : clamp applied to power terms; default ``1e-12``.
    min_signal_energy : per-sample target energy below which samples are skipped;
                        only applies to ``reduction="per_sample"``. Default 0
                        preserves the original behavior.
    """

    higher_is_better = True

    def __init__(
        self,
        reduction: Reduction = "per_sample",
        eps: float = _EPS,
        min_signal_energy: float = 0.0,
    ) -> None:
        _check_reduction("SNR", reduction)
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        if min_signal_energy < 0:
            raise ValueError(
                f"min_signal_energy must be >= 0, got {min_signal_energy}."
            )
        self.reduction = reduction
        self.eps = eps
        self.min_signal_energy = float(min_signal_energy)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred, target = _prepare(pred, target, "SNR")
        if self.reduction == "global":
            return _snr_numpy(pred.cpu().numpy(), target.cpu().numpy(), self.eps)
        values = _snr_per_sample_numpy(
            pred.cpu().numpy(),
            target.cpu().numpy(),
            self.eps,
            self.min_signal_energy,
        )
        if self.min_signal_energy > 0.0:
            if np.all(np.isnan(values)):
                return float("nan")
            return float(np.nanmean(values))
        return float(values.mean())


@register_metric("psnr")
class PSNR(BaseMetric):
    """Peak signal-to-noise ratio, in dB.

    Formulas
    --------
    per_sample : PSNR = mean_b( 10 * log10( peak**2 / mean_i (pred_b - target_b)**2 ) )
    global     : PSNR = 10 * log10( peak**2 / mean_all (pred - target)**2 )

    ``data_range`` is the **peak amplitude** (maximum absolute value) of the
    reference signal, **not** the peak-to-peak range.
    Examples: ``1.0`` for both [0, 1] and [-1, 1] data; ``255`` for 8-bit images.
    Only ``reduction="global"`` satisfies ``PSNR == 10*log10(peak**2 / MSE)``.

    Parameters
    ----------
    data_range : positive float; peak amplitude (max absolute value) of the reference signal.
    reduction  : ``"per_sample"`` (default) or ``"global"``.
    eps        : clamp applied to MSE before the logarithm; default ``1e-12``.
    """

    higher_is_better = True

    def __init__(
        self,
        data_range: float = 1.0,
        reduction: Reduction = "per_sample",
        eps: float = _EPS,
    ) -> None:
        if data_range <= 0:
            raise ValueError(f"data_range must be > 0, got {data_range}.")
        _check_reduction("PSNR", reduction)
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.data_range = float(data_range)
        self.reduction = reduction
        self.eps = eps

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred, target = _prepare(pred, target, "PSNR")
        if self.reduction == "global":
            return _psnr_numpy(
                pred.cpu().numpy(), target.cpu().numpy(), self.data_range, self.eps
            )
        return float(
            _psnr_per_sample_numpy(
                pred.cpu().numpy(), target.cpu().numpy(), self.data_range, self.eps
            ).mean()
        )


# ----------------------------------------------------------------------
# Structural similarity (Wang et al., IEEE TIP 2004)
# ----------------------------------------------------------------------

@register_metric("ssim")
class SSIM(BaseMetric):
    """Structural similarity index (SSIM) averaged over samples, channels, and windows.

    Formula (per window, per channel)
    ---------------------------------
    SSIM(x, y) =
        ( (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) )
      / ( (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2) )

    where ``mu_x``, ``mu_y`` are Gaussian-weighted local means, ``sigma_x``,
    ``sigma_y`` local stds, ``sigma_xy`` the local covariance, and
    ``C1 = (K1 * data_range) ** 2``, ``C2 = (K2 * data_range) ** 2``.

    Reference: Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment:
    From Error Visibility to Structural Similarity", IEEE TIP 13(4), 2004.

    Parameters
    ----------
    data_range  : peak-to-peak range of the target signal (see :class:`PSNR`).
    window_size : odd integer >= 3; side length of the Gaussian window. Default 11.
    sigma       : stddev of the Gaussian window in pixels. Default 1.5.
    k1, k2      : stability constants from the paper. Defaults 0.01 / 0.03.

    Inputs / Returns
    ----------------
    pred, target : ``(B, C, H, W)`` or ``(B, H, W)`` (auto-promoted to ``C=1``);
                   spatial sides must be >= ``window_size``.
    returns      : python ``float`` in roughly ``[-1, 1]`` (1 = identical);
                   averaged over every (B, C, window) position. Higher is better.
    """

    higher_is_better = True

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        if data_range <= 0:
            raise ValueError(f"data_range must be > 0, got {data_range}.")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd and >= 3, got {window_size}."
            )
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.k1 = float(k1)
        self.k2 = float(k2)
        # (channels, dtype, device) -> depthwise Gaussian kernel (C, 1, W, W).
        self._kernel_cache: Dict[Tuple[int, torch.dtype, str], torch.Tensor] = {}

    def _gaussian_kernel(
        self, channels: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        key = (channels, dtype, str(device))
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        coords = (
            torch.arange(self.window_size, dtype=dtype, device=device)
            - (self.window_size - 1) / 2.0
        )
        g1d = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] * g1d[None, :]                          # (W, W)
        kernel = g2d.expand(channels, 1, self.window_size, self.window_size).contiguous()
        self._kernel_cache[key] = kernel
        return kernel

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred, target = _prepare(pred, target, "SSIM")
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        elif pred.dim() != 4:
            raise ValueError(
                f"SSIM expects (B, C, H, W) or (B, H, W); got ndim={pred.dim()}."
            )
        _, c, h, w = pred.shape
        if h < self.window_size or w < self.window_size:
            raise ValueError(
                f"SSIM: spatial size ({h}, {w}) smaller than window_size {self.window_size}."
            )

        kernel = self._gaussian_kernel(c, pred.dtype, pred.device)
        # Valid (no-padding) convolution: every window is fully supported.
        def _filt(x: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, kernel, padding=0, groups=c)

        mu_x = _filt(pred)
        mu_y = _filt(target)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = _filt(pred * pred) - mu_x2
        sigma_y2 = _filt(target * target) - mu_y2
        sigma_xy = _filt(pred * target) - mu_xy

        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        return (numerator / denominator).mean().item()


# ----------------------------------------------------------------------
# Build / evaluate helpers
# ----------------------------------------------------------------------

def build_metrics(cfg_list: List[Dict[str, Any]]) -> "OrderedDict[str, BaseMetric]":
    """Instantiate metrics from a list of ``{name, params}`` entries."""
    metrics: "OrderedDict[str, BaseMetric]" = OrderedDict()
    for item in cfg_list:
        name = item["name"]
        if name not in METRIC_REGISTRY:
            raise KeyError(
                f"Unknown metric '{name}'. Available: {sorted(METRIC_REGISTRY)}"
            )
        metrics[name] = METRIC_REGISTRY[name](**item.get("params", {}))
    return metrics


def compute_metrics(
    metrics: "OrderedDict[str, BaseMetric]",
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Run every metric on a single batch; returns ``{metric_name: scalar}``."""
    return {name: metric(pred, target) for name, metric in metrics.items()}


def format_metric_value(name: str, value: float) -> str:
    """Format a metric for human-readable output or file saving.

    - ``mse``: 6 significant digits so small values are not truncated to 0.00.
    - All others: 2 decimal places.
    """
    if name.lower() == "mse":
        return f"{value:.6g}"
    return f"{value:.2f}"


BAD_FIRST_BREAK_PICK_INDEX = -1


class _BinaryMetric(BaseMetric):
    higher_is_better = True

    def __init__(self, threshold: float = 0.5, eps: float = 1.0e-6) -> None:
        self.threshold = float(threshold)
        self.eps = float(eps)

    def _masks(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = target.detach() >= 0
        pred_mask = (torch.sigmoid(pred.detach()) >= self.threshold) & valid
        target_mask = (target.detach() >= 0.5) & valid
        return pred_mask, target_mask, valid

    def _pred_picks(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_mask, _, _ = self._masks(pred, target)
        pred_flat = pred_mask[:, 0].reshape(-1, pred_mask.size(-1))
        pred_has = pred_flat.any(dim=1)
        pred_pick = pred_flat.float().argmax(dim=1).to(dtype=torch.long)
        pred_pick = torch.where(
            pred_has,
            pred_pick,
            torch.full_like(pred_pick, BAD_FIRST_BREAK_PICK_INDEX),
        )
        return pred_pick, pred_has

    def _pick_data(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_pick: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if target_pick is None:
            raise ValueError("First-break pick metrics require target_pick.")
        pred_pick, pred_has = self._pred_picks(pred, target)
        target_pick = target_pick.detach().to(device=pred_pick.device).reshape(-1).to(dtype=torch.long)
        if target_pick.numel() != pred_pick.numel():
            raise ValueError(
                "target_pick shape is incompatible with prediction traces: "
                f"{tuple(target_pick.shape)} vs {tuple(pred_pick.shape)}."
            )
        target_valid = target_pick >= 0
        return pred_pick.float(), pred_has, target_pick.float(), target_valid


@register_metric("dice")
class DiceMetric(_BinaryMetric):
    """Mean binary Dice score, ignoring target pixels < 0."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        pred_mask, target_mask, valid = self._masks(pred, target)
        pred_f = pred_mask.float().reshape(pred_mask.size(0), -1)
        target_f = target_mask.float().reshape(target_mask.size(0), -1)
        valid_f = valid.float().reshape(valid.size(0), -1)
        sample_has_valid = valid_f.sum(dim=1) > 0
        if not bool(sample_has_valid.any()):
            return float("nan")
        inter = (pred_f * target_f).sum(dim=1)
        denom = pred_f.sum(dim=1) + target_f.sum(dim=1)
        score = (2.0 * inter + self.eps) / (denom + self.eps)
        return float(score[sample_has_valid].mean().item())


@register_metric("f1")
class F1Metric(DiceMetric):
    """Binary F1 score, equivalent to Dice for one foreground class."""


@register_metric("iou")
class IoUMetric(_BinaryMetric):
    """Mean binary intersection-over-union, ignoring target pixels < 0."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        pred_mask, target_mask, valid = self._masks(pred, target)
        pred_f = pred_mask.reshape(pred_mask.size(0), -1)
        target_f = target_mask.reshape(target_mask.size(0), -1)
        valid_f = valid.reshape(valid.size(0), -1)
        sample_has_valid = valid_f.sum(dim=1) > 0
        if not bool(sample_has_valid.any()):
            return float("nan")
        inter = (pred_f & target_f).float().sum(dim=1)
        union = (pred_f | target_f).float().sum(dim=1)
        score = (inter + self.eps) / (union + self.eps)
        return float(score[sample_has_valid].mean().item())


@register_metric("pick_mae")
class PickMAE(_BinaryMetric):
    """Mean absolute first-break sample error over traces with a target pick."""

    higher_is_better = False

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        target_pick = kwargs.get("target_pick")
        pred_pick, _, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = pred_pick[valid] - target_pick_f[valid]
        return float(err.abs().mean().item())


@register_metric("pick_rmse")
class PickRMSE(_BinaryMetric):
    """Root mean squared first-break sample error over traces with a target pick."""

    higher_is_better = False

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        target_pick = kwargs.get("target_pick")
        pred_pick, _, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = pred_pick[valid] - target_pick_f[valid]
        return float(torch.sqrt((err ** 2).mean()).item())


@register_metric("pick_bias")
class MeanBiasError(_BinaryMetric):
    """Mean signed first-break sample error, computed as prediction minus target."""

    higher_is_better = False

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        target_pick = kwargs.get("target_pick")
        pred_pick, _, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = pred_pick[valid] - target_pick_f[valid]
        return float(err.mean().item())


@register_metric("gather_coverage")
class GatherCoverage(_BinaryMetric):
    """Fraction of target-pick traces for which the model produced any pick."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        target_pick = kwargs.get("target_pick")
        _, pred_has, _, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        return float(pred_has[valid].float().mean().item())


# Backward-compatible aliases used in existing first-break configs.
METRIC_REGISTRY["GatherCoverage"] = GatherCoverage
METRIC_REGISTRY["MeanAbsoluteError"] = PickMAE
METRIC_REGISTRY["MeanBiasError"] = MeanBiasError
METRIC_REGISTRY["RootMeanSquaredError"] = PickRMSE
METRIC_REGISTRY["MeanBiasError"] = MeanBiasError
METRIC_REGISTRY["GatherCoverage"] = GatherCoverage


@register_metric("pick_within")
class PickWithin(_BinaryMetric):
    """Fraction of target-pick traces picked within a sample tolerance."""

    def __init__(
        self,
        threshold: float = 0.5,
        tolerance: int = 5,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__(threshold=threshold, eps=eps)
        self.tolerance = int(tolerance)

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        target_pick = kwargs.get("target_pick")
        pred_pick, pred_has, target_pick_f, valid = self._pick_data(pred, target, target_pick)
        if not bool(valid.any()):
            return float("nan")
        err = (pred_pick[valid] - target_pick_f[valid]).abs()
        hit = pred_has[valid] & (err < self.tolerance)
        return float(hit.float().mean().item())


def build_first_break_metrics(cfg_list: List[Dict[str, Any]]) -> "OrderedDict[str, Any]":
    """Instantiate first-break metrics from a list of ``{name, params}`` entries.

    Supports convenience aliases ``pick_within_<N>`` and ``HitRate<N>px``,
    which are rewritten to ``pick_within`` with ``tolerance=<N>``.
    """
    metrics: "OrderedDict[str, Any]" = OrderedDict()
    for item in cfg_list:
        name = str(item["name"])
        metric_key = name
        params = dict(item.get("params") or {})
        if name.startswith("pick_within_"):
            metric_key = "pick_within"
            params.setdefault("tolerance", int(name.rsplit("_", 1)[-1]))
        hit_rate_match = re.fullmatch(r"HitRate(\d+)px", name)
        if hit_rate_match is not None:
            metric_key = "pick_within"
            params.setdefault("tolerance", int(hit_rate_match.group(1)))
        if metric_key not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown first-break metric {name!r}. "
                f"Available: {sorted(METRIC_REGISTRY)}"
            )
        metrics[name] = METRIC_REGISTRY[metric_key](**params)
    return metrics
