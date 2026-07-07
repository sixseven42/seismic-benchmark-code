"""Loss registry + factory with basic supervised reconstruction losses.

See ``utils/README.md`` for the registry workflow (how to add a new loss).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn

LOSS_REGISTRY: Dict[str, Type["BaseLoss"]] = {}


def register_loss(name: str) -> Callable[[Type["BaseLoss"]], Type["BaseLoss"]]:
    """Class decorator that registers a loss under ``name``."""

    def _decorator(cls: Type["BaseLoss"]) -> Type["BaseLoss"]:
        if name in LOSS_REGISTRY:
            raise KeyError(f"Loss '{name}' already registered.")
        LOSS_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseLoss(nn.Module):
    """Common loss interface: ``forward(pred, target=None, **extras) -> Tensor``; ``extras`` passes optional mask / weight."""

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        raise NotImplementedError


@register_loss("mse")
class MSELoss(BaseLoss):
    """Mean squared error."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MSELoss requires `target`.")
        return nn.functional.mse_loss(pred, target, reduction=self.reduction)


@register_loss("l1")
class L1Loss(BaseLoss):
    """Mean absolute error."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("L1Loss requires `target`.")
        return nn.functional.l1_loss(pred, target, reduction=self.reduction)


@register_loss("weighted_mse")
class WeightedMSELoss(BaseLoss):
    """MSE weighted by ``extras["weight"]``."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("WeightedMSELoss requires `target`.")
        weight = extras.get("weight")
        if weight is None:
            raise ValueError("WeightedMSELoss requires extras['weight'].")
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, device=pred.device, dtype=pred.dtype)
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        err2 = (pred - target).pow(2)
        weighted = weight * err2
        denom = weight.sum().clamp_min(self.eps)
        return weighted.sum() / denom


# ----------------------------------------------------------------------
# First-break binary mask segmentation losses
# ----------------------------------------------------------------------


@register_loss("bce_dice")
class BCEDiceLoss(BaseLoss):
    """Weighted sum of masked BCE-with-logits and soft Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("bce_weight and dice_weight must be non-negative.")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one of bce_weight or dice_weight must be > 0.")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("BCEDiceLoss requires `target`.")
        valid = target >= 0
        if not bool(valid.any()):
            return pred.sum() * 0.0

        target = target.to(dtype=pred.dtype)
        valid_f = valid.to(dtype=pred.dtype)
        target_valid = target.clamp_min(0.0)
        loss = pred.new_tensor(0.0)
        if self.bce_weight:
            pos_weight = self.pos_weight
            if pos_weight is not None:
                pos_weight = pos_weight.to(device=pred.device, dtype=pred.dtype)
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred[valid],
                target_valid[valid],
                pos_weight=pos_weight,
                reduction="mean",
            )
            loss = loss + self.bce_weight * bce
        if self.dice_weight:
            prob = torch.sigmoid(pred) * valid_f
            target_valid = target_valid * valid_f
            dims = tuple(range(1, prob.dim()))
            intersection = (prob * target_valid).sum(dim=dims)
            denom = prob.sum(dim=dims) + target_valid.sum(dim=dims)
            dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            sample_has_valid = valid_f.sum(dim=dims) > 0
            if bool(sample_has_valid.any()):
                loss = loss + self.dice_weight * (1.0 - dice[sample_has_valid].mean())
        return loss


@register_loss("masked_bce")
class MaskedBCEWithLogitsLoss(BaseLoss):
    """BCE-with-logits that ignores pixels where target < 0."""

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Masked BCE supports reduction 'mean' or 'sum', got {reduction!r}.")
        self.reduction = reduction
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("MaskedBCEWithLogitsLoss requires `target`.")
        valid = target >= 0
        if not bool(valid.any()):
            return pred.sum() * 0.0
        target_valid = target.to(dtype=pred.dtype).clamp_min(0.0)
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=pred.device, dtype=pred.dtype)
        return nn.functional.binary_cross_entropy_with_logits(
            pred[valid],
            target_valid[valid],
            pos_weight=pos_weight,
            reduction=self.reduction,
        )


# ----------------------------------------------------------------------
# First-break HUNet deep supervision loss
# ----------------------------------------------------------------------
@register_loss("first_break_hunet")
class FirstBreakHUNetLoss(BaseLoss):
    """First-break HUNet deep supervision loss.

    L = BCE_Dice(fused, target) + side_weight × Σ BCE_Dice(side_i, target)

    Expects pred as ``(fused, side_outputs)`` when the model uses
    ``return_sides=True``, where ``side_outputs`` is a list of tensors
    from deep supervision branches. If pred is a plain tensor, falls
    back to simple BCE+Dice.
    """

    def __init__(
        self,
        side_weight: float = 0.5,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.side_weight = float(side_weight)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    @staticmethod
    def _bce_dice(
        pred: torch.Tensor,
        target: torch.Tensor,
        bce_weight: float,
        dice_weight: float,
        smooth: float,
        pos_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        valid = target >= 0
        if not bool(valid.any()):
            return pred.sum() * 0.0
        target_clamped = target.clamp_min(0.0).to(dtype=pred.dtype)
        valid_f = valid.to(dtype=pred.dtype)
        loss = pred.new_tensor(0.0)
        if bce_weight:
            pw = pos_weight
            if pw is not None:
                pw = pw.to(device=pred.device, dtype=pred.dtype)
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred[valid], target_clamped[valid],
                pos_weight=pw, reduction="mean",
            )
            loss = loss + bce_weight * bce
        if dice_weight:
            prob = torch.sigmoid(pred) * valid_f
            target_valid = target_clamped * valid_f
            dims = tuple(range(1, prob.dim()))
            intersection = (prob * target_valid).sum(dim=dims)
            denom = prob.sum(dim=dims) + target_valid.sum(dim=dims)
            dice = (2.0 * intersection + smooth) / (denom + smooth)
            sample_has_valid = valid_f.sum(dim=dims) > 0
            if bool(sample_has_valid.any()):
                loss = loss + dice_weight * (1.0 - dice[sample_has_valid].mean())
        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **extras: Any,
    ) -> torch.Tensor:
        if target is None:
            raise ValueError("FirstBreakHUNetLoss requires `target`.")
        if isinstance(pred, (tuple, list)):
            fused, sides = pred
            loss = self._bce_dice(
                fused, target, self.bce_weight, self.dice_weight,
                self.smooth, self.pos_weight,
            )
            for s in sides:
                loss = loss + self.side_weight * self._bce_dice(
                    s, target, self.bce_weight, self.dice_weight,
                    self.smooth, self.pos_weight,
                )
            return loss
        return self._bce_dice(
            pred, target, self.bce_weight, self.dice_weight,
            self.smooth, self.pos_weight,
        )


def build_loss(cfg: Dict[str, Any]) -> BaseLoss:
    """Instantiate a loss from a ``{type, params}`` config block."""
    name = cfg["type"]
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss '{name}'. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name](**cfg.get("params", {}))
