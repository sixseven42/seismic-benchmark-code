"""DSU-Net adapted to seismic gather patches and step-mask supervision.

Network structure follows Wang et al., *DSU-Net: Dynamic Snake U-Net for 2-D
Seismic First Break Picking*, IEEE TGRS 2024.  Paper-specific preprocessing,
point labels and output post-processing are deliberately outside this model.
"""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


SnakeAxis = Literal["horizontal", "vertical"]


def accumulate_snake_offsets(offsets: torch.Tensor) -> torch.Tensor:
    """Accumulate ``(B, K, H, W)`` offsets away from the kernel centre.

    The centre is fixed to zero. Each point on either side is displaced
    relative to its immediately preceding point, which is the continuity
    constraint that distinguishes DSConv from free deformable convolution.
    """
    if offsets.ndim != 4:
        raise ValueError(
            f"Snake offsets must have shape (B,K,H,W), got {tuple(offsets.shape)}."
        )
    kernel_size = int(offsets.shape[1])
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"Snake kernel size must be a positive odd integer, got {kernel_size}."
        )

    centre = kernel_size // 2
    left_raw = offsets[:, :centre]
    right_raw = offsets[:, centre + 1 :]
    left = torch.flip(
        torch.cumsum(torch.flip(left_raw, dims=(1,)), dim=1),
        dims=(1,),
    )
    middle = torch.zeros_like(offsets[:, centre : centre + 1])
    right = torch.cumsum(right_raw, dim=1)
    return torch.cat((left, middle, right), dim=1)


def _normalise_coordinate(coordinate: torch.Tensor, size: int) -> torch.Tensor:
    """Map pixel coordinates to the ``[-1, 1]`` grid_sample convention."""
    if size <= 1:
        return torch.zeros_like(coordinate)
    return coordinate.mul(2.0 / float(size - 1)).sub(1.0)


class DynamicSnakeConv2d(nn.Module):
    """One directional 2-D Dynamic Snake Convolution branch.

    The sampling rule follows Qi et al., *Dynamic Snake Convolution Based on
    Topological Geometric Constraints for Tubular Structure Segmentation*,
    ICCV 2023. The official reference implementation is MIT licensed:
    https://github.com/YaoleiQi/DSCNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        extension_scope: float = 4.0,
        axis: SnakeAxis = "horizontal",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                "DynamicSnakeConv2d kernel_size must be positive and odd, "
                f"got {kernel_size}."
            )
        if extension_scope < 0:
            raise ValueError(
                "DynamicSnakeConv2d extension_scope must be non-negative, "
                f"got {extension_scope}."
            )
        if axis not in ("horizontal", "vertical"):
            raise ValueError(f"Unknown snake axis {axis!r}.")

        self.kernel_size = int(kernel_size)
        self.extension_scope = float(extension_scope)
        self.axis: SnakeAxis = axis
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.kernel_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.offset_norm = nn.GroupNorm(
            num_groups=self.kernel_size,
            num_channels=2 * self.kernel_size,
        )

        if axis == "horizontal":
            kernel = (1, self.kernel_size)
            stride = (1, self.kernel_size)
        else:
            kernel = (self.kernel_size, 1)
            stride = (self.kernel_size, 1)
        self.collapse = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU(inplace=True)

    def _sampling_grid(self, offsets: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = offsets.shape
        y_offsets, x_offsets = offsets.chunk(2, dim=1)
        dtype = offsets.dtype
        device = offsets.device
        y_base = torch.arange(height, dtype=dtype, device=device).view(
            1, 1, height, 1
        )
        x_base = torch.arange(width, dtype=dtype, device=device).view(
            1, 1, 1, width
        )
        spread = torch.arange(
            -(self.kernel_size // 2),
            self.kernel_size // 2 + 1,
            dtype=dtype,
            device=device,
        ).view(1, self.kernel_size, 1, 1)

        if self.axis == "horizontal":
            snake = accumulate_snake_offsets(y_offsets) * self.extension_scope
            y = y_base + snake
            x = x_base + spread
            y = y.permute(0, 2, 3, 1).reshape(
                batch, height, width * self.kernel_size
            )
            x = x.expand(batch, -1, height, -1).permute(0, 2, 3, 1).reshape(
                batch, height, width * self.kernel_size
            )
        else:
            snake = accumulate_snake_offsets(x_offsets) * self.extension_scope
            y = y_base + spread
            x = x_base + snake
            y = y.expand(batch, -1, -1, width).permute(0, 2, 1, 3).reshape(
                batch, height * self.kernel_size, width
            )
            x = x.permute(0, 2, 1, 3).reshape(
                batch, height * self.kernel_size, width
            )

        # The reference DSCNet implementation clamps deformed coordinates to
        # the valid image extent before converting them for grid_sample.
        grid_x = _normalise_coordinate(x.clamp(0, max(width - 1, 0)), width)
        grid_y = _normalise_coordinate(y.clamp(0, max(height - 1, 0)), height)
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"DynamicSnakeConv2d expects BCHW input, got {tuple(x.shape)}."
            )
        offsets = torch.tanh(self.offset_norm(self.offset_conv(x)))
        sampled = F.grid_sample(
            x,
            self._sampling_grid(offsets),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return self.activation(self.norm(self.collapse(sampled)))


class TraConv(nn.Module):
    """Paper's 3x3 Conv -> BN -> ELU module."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSConvModule(nn.Module):
    """Fuse horizontal snake, vertical snake and local 3x3 features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        extension_scope: float,
    ) -> None:
        super().__init__()
        self.horizontal = DynamicSnakeConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            extension_scope=extension_scope,
            axis="horizontal",
        )
        self.vertical = DynamicSnakeConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            extension_scope=extension_scope,
            axis="vertical",
        )
        self.local = TraConv(in_channels, out_channels)
        self.fuse = TraConv(3 * out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = (self.horizontal(x), self.vertical(x), self.local(x))
        return self.fuse(torch.cat(features, dim=1))


class _TwoTraConv(nn.Module):
    """Two sequential TraConv modules, as used in each paper layer."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            TraConv(in_channels, out_channels),
            TraConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpModule(nn.Module):
    """Paper's 2x2 transposed convolution followed by ELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            ),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("dsu_net")
class DSUNet(nn.Module):
    """Three-level DSU-Net returning input-resolution segmentation logits.

    Input convention is ``(batch, channel, trace, time)``.  Inputs are padded
    only on the bottom/right to a multiple of eight for the three pooling
    operations, then cropped back so output and target have identical shapes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        extension_scope: float = 4.0,
        pad_multiple: int = 8,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"DSUNet requires four channel levels, got {list(channels)}.")
        c0, c1, c2, c3 = (int(c) for c in channels)
        if min(c0, c1, c2, c3) <= 0:
            raise ValueError(f"DSUNet channels must be positive, got {list(channels)}.")
        if pad_multiple < 8 or pad_multiple % 8 != 0:
            raise ValueError(
                f"DSUNet pad_multiple must be a positive multiple of 8, got {pad_multiple}."
            )
        self.pad_multiple = int(pad_multiple)

        # The first encoder layer is the only layer using Dynamic Snake
        # modules; the paper's ablation found shallow-only DSConv preferable.
        self.encoder0 = nn.Sequential(
            DSConvModule(
                in_channels,
                c0,
                kernel_size=kernel_size,
                extension_scope=extension_scope,
            ),
            DSConvModule(
                c0,
                c0,
                kernel_size=kernel_size,
                extension_scope=extension_scope,
            ),
        )
        self.encoder1 = _TwoTraConv(c0, c1)
        self.encoder2 = _TwoTraConv(c1, c2)
        self.encoder3 = _TwoTraConv(c2, c3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up2 = _UpModule(c3, c2)
        self.decoder2 = _TwoTraConv(c2 + c2, c2)
        self.up1 = _UpModule(c2, c1)
        self.decoder1 = _TwoTraConv(c1 + c1, c1)
        self.up0 = _UpModule(c1, c0)
        self.decoder0 = _TwoTraConv(c0 + c0, c0)
        self.segmentation_head = nn.Conv2d(c0, out_channels, kernel_size=1)

    def _pad_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.pad_multiple
        pad_w = (-width) % self.pad_multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x, height, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DSUNet expects BCHW input, got {tuple(x.shape)}.")
        x, original_h, original_w = self._pad_input(x)

        skip0 = self.encoder0(x)
        skip1 = self.encoder1(self.pool(skip0))
        skip2 = self.encoder2(self.pool(skip1))
        encoded = self.encoder3(self.pool(skip2))

        decoded = self.decoder2(torch.cat((skip2, self.up2(encoded)), dim=1))
        decoded = self.decoder1(torch.cat((skip1, self.up1(decoded)), dim=1))
        decoded = self.decoder0(torch.cat((skip0, self.up0(decoded)), dim=1))
        logits = self.segmentation_head(decoded)
        return logits[..., :original_h, :original_w]


__all__ = [
    "DSConvModule",
    "DSUNet",
    "DynamicSnakeConv2d",
    "SnakeAxis",
    "TraConv",
    "accumulate_snake_offsets",
]
