"""DNNDAT-style convolutional encoder-decoder for multiple suppression.

Network-level reproduction of Wang et al. (2022): 14 convolutional encoder
layers, four max-pooling layers, 14 convolutional decoder layers, four
upsampling layers, U-Net-style concatenations, dropout after the 12th and 15th
convolutions, ReLU on the first 27 convolutions, and tanh on the output.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model

UpsampleMode = Literal["nearest", "bilinear"]


class _SameConv2d(nn.Module):
    """Conv2d with TensorFlow-style same padding, including even kernels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = (int(kernel_size), int(kernel_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.kernel_size
        pad_h = kh - 1
        pad_w = kw - 1
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return self.conv(F.pad(x, (left, right, top, bottom)))


class _ConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _SameConv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("dnndat")
class DNNDAT(nn.Module):
    """Wang et al. DNNDAT encoder-decoder adapted to the benchmark tensor API."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dropout: float = 0.5,
        upsample_mode: UpsampleMode = "nearest",
        final_activation: Literal["tanh", "identity"] = "tanh",
    ) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if upsample_mode not in {"nearest", "bilinear"}:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode!r}.")
        if final_activation not in {"tanh", "identity"}:
            raise ValueError(f"Unsupported final_activation: {final_activation!r}.")

        self.conv1 = _ConvReLU(in_channels, 32)
        self.conv2 = _ConvReLU(32, 32)
        self.conv3 = _ConvReLU(32, 32)
        self.conv4 = _ConvReLU(32, 64)
        self.conv5 = _ConvReLU(64, 128)
        self.conv6 = _ConvReLU(128, 128)
        self.conv7 = _ConvReLU(128, 128)
        self.conv8 = _ConvReLU(128, 256)
        self.conv9 = _ConvReLU(256, 256)
        self.conv10 = _ConvReLU(256, 256)
        self.conv11 = _ConvReLU(256, 512)
        self.conv12 = _ConvReLU(512, 512)
        self.conv13 = _ConvReLU(512, 1024)
        self.conv14 = _ConvReLU(1024, 1024)

        self.conv15 = _ConvReLU(1024, 1024)
        self.conv16 = _ConvReLU(1024 + 256, 512, kernel_size=2)
        self.conv17 = _ConvReLU(512, 512)
        self.conv18 = _ConvReLU(512 + 128, 256)
        self.conv19 = _ConvReLU(256, 256, kernel_size=2)
        self.conv20 = _ConvReLU(256, 256)
        self.conv21 = _ConvReLU(256 + 64, 128)
        self.conv22 = _ConvReLU(128, 128, kernel_size=2)
        self.conv23 = _ConvReLU(128, 128)
        self.conv24 = _ConvReLU(128 + 32, 64)
        self.conv25 = _ConvReLU(64, 64, kernel_size=2)
        self.conv26 = _ConvReLU(64, 64)
        self.conv27 = _ConvReLU(64, 32)
        self.conv28 = _SameConv2d(32, out_channels, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.upsample_mode = upsample_mode
        self.final_activation = final_activation
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        skip1 = self.conv3(h)

        h = self.pool(skip1)
        skip2 = self.conv4(h)

        h = self.pool(skip2)
        h = self.conv5(h)
        h = self.conv6(h)
        skip3 = self.conv7(h)

        h = self.pool(skip3)
        h = self.conv8(h)
        h = self.conv9(h)
        skip4 = self.conv10(h)

        h = self.pool(skip4)
        h = self.conv11(h)
        h = self.dropout(self.conv12(h))
        h = self.conv13(h)
        h = self.conv14(h)

        h = self.dropout(self.conv15(h))
        h = self._up_to(h, skip4)
        h = self.conv16(torch.cat([h, skip4], dim=1))
        h = self.conv17(h)

        h = self._up_to(h, skip3)
        h = self.conv18(torch.cat([h, skip3], dim=1))
        h = self.conv19(h)
        h = self.conv20(h)

        h = self._up_to(h, skip2)
        h = self.conv21(torch.cat([h, skip2], dim=1))
        h = self.conv22(h)
        h = self.conv23(h)

        h = self._up_to(h, skip1)
        h = self.conv24(torch.cat([h, skip1], dim=1))
        h = self.conv25(h)
        h = self.conv26(h)
        h = self.conv27(h)
        h = self.conv28(h)
        if self.final_activation == "tanh":
            h = torch.tanh(h)
        return h

    def _up_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        kwargs = {"scale_factor": 2.0, "mode": self.upsample_mode}
        if self.upsample_mode == "bilinear":
            kwargs["align_corners"] = False
        out = F.interpolate(x, **kwargs)
        if out.shape[-2:] == ref.shape[-2:]:
            return out
        kwargs = {"size": ref.shape[-2:], "mode": self.upsample_mode}
        if self.upsample_mode == "bilinear":
            kwargs["align_corners"] = False
        return F.interpolate(out, **kwargs)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, _SameConv2d):
                nn.init.kaiming_normal_(module.conv.weight, nonlinearity="relu")
                if module.conv.bias is not None:
                    nn.init.constant_(module.conv.bias, 0.0)

