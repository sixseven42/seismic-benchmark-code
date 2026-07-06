"""Self-attention GAN network components for seismic multiple suppression.

The registered ``sagan`` model is the U-Net generator from Tao et al. (2022).
The Markov discriminator is included for completeness, but the existing paired
denoise training scripts use only the generator with supervised losses.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model

NormType = Literal["none", "batch", "instance"]


class SelfAttention2d(nn.Module):
    """SAGAN-style self-attention over flattened spatial locations."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        attn_channels = max(1, int(channels) // int(reduction))
        self.query = nn.Conv2d(channels, attn_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, attn_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        query = self.query(x).view(b, -1, n).transpose(1, 2)
        key = self.key(x).view(b, -1, n)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(b, c, n)
        out = torch.bmm(value, attention.transpose(1, 2)).view(b, c, h, w)
        return self.gamma * out + x


def _norm_layer(channels: int, norm: NormType) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unknown norm type: {norm!r}.")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: NormType) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=norm == "none"),
            _norm_layer(out_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=norm == "none"),
            _norm_layer(out_channels, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("sagan")
class SAGAN(nn.Module):
    """U-Net generator with a bottleneck self-attention block."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        norm: NormType = "none",
        final_activation: Literal["identity", "tanh"] = "identity",
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")
        if final_activation not in {"identity", "tanh"}:
            raise ValueError(f"Unsupported final_activation: {final_activation!r}.")

        channels = [base_channels * (2**i) for i in range(5)]
        self.encoders = nn.ModuleList()
        prev = int(in_channels)
        for c in channels:
            self.encoders.append(_ConvBlock(prev, c, norm=norm))
            prev = c
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention = SelfAttention2d(channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.upconvs.append(
                nn.ConvTranspose2d(prev, skip_channels, kernel_size=2, stride=2)
            )
            self.decoders.append(
                _ConvBlock(skip_channels * 2, skip_channels, norm=norm)
            )
            prev = skip_channels

        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.final_activation = final_activation
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for idx, encoder in enumerate(self.encoders):
            h = encoder(h)
            if idx < len(self.encoders) - 1:
                skips.append(h)
                h = self.pool(h)

        h = self.attention(h)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = upconv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = decoder(torch.cat([skip, h], dim=1))

        h = self.head(h)
        if self.final_activation == "tanh":
            h = torch.tanh(h)
        return h

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


class SAGANMarkovDiscriminator(nn.Module):
    """Patch/Markov discriminator from the SAGAN paper."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = []
        prev = int(in_channels)
        for c in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, c, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            prev = c
        self.features = nn.Sequential(*layers)
        self.attention = SelfAttention2d(channels[-1])
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=1)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.attention(self.features(x)))

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.constant_(module.bias, 0.0)

