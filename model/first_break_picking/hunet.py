""" HU-Net: A nested U-Net with side outputs at each decoder level and bottleneck. 
    These side outputs are fused into a final prediction. forward returns (fused, sides).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model

class _DoubleConv(nn.Module):
    """
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("hunet")
class HUNet(nn.Module):
    """Holistically-nested U-Net。

    Configurable depth / base_channels, matching the other UNet variants.
    Every decoder layer (including bottleneck) produces a side output; all
    side outputs are upsampled to the input resolution and fused.

    Parameters
    ----------
    in_channels   : Number of input channels (default: 1)
    out_channels  : Number of output channels (default: 1)
    depth         : Number of encoder levels (default: 4); channels = [base * 2^i]
    base_channels : Number of channels in the first layer (default: 32)
    return_sides  : Whether to return side outputs (default: True; if False, only the fused output is returned)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        base_channels: int = 32,
        return_sides: bool = True,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"HUNet depth must be >= 2, got {depth}.")

        chans: List[int] = [base_channels * (2 ** i) for i in range(depth)]
        self.depth = depth
        self.return_sides = return_sides

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(_DoubleConv(prev, c))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = c

        self.bottleneck = _DoubleConv(chans[-1], chans[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = chans[-1] * 2
        for c in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(dec_in, c, kernel_size=2, stride=2))
            self.decoders.append(_DoubleConv(c * 2, c))
            dec_in = c

        # Side-output heads (depth decoder layers + 1 bottleneck)
        self.side_heads = nn.ModuleList()
        self.side_heads.append(nn.Conv2d(chans[-1] * 2, out_channels, kernel_size=1))
        for c in reversed(chans):
            self.side_heads.append(nn.Conv2d(c, out_channels, kernel_size=1))

        num_sides = depth + 1
        self.fuse = nn.Conv2d(out_channels * num_sides, out_channels, kernel_size=1)
        with torch.no_grad():
            self.fuse.weight.fill_(1.0 / num_sides)

    @staticmethod
    def _cat_skip(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    @staticmethod
    def _upsample_to_input(side: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if side.shape[-2:] != size:
            side = F.interpolate(side, size=size, mode="bilinear", align_corners=False)
        return side

    def forward(
        self, x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        input_size = x.shape[-2:]

        # Encoder: collect skip connections
        skips: List[torch.Tensor] = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        b = self.bottleneck(h)

        # Decoder
        h = b
        decoder_outputs: List[torch.Tensor] = []
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = self._cat_skip(up(h), skip)
            h = dec(h)
            decoder_outputs.append(h)

        # Side outputs (bottleneck + each decoder layer, upsampled to input size)
        sides: List[torch.Tensor] = [
            self._upsample_to_input(head(src), input_size)
            for head, src in zip(self.side_heads, [b] + decoder_outputs)
        ]

        fused = self.fuse(torch.cat(sides, dim=1))

        if self.return_sides:
            return fused, sides
        return fused
