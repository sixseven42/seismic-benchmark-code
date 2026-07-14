"""STUNet with a Swin Transformer encoder for first-break segmentation."""

from __future__ import annotations

from typing import List, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


def _to_2tuple(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, c)


def _window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    b = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        b, height // window_size, width // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, height, width, -1)


class _DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _Mlp(nn.Module):
    """Two-layer feed-forward network used inside Swin blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class _WindowAttention(nn.Module):
    """Window multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b_windows, n_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            b_windows, n_tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        position_index = cast(torch.Tensor, self.relative_position_index).reshape(-1)
        relative_position_bias = self.relative_position_bias_table[position_index]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(
                b_windows // n_windows, n_windows, self.num_heads, n_tokens, n_tokens
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b_windows, n_tokens, channels)
        x = self.proj_drop(self.proj(x))
        return x


class _SwinTransformerBlock(nn.Module):
    """Swin Transformer block with regular or shifted window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if shift_size >= window_size:
            raise ValueError("shift_size must be smaller than window_size.")
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def _attention_mask(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = _window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        return attn_mask.masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, height, width, channels = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, hp, wp, _ = x.shape

        shift_size = self.shift_size
        if min(hp, wp) <= self.window_size:
            shift_size = 0
        shifted_x = (
            torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            if shift_size > 0
            else x
        )
        x_windows = _window_partition(shifted_x, self.window_size)
        attn_mask = self._attention_mask(hp, wp, x.device) if shift_size > 0 else None
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = _window_reverse(attn_windows, self.window_size, hp, wp)
        x = (
            torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
            if shift_size > 0
            else shifted_x
        )
        if pad_h or pad_w:
            x = x[:, :height, :width, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Patch projection from an image tensor to NHWC Swin features."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Tuple[int, int] = 4,
    ) -> None:
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        patch_h, patch_w = self.patch_size
        self.patch_dim = in_channels * patch_h * patch_w
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - x.shape[-2] % patch_h) % patch_h
        pad_w = (patch_w - x.shape[-1] % patch_w) % patch_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height = x.shape[-2] // patch_h
        width = x.shape[-1] // patch_w
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.transpose(1, 2).contiguous()
        x = self.proj(x).view(x.shape[0], height, width, -1)
        return self.norm(x)


class _PatchMerging(nn.Module):
    """Swin patch merging layer that halves spatial resolution."""

    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, out_dim or 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, _ = x.shape
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class _BasicLayer(nn.Module):
    """A Swin stage with optional patch merging before the blocks."""

    def __init__(
        self,
        in_dim: int,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: Sequence[float],
        downsample: bool,
    ) -> None:
        super().__init__()
        self.downsample = _PatchMerging(in_dim, dim) if downsample else None
        self.blocks = nn.ModuleList(
            [
                _SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x, x


class _DilatedResidualBlock(nn.Module):
    """Residual dilated convolution block used in the STUNet bottleneck."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class _DecoderBlock(nn.Module):
    """Upsampling block with one optional encoder skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)
        self.post = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(self.pre(x))
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
        return self.post(x)


@register_model("stunet")
class STUNet(nn.Module):
    """Swin Transformer U-Net with a dilated-convolution bottleneck."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 96,
        patch_size: int = 4,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("STUNet expects four Swin stages.")

        self.patch_size = patch_size
        self.patch_embed = _PatchEmbed(in_channels, embed_dim, patch_size)
        dims = [embed_dim * (2**i) for i in range(4)]
        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()

        self.layers = nn.ModuleList()
        cursor = 0
        for i, depth in enumerate(depths):
            self.layers.append(
                _BasicLayer(
                    in_dim=dims[i - 1] if i > 0 else dims[i],
                    dim=dims[i],
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cursor : cursor + depth],
                    downsample=i > 0,
                )
            )
            cursor += depth

        self.bottleneck = nn.Sequential(
            *[_DilatedResidualBlock(dims[-1], dilation=d) for d in dilations]
        )
        self.decoders = nn.ModuleList(
            [
                _DecoderBlock(dims[3], dims[2], dims[2]),
                _DecoderBlock(dims[2], dims[1], dims[1]),
                _DecoderBlock(dims[1], dims[0], dims[0]),
            ]
        )
        self.segmentation_head = nn.Conv2d(dims[0], out_channels, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        h = self.patch_embed(x)

        stage_outputs: List[torch.Tensor] = []
        for layer in self.layers:
            h, stage_output = layer(h)
            stage_outputs.append(stage_output)

        h = self._nhwc_to_nchw(h)
        h = self.bottleneck(h)
        skips: List[torch.Tensor | None] = [
            self._nhwc_to_nchw(stage_outputs[2]),
            self._nhwc_to_nchw(stage_outputs[1]),
            self._nhwc_to_nchw(stage_outputs[0]),
        ]

        for decoder, skip in zip(self.decoders, skips):
            h = decoder(h, skip)

        if h.shape[-2:] != input_size:
            h = F.interpolate(h, size=input_size, mode="bilinear", align_corners=False)
        return self.segmentation_head(h)
