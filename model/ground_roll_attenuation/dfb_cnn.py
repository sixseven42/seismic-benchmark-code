"""DFB-CNN: Dual-Filter-Bank CNN for ground-roll attenuation (Zhang & van der Baan, 2022).

Two DnCNN-style subnetworks with different kernel sizes process low- and high-frequency
components separately after a radial trace (RT) transform that compacts ground-roll.
The model predicts additive noise (residual learning), compatible with the existing
denoise pipeline (``metrics_on_denoised_signal=True``).

Reference: IEEE Trans. Geosci. Remote Sens., vol. 60, 2022, 5907511.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Helper: Gaussian low-pass filter
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """1-D Gaussian kernel (unnormalized on purpose — sum-to-1 later)."""
    radius = int(sigma * truncate + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class _GaussianLowPass(nn.Module):
    """Spatial-domain Gaussian low-pass filter (separable, fixed kernel)."""

    def __init__(self, sigma: float = 5.0, channels: int = 1) -> None:
        super().__init__()
        k1d = _gaussian_kernel(sigma)
        k2d = k1d[:, None] * k1d[None, :]  # (K, K)
        weight = k2d.view(1, 1, *k2d.shape).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = channels
        self.padding = k2d.shape[0] // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


# ---------------------------------------------------------------------------
# Radial Trace Transform (differentiable via grid_sample)
# ---------------------------------------------------------------------------

class _RadialTraceTransform(nn.Module):
    """Differentiable forward/inverse RT transform using ``grid_sample``.

    Forward:  (x, t) → (v, t')  — maps offset–time to velocity–shifted-time.
    Inverse:  (v, t') → (x, t)  — maps back.

    The origin ``(x0, t0)`` defaults to the bottom-centre of each patch,
    approximating the source-point origin for a local patch context.
    """

    def __init__(self, v_max: float = 3200.0, n_velocities: int = 128) -> None:
        super().__init__()
        self.v_max = v_max
        self.n_velocities = n_velocities

    def _normalized_grid(
        self, coords_xy: torch.Tensor, H_out: int, W_out: int, device: torch.device
    ) -> torch.Tensor:
        """Convert (x, t) physical coords to [-1, 1] sampling grid."""
        gx = 2.0 * coords_xy[..., 0] / (W_out - 1) - 1.0
        gy = 2.0 * coords_xy[..., 1] / (H_out - 1) - 1.0
        return torch.stack([gx, gy], dim=-1)  # (H, W, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward RT: (B, C, H, W)_(t,x) → (B, C, H, W_out)_(t',v)."""
        B, C, H, W = x.shape
        device = x.device
        W_out = min(self.n_velocities, W)

        # Origin at bottom-centre: (x0, t0) = (W//2, H-1)
        x0 = float(W // 2)
        t0 = float(H - 1)

        # Velocity axis
        vs = torch.linspace(-self.v_max, self.v_max, W_out, device=device)

        # Build sampling coordinates: for each (t', v), compute (x, t) = (x0+v*t', t'+t0)
        t_prime = torch.arange(H, device=device, dtype=torch.float32)
        # (H, W_out)
        x_coord = x0 + vs[None, :] * t_prime[:, None]
        t_coord = (t0 + t_prime[:, None]).expand(-1, W_out)
        # (H, W_out) → (H, W_out, 2) — note grid_sample expects (x, y) = (W, H) order
        grid = torch.stack(
            [
                2.0 * x_coord / (W - 1) - 1.0,
                2.0 * t_coord / (H - 1) - 1.0,
            ],
            dim=-1,
        )
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W_out, 2)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def inverse(self, x_rt: torch.Tensor, H_orig: int, W_orig: int) -> torch.Tensor:
        """Inverse RT: (B, C, H, W_rt)_(t',v) → (B, C, H_orig, W_orig)_(t,x)."""
        B, C, H_rt, W_rt = x_rt.shape
        device = x_rt.device

        x0 = float(W_orig // 2)
        t0 = float(H_orig - 1)

        xs = torch.arange(W_orig, device=device, dtype=torch.float32)
        ts = torch.arange(H_orig, device=device, dtype=torch.float32)

        # t' = t - t0
        t_prime = ts[:, None] - t0  # (H_orig, 1)
        # v = (x - x0) / (t' + eps)
        eps = 1e-6
        v_coord = (xs[None, :] - x0) / (t_prime + eps)  # (H_orig, W_orig)

        # Map v ∈ [-v_max, v_max] → index [0, W_rt-1]
        v_idx = (v_coord / self.v_max) * ((W_rt - 1) / 2.0) + (W_rt - 1) / 2.0
        v_idx_norm = 2.0 * v_idx / (W_rt - 1) - 1.0  # (H_orig, W_orig)

        # t' index: simple 1:1 mapping t' ∈ [0, H_rt)  → assume H_rt == H_orig
        t_idx = ts[:, None].expand(-1, W_orig)  # (H_orig, W_orig)
        t_idx_norm = 2.0 * t_idx / (H_rt - 1) - 1.0

        grid = torch.stack([v_idx_norm, t_idx_norm], dim=-1)  # (H_orig, W_orig, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        return F.grid_sample(x_rt, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# DnCNN-style sub-network (residual, noise-predicting)
# ---------------------------------------------------------------------------

class _DnCNNBlock(nn.Module):
    """Stack of Conv–BN–ReLU + final Conv with residual output ``x - net(x)``."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 5,
        base_channels: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"DnCNN depth must be >= 2, got {depth}.")
        pad = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, padding=pad, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, padding=pad, bias=False)
            )
            layers.append(nn.BatchNorm2d(base_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(base_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        )
        self.net = nn.Sequential(*layers)
        # Learnable output gain — initialised to 1.0 since BN normalises RT features
        self.gain = nn.Parameter(torch.tensor(1.0))
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.gain

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# DFB-CNN
# ---------------------------------------------------------------------------

@register_model("dfb_cnn")
class DFBCNN(nn.Module):
    """Dual-Filter-Bank CNN for ground-roll attenuation.

    Splits input into low/high-frequency branches via a Gaussian low-pass filter,
    transforms both branches to the radial-trace (RT) domain where ground-roll is
    compacted, predicts noise with two DnCNN-style sub-networks of different kernel
    sizes, applies inverse RT, and combines the branches back to x–t domain.

    Parameters
    ----------
    in_channels : int
        Input channels (default 1).
    out_channels : int
        Output channels, must equal ``in_channels`` (default 1).
    lowpass_sigma : float
        Spatial sigma for the Gaussian low-pass filter in pixels (default 5.0).
    v_max : float
        Maximum apparent velocity in RT domain m/s (default 3200.0).
    n_velocities : int
        Number of velocity bins; when ≤0, uses input patch width (default 0 → auto).
    cnn1_kernel_size : int
        CNN1 (low-freq) kernel size (default 5).
    cnn1_depth : int
        CNN1 conv layer count (default 9).
    cnn1_base_channels : int
        CNN1 feature maps per hidden layer (default 100).
    cnn2_kernel_size : int
        CNN2 (high-freq) kernel size (default 3).
    cnn2_depth : int
        CNN2 conv layer count (default 5).
    cnn2_base_channels : int
        CNN2 feature maps per hidden layer (default 64).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        lowpass_sigma: float = 1.5,
        v_max: float = 3200.0,
        n_velocities: int = 0,
        cnn1_kernel_size: int = 5,
        cnn1_depth: int = 9,
        cnn1_base_channels: int = 100,
        cnn2_kernel_size: int = 3,
        cnn2_depth: int = 5,
        cnn2_base_channels: int = 64,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                f"DFB-CNN residual form requires in_channels == out_channels; "
                f"got {in_channels} vs {out_channels}."
            )

        self.in_channels = in_channels
        self.lowpass = _GaussianLowPass(sigma=lowpass_sigma, channels=in_channels)
        self.rt = _RadialTraceTransform(v_max=v_max, n_velocities=n_velocities if n_velocities > 0 else 9999)

        # BN before DnCNN to normalise RT-domain features (compensates Gaussian + RT attenuation).
        # track_running_stats=False → always use batch stats so train/eval behaviour is identical.
        self.norm_low = nn.BatchNorm2d(in_channels, track_running_stats=False)
        self.norm_high = nn.BatchNorm2d(in_channels, track_running_stats=False)

        self.cnn1 = _DnCNNBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=cnn1_depth,
            base_channels=cnn1_base_channels,
            kernel_size=cnn1_kernel_size,
        )
        self.cnn2 = _DnCNNBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=cnn2_depth,
            base_channels=cnn2_base_channels,
            kernel_size=cnn2_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict additive noise in x–t domain.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input patches ``(B, C, H, W)`` in offset–time domain.

        Returns
        -------
        torch.Tensor
            Predicted noise ``(B, C, H, W)`` in the same domain.
        """
        B, C, H, W = x.shape

        # 1. Frequency separation
        x_low = self.lowpass(x)                     # low-frequency component
        x_high = x - x_low                          # high-frequency component

        # 2. Forward RT transform
        rt_low = self.rt(x_low)
        rt_high = self.rt(x_high)

        # 2b. Batch-norm RT features → normalised input for DnCNN blocks
        rt_low = self.norm_low(rt_low)
        rt_high = self.norm_high(rt_high)

        # 3. Noise prediction in RT domain (residual sub-networks)
        noise_rt_low = self.cnn1(rt_low)
        noise_rt_high = self.cnn2(rt_high)

        # 4. Inverse RT → noise in x–t domain
        noise_low = self.rt.inverse(noise_rt_low, H, W)
        noise_high = self.rt.inverse(noise_rt_high, H, W)

        # 5. Combine branches
        return noise_low + noise_high
