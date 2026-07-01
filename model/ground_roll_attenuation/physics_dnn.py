"""Physics-constrained DNN for nonstationary coherent noise suppression (Liu et al., 2025).

MPIC (Multi-modality Physical Information Constraint) Blocks process multi-channel
feature matrices through parallel spatial, frequency, and Hilbert-domain branches.
Physical constraints (frequency masks, Hilbert envelopes) limit the solution space
to avoid underfitting.  The model predicts additive noise (residual learning),
compatible with the existing denoise pipeline.

Reference: IEEE Trans. Geosci. Remote Sens., vol. 63, 2025, 5903010.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


# ---------------------------------------------------------------------------
# Frequency-domain physical constraint mask (per-frequency-bin learnable)
# ---------------------------------------------------------------------------

class _FrequencyMask(nn.Module):
    """Learnable frequency-domain mask with per-frequency-bin weights.

    Each frequency bin ``(h, w)`` in the rFFT output gets its own learnable
    weight, initialised to 1.0 (pass-through).  A sigmoid gate parameterized
    by ``(center, sigma)`` provides the *physical constraint* — it suppresses
    low frequencies (ground-roll) while passing higher frequencies (reflections).

    The effective mask is ``sigmoid_gate * per_bin_weight``, where the gate
    encodes prior knowledge and the per-bin weights are learned from data.

    Parameters
    ----------
    init_center : float
        Initial sigmoid centre (fraction of Nyquist, default 0.15).
    init_sigma : float
        Initial sigmoid width (default 0.05).
    """

    def __init__(self, init_center: float = 0.15, init_sigma: float = 0.05) -> None:
        super().__init__()
        # Sigmoid gate parameters (physical constraint)
        self.logit_center = nn.Parameter(torch.tensor(
            [math.log(init_center / (1 - init_center + 1e-8))]
        ))
        self.logit_sigma = nn.Parameter(torch.tensor(
            [math.log(init_sigma / (1 - init_sigma + 1e-8))]
        ))
        # Per-frequency-bin learnable weights are allocated lazily on first forward
        self._weight: nn.Parameter | None = None

    def _ensure_weight(self, H_rfft: int, W_rfft: int, device: torch.device) -> torch.Tensor:
        if self._weight is None:
            self._weight = nn.Parameter(torch.ones(1, 1, H_rfft, W_rfft, device=device))
            return self._weight
        # Interpolate existing weights if spatial dimensions changed
        if self._weight.shape[2] != H_rfft or self._weight.shape[3] != W_rfft:
            return F.interpolate(self._weight, size=(H_rfft, W_rfft),
                                 mode="bilinear", align_corners=True)
        return self._weight

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return frequency mask ``(1, 1, H, W//2+1)`` in [0, 1]."""
        fy = torch.fft.fftfreq(H, device=device).view(-1, 1).abs()
        fx = torch.fft.rfftfreq(W, device=device).view(1, -1).abs()
        radius = torch.sqrt(fx ** 2 + fy ** 2)
        radius = radius / (radius.max() + 1e-8)

        center = torch.sigmoid(self.logit_center)
        sigma = torch.sigmoid(self.logit_sigma) * 0.3 + 0.01
        gate = torch.sigmoid((radius - center) / sigma)      # (H, W//2+1)
        gate = gate.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W//2+1)

        w = self._ensure_weight(gate.shape[2], gate.shape[3], device)
        return gate * w                                        # physical constraint × learned weights


# ---------------------------------------------------------------------------
# MPIC Block modalities
# ---------------------------------------------------------------------------

class _SpatialModality(nn.Module):
    """Spatial-domain convolution (T modality) — learns local features."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _FrequencyModality(nn.Module):
    """Frequency-domain constraint (F modality): FFT → γ_F ⊙ W_freq → IFFT.

    Applies a learnable per-frequency-bin weight multiplied by a smooth
    sigmoid gate that suppresses low frequencies (the physical constraint).
    """

    def __init__(self, channels: int, init_center: float = 0.15, init_sigma: float = 0.05) -> None:
        super().__init__()
        self.freq_mask = _FrequencyMask(init_center=init_center, init_sigma=init_sigma)
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x, norm="ortho")                  # (B, C, H, W//2+1) complex
        mask = self.freq_mask(H, W, x.device)                  # (1, 1, H, W//2+1)
        X_masked = X * mask
        out = torch.fft.irfft2(X_masked, s=(H, W), norm="ortho")
        return out * self.scale.view(1, -1, 1, 1)


class _HilbertModality(nn.Module):
    """Hilbert-domain constraint (H modality) along the time axis.

    Computes the analytic signal via 1-D Hilbert transform along the time
    dimension (dim=-2), processes the envelope with a learnable per-channel
    weight, then reconstructs the real signal.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — H is time, W is trace
        B, C, H_orig, W_orig = x.shape

        # 1-D rFFT along time axis (dim=-2)
        X = torch.fft.rfft(x, n=None, dim=-2, norm="ortho")   # (B, C, H_rfft, W)

        # Analytic signal: zero negative freqs & double positive (already rFFT)
        # Apply learnable per-channel envelope scaling in frequency domain
        X_h = X * self.scale.view(1, -1, 1, 1)

        # Inverse FFT back to time domain
        out = torch.fft.irfft(X_h, n=H_orig, dim=-2, norm="ortho")
        return out


# ---------------------------------------------------------------------------
# MPIC Block
# ---------------------------------------------------------------------------

class _MPICBlock(nn.Module):
    """Multi-modality Physical Information Constraint Block.

    Parallel spatial / frequency / Hilbert branches are summed, then
    BN → ReLU → + residual (short-circuit), following the paper's
    formula (3) with a ResNet-inspired skip connection.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    kernel_size : int
        Spatial conv kernel size (default 5 for wider receptive field).
    use_spatial : bool
    use_frequency : bool
    use_hilbert : bool
    freq_init_center : float
    freq_init_sigma : float
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_hilbert: bool = True,
        freq_init_center: float = 0.15,
        freq_init_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.use_hilbert = use_hilbert

        if use_spatial:
            self.spatial = _SpatialModality(channels, kernel_size=kernel_size)
        if use_frequency:
            self.frequency = _FrequencyModality(channels, freq_init_center, freq_init_sigma)
        if use_hilbert:
            self.hilbert = _HilbertModality(channels)

        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = torch.zeros_like(x)

        if self.use_spatial:
            out = out + self.spatial(x)
        if self.use_frequency:
            out = out + self.frequency(x)
        if self.use_hilbert:
            out = out + self.hilbert(x)

        # MPIC formula (3): ReLU(BN(Σ M_j + b)) — bias folded into Conv layers
        out = self.bn(out)
        out = self.activation(out)
        # Residual short-circuit (ResNet-inspired)
        out = out + identity
        return out


# ---------------------------------------------------------------------------
# Physics-Constrained DNN
# ---------------------------------------------------------------------------

@register_model("physics_dnn")
class PhysicsConstrainedDNN(nn.Module):
    """Physically constrained deep neural network for nonstationary noise suppression.

    Architecture:  FC_in (1×1 Conv + coord grid) → MPIC Block × N → FC_out.
    Physical constraints (frequency mask, Hilbert envelope) in each MPIC Block
    guide the optimizer toward physically plausible solutions.

    Parameters
    ----------
    in_channels : int
        Input channels (default 1).
    out_channels : int
        Output channels, must equal ``in_channels`` (default 1).
    n_channels : int
        Feature channels after the first FC layer (default 64).
    n_mpic_blocks : int
        Number of cascaded MPIC Blocks (default 6).
    spatial_kernel_size : int
        Kernel size for spatial modality conv (default 5).
    use_spatial : bool
    use_frequency : bool
    use_hilbert : bool
    freq_init_center : float
        Initial frequency mask centre as fraction of Nyquist (default 0.15).
    freq_init_sigma : float
        Initial frequency mask transition width (default 0.05).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_channels: int = 64,
        n_mpic_blocks: int = 6,
        spatial_kernel_size: int = 5,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_hilbert: bool = True,
        freq_init_center: float = 0.15,
        freq_init_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                f"PhysicsConstrainedDNN requires in_channels == out_channels; "
                f"got {in_channels} vs {out_channels}."
            )

        self.in_channels = in_channels
        self.n_channels = n_channels

        # FC1: 1×1 conv maps input (1 ch) + coord grid (2 ch) → n_channels
        self.fc_in = nn.Conv2d(in_channels + 2, n_channels, kernel_size=1, bias=True)

        # MPIC Blocks — built lazily on first forward
        self.n_mpic_blocks = n_mpic_blocks
        self._mpic_kwargs = dict(
            channels=n_channels,
            kernel_size=spatial_kernel_size,
            use_spatial=use_spatial,
            use_frequency=use_frequency,
            use_hilbert=use_hilbert,
            freq_init_center=freq_init_center,
            freq_init_sigma=freq_init_sigma,
        )
        self.mpic_blocks: nn.ModuleList | None = None

        # Output projection: BN → ReLU → 1×1 Conv (better than bare 1×1 Conv)
        self.output_bn = nn.BatchNorm2d(n_channels)
        self.output_act = nn.ReLU(inplace=True)
        self.fc_out = nn.Conv2d(n_channels, out_channels, kernel_size=1, bias=True)

        self._initialized = False
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1):
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _ensure_mpic_blocks(self, device: torch.device) -> None:
        """Build MPIC blocks on first forward pass and move them to ``device``."""
        if self._initialized:
            return
        blocks: List[nn.Module] = []
        for _ in range(self.n_mpic_blocks):
            blocks.append(_MPICBlock(**self._mpic_kwargs))
        self.mpic_blocks = nn.ModuleList(blocks)
        self.mpic_blocks.to(device)
        self._initialized = True

    def _coord_grid(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Normalised 2D coordinate grid ``(B, 2, H, W)``."""
        gy = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([gy, gx], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict additive noise in x–t domain.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input patches ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted noise ``(B, C, H, W)``.
        """
        B, C, H, W = x.shape

        self._ensure_mpic_blocks(x.device)

        # FC1: concatenate with coordinate grid, then 1×1 conv
        grid = self._coord_grid(B, H, W, x.device)
        h = torch.cat([x, grid], dim=1)        # (B, C+2, H, W)
        h = self.fc_in(h)                       # (B, n_channels, H, W)

        # Cascaded MPIC Blocks
        for block in self.mpic_blocks:
            h = block(h)

        # Output projection
        h = self.output_bn(h)
        h = self.output_act(h)
        out = self.fc_out(h)                    # (B, out_channels, H, W)

        return out
