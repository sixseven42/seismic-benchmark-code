"""Trace-by-trace 1D CNN for free-surface multiple attenuation and deghosting.

Reproduces the network-level method described by Kiraz et al. (2024): nine
hidden 1D convolutional layers applied independently to each seismic trace,
followed by a linear 1x1 output convolution.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..registry import register_model


@register_model("kiraz_trace_cnn")
class KirazTraceCNN(nn.Module):
    """Trace-wise 1D CNN that preserves the existing 4D patch tensor format.

    The denoise training pipeline provides patches as ``(batch, channel, trace,
    time)``. This module folds the trace axis into the batch axis, applies the
    paper's 1D convolutional network along time, then restores the original
    patch layout.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: Sequence[int] = (32, 32, 8, 32, 32, 8, 32, 32, 8),
        kernel_sizes: Sequence[int] = (1125, 1125, 1125, 101, 101, 101, 101, 101, 51),
        output_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 9:
            raise ValueError(
                "KirazTraceCNN expects nine hidden convolutional layers; "
                f"got {len(hidden_channels)}."
            )
        if len(kernel_sizes) != len(hidden_channels):
            raise ValueError(
                "kernel_sizes must match hidden_channels; "
                f"got {len(kernel_sizes)} vs {len(hidden_channels)}."
            )
        for kernel_size in (*kernel_sizes, output_kernel_size):
            if int(kernel_size) <= 0:
                raise ValueError(f"kernel sizes must be positive, got {kernel_size}.")

        layers: list[nn.Module] = []
        prev = int(in_channels)
        for channels, kernel_size in zip(hidden_channels, kernel_sizes):
            k = int(kernel_size)
            layers.extend(
                [
                    nn.Conv1d(prev, int(channels), kernel_size=k, padding=k // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            prev = int(channels)

        out_k = int(output_kernel_size)
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev, int(out_channels), kernel_size=out_k, padding=out_k // 2)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return self._forward_traces(x)
        if x.ndim != 4:
            raise ValueError(
                "KirazTraceCNN expects input with shape (B, C, T) or "
                f"(B, C, trace, time); got {tuple(x.shape)}."
            )

        b, c, n_traces, n_time = x.shape
        traces = x.permute(0, 2, 1, 3).reshape(b * n_traces, c, n_time)
        out = self._forward_traces(traces)
        if out.shape[-1] != n_time:
            out = self._match_time(out, n_time)
        return out.reshape(b, n_traces, out.shape[1], n_time).permute(0, 2, 1, 3)

    def _forward_traces(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.features(x))
        if out.shape[-1] != x.shape[-1]:
            out = self._match_time(out, x.shape[-1])
        return out

    @staticmethod
    def _match_time(x: torch.Tensor, n_time: int) -> torch.Tensor:
        if x.shape[-1] > n_time:
            start = (x.shape[-1] - n_time) // 2
            return x[..., start : start + n_time]
        pad = n_time - x.shape[-1]
        left = pad // 2
        right = pad - left
        return torch.nn.functional.pad(x, (left, right))

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

