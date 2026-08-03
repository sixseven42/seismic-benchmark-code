"""UNet ground-roll attenuation inference: noisy SEG-Y → denoised SEG-Y.

Example::

    python scripts/ground_roll_attenuation/inference/inference_denoise_unet.py \\
        --config configs/ground_roll_attenuation/denoise_unet.yaml \\
        --checkpoint results/.../checkpoints/best.pt \\
        --input-sgy /path/to/noisy.sgy \\
        --output-sgy /path/to/denoised.sgy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = next((p for p in Path(__file__).resolve().parents
                   if (p / "model").is_dir() and (p / "utils").is_dir()), None)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.ground_roll_attenuation import build_model  # noqa: E402

_DEFAULT_CONFIG = "configs/ground_roll_attenuation/denoise_unet.yaml"


def _forward(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """UNet predicts noise component; denoised = input - predicted noise."""
    return batch - model(batch)


def main() -> None:
    p = argparse.ArgumentParser(description="UNet ground-roll attenuation inference.")
    from _common import add_inference_args, run_inference
    add_inference_args(p)
    args = p.parse_args()
    run_inference(args, _DEFAULT_CONFIG, build_model, _forward, model_type_label="unet")


if __name__ == "__main__":
    main()
