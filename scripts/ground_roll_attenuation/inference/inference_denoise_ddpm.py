"""DDPM ground-roll attenuation inference: noisy SEG-Y → denoised SEG-Y.

Uses DDIM (deterministic) sampling with a few steps for efficiency.
The diffusion model takes the noisy input as condition and iteratively
denoises from pure Gaussian noise.

Example::

    python scripts/ground_roll_attenuation/inference/inference_denoise_ddpm.py \\
        --config configs/ground_roll_attenuation/denoise_ddpm.yaml \\
        --checkpoint results/.../checkpoints/best.pt \\
        --input-sgy /path/to/noisy.sgy \\
        --output-sgy /path/to/denoised.sgy \\
        --sample-steps 10
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
from model.ground_roll_attenuation.ddpm import DDPMNoiseScheduler  # noqa: E402

_DEFAULT_CONFIG = "configs/ground_roll_attenuation/denoise_ddpm.yaml"

# ---------------------------------------------------------------------------
# global so _forward can access it (set in main before run_inference)
# ---------------------------------------------------------------------------
_scheduler: DDPMNoiseScheduler = None  # type: ignore[assignment]
_sample_steps: int = 10
_use_ddim: bool = True
_ddim_eta: float = 0.0


def _forward(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """DDIM sampling: condition on noisy input, sample clean signal x_0."""
    x_0_pred, _z_0 = _scheduler.sample_full(
        model, batch,
        num_steps=_sample_steps,
        use_ddim=_use_ddim,
        ddim_eta=_ddim_eta,
        progress=False,
    )
    return x_0_pred


def main() -> None:
    global _scheduler, _sample_steps, _use_ddim, _ddim_eta

    p = argparse.ArgumentParser(description="DDPM ground-roll attenuation inference.")
    from _common import add_inference_args, run_inference
    add_inference_args(p)
    p.add_argument("--sample-steps", type=int, default=10,
                   help="Number of DDIM sampling steps (default: 10).")
    p.add_argument("--no-ddim", action="store_true",
                   help="Use full DDPM sampling instead of DDIM.")
    p.add_argument("--ddim-eta", type=float, default=0.0,
                   help="DDIM stochasticity (0 = deterministic, default: 0).")
    args = p.parse_args()

    # Build scheduler (must match training config)
    from utils import load_config
    cfg = load_config(args.config if args.config else _DEFAULT_CONFIG)
    ddpm_cfg = cfg.get("ddpm", {})

    num_timesteps = int(ddpm_cfg.get("num_timesteps", 1000))
    beta_start = float(ddpm_cfg.get("beta_start", 1e-4))
    beta_end = float(ddpm_cfg.get("beta_end", 0.02))
    _scheduler = DDPMNoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    _sample_steps = args.sample_steps
    _use_ddim = not args.no_ddim
    _ddim_eta = args.ddim_eta

    run_inference(args, _DEFAULT_CONFIG, build_model, _forward, model_type_label="ddpm")


if __name__ == "__main__":
    main()
