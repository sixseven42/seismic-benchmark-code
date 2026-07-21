"""SEG-Y interpolation inference: full-shot reconstruction, per-shot metrics, and viz.

Example (single GPU):
    python scripts/interpolation/inference_interpolation.py \
        --config configs/interpolation/interpolation_unet.yaml \
        --checkpoint results/interp_unet_base/checkpoints/epoch_0049.pt \
        --output-dir results/interp_unet_base/inference \
        --n-viz-shots 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Bootstrap repo root into sys.path BEFORE importing utils/model. Walks up from
# this file looking for a directory that contains both ``model/`` and ``utils/``.
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(
        "Cannot find repo root (a directory containing both ``model/`` and ``utils/``)."
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation import build_model  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.preprocessing import (
    denormalize,
    inverse_spherical_divergence_correction,
    mask_traces,
    normalize,
    spherical_divergence_correction,
)
from utils import count_parameters, load_checkpoint, load_config  # noqa: E402
from utils.inference_utils import (
    compute_shot_metrics,
    inference_on_shots,
    save_shot_visualizations,
    select_random_shots,
)
from utils.metrics import format_metric_value  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interpolation inference on a volume and report per-shot metrics. "
        "All parameters can be set in the YAML config under the ``inference`` block; "
        "CLI arguments override config values."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/interpolation/interpolation_unet.yaml",
        help="Path to interpolation config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Required if not set in config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for inference outputs. Overrides config inference.output_dir.",
    )
    parser.add_argument(
        "--n-viz-shots",
        type=int,
        default=None,
        help="Number of random shots to visualize. Overrides config inference.n_viz_shots.",
    )
    parser.add_argument(
        "--save-viz-npy",
        dest="save_viz_npy",
        action="store_true",
        default=True,
        help="Save per-shot visualization arrays as .npy files (default: True).",
    )
    parser.add_argument(
        "--no-save-viz-npy",
        dest="save_viz_npy",
        action="store_false",
        help="Disable saving per-shot visualization arrays as .npy files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shot selection. Overrides config inference.seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cuda:0', 'cpu'). Overrides config inference.device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Overrides config inference.batch_size.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        default=None,
        help="Save predicted/input/target shots as .npy files. Overrides config inference.save_npy.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default=None,
        choices=["uniform", "random", "continuous"],
        help="Trace masking mode. Must match training. Defaults to config preprocess.mask_mode.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Trace missing ratio in (0, 1). Must match training. Defaults to config preprocess.mask_ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    infer_cfg = cfg.get("inference", {})

    # Resolve parameters: CLI > config.inference > config.experiment > defaults
    checkpoint = args.checkpoint if args.checkpoint is not None else infer_cfg.get("checkpoint")
    if checkpoint is None:
        parser = argparse.ArgumentParser()
        parser.error("--checkpoint is required (or set inference.checkpoint in config).")

    output_dir = args.output_dir if args.output_dir is not None else infer_cfg.get("output_dir")
    n_viz_shots = (
        args.n_viz_shots
        if args.n_viz_shots is not None
        else infer_cfg.get("n_viz_shots", 5)
    )
    seed = (
        args.seed
        if args.seed is not None
        else infer_cfg.get("seed", cfg["experiment"]["seed"])
    )
    device = torch.device(
        args.device
        if args.device is not None
        else infer_cfg.get("device", cfg["experiment"].get("device", "cpu"))
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else infer_cfg.get("batch_size", int(cfg["data"].get("loader", {}).get("batch_size", 8)))
    )
    save_npy = (
        args.save_npy
        if args.save_npy is not None
        else infer_cfg.get("save_npy", False)
    )

    # Mask params: CLI > config preprocess > defaults
    prep = cfg.get("preprocess", {})
    mask_mode = args.mask_mode if args.mask_mode is not None else str(prep.get("mask_mode", "uniform"))
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else float(prep.get("mask_ratio", 0.5))

    # ------------------------------------------------------------------
    # 1. Build model and load checkpoint
    # ------------------------------------------------------------------
    model = build_model(cfg["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model_type = str(cfg["model"]["type"])
    print(f"Model parameters: {count_parameters(model)}")

    # ------------------------------------------------------------------
    # 2. Load volume and preprocess (same pipeline as training)
    # ------------------------------------------------------------------
    # Prefer inference.data over data (allows independent inference volume).
    infer_data_cfg = infer_cfg.get("data", {})
    data_cfg = None
    for key in ("segy", "npy", "mat"):
        if key in infer_data_cfg:
            data_cfg = infer_data_cfg[key]
            break
    if data_cfg is None:
        for key in ("segy", "npy", "mat"):
            if key in cfg["data"]:
                data_cfg = cfg["data"][key]
                break
    if data_cfg is None:
        raise ValueError("No data source found in config.")

    shots = load_volume(data_cfg)
    prep = cfg["preprocess"]
    if prep.get("max_shots") is not None:
        shots = shots[: int(prep["max_shots"])]

    skip = set(prep.get("skip", []))

    if "spherical_divergence_correction" not in skip:
        shots = spherical_divergence_correction(
            shots,
            dt=float(prep["dt"]),
            power=float(prep.get("spherical_power", 1.2)),
            t0=float(prep.get("t0", 0.0)),
        )

    stats: Optional[Dict[str, Any]] = None
    norm_mode = str(prep.get("normalize_mode", "max_abs"))
    norm_scope = str(prep.get("normalize_scope", "global"))
    if "normalize" not in skip:
        shots, stats = normalize(shots, mode=norm_mode, per=norm_scope)

    # input = masked traces; target = unmasked (original) traces
    if "mask_traces" not in skip:
        mask_kwargs: Dict[str, Any] = {"mode": mask_mode, "ratio": mask_ratio}
        if mask_mode == "uniform":
            mask_kwargs["uniform_stride"] = int(prep.get("uniform_stride", 2))
        masked, _ = mask_traces(shots, **mask_kwargs)
    else:
        masked = shots
    shots_norm = shots
    masked_norm = masked

    # ------------------------------------------------------------------
    # 3. Inference on patches -> reconstruct full shots
    # ------------------------------------------------------------------
    patch_trace = int(prep.get("patch_trace", 128))
    patch_time = int(prep.get("patch_time", 256))
    overlap = float(prep.get("patch_overlap", 0.5))

    infer_start = time.time()
    pred_norm = inference_on_shots(
        model=model,
        input_shots=masked_norm,
        patch_size=(patch_trace, patch_time),
        overlap=overlap,
        device=device,
        batch_size=batch_size,
    )
    infer_elapsed = time.time() - infer_start
    print(f"Inference time: {infer_elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 4. Inverse preprocessing (back to original amplitude domain)
    # ------------------------------------------------------------------
    def _inverse(arr: np.ndarray) -> np.ndarray:
        if "normalize" not in skip and stats is not None:
            arr = denormalize(
                arr,
                stats,
                mode=norm_mode,
                per=norm_scope,
            )
        if "spherical_divergence_correction" not in skip:
            arr = inverse_spherical_divergence_correction(
                arr,
                dt=float(prep["dt"]),
                power=float(prep.get("spherical_power", 1.2)),
                t0=float(prep.get("t0", 0.0)),
            )
        return arr

    # ------------------------------------------------------------------
    # 5. Per-shot metrics (normalized domain)
    # ------------------------------------------------------------------
    metric_cfg = cfg.get("metrics", [])
    metric_names = [m["name"] for m in metric_cfg]
    # PSNR uses peak amplitude (max_abs), SSIM uses peak-to-peak range (L).
    if norm_mode == "max_abs":
        psnr_peak = 1.0
        ssim_data_range = 2.0
    elif norm_mode == "minmax":
        psnr_peak = 1.0
        ssim_data_range = 1.0
    else:  # mean_std — unbounded; infer from the actual target volume
        psnr_peak = float(np.max(np.abs(shots_norm)))
        ssim_data_range = float(np.max(shots_norm) - np.min(shots_norm))
        # Guard against constant-zero / near-constant data
        if psnr_peak <= 0.0:
            psnr_peak = 1.0
        if ssim_data_range <= 0.0:
            ssim_data_range = 1.0

    for m in metric_cfg:
        if m["name"] == "psnr" and "data_range" in m.get("params", {}):
            psnr_peak = float(m["params"]["data_range"])
        elif m["name"] == "ssim" and "data_range" in m.get("params", {}):
            ssim_data_range = float(m["params"]["data_range"])

    per_shot, mean = compute_shot_metrics(
        pred_norm,
        shots_norm,
        metric_names=metric_names,
        psnr_peak=psnr_peak,
        ssim_data_range=ssim_data_range,
    )

    # ------------------------------------------------------------------
    # 6. Inverse preprocessing (back to original amplitude domain)
    # ------------------------------------------------------------------
    pred_shots = _inverse(pred_norm)
    target_shots = _inverse(shots_norm)
    input_shots = _inverse(masked_norm)

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        exp = cfg.get("experiment", {})
        out_dir = Path(exp.get("output_dir", "results")) / exp.get("name", "exp") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_shots = pred_norm.shape[0]

    # CSV: one row per shot
    csv_path = out_dir / "metrics_per_shot.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = ["shot_idx"] + list(per_shot.keys())
        f.write(",".join(header) + "\n")
        for i in range(n_shots):
            row = [str(i)] + [
                format_metric_value(k, float(per_shot[k][i])) for k in per_shot.keys()
            ]
            f.write(",".join(row) + "\n")

    # JSON: summary
    summary = dict(mean)
    summary["inference_time_seconds"] = round(infer_elapsed, 3)
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional: save .npy files (original amplitude domain)
    if save_npy:
        npy_dir = out_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / "pred_shots.npy", pred_shots)
        np.save(npy_dir / "target_shots.npy", target_shots)
        np.save(npy_dir / "input_shots.npy", input_shots)
        print(f"Saved .npy files to {npy_dir}")

    # ------------------------------------------------------------------
    # 8. Visualize random shots (original amplitude domain)
    # ------------------------------------------------------------------
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    indices = select_random_shots(n_shots, n_viz_shots, seed=seed)

    # Use a single symmetric color scale for all visualizations in this run.
    vmax = float(np.quantile(np.abs(np.concatenate([
        input_shots.ravel(), pred_shots.ravel(), target_shots.ravel()
    ])), 0.995))

    save_shot_visualizations(
        input_shots=input_shots,
        pred_shots=pred_shots,
        target_shots=target_shots,
        indices=indices,
        save_dir=viz_dir,
        title_prefix=f"interp_{model_type}",
        vmin=-vmax,
        vmax=vmax,
        save_npy=args.save_viz_npy,
    )

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print(f"Inference complete. Outputs saved to: {out_dir}")
    print(f"Visualized shots: {list(indices)}")
    print("Mean metrics (normalized domain):")
    for k, v in mean.items():
        print(f"  {k}: {format_metric_value(k, v)}")


if __name__ == "__main__":
    main()
