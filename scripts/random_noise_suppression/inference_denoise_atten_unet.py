"""Random-noise suppression inference: full-shot denoising, per-shot metrics, and viz.

Example (single GPU):
    python scripts/random_noise_suppression/inference_denoise_atten_unet.py \
        --config configs/random_noise_suppression/denoise_atten_unet.yaml \
        --checkpoint results/random_noise_atten_unet_base/checkpoints/epoch_0049.pt \
        --output-dir results/random_noise_atten_unet_base/inference \
        --noise-kind gaussian \
        --snr-db 5 \
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

from model.random_noise_suppression import build_model  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.preprocessing import (  # noqa: E402
    add_noise,
    denormalize,
    inverse_spherical_divergence_correction,
    normalize,
    spherical_divergence_correction,
)
from utils import count_parameters, load_checkpoint, load_config, set_seed  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    compute_shot_metrics,
    inference_on_shots,
    save_shot_visualizations,
    select_random_shots,
)
from utils.metrics import format_metric_value  # noqa: E402


_HIGHER_IS_BETTER_METRICS = {"snr", "psnr", "ssim"}


def _resolve_data_cfg(cfg: Dict[str, Any], infer_cfg: Dict[str, Any]) -> Dict[str, Any]:
    infer_data_cfg = infer_cfg.get("data", {})
    for key in ("segy", "npy", "mat"):
        if key in infer_data_cfg:
            return infer_data_cfg[key]
    for key in ("segy", "npy", "mat"):
        if key in cfg["data"]:
            return cfg["data"][key]
    raise ValueError("No data source found in config.")


def _extract_per_shot_ffid(data_cfg: Dict[str, Any], n_shots: int) -> np.ndarray:
    path = str(data_cfg.get("path", ""))
    if path.lower().endswith((".sgy", ".segy")):
        from tools.segy_read import read_regular_shots

        _, headers = read_regular_shots(
            path,
            traces_per_shot=int(data_cfg.get("traces_per_shot", 201)),
            time_downsample=int(data_cfg.get("time_downsample", 1)),
            return_headers=True,
        )
        return headers["FieldRecord"][:, 0]
    return np.arange(n_shots)


def _select_inference_shots(
    shots: np.ndarray,
    data_cfg: Dict[str, Any],
    infer_cfg: Dict[str, Any],
) -> np.ndarray:
    shot_split = infer_cfg.get("shot_split")
    if shot_split is None:
        return shots

    n_train = int(shot_split["train"])
    n_val = int(shot_split["val"])
    n_test = int(shot_split["test"])
    per_shot_ffid = _extract_per_shot_ffid(data_cfg, shots.shape[0])
    unique_ffids = np.unique(per_shot_ffid)
    if n_train + n_val + n_test > unique_ffids.size:
        raise ValueError(
            f"inference.shot_split asks for {n_train}+{n_val}+{n_test} shots "
            f"but only {unique_ffids.size} unique FFIDs available."
        )
    test_ffids = unique_ffids[n_train + n_val : n_train + n_val + n_test]
    return shots[np.isin(per_shot_ffid, test_ffids)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run random-noise suppression inference on a clean volume by adding "
            "synthetic noise using the same preprocessing parameters as training. "
            "All parameters can be set in the YAML config under the ``inference`` "
            "block; CLI arguments override config values."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/random_noise_suppression/denoise_atten_unet.yaml",
        help="Path to random-noise suppression config.",
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
        "--noise-kind",
        type=str,
        default=None,
        choices=["gaussian", "poisson"],
        help="Synthetic noise kind. Must match training. Defaults to config preprocess.noise_kind.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=None,
        help="Synthetic noise SNR in dB. Must match training. Defaults to config preprocess.snr_db.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    infer_cfg = cfg.get("inference", {})
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
    set_seed(int(seed))
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

    prep = cfg.get("preprocess", {})
    noise_kind = (
        args.noise_kind
        if args.noise_kind is not None
        else str(prep.get("noise_kind", "gaussian"))
    )
    snr_db = (
        args.snr_db
        if args.snr_db is not None
        else float(prep.get("snr_db", 20.0))
    )

    model = build_model(cfg["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model_type = str(cfg["model"]["type"])
    print(f"Model parameters: {count_parameters(model)}")

    data_cfg = _resolve_data_cfg(cfg, infer_cfg)
    skip = set(prep.get("skip", []))
    stats: Optional[Dict[str, Any]] = None
    norm_mode = str(prep.get("normalize_mode", "max_abs"))
    norm_scope = str(prep.get("normalize_scope", "global"))
    shots = load_volume(data_cfg)
    shots = _select_inference_shots(shots, data_cfg, infer_cfg)
    if prep.get("max_shots") is not None:
        shots = shots[: int(prep["max_shots"])]

    if "spherical_divergence_correction" not in skip:
        shots = spherical_divergence_correction(
            shots,
            dt=float(prep["dt"]),
            power=float(prep.get("spherical_power", 1.2)),
            t0=float(prep.get("t0", 0.0)),
        )

    if "normalize" not in skip:
        shots, stats = normalize(shots, mode=norm_mode, per=norm_scope)

    if "add_noise" not in skip:
        noisy, _ = add_noise(shots, kind=noise_kind, snr_db=snr_db, rng=int(seed))
    else:
        noisy = shots
    shots_norm = shots
    noisy_norm = noisy

    patch_trace = int(prep.get("patch_trace", 128))
    patch_time = int(prep.get("patch_time", 256))
    overlap = float(prep.get("patch_overlap", 0.5))

    infer_start = time.time()
    pred_norm = inference_on_shots(
        model=model,
        input_shots=noisy_norm,
        patch_size=(patch_trace, patch_time),
        overlap=overlap,
        device=device,
        batch_size=batch_size,
    )
    infer_elapsed = time.time() - infer_start
    print(f"Inference time: {infer_elapsed:.2f}s")

    def _inverse(arr: np.ndarray) -> np.ndarray:
        if "normalize" not in skip and stats is not None:
            arr = denormalize(arr, stats, mode=norm_mode, per=norm_scope)
        if "spherical_divergence_correction" not in skip:
            arr = inverse_spherical_divergence_correction(
                arr,
                dt=float(prep["dt"]),
                power=float(prep.get("spherical_power", 1.2)),
                t0=float(prep.get("t0", 0.0)),
            )
        return arr

    metric_cfg = cfg.get("metrics", [])
    metric_names = [m["name"] for m in metric_cfg]
    if norm_mode == "max_abs":
        psnr_peak = 1.0
        ssim_data_range = 2.0
    elif norm_mode == "minmax":
        psnr_peak = 1.0
        ssim_data_range = 1.0
    else:
        psnr_peak = float(np.max(np.abs(shots_norm)))
        ssim_data_range = float(np.max(shots_norm) - np.min(shots_norm))
        if psnr_peak <= 0.0:
            psnr_peak = 1.0
        if ssim_data_range <= 0.0:
            ssim_data_range = 1.0

    for m in metric_cfg:
        if m["name"] == "psnr" and "data_range" in m.get("params", {}):
            psnr_peak = float(m["params"]["data_range"])
        elif m["name"] == "ssim" and "data_range" in m.get("params", {}):
            ssim_data_range = float(m["params"]["data_range"])

    noisy_per_shot, noisy_mean = compute_shot_metrics(
        noisy_norm,
        shots_norm,
        metric_names=metric_names,
        psnr_peak=psnr_peak,
        ssim_data_range=ssim_data_range,
    )

    denoised_per_shot, denoised_mean = compute_shot_metrics(
        pred_norm,
        shots_norm,
        metric_names=metric_names,
        psnr_peak=psnr_peak,
        ssim_data_range=ssim_data_range,
    )

    delta_per_shot: Dict[str, np.ndarray] = {}
    delta_mean: Dict[str, float] = {}
    for name in metric_names:
        key = name.lower()
        delta_arr = denoised_per_shot[key] - noisy_per_shot[key]
        delta_per_shot[key] = delta_arr
        delta_mean[key] = round(float(np.nanmean(delta_arr)), 6)

    pred_shots = _inverse(pred_norm)
    target_shots = _inverse(shots_norm)
    input_shots = _inverse(noisy_norm)

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        exp = cfg.get("experiment", {})
        out_dir = Path(exp.get("output_dir", "results")) / exp.get("name", "exp") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_shots = pred_norm.shape[0]
    csv_path = out_dir / "metrics_per_shot.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = ["shot_idx"]
        header += [f"noisy_{k}" for k in noisy_per_shot.keys()]
        header += [f"denoised_{k}" for k in denoised_per_shot.keys()]
        header += [f"delta_{k}" for k in delta_per_shot.keys()]
        f.write(",".join(header) + "\n")
        for i in range(n_shots):
            row = [str(i)]
            row += [
                format_metric_value(k, float(noisy_per_shot[k][i]))
                for k in noisy_per_shot.keys()
            ]
            row += [
                format_metric_value(k, float(denoised_per_shot[k][i]))
                for k in denoised_per_shot.keys()
            ]
            row += [
                format_metric_value(k, float(delta_per_shot[k][i]))
                for k in delta_per_shot.keys()
            ]
            f.write(",".join(row) + "\n")

    summary = {
        "noisy": dict(noisy_mean),
        "denoised": dict(denoised_mean),
        "delta": dict(delta_mean),
    }
    summary["noise_kind"] = noise_kind
    summary["snr_db"] = snr_db
    summary["input_source"] = "volume_with_synthetic_noise"
    summary["inference_time_seconds"] = round(infer_elapsed, 3)
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if save_npy:
        npy_dir = out_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / "pred_shots.npy", pred_shots)
        np.save(npy_dir / "target_shots.npy", target_shots)
        np.save(npy_dir / "input_shots.npy", input_shots)
        print(f"Saved .npy files to {npy_dir}")

    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    indices = select_random_shots(n_shots, n_viz_shots, seed=seed)
    vmax = float(np.quantile(np.abs(np.concatenate([
        input_shots.ravel(), pred_shots.ravel(), target_shots.ravel()
    ])), 0.995))

    save_shot_visualizations(
        input_shots=input_shots,
        pred_shots=pred_shots,
        target_shots=target_shots,
        indices=indices,
        save_dir=viz_dir,
        title_prefix=f"denoise_{model_type}_{noise_kind}_snr{snr_db:g}",
        vmin=-vmax,
        vmax=vmax,
        save_npy=args.save_viz_npy,
    )

    print(f"Inference complete. Outputs saved to: {out_dir}")
    print("Input source: volume_with_synthetic_noise")
    print(f"Noise setting: kind={noise_kind}, snr_db={snr_db}")
    print(f"Visualized shots: {list(indices)}")
    print("Mean metrics (normalized domain):")
    print("  Noisy vs target:")
    for k, v in noisy_mean.items():
        print(f"    {k}: {format_metric_value(k, v)}")
    print("  Denoised vs target:")
    for k, v in denoised_mean.items():
        print(f"    {k}: {format_metric_value(k, v)}")
    print("  Delta (denoised - noisy):")
    for k, v in delta_mean.items():
        print(f"    {k}: {format_metric_value(k, v)}")


if __name__ == "__main__":
    main()
