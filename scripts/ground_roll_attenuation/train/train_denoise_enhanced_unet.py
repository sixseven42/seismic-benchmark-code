"""Enhanced Attention U-Net with hybrid MSE+AFM loss for ground-roll attenuation.

The model predicts the **additive noise component** (aligned with other models
in this benchmark).  Denoised = input - predicted_noise.
Loss = MSE + λ * AFM (adaptive frequency modulation in f-x domain via torch.fft).

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \\
    scripts/ground_roll_attenuation/train/train_denoise_enhanced_unet.py \\
    --config configs/ground_roll_attenuation/denoise_enhanced_unet.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

from model.ground_roll_attenuation import build_model  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.patching import patchify_uniform  # noqa: E402
from tools.preprocessing import normalize  # noqa: E402
from utils import (  # noqa: E402
    TrainingLogger,
    apply_denoise_experiment_name_from_model,
    build_loss,
    build_loaders,
    build_metrics,
    build_optimizer,
    build_scheduler,
    build_shot_split_loaders,
    default_config_relpath_for_train_script,
    destroy_distributed,
    evaluate,
    init_distributed,
    load_config,
    maybe_wrap_ddp,
    resolve_denoise_metrics,
    maybe_save_best_checkpoint,
    save_checkpoint,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir_distributed,
    train_one_epoch,
    training_device,
    visualize_random_sample,
)


# ---------------------------------------------------------------------------
# data pipeline — returns noise-label target (aligned with other models)
# ---------------------------------------------------------------------------

def _preprocess_shots(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prep = cfg["preprocess"]

    pair_cfg = None
    for key in ("segy_pair", "npy_pair", "mat_pair"):
        if key in cfg["data"]:
            pair_cfg = cfg["data"][key]
            break
    if pair_cfg is None:
        raise ValueError("No paired data source found in config.")

    input_cfg = dict(pair_cfg)
    input_cfg["path"] = pair_cfg["input_path"]
    target_cfg = dict(pair_cfg)
    target_cfg["path"] = pair_cfg["target_path"]

    input_shots = load_volume(input_cfg)
    noise_shots = load_volume(target_cfg)

    if input_shots.shape != noise_shots.shape:
        raise ValueError(
            f"Paired volume shape mismatch: "
            f"input {input_shots.shape} vs target {noise_shots.shape}."
        )
    if prep.get("max_shots") is not None:
        m = int(prep["max_shots"])
        input_shots = input_shots[:m]
        noise_shots = noise_shots[:m]

    skip = set(prep.get("skip", []))

    if "normalize" not in skip:
        mode = str(prep.get("normalize_mode", "max_abs"))
        per = str(prep.get("normalize_scope", "global"))
        clip_raw = prep.get("clip_percentile")
        clip_p = float(clip_raw) if clip_raw is not None else None

        input_shots, in_stats = normalize(
            input_shots, mode=mode, per=per, clip_percentile=clip_p
        )
        noise_shots, _ = normalize(
            noise_shots, mode=mode, per=per, override_stats=in_stats,
        )

    if input_cfg.get("path", "").lower().endswith((".sgy", ".segy")):
        from tools.segy_read import read_regular_shots
        _, headers = read_regular_shots(
            input_cfg["path"],
            traces_per_shot=int(input_cfg.get("traces_per_shot", 201)),
            time_downsample=int(input_cfg.get("time_downsample", 1)),
            return_headers=True,
        )
        per_shot_ffid = headers["FieldRecord"][:, 0]
    else:
        per_shot_ffid = np.arange(input_shots.shape[0])

    return input_shots, noise_shots, per_shot_ffid


def _patchify_pairs(
    input_shots: np.ndarray, target_shots: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    prep = cfg["preprocess"]
    patch_t = int(prep.get("patch_time", 256))
    patch_x = int(prep.get("patch_trace", 128))
    overlap = float(prep.get("patch_overlap", 0.5))

    target_patches, _ = patchify_uniform(
        target_shots, patch_size=(patch_x, patch_t), overlap=overlap, output_ndim=4
    )
    input_patches, _ = patchify_uniform(
        input_shots, patch_size=(patch_x, patch_t), overlap=overlap, output_ndim=4
    )
    return input_patches.astype(np.float32), target_patches.astype(np.float32)


def _build_denoise_patch_pairs(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    inp, tgt, _ = _preprocess_shots(cfg)
    return _patchify_pairs(inp, tgt, cfg)


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Enhanced Attention U-Net with AFM loss."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ground_roll_attenuation/denoise_enhanced_unet.yaml",
        help="Path to config YAML.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main (follows train_denoise_unet.py exactly except: loss + eval mode)
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_denoise_experiment_name_from_model(cfg)
    cfg["metrics"] = resolve_denoise_metrics(cfg)

    distributed, rank, local_rank, world_size = init_distributed()

    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=_REPO_ROOT)
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    if "shot_split" in cfg.get("data", {}):
        train_loader, val_loader, train_sampler, eval_train_loader = build_shot_split_loaders(
            cfg,
            preprocess_fn=_preprocess_shots,
            patchify_fn=_patchify_pairs,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            test_set_dir=exp_dir / "test_set",
        )
    else:
        train_loader, val_loader, train_sampler, eval_train_loader = build_loaders(
            cfg,
            build_patch_pairs_fn=_build_denoise_patch_pairs,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )

    model = build_model(cfg["model"]).to(device)
    model = maybe_wrap_ddp(model, distributed=distributed, device=device, local_rank=local_rank)
    model_type = str(cfg["model"]["type"])
    loss_fn = build_loss(cfg["loss"]).to(device)
    metrics = build_metrics(cfg["metrics"])
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], int(cfg["train"]["epochs"]))

    metric_names = list(metrics.keys())
    logger: Optional[TrainingLogger] = None
    if rank == 0:
        logger = TrainingLogger(
            log_dir=exp_dir / cfg["log"].get("log_dir", "logs"),
            loss_keys=["train", "val"],
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )
    if logger is not None:
        logger.info(
            f"Model {model_type} | train/val patches: "
            f"{len(train_loader.dataset)} / {len(val_loader.dataset)}"
        )

    total_epochs = int(cfg["train"]["epochs"])
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 5))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    log_step = bool(cfg["train"].get("log_step", False))

    # Model predicts noise residual → denoised = input - pred (aligned with other models)
    best_val_loss = float("inf")
    start_time = time.time()
    for epoch in range(total_epochs):
        sampler_set_epoch(train_sampler, epoch)
        train_stats = train_one_epoch(
            model=model, loader=train_loader, loss_fn=loss_fn,
            optimizer=optimizer, device=device, epoch=epoch,
            scheduler=scheduler,
            grad_clip=cfg["train"].get("grad_clip"),
            log_interval=int(cfg["train"].get("log_interval", 20)),
            logger=logger if log_step else None,
        )
        val_losses = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}

        if rank == 0 and eval_train_loader is not None:
            _, train_metrics = evaluate(
                model=model, loader=eval_train_loader, loss_fn=loss_fn,
                metrics=metrics, device=device,
                metrics_on_denoised_signal=True,
            )
            if (epoch + 1) % eval_interval == 0:
                val_losses, val_metrics = evaluate(
                    model=model, loader=val_loader, loss_fn=loss_fn,
                    metrics=metrics, device=device,
                    metrics_on_denoised_signal=True,
                )
                best_val_loss = maybe_save_best_checkpoint(
                    exp_dir / "checkpoints" / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, val_loss=val_losses["val"],
                    best_val_loss=best_val_loss,
                    extras={"config": cfg}, logger=logger,
                )

        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                losses={"train": train_stats["train"], "val": val_losses.get("val", float("nan"))},
                metrics=metric_row,
                extras={"lr": optimizer.param_groups[0]["lr"]},
            )

        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, extras={"config": cfg},
            )

        if rank == 0 and (epoch + 1) % vis_interval == 0:
            visualize_random_sample(
                model=model, loader=val_loader,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device, title=f"Enhanced-AttenUNet {model_type} epoch {epoch}",
                seed=None,
            )

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(f"Enhanced-AttenUNet training finished in {elapsed:.2f}s ({elapsed/60:.2f} min).")
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
