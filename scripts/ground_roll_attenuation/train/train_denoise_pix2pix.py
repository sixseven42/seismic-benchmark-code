"""Pix2Pix cGAN for ground-roll attenuation (paired volumes + DDP via ``torchrun``).

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \\
    scripts/ground_roll_attenuation/train/train_denoise_pix2pix.py \\
    --config configs/ground_roll_attenuation/denoise_pix2pix.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Bootstrap repo root into sys.path BEFORE importing utils/model.
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
from model.ground_roll_attenuation.pix2pix import Pix2PixDiscriminator  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.patching import patchify_uniform  # noqa: E402
from tools.preprocessing import normalize  # noqa: E402
from utils import (  # noqa: E402
    TrainingLogger,
    apply_denoise_experiment_name_from_model,
    barrier_if_distributed,
    build_loaders,
    build_metrics,
    build_shot_split_loaders,
    default_config_relpath_for_train_script,
    destroy_distributed,
    evaluate,
    init_distributed,
    load_config,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir_distributed,
    training_device,
    unwrap_ddp,
    visualize_random_sample,
)


# ---------------------------------------------------------------------------
# data pipeline (identical to train_denoise_unet.py)
# ---------------------------------------------------------------------------

def _preprocess_shots(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load paired denoise volumes, preprocess, and return ``(input_shots, target_shots, per_shot_ffid)``."""
    prep = cfg["preprocess"]

    pair_cfg = None
    for key in ("segy_pair", "npy_pair", "mat_pair"):
        if key in cfg["data"]:
            pair_cfg = cfg["data"][key]
            break
    if pair_cfg is None:
        raise ValueError(
            "No paired data source found in config (expected data.segy_pair, "
            "data.npy_pair, or data.mat_pair)."
        )

    input_cfg = dict(pair_cfg)
    input_cfg["path"] = pair_cfg["input_path"]
    target_cfg = dict(pair_cfg)
    target_cfg["path"] = pair_cfg["target_path"]

    input_shots = load_volume(input_cfg)
    target_shots = load_volume(target_cfg)

    if input_shots.shape != target_shots.shape:
        raise ValueError(
            "Paired volume shape mismatch: "
            f"input {input_shots.shape} vs target {target_shots.shape}."
        )
    if prep.get("max_shots") is not None:
        m = int(prep["max_shots"])
        input_shots = input_shots[:m]
        target_shots = target_shots[:m]

    skip = set(prep.get("skip", []))

    if "normalize" not in skip:
        mode = str(prep.get("normalize_mode", "max_abs"))
        per = str(prep.get("normalize_scope", "global"))
        clip_raw = prep.get("clip_percentile")
        clip_p = float(clip_raw) if clip_raw is not None else None

        mode_keys = {
            "minmax": ("min", "max"),
            "max_abs": ("max_abs",),
            "mean_std": ("mean", "std"),
        }
        if mode not in mode_keys:
            raise ValueError(
                f"Unknown normalize_mode {mode!r} for paired denoise pipeline."
            )

        input_shots, in_stats = normalize(
            input_shots, mode=mode, per=per, clip_percentile=clip_p
        )
        target_shots, _ = normalize(
            target_shots,
            mode=mode,
            per=per,
            override_stats=in_stats,
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

    return input_shots, target_shots, per_shot_ffid


def _patchify_pairs(
    input_shots: np.ndarray, target_shots: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Patchify given shot subsets."""
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
# GAN helpers
# ---------------------------------------------------------------------------

def _save_pix2pix_checkpoint(
    path: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optim: torch.optim.Optimizer,
    d_optim: torch.optim.Optimizer,
    g_scheduler: Any,
    d_scheduler: Any,
    epoch: int,
    extras: Dict[str, Any],
) -> None:
    """Save G+D states + optimizers + schedulers."""
    ckpt = {
        "model": unwrap_ddp(generator).state_dict(),
        "discriminator": unwrap_ddp(discriminator).state_dict(),
        "optimizer": g_optim.state_dict(),
        "d_optimizer": d_optim.state_dict(),
        "scheduler": g_scheduler.state_dict() if g_scheduler is not None else None,
        "d_scheduler": d_scheduler.state_dict() if d_scheduler is not None else None,
        "epoch": int(epoch),
        "extras": extras or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pix2Pix cGAN for ground-roll attenuation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ground_roll_attenuation/denoise_pix2pix.yaml",
        help="Path to pix2pix config YAML.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_denoise_experiment_name_from_model(cfg)

    distributed, rank, local_rank, world_size = init_distributed()

    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=_REPO_ROOT)
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    # --- data loaders -------------------------------------------------------
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

    # --- models ------------------------------------------------------------
    generator = build_model(cfg["model"]).to(device)
    disc_cfg = cfg["discriminator"]
    discriminator = Pix2PixDiscriminator(
        in_channels=int(disc_cfg.get("in_channels", 1)),
    ).to(device)

    g_type = str(cfg["model"]["type"])

    # wrap with DDP
    from utils.train_utils import maybe_wrap_ddp as _wrap
    generator = _wrap(generator, distributed=distributed, device=device, local_rank=local_rank)
    discriminator = _wrap(discriminator, distributed=distributed, device=device, local_rank=local_rank)

    # --- metrics (evaluation uses G only) ---------------------------------
    metrics_spec = cfg.get("metrics", [])
    metrics = build_metrics(metrics_spec)
    metric_names = list(metrics.keys())

    # --- losses -----------------------------------------------------------
    gan_cfg = cfg["gan"]
    lambda_l1 = float(gan_cfg.get("lambda_l1", 100.0))
    bce_loss = nn.BCEWithLogitsLoss()

    # --- optimizers & schedulers -------------------------------------------
    g_lr = float(cfg["optim"]["params"].get("lr", 0.0002))
    g_betas = tuple(cfg["optim"]["params"].get("betas", [0.5, 0.999]))
    g_optim = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=g_betas)

    d_lr = float(cfg["d_optim"]["params"].get("lr", 0.0002))
    d_betas = tuple(cfg["d_optim"]["params"].get("betas", [0.5, 0.999]))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=d_betas)

    total_epochs = int(cfg["train"]["epochs"])
    from utils.train_utils import build_scheduler
    g_scheduler = build_scheduler(g_optim, cfg.get("scheduler", {}), total_epochs)
    d_scheduler = build_scheduler(d_optim, cfg.get("d_scheduler", cfg.get("scheduler", {})), total_epochs)

    # --- logger ------------------------------------------------------------
    loss_keys = ["d_loss", "g_gan", "g_l1", "g_total", "val"]
    logger: Optional[TrainingLogger] = None
    if rank == 0:
        logger = TrainingLogger(
            log_dir=exp_dir / cfg["log"].get("log_dir", "logs"),
            loss_keys=loss_keys,
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )
    if logger is not None:
        logger.info(
            f"Pix2Pix Generator {g_type} | "
            f"train/val patches: {len(train_loader.dataset)} / {len(val_loader.dataset)}"
        )

    # --- training settings -------------------------------------------------
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 20))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    grad_clip = cfg["train"].get("grad_clip")

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(total_epochs):
        sampler_set_epoch(train_sampler, epoch)
        generator.train()
        discriminator.train()

        d_loss_sum = 0.0
        g_gan_sum = 0.0
        g_l1_sum = 0.0
        g_total_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---- Discriminator step -----------------------------------------
            # Two forwards back-propagated via separate backward() calls
            # inside no_sync() to avoid the BN running-stat version conflict
            # that occurs when a single backward traverses both computation
            # graphs.  Gradients accumulate locally; a manual all-reduce
            # (AVG) replicates the DDP sync that was suppressed.
            d_optim.zero_grad(set_to_none=True)

            with discriminator.no_sync():
                # real
                real_out = discriminator(x, y)
                d_real_loss = bce_loss(real_out, torch.ones_like(real_out)) * 0.5
                d_real_loss.backward()

                # fake
                with torch.no_grad():
                    fake_y = generator(x)
                fake_out = discriminator(x, fake_y.detach())
                d_fake_loss = bce_loss(fake_out, torch.zeros_like(fake_out)) * 0.5
                d_fake_loss.backward()

            if distributed:
                for p in discriminator.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(discriminator.parameters(), float(grad_clip))
            d_optim.step()

            # ---- Generator step ---------------------------------------------
            g_optim.zero_grad(set_to_none=True)

            fake_y = generator(x)
            fake_out = discriminator(x, fake_y)
            g_gan_loss = bce_loss(fake_out, torch.ones_like(fake_out))
            g_l1_loss = F.l1_loss(fake_y, y)
            g_loss = g_gan_loss + lambda_l1 * g_l1_loss

            g_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(generator.parameters(), float(grad_clip))
            g_optim.step()

            # accumulate
            d_loss_sum += (d_real_loss.item() + d_fake_loss.item())
            g_gan_sum += g_gan_loss.item()
            g_l1_sum += g_l1_loss.item()
            g_total_sum += g_loss.item()
            n_batches += 1

        # end of epoch: average losses
        d_loss_avg = d_loss_sum / max(n_batches, 1)
        g_gan_avg = g_gan_sum / max(n_batches, 1)
        g_l1_avg = g_l1_sum / max(n_batches, 1)
        g_total_avg = g_total_sum / max(n_batches, 1)

        g_scheduler.step()
        d_scheduler.step()

        # --- evaluation (G only, all ranks participate, DDP aggregation) ----
        val_losses: Dict[str, float] = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}

        # Training metrics: rank 0 only (acceptable for monitoring; barrier
        # below keeps other ranks from drifting ahead).
        if rank == 0 and eval_train_loader is not None:
            _, train_metrics = evaluate(
                model=generator,
                loader=eval_train_loader,
                loss_fn=nn.MSELoss().to(device),
                metrics=metrics,
                device=device,
                metrics_on_denoised_signal=True,
            )

        # Validation: every rank evaluates on its split of val_loader; results
        # are aggregated via all_reduce so no single rank becomes a bottleneck.
        if (epoch + 1) % eval_interval == 0:
            val_losses, val_metrics = evaluate(
                model=generator,
                loader=val_loader,
                loss_fn=nn.MSELoss().to(device),
                metrics=metrics,
                device=device,
                metrics_on_denoised_signal=True,
                distributed=distributed,
            )

            # best checkpoint by validation MSE (rank 0 only)
            if rank == 0 and val_losses["val"] < best_val_loss:
                best_val_loss = val_losses["val"]
                _save_pix2pix_checkpoint(
                    exp_dir / "checkpoints" / "best.pt",
                    generator=generator,
                    discriminator=discriminator,
                    g_optim=g_optim,
                    d_optim=d_optim,
                    g_scheduler=g_scheduler,
                    d_scheduler=d_scheduler,
                    epoch=epoch,
                    extras={"config": cfg},
                )
                if logger is not None:
                    logger.info(
                        f"Best checkpoint saved (val_loss={best_val_loss:.6f})"
                    )

        # --- logging --------------------------------------------------------
        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                losses={
                    "d_loss": d_loss_avg,
                    "g_gan": g_gan_avg,
                    "g_l1": g_l1_avg,
                    "g_total": g_total_avg,
                    "val": val_losses["val"],
                },
                metrics=metric_row,
                extras={"lr": g_optim.param_groups[0]["lr"]},
            )

        # --- periodic checkpointing (rank 0) --------------------------------
        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            _save_pix2pix_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                generator=generator,
                discriminator=discriminator,
                g_optim=g_optim,
                d_optim=d_optim,
                g_scheduler=g_scheduler,
                d_scheduler=d_scheduler,
                epoch=epoch,
                extras={"config": cfg},
            )

        # --- visualization (rank 0) -----------------------------------------
        if rank == 0 and (epoch + 1) % vis_interval == 0:
            visualize_random_sample(
                model=generator,
                loader=val_loader,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device,
                title=f"Pix2Pix {g_type} epoch {epoch}",
                seed=None,
            )

        # Ensure all ranks finish the epoch before the next one begins,
        # preventing rank skew that can cause NCCL collective timeouts.
        barrier_if_distributed()

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(
            f"Pix2Pix {g_type} training finished in {elapsed:.2f}s ({elapsed/60:.2f} min)."
        )
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
