"""Conditional DDPM (cDDPM-2c) for ground-roll attenuation (DDP via ``torchrun``).

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \\
    scripts/ground_roll_attenuation/train/train_denoise_ddpm.py \\
    --config configs/ground_roll_attenuation/denoise_ddpm.yaml
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from model.ground_roll_attenuation.ddpm import DDPMNoiseScheduler  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.patching import patchify_uniform  # noqa: E402
from tools.preprocessing import normalize  # noqa: E402
from utils import (  # noqa: E402
    TrainingLogger,
    apply_denoise_experiment_name_from_model,
    build_loaders,
    build_metrics,
    build_shot_split_loaders,
    default_config_relpath_for_train_script,
    destroy_distributed,
    load_config,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir_distributed,
    training_device,
    unwrap_ddp,
    visualize_random_sample,
    plot_sample,
)


# ---------------------------------------------------------------------------
# DDPM-local distributed init with a long timeout (DDIM evaluation is slow)
# ---------------------------------------------------------------------------

def _init_distributed_ddpm(backend=None, timeout=7200):
    """Like ``utils.init_distributed`` but with a longer process-group timeout."""
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    td = datetime.timedelta(seconds=int(timeout))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", device_id=device, timeout=td,
        )
    else:
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", timeout=td,
        )
    return True, rank, local_rank, world_size


# ---------------------------------------------------------------------------
# data pipeline
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
    target_shots = load_volume(target_cfg)

    if input_shots.shape != target_shots.shape:
        raise ValueError(
            f"Paired volume shape mismatch: "
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

        input_shots, in_stats = normalize(
            input_shots, mode=mode, per=per, clip_percentile=clip_p
        )
        target_shots, _ = normalize(
            target_shots, mode=mode, per=per, override_stats=in_stats,
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
# DDPM evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _evaluate_ddpm(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    scheduler: DDPMNoiseScheduler,
    metrics: Dict[str, Any],
    device: torch.device,
    sample_steps: int = 10,
    use_ddim: bool = True,
    ddim_eta: float = 0.0,
    distributed: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate DDPM on a dataloader (few-step DDIM sampling for monitoring).

    When ``distributed=True``, each rank processes its own data shard and
    results are all-reduced so every rank sees the global averages.
    """
    model.eval()
    metric_sums: Dict[str, float] = {}
    mse_sum = 0.0
    n_total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = x.shape[0]

        # x = noisy input (condition), y = noise label (ground-roll z_0)
        # x_0 = clean signal = noisy - noise
        x_0_true = x - y

        # DDIM (or DDPM) sampling
        x_0_pred, _ = scheduler.sample_full(
            model, x, num_steps=sample_steps, use_ddim=use_ddim, ddim_eta=ddim_eta, progress=False
        )

        # Compute MSE on clean signal
        mse_batch = F.mse_loss(x_0_pred, x_0_true)
        mse_sum += mse_batch.item() * B
        n_total += B

        # compute metrics on denoised signal vs clean reference
        pred_m = x_0_pred
        targ_m = x_0_true
        from utils.metrics import compute_metrics
        batch_metrics = compute_metrics(metrics, pred_m, targ_m)
        for k, v in batch_metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + v * B

    # all-reduce partial sums across ranks
    if distributed:
        metric_keys = sorted(metric_sums.keys())
        stat = torch.tensor(
            [mse_sum, float(n_total)] + [metric_sums[k] for k in metric_keys],
            device=device, dtype=torch.float64,
        )
        torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
        mse_sum = float(stat[0].item())
        n_total = int(stat[1].item())
        for i, k in enumerate(metric_keys):
            metric_sums[k] = float(stat[2 + i].item())

    val_loss = mse_sum / max(n_total, 1)
    out_metrics = {k: v / max(n_total, 1) for k, v in metric_sums.items()}
    return {"val": val_loss}, out_metrics


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _visualize_ddpm_sample(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    scheduler: DDPMNoiseScheduler,
    save_path: Path,
    device: torch.device,
    sample_steps: int = 10,
    use_ddim: bool = True,
    ddim_eta: float = 0.0,
    title: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """Pick a random sample from ``loader``, run DDIM sampling, save a 4-panel plot.

    Panels: noisy input | denoised prediction | clean reference | residual.
    """
    dataset = loader.dataset
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(dataset)))
    sample = dataset[idx]
    x, y = sample
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    x_0_true = x - y
    x_0_pred, _ = scheduler.sample_full(
        model, x, num_steps=sample_steps, use_ddim=use_ddim, ddim_eta=ddim_eta, progress=False
    )

    suffix = f"sample idx={idx}"
    full_title = f"{title} | {suffix}" if title else suffix
    plot_sample(
        input_data=x,
        prediction=x_0_pred,
        target=x_0_true,
        save_path=save_path,
        title=full_title,
        cmap="gray",
    )


# ---------------------------------------------------------------------------
# DDPM-specific plotting (train=line, val=scatter; dual-axis loss)
# ---------------------------------------------------------------------------

def _plot_ddpm_loss_curve(
    loss_history: Dict[str, list],
    save_path: Path,
) -> None:
    """Loss curve: train_l1 on left y-axis (line), val on right y-axis (scatter)."""
    import matplotlib.pyplot as plt

    train_vals = loss_history.get("train_l1", [])
    val_vals = loss_history.get("val", [])

    fig, ax1 = plt.subplots(figsize=(7, 4))

    if train_vals:
        train_arr = np.asarray(train_vals, dtype=float)
        train_valid = np.isfinite(train_arr)
        if train_valid.any():
            ax1.plot(np.where(train_valid)[0], train_arr[train_valid], "b-", label="train_l1", linewidth=1.5)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train_l1", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if val_vals:
        val_arr = np.asarray(val_vals, dtype=float)
        valid = np.isfinite(val_arr)
        if valid.any():
            ax2.scatter(
                np.where(valid)[0], val_arr[valid],
                c="r", marker="x", s=30, label="val (MSE)",
            )
    ax2.set_ylabel("val MSE", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle("Loss")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_ddpm_metric_curves(
    metric_history: Dict[str, list],
    save_dir: Path,
) -> None:
    """Metric curves: train_xxx as continuous line, val_xxx as scatter points."""
    import matplotlib.pyplot as plt

    groups: Dict[str, Dict[str, list]] = {}
    for key in metric_history:
        for prefix in ("train_", "val_"):
            if key.startswith(prefix):
                base = key[len(prefix):]
                groups.setdefault(base, {})[key] = metric_history[key]
                break

    save_dir.mkdir(parents=True, exist_ok=True)

    for base, sub_history in groups.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for name, values in sub_history.items():
            arr = np.asarray(values, dtype=float)
            valid = np.isfinite(arr)
            if not valid.any():
                continue
            if name.startswith("train_"):
                ax.plot(np.where(valid)[0], arr[valid], "-", label=name, linewidth=1.5)
            else:
                ax.scatter(
                    np.where(valid)[0], arr[valid],
                    marker="x", s=30, label=name,
                )
            plotted = True

        if plotted:
            ax.set_xlabel("epoch")
            ax.set_ylabel(base)
            ax.set_title(base.upper())
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(save_dir / f"{base}_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------

def _save_ddpm_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_obj: Any,
    ddpm_scheduler: DDPMNoiseScheduler,
    epoch: int,
    extras: Dict[str, Any],
) -> None:
    ckpt = {
        "model": unwrap_ddp(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_obj.state_dict() if scheduler_obj is not None else None,
        "ddpm_scheduler": ddpm_scheduler.state_dict(),
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
        description="Train conditional DDPM (DDPM-2c) for ground-roll attenuation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ground_roll_attenuation/denoise_ddpm.yaml",
        help="Path to DDPM config YAML.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_denoise_experiment_name_from_model(cfg)

    distributed, rank, local_rank, world_size = _init_distributed_ddpm()

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

    # --- patch val_loader & eval_train_loader for DDP eval -------------------
    if distributed:
        loader_cfg = cfg["data"].get("loader", {})
        _bs = int(loader_cfg.get("batch_size", 8))
        _nw = int(loader_cfg.get("num_workers", 0))
        _pm = bool(loader_cfg.get("pin_memory", True))
        _seed = int(cfg["experiment"]["seed"])

        val_ds = val_loader.dataset
        val_sampler = torch.utils.data.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=_seed,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=_bs, sampler=val_sampler, num_workers=_nw,
            pin_memory=_pm, drop_last=False,
        )

        train_ds = train_loader.dataset
        eval_train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=_seed,
        )
        eval_train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=_bs, sampler=eval_train_sampler, num_workers=_nw,
            pin_memory=_pm, drop_last=False,
        )

    # --- model & scheduler -------------------------------------------------
    model = build_model(cfg["model"]).to(device)
    from utils.train_utils import maybe_wrap_ddp as _wrap
    model = _wrap(model, distributed=distributed, device=device, local_rank=local_rank)
    _eval_model = unwrap_ddp(model)

    diff_cfg = cfg["diffusion"]
    ddpm_scheduler = DDPMNoiseScheduler(
        num_timesteps=int(diff_cfg.get("num_timesteps", 1000)),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 0.02)),
    )
    ddpm_scheduler.to(device)

    g_type = str(cfg["model"]["type"])

    # --- metrics -----------------------------------------------------------
    metrics_spec = cfg.get("metrics", [])
    metrics = build_metrics(metrics_spec)
    metric_names = list(metrics.keys())

    # --- optimizer & scheduler ---------------------------------------------
    from utils.train_utils import build_optimizer, build_scheduler
    optimizer = build_optimizer(model, cfg["optim"])
    lr_scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), int(cfg["train"]["epochs"]))

    # --- logger ------------------------------------------------------------
    loss_keys = ["train_l1", "val"]
    logger: Optional[TrainingLogger] = None
    if rank == 0:
        is_resume = bool(cfg["train"].get("resume"))
        logger = TrainingLogger(
            log_dir=exp_dir / cfg["log"].get("log_dir", "logs"),
            loss_keys=loss_keys,
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
            clear_existing=not is_resume,
        )
    if logger is not None:
        logger.info(
            f"DDPM {g_type} | train/val patches: "
            f"{len(train_loader.dataset)} / {len(val_loader.dataset)}"
        )

    # --- training settings -------------------------------------------------
    total_epochs = int(cfg["train"]["epochs"])
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 20))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    grad_clip = cfg["train"].get("grad_clip")
    eval_sample_steps = int(cfg["train"].get("eval_sample_steps", 10))
    eval_use_ddim = bool(cfg["train"].get("eval_use_ddim", True))
    eval_ddim_eta = float(cfg["train"].get("eval_ddim_eta", 0.0))

    best_val_loss = float("inf")
    T = ddpm_scheduler.num_timesteps
    start_time = time.time()

    for epoch in range(total_epochs):
        sampler_set_epoch(train_sampler, epoch)
        model.train()

        l1_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)         # noisy input (condition y)
            y = y.to(device, non_blocking=True)         # noise label (ground-roll z_0)
            B = x.shape[0]

            # derive clean signal and ground-roll
            x_0 = x - y   # clean signal
            z_0 = y       # ground-roll noise

            # sample timesteps and noise
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            eps_sig = torch.randn_like(x_0)
            eps_gr = torch.randn_like(z_0)

            # forward diffusion
            x_t, z_t = ddpm_scheduler.add_noise(x_0, z_0, t, eps_sig, eps_gr)

            # model forward: concat(y, x_t, z_t) → predict noise
            inp = torch.cat([x, x_t, z_t], dim=1)       # (B, 3, H, W)
            eps_pred = model(inp, t)                     # (B, 2, H, W)

            eps_sig_pred = eps_pred[:, :1, :, :]
            eps_gr_pred = eps_pred[:, 1:, :, :]

            loss = F.l1_loss(eps_sig_pred, eps_sig) + F.l1_loss(eps_gr_pred, eps_gr)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            l1_sum += loss.item()
            n_batches += 1

        l1_avg: float
        if distributed:
            stat = torch.tensor([float(l1_sum), float(n_batches)], device=device, dtype=torch.float64)
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            l1_avg = float(stat[0].item() / max(stat[1].item(), 1.0))
        else:
            l1_avg = l1_sum / max(n_batches, 1)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # --- evaluation (all ranks, every eval_interval) ------------------
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}
        val_losses: Dict[str, float] = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        do_eval = (epoch == 0 or (epoch + 1) % eval_interval == 0)
        if do_eval:
            if eval_train_loader is not None:
                _, train_metrics = _evaluate_ddpm(
                    model=_eval_model,
                    loader=eval_train_loader,
                    scheduler=ddpm_scheduler,
                    metrics=metrics,
                    device=device,
                    sample_steps=eval_sample_steps,
                    use_ddim=eval_use_ddim,
                    ddim_eta=eval_ddim_eta,
                    distributed=distributed,
                )
            val_losses, val_metrics = _evaluate_ddpm(
                model=_eval_model,
                loader=val_loader,
                scheduler=ddpm_scheduler,
                metrics=metrics,
                device=device,
                sample_steps=eval_sample_steps,
                use_ddim=eval_use_ddim,
                ddim_eta=eval_ddim_eta,
                distributed=distributed,
            )
            if rank == 0 and val_losses["val"] < best_val_loss:
                best_val_loss = val_losses["val"]
                _save_ddpm_checkpoint(
                    exp_dir / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler_obj=lr_scheduler,
                    ddpm_scheduler=ddpm_scheduler,
                    epoch=epoch,
                    extras={"config": cfg},
                )
                if logger is not None:
                    logger.info(f"Best checkpoint saved (val_mse={best_val_loss:.6f})")

        # Barrier: ensure rank 0 eval is done before all ranks enter next epoch
        if distributed:
            torch.distributed.barrier()

        # --- logging --------------------------------------------------------
        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                losses={
                    "train_l1": l1_avg,
                    "val": val_losses["val"],
                },
                metrics=metric_row,
                extras={"lr": optimizer.param_groups[0]["lr"]},
            )

        # --- periodic checkpointing -----------------------------------------
        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            _save_ddpm_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler_obj=lr_scheduler,
                ddpm_scheduler=ddpm_scheduler,
                epoch=epoch,
                extras={"config": cfg},
            )

        # --- visualization (rank 0) ----------------------------------------
        if rank == 0 and vis_interval > 0 and (epoch == 0 or (epoch + 1) % vis_interval == 0):
            _visualize_ddpm_sample(
                model=_eval_model,
                loader=val_loader,
                scheduler=ddpm_scheduler,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device,
                sample_steps=eval_sample_steps,
                use_ddim=eval_use_ddim,
                ddim_eta=eval_ddim_eta,
                title=f"DDPM {g_type} epoch {epoch}",
                seed=None,
            )

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(
            f"DDPM {g_type} training finished in {elapsed:.2f}s ({elapsed/60:.2f} min)."
        )
        # Overwrite auto-generated curves with DDPM-specific versions
        _plot_ddpm_loss_curve(
            logger._loss_history,
            logger._loss_curve_path,
        )
        _plot_ddpm_metric_curves(
            logger._metric_history,
            logger._metric_curve_dir,
        )
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
