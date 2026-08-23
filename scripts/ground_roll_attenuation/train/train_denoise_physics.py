"""Physics-constrained deep learning for ground-roll attenuation (DDP via ``torchrun``).

Two phases:
  1. Pre-train f-k classifier (--pretrain_classifier)
  2. Train separation network CNN1+CNN2+CNN3 with full joint loss

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \\
    scripts/ground_roll_attenuation/train/train_denoise_physics.py \\
    --config configs/ground_roll_attenuation/denoise_physics.yaml
"""

from __future__ import annotations

import argparse
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
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.ground_roll_attenuation import build_model  # noqa: E402
from model.ground_roll_attenuation.physics_unet import FKClassifier  # noqa: E402
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
    evaluate,
    init_distributed,
    load_config,
    maybe_wrap_ddp,
    resolve_denoise_metrics,
    save_checkpoint,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir_distributed,
    training_device,
    unwrap_ddp,
    visualize_random_sample,
)

# ---------------------------------------------------------------------------
# data pipeline — provides (noisy, clean_signal, ground_roll_noise)
# ---------------------------------------------------------------------------

def _preprocess_shots(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (noisy, clean_signal, ground_roll, per_shot_ffid)."""
    prep = cfg["preprocess"]
    pair_cfg = None
    for key in ("segy_pair", "npy_pair", "mat_pair"):
        if key in cfg["data"]:
            pair_cfg = cfg["data"][key]
            break
    if pair_cfg is None:
        raise ValueError("No paired data source found.")

    input_cfg = dict(pair_cfg)
    input_cfg["path"] = pair_cfg["input_path"]
    target_cfg = dict(pair_cfg)
    target_cfg["path"] = pair_cfg["target_path"]

    noisy = load_volume(input_cfg)
    gr_noise = load_volume(target_cfg)

    if noisy.shape != gr_noise.shape:
        raise ValueError(f"Shape mismatch: noisy {noisy.shape} vs noise {gr_noise.shape}.")
    if prep.get("max_shots") is not None:
        m = int(prep["max_shots"])
        noisy, gr_noise = noisy[:m], gr_noise[:m]

    skip = set(prep.get("skip", []))
    if "normalize" not in skip:
        mode = str(prep.get("normalize_mode", "max_abs"))
        per = str(prep.get("normalize_scope", "global"))
        clip = float(prep["clip_percentile"]) if prep.get("clip_percentile") else None
        noisy, stats = normalize(noisy, mode=mode, per=per, clip_percentile=clip)
        gr_noise, _ = normalize(gr_noise, mode=mode, per=per, override_stats=stats)

    clean = noisy - gr_noise

    if input_cfg["path"].lower().endswith((".sgy", ".segy")):
        from tools.segy_read import read_regular_shots
        _, hdrs = read_regular_shots(input_cfg["path"], traces_per_shot=int(input_cfg.get("traces_per_shot", 201)), time_downsample=int(input_cfg.get("time_downsample", 1)), return_headers=True)
        ffid = hdrs["FieldRecord"][:, 0]
    else:
        ffid = np.arange(noisy.shape[0])
    return noisy, clean, gr_noise, ffid


def _patchify_triplets(
    noisy: np.ndarray, clean: np.ndarray, gr_noise: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prep = cfg["preprocess"]
    pt, px = int(prep.get("patch_time", 256)), int(prep.get("patch_trace", 128))
    ov = float(prep.get("patch_overlap", 0.5))
    size = (px, pt)
    n_p, _ = patchify_uniform(noisy, patch_size=size, overlap=ov, output_ndim=4)
    c_p, _ = patchify_uniform(clean, patch_size=size, overlap=ov, output_ndim=4)
    g_p, _ = patchify_uniform(gr_noise, patch_size=size, overlap=ov, output_ndim=4)
    return n_p.astype(np.float32), c_p.astype(np.float32), g_p.astype(np.float32)


# ---------------------------------------------------------------------------
# f-k classifier helpers
# ---------------------------------------------------------------------------

def _to_fk(x: torch.Tensor) -> torch.Tensor:
    """2D orthonormal FFT → stack real + imag as 2-channel tensor.

    ``norm="ortho"`` makes the transform norm-preserving so gradients
    through ``fft2`` have O(1) magnitude regardless of patch size.
    """
    f = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    return torch.cat([f.real, f.imag], dim=1)  # (B, 2, H, W)


# ---------------------------------------------------------------------------
# pre-train f-k classifier
# ---------------------------------------------------------------------------

def pretrain_classifier(
    cfg: Dict[str, Any], device: torch.device, exp_dir: Path, rank: int, distributed: bool
) -> None:
    """Pre-train f-k classifier on clean-signal vs ground-roll patches."""
    if rank != 0:
        return

    # When running single-GPU (not under torchrun), the config's
    # experiment.device may reference a physical GPU index that doesn't
    # match CUDA_VISIBLE_DEVICES set by the launcher.  Use the first
    # visible CUDA device instead.
    if not distributed and torch.cuda.is_available():
        device = torch.device("cuda:0")

    print("=== Phase 1: Pre-training f-k Classifier ===")
    noisy_p, clean_p, gr_p = _build_triplet_patches(cfg)

    # build balanced dataset: half signal, half noise
    n_samples = min(len(clean_p), len(gr_p), 2000)
    idx_sig = np.random.RandomState(42).choice(len(clean_p), n_samples // 2, replace=False)
    idx_noise = np.random.RandomState(43).choice(len(gr_p), n_samples // 2, replace=False)

    sig = torch.from_numpy(clean_p[idx_sig])
    noise = torch.from_numpy(gr_p[idx_noise])
    all_data = torch.cat([sig, noise], dim=0)  # (N, 1, H, W)
    labels = torch.cat([torch.ones(n_samples // 2), torch.zeros(n_samples // 2)], dim=0)  # 1=signal, 0=noise

    clf_cfg = cfg["fk_classifier"]
    classifier = FKClassifier(
        in_channels=2,
        base_channels=int(clf_cfg.get("base_channels", 32)),
        dropout=float(clf_cfg.get("dropout", 0.5)),
    ).to(device)

    optim = torch.optim.Adam(classifier.parameters(), lr=float(clf_cfg.get("lr", 1e-4)))
    epochs = int(clf_cfg.get("epochs", 20))
    bs = int(clf_cfg.get("batch_size", 64))
    bce = nn.BCEWithLogitsLoss()

    classifier.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(all_data))
        total_loss = 0.0
        n_batch = 0
        for start in range(0, len(all_data), bs):
            idx = perm[start : start + bs]
            batch = all_data[idx].to(device)
            lbl = labels[idx].to(device)
            fk = _to_fk(batch)
            out = classifier(fk).squeeze(-1)
            loss = bce(out, lbl)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batch += 1
        # Batched accuracy computation to avoid OOM
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(all_data), bs):
                idx_b = torch.arange(start, min(start + bs, len(all_data)))
                batch = all_data[idx_b].to(device)
                lbl = labels[idx_b].to(device)
                fk = _to_fk(batch)
                out = classifier(fk).squeeze(-1)
                pred = (torch.sigmoid(out) > 0.5).float()
                correct += (pred == lbl).sum().item()
                total += len(idx_b)
        acc = correct / max(total, 1)
        classifier.train()

        print(f"  Classifier epoch {epoch + 1}/{epochs}: loss={total_loss / max(n_batch, 1):.4f}, acc={acc:.4f}")

    save_path = exp_dir / "fk_classifier.pt"
    torch.save(classifier.state_dict(), save_path)
    print(f"  Classifier saved to {save_path}")


def _build_triplet_patches(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    noisy, clean, gr, _ = _preprocess_shots(cfg)
    return _patchify_triplets(noisy, clean, gr, cfg)


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def _visualize_physics_sample(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    save_path: Path,
    device: torch.device,
    title: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """Pick a random sample from ``loader``, run denoise, save a 4-panel plot.

    Panels: noisy input | denoised prediction | clean reference | residual.
    """
    dataset = loader.dataset
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(dataset)))
    sample = dataset[idx]
    z, x_true, _ = sample
    z = z.unsqueeze(0).to(device)
    x_true = x_true.unsqueeze(0).to(device)

    x_pred = model.denoise(z)

    from utils import plot_sample
    suffix = f"sample idx={idx}"
    full_title = f"{title} | {suffix}" if title else suffix
    plot_sample(
        input_data=z,
        prediction=x_pred,
        target=x_true,
        save_path=save_path,
        title=full_title,
        cmap="gray",
    )


# ---------------------------------------------------------------------------
# main training
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physics-constrained separation network.")
    parser.add_argument("--config", type=str, default="configs/ground_roll_attenuation/denoise_physics.yaml")
    parser.add_argument("--pretrain_classifier", action="store_true", help="Pre-train f-k classifier only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_denoise_experiment_name_from_model(cfg)

    distributed, rank, local_rank, world_size = init_distributed()
    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=_REPO_ROOT)
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    # --- pre-train classifier -----------------------------------------------
    if args.pretrain_classifier:
        pretrain_classifier(cfg, device, exp_dir, rank, distributed)
        destroy_distributed()
        return

    # --- phase 2: train separation network ----------------------------------
    # load pre-trained classifier
    clf_path = exp_dir / "fk_classifier.pt"
    if not clf_path.is_file():
        raise FileNotFoundError(f"FK classifier not found at {clf_path}. Run with --pretrain_classifier first.")
    fk_clf = FKClassifier(
        in_channels=2,
        base_channels=int(cfg["fk_classifier"].get("base_channels", 32)),
    ).to(device)
    fk_clf.load_state_dict(torch.load(clf_path, map_location=device))
    fk_clf.eval()
    for p in fk_clf.parameters():
        p.requires_grad = False
    if rank == 0:
        print(f"Loaded pre-trained f-k classifier from {clf_path}")

    # data loaders — custom build since we need triplets
    noisy, clean, gr, per_shot_ffid = _preprocess_shots(cfg)
    loader_cfg = cfg["data"].get("loader", {})
    bs = int(loader_cfg.get("batch_size", 16))
    nw = int(loader_cfg.get("num_workers", 4))
    pm = bool(loader_cfg.get("pin_memory", True))
    seed = int(cfg["experiment"]["seed"])

    if "shot_split" in cfg.get("data", {}):
        ss = cfg["data"]["shot_split"]
        n_train, n_val = int(ss["train"]), int(ss["val"])
        unique = np.unique(per_shot_ffid)
        train_mask = np.isin(per_shot_ffid, unique[:n_train])
        val_mask = np.isin(per_shot_ffid, unique[n_train:n_train + n_val])

        train_n, train_c, train_g = _patchify_triplets(noisy[train_mask], clean[train_mask], gr[train_mask], cfg)
        val_n, val_c, val_g = _patchify_triplets(noisy[val_mask], clean[val_mask], gr[val_mask], cfg)
    else:
        noisy_p, clean_p, gr_p = _patchify_triplets(noisy, clean, gr, cfg)
        n_total = len(noisy_p)
        n_val = int(n_total * 0.1)
        indices = np.random.RandomState(seed).permutation(n_total)
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        train_n, train_c, train_g = noisy_p[train_idx], clean_p[train_idx], gr_p[train_idx]
        val_n, val_c, val_g = noisy_p[val_idx], clean_p[val_idx], gr_p[val_idx]

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(train_n), torch.from_numpy(train_c), torch.from_numpy(train_g)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(val_n), torch.from_numpy(val_c), torch.from_numpy(val_g)
    )

    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=seed,
        )
        eval_train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=seed,
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_sampler, num_workers=nw, pin_memory=pm, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_sampler, num_workers=nw, pin_memory=pm, drop_last=False)
        eval_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=eval_train_sampler, num_workers=nw, pin_memory=pm, drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
        eval_train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    # model
    model = build_model(cfg["model"]).to(device)
    model = maybe_wrap_ddp(model, distributed=distributed, device=device, local_rank=local_rank)
    model_type = str(cfg["model"]["type"])

    metrics_spec = cfg.get("metrics", [])
    metrics = build_metrics(metrics_spec)
    metric_names = list(metrics.keys())

    # optimizers
    lr = float(cfg["optim"]["params"].get("lr", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_epochs = int(cfg["train"]["epochs"])
    from utils.train_utils import build_scheduler
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), total_epochs)

    # loss weights
    loss_w = cfg.get("physics_loss", {})
    w1 = float(loss_w.get("lambda_signal_class", 0.1))
    w2 = float(loss_w.get("lambda_noise_class", 0.1))
    w3 = float(loss_w.get("lambda_recover_data", 0.5))
    w4 = float(loss_w.get("lambda_recover_noise_class", 0.1))

    bce = nn.BCEWithLogitsLoss()

    # logger
    logger: Optional[TrainingLogger] = None
    if rank == 0:
        logger = TrainingLogger(
            log_dir=exp_dir / cfg["log"].get("log_dir", "logs"),
            loss_keys=["train", "val"],
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )

    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 20))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    grad_clip = cfg["train"].get("grad_clip")
    detect_anomaly = bool(cfg["train"].get("detect_anomaly", False))
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if rank == 0:
            print("  [DEBUG] torch.autograd anomaly detection enabled")

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(total_epochs):
        if train_sampler is not None:
            sampler_set_epoch(train_sampler, epoch)
        model.train()

        total_loss_sum = 0.0
        n_batches = 0

        for z_batch, x_batch, y_batch in train_loader:
            z = z_batch.to(device, non_blocking=True)
            x_true = x_batch.to(device, non_blocking=True)
            y_true = y_batch.to(device, non_blocking=True)

            x_pred, y_pred, y_rec = model(z)

            # Guard: skip batch if any model output is non-finite
            if not torch.isfinite(x_pred).all() or not torch.isfinite(y_pred).all():
                if rank == 0:
                    print(f"  [WARNING] Non-finite model output at epoch {epoch}, skipping batch")
                continue

            # data fidelity
            l_signal = F.mse_loss(x_pred, x_true)
            l_gr = F.mse_loss(y_pred, y_true)
            l_rec = F.mse_loss(y_rec, y_pred.detach())

            # classification losses via frozen f-k classifier
            fk_x = _to_fk(x_pred)
            fk_y = _to_fk(y_pred)
            fk_yrec = _to_fk(y_rec)

            l_sig_cls = bce(
                torch.clamp(fk_clf(fk_x).squeeze(-1), -10, 10),
                torch.ones(z.shape[0], device=device),
            )
            l_noise_cls = bce(
                torch.clamp(fk_clf(fk_y).squeeze(-1), -10, 10),
                torch.zeros(z.shape[0], device=device),
            )
            l_rec_cls = bce(
                torch.clamp(fk_clf(fk_yrec).squeeze(-1), -10, 10),
                torch.zeros(z.shape[0], device=device),
            )

            loss = l_signal + l_gr + w1 * l_sig_cls + w2 * l_noise_cls + w3 * l_rec + w4 * l_rec_cls

            # Guard: skip batch if loss is non-finite
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"  [WARNING] Non-finite loss at epoch {epoch}, skipping batch")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            total_loss_sum += loss.item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        if distributed:
            stat = torch.tensor([float(total_loss_sum), float(n_batches)], device=device, dtype=torch.float64)
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            train_loss = float(stat[0].item() / max(stat[1].item(), 1.0))
        else:
            train_loss = total_loss_sum / max(n_batches, 1)

        # --- evaluation: use CNN1 output as denoised signal (all ranks) ----
        val_losses = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {}

        if epoch == 0 or (epoch + 1) % eval_interval == 0:
            model.eval()

            from utils.metrics import compute_metrics

            # --- train set evaluation ---
            t_mse_sum = 0.0
            t_total = 0
            t_metric_sums: Dict[str, float] = {}
            with torch.no_grad():
                for z_b, x_b, _ in eval_train_loader:
                    z_b = z_b.to(device, non_blocking=True)
                    x_b = x_b.to(device, non_blocking=True)
                    B = z_b.shape[0]
                    x_pr = model.module.denoise(z_b) if hasattr(model, "module") else model.denoise(z_b)
                    t_mse_sum += F.mse_loss(x_pr, x_b).item() * B
                    t_total += B
                    bm = compute_metrics(metrics, x_pr, x_b)
                    for k, v in bm.items():
                        t_metric_sums[k] = t_metric_sums.get(k, 0.0) + v * B

            # --- val set evaluation ---
            v_mse_sum = 0.0
            v_total = 0
            v_metric_sums: Dict[str, float] = {}
            with torch.no_grad():
                for z_b, x_b, _ in val_loader:
                    z_b = z_b.to(device, non_blocking=True)
                    x_b = x_b.to(device, non_blocking=True)
                    B = z_b.shape[0]
                    x_pr = model.module.denoise(z_b) if hasattr(model, "module") else model.denoise(z_b)
                    v_mse_sum += F.mse_loss(x_pr, x_b).item() * B
                    v_total += B
                    bm = compute_metrics(metrics, x_pr, x_b)
                    for k, v in bm.items():
                        v_metric_sums[k] = v_metric_sums.get(k, 0.0) + v * B

            # all-reduce partial sums
            if distributed:
                t_keys = sorted(t_metric_sums.keys())
                v_keys = sorted(v_metric_sums.keys())
                stat = torch.tensor(
                    [t_mse_sum, float(t_total), v_mse_sum, float(v_total)]
                    + [t_metric_sums.get(k, 0.0) for k in t_keys]
                    + [v_metric_sums.get(k, 0.0) for k in v_keys],
                    device=device, dtype=torch.float64,
                )
                torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
                t_mse_sum = float(stat[0].item())
                t_total = int(stat[1].item())
                v_mse_sum = float(stat[2].item())
                v_total = int(stat[3].item())
                for i, k in enumerate(t_keys):
                    t_metric_sums[k] = float(stat[4 + i].item())
                for i, k in enumerate(v_keys):
                    v_metric_sums[k] = float(stat[4 + len(t_keys) + i].item())

            if t_total > 0:
                train_metrics = {k: v / max(t_total, 1) for k, v in t_metric_sums.items()}
            val_losses["val"] = v_mse_sum / max(v_total, 1)
            val_metrics = {k: v / max(v_total, 1) for k, v in v_metric_sums.items()}

            if rank == 0 and val_losses["val"] < best_val_loss:
                from utils.train_utils import maybe_save_best_checkpoint as _best
                best_val_loss = _best(
                    exp_dir / "checkpoints" / "best.pt",
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, val_loss=val_losses["val"],
                    best_val_loss=best_val_loss, extras={"config": cfg}, logger=logger,
                )

        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(epoch=epoch, losses={"train": train_loss, "val": val_losses["val"]}, metrics=metric_row, extras={"lr": optimizer.param_groups[0]["lr"]})

        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, extras={"config": cfg})

        # --- visualization (rank 0) ----------------------------------------
        if rank == 0 and vis_interval > 0 and (epoch == 0 or (epoch + 1) % vis_interval == 0):
            _visualize_physics_sample(
                model=model.module if hasattr(model, "module") else model,
                loader=val_loader,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device,
                title=f"Physics {model_type} epoch {epoch}",
                seed=None,
            )

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(f"Physics-constrained training finished in {elapsed:.2f}s.")
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
