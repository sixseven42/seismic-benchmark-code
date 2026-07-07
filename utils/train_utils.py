"""Training helpers (seeding, config, optim / sched / ckpt / loops)."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .logger import _safe_value
from .metrics import compute_metrics


# ----------------------------------------------------------------------
# Environment / config
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed ``random`` / ``numpy`` / ``torch`` (CPU + CUDA)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg)}.")
    return cfg


def resolve_repo_root(script_file: Union[str, Path]) -> Path:
    """Walk upward from ``script_file`` until a directory containing ``model/`` and ``utils/`` is found."""
    script_path = Path(script_file).resolve()
    current = script_path.parent
    for _ in range(10):
        if (current / "model").is_dir() and (current / "utils").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback to two levels above the script (legacy flat layout).
    return script_path.parent.parent


def setup_experiment_dir(
    cfg: Dict[str, Any],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Create ``<output_dir>/<experiment.name>/`` and dump the config.

    Parameters
    ----------
    cfg      : experiment config dict.
    base_dir : when ``output_dir`` is relative, resolve it against this directory.
               Defaults to the current working directory.
    """
    exp = cfg.get("experiment", {})
    name = exp.get("name", "exp")
    output_dir_raw = exp.get("output_dir", "results")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute() and base_dir is not None:
        output_dir = Path(base_dir).resolve() / output_dir
    exp_dir = output_dir / name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    with (exp_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return exp_dir


def default_config_relpath_for_train_script(
    script_file: Union[str, Path],
    *,
    repo_root: Optional[Union[str, Path]] = None,
) -> str:
    """Resolve the default config path for a training/inference script.

    Layouts supported:
    - flat:    ``scripts/train_<stem>.py``                 -> ``configs/<stem>.yaml``
    - subtask: ``scripts/<subtask>/train_<stem>.py``       -> ``configs/<subtask>/<stem>.yaml``
    - subtask: ``scripts/<a>/<b>/train_<stem>.py``         -> ``configs/<a>/<b>/<stem>.yaml``

    Subtask folders between ``scripts/`` and the script file are mirrored under
    ``configs/``. Scripts whose stem does not start with ``train_`` keep the full
    stem (used by inference entry points such as ``inference_interpolation.py``).
    """
    script_path = Path(script_file).resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else resolve_repo_root(script_file)
    stem = script_path.stem
    cfg_stem = stem[6:] if stem.startswith("train_") else stem

    subdirs: list[str] = []
    try:
        rel_parts = script_path.relative_to(repo).parts
    except ValueError:
        rel_parts = ()
    if rel_parts and rel_parts[0] == "scripts":
        subdirs = list(rel_parts[1:-1])

    cfg_path = repo.joinpath("configs", *subdirs, f"{cfg_stem}.yaml")
    try:
        return str(cfg_path.relative_to(repo))
    except ValueError:
        return str(cfg_path)


def apply_denoise_experiment_name_from_model(cfg: Dict[str, Any]) -> None:
    """If ``experiment.name`` is still ``denoise_unet_base``, align it with ``model.type``."""
    exp = cfg.setdefault("experiment", {})
    m_raw = cfg.get("model") or {}
    mtype = m_raw.get("type")
    if exp.get("name") != "denoise_unet_base" or not mtype:
        return
    if str(mtype) == "unet":
        return
    exp["name"] = f"denoise_{mtype}_base"


# Core metrics for SEG-Y patch denoise scripts (baseline + YAML overrides); see ``resolve_denoise_metrics``.
_DENOISE_METRICS_BASELINE: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("snr", {"reduction": "per_sample"}),
    ("psnr", {"data_range": 1.0, "reduction": "per_sample"}),
    ("ssim", {"data_range": 2.0, "window_size": 11, "sigma": 1.5}),
    ("mae", {}),
    ("mse", {}),
    ("rmse", {"reduction": "per_sample"}),
)


def resolve_denoise_metrics(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ``metrics`` spec: six defaults (SNR … RMSE) merged with YAML ``cfg['metrics']`` params.

    Order: baseline metrics first (``snr`` … ``rmse``), then any extra entries only listed in YAML.
    """
    yaml_list = list(cfg.get("metrics") or [])
    overrides = {entry["name"]: entry for entry in yaml_list}

    baseline_order = [n for n, _ in _DENOISE_METRICS_BASELINE]

    merged: List[Dict[str, Any]] = []
    for name, default_params in _DENOISE_METRICS_BASELINE:
        params = dict(default_params)
        if name in overrides:
            params.update(overrides[name].get("params") or {})
        merged.append({"name": name, "params": params})

    for entry in yaml_list:
        name = entry["name"]
        if name not in baseline_order:
            merged.append({"name": name, "params": dict(entry.get("params") or {})})

    return merged


def init_distributed(backend: Optional[str] = None) -> Tuple[bool, int, int, int]:
    """Initialize ``torch.distributed`` when launched with ``WORLD_SIZE`` > 1 (e.g. ``torchrun``).

    Returns
    -------
    tuple
        ``(distributed, rank, local_rank, world_size)``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", device_id=device
        )
    else:
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    return True, rank, local_rank, world_size


def barrier_if_distributed() -> None:
    """Synchronize all processes when a process group exists."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def destroy_distributed() -> None:
    """Tear down the default process group (no-op when not initialized)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def training_device(cfg: Dict[str, Any], *, distributed: bool, local_rank: int) -> torch.device:
    """Under DDP use ``CUDA:local_rank``; otherwise ``experiment.device`` (CPU/GPU ID)."""
    if distributed and torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device(cfg.get("experiment", {}).get("device", "cpu"))


def setup_experiment_dir_distributed(
    cfg: Dict[str, Any],
    rank: int,
    distributed: bool,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Create dirs and dump config only on rank 0; barrier so all ranks see the paths.

    Parameters
    ----------
    cfg      : experiment config dict.
    rank     : process rank.
    distributed : whether running under DDP.
    base_dir : when ``output_dir`` is relative, resolve it against this directory.
    """
    exp = cfg.get("experiment", {})
    output_dir_raw = exp.get("output_dir", "results")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute() and base_dir is not None:
        output_dir = Path(base_dir).resolve() / output_dir
    name = exp.get("name", "exp")
    exp_dir = output_dir / name
    if not distributed:
        return setup_experiment_dir(cfg, base_dir=base_dir)
    if rank == 0:
        setup_experiment_dir(cfg, base_dir=base_dir)
    barrier_if_distributed()
    return exp_dir


def unwrap_ddp(module: nn.Module) -> nn.Module:
    """Return underlying module from ``DistributedDataParallel`` when applicable."""
    if isinstance(module, nn.parallel.DistributedDataParallel):
        return module.module
    return module


def maybe_wrap_ddp(
    module: nn.Module,
    *,
    distributed: bool,
    device: torch.device,
    local_rank: int,
) -> nn.Module:
    """Wrap ``module`` with ``DistributedDataParallel`` when ``distributed``."""
    if not distributed:
        return module
    if device.type != "cuda":
        return nn.parallel.DistributedDataParallel(module)
    return nn.parallel.DistributedDataParallel(
        module, device_ids=[local_rank], output_device=local_rank
    )


def sampler_set_epoch(sampler: Optional[DistributedSampler], epoch: int) -> None:
    """Shard rotation for DistributedSampler."""
    if sampler is not None:
        sampler.set_epoch(epoch)


# ----------------------------------------------------------------------
# Optimizer / scheduler builders
# ----------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> Optimizer:
    """Build an optimizer from a ``{type, params}`` block."""
    name = str(cfg.get("type", "adamw")).lower()
    params = cfg.get("params", {})
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **params)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    raise ValueError(f"Unknown optimizer type: {name!r}.")


def build_scheduler(
    optimizer: Optimizer,
    cfg: Dict[str, Any],
    total_epochs: int,
) -> Optional[_LRScheduler]:
    """Build an LR scheduler (``cosine`` / ``step`` / ``none``)."""
    name = str(cfg.get("type", "none")).lower()
    params = cfg.get("params", {})
    if name == "none":
        return None
    if name == "step":
        step_size = int(params.get("step_size", 30))
        gamma = float(params.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        min_lr = float(params.get("min_lr", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(total_epochs)), eta_min=min_lr
        )
    raise ValueError(f"Unknown scheduler type: {name!r}.")


# ----------------------------------------------------------------------
# Checkpoint I/O
# ----------------------------------------------------------------------

def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a self-contained training snapshot."""
    ckpt = {
        "model": unwrap_ddp(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "extras": extras or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, p)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Restore a checkpoint in-place and return its extras dict."""
    ckpt = torch.load(Path(path), map_location=map_location)
    unwrap_ddp(model).load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    extras = dict(ckpt.get("extras", {}))
    extras.setdefault("epoch", int(ckpt.get("epoch", -1)))
    return extras


def find_latest_checkpoint(ckpt_dir: Union[str, Path]) -> Optional[Path]:
    """Return the path of the most recent checkpoint, or ``None`` if empty."""
    root = Path(ckpt_dir)
    if not root.exists():
        return None
    files = sorted(root.glob("*.pt"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def maybe_save_best_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    extras: Optional[Dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> float:
    """Save checkpoint when ``val_loss`` improves; returns updated best_val_loss."""
    if val_loss < best_val_loss:
        save_checkpoint(path, model, optimizer, scheduler, epoch, extras)
        if logger is not None:
            logger.info(
                f"New best val_loss={val_loss:.6g} (prev={best_val_loss:.6g}) at epoch {epoch}; "
                f"saved {Path(path).name}"
            )
        return val_loss
    return best_val_loss


# ----------------------------------------------------------------------
# Training / evaluation loops
# ----------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[_LRScheduler] = None,
    grad_clip: Optional[float] = None,
    log_interval: int = 50,
    logger: Optional[Any] = None,
) -> Dict[str, float]:
    """Run one training epoch; returns ``{'train': mean_loss}``."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for step, batch in enumerate(loader):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError("train_one_epoch expects loader batch as (input, target).")
        if y is None:
            raise ValueError("train_one_epoch requires supervised targets.")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1
        if logger is not None and (step + 1) % max(1, int(log_interval)) == 0:
            logger.info(
                f"[epoch={epoch} step={step + 1}/{len(loader)}] "
                f"train_step_loss={loss.detach().item():.6g}"
            )

    if scheduler is not None:
        scheduler.step()
    mean_loss = total_loss / max(n_batches, 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        if ws > 1:
            stat = torch.tensor(
                [float(total_loss), float(n_batches)],
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            mean_loss = float(stat[0].item() / max(stat[1].item(), 1.0))
    return {"train": float(mean_loss)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics: Optional[Dict[str, Any]],
    device: torch.device,
    *,
    metrics_on_denoised_signal: bool = False,
    distributed: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run evaluation; returns ``(loss_dict, metric_dict)``.

    Parameters
    ----------
    metrics_on_denoised_signal
        If ``False`` (default), metrics use ``(pred, target)`` directly.
        If ``True``, metrics use ``(input - pred, input - target)``. Use this when
        the model predicts additive noise and targets are noise maps: then
        ``input - target`` is the reference signal and ``input - pred`` is the
        denoised estimate (same normalization scale as training).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metric_sums: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    metric_counts: Dict[str, int] = {k: 0 for k in (metrics or {})}
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError("evaluate expects loader batch as (input, target).")
        if y is None:
            raise ValueError("evaluate requires supervised targets.")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += float(loss.detach().item())
        n_batches += 1

        if metrics:
            if metrics_on_denoised_signal:
                pred_m = x - pred
                targ_m = x - y
            else:
                pred_m, targ_m = pred, y
            batch_metrics = compute_metrics(metrics, pred_m, targ_m)
            for k, v in batch_metrics.items():
                value = float(v)
                if math.isfinite(value):
                    metric_sums[k] += value
                    metric_counts[k] += 1

    # Aggregate across DDP ranks when each rank evaluates a different subset.
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        if ws > 1:
            stat = torch.tensor(
                [float(total_loss), float(n_batches)], device=device, dtype=torch.float64
            )
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            total_loss = float(stat[0].item())
            n_batches = int(stat[1].item())
            for k in list(metric_sums.keys()):
                s = torch.tensor(
                    [float(metric_sums[k]), float(metric_counts[k])],
                    device=device,
                    dtype=torch.float64,
                )
                torch.distributed.all_reduce(s, op=torch.distributed.ReduceOp.SUM)
                metric_sums[k] = float(s[0].item())
                metric_counts[k] = int(s[1].item())

    denom = max(n_batches, 1)
    losses = {"val": float(total_loss / denom)}
    out_metrics = {
        k: float(metric_sums[k] / metric_counts[k])
        if metric_counts[k] > 0 else float("nan")
        for k in metric_sums
    }
    return losses, out_metrics


def _make_dataloader(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Dict[str, Any],
    shuffle: bool,
    sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    """Build a DataLoader from tensors and config."""
    loader_cfg = cfg["data"].get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 8))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    ds = TensorDataset(x, y)
    if sampler is not None:
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_loaders(
    cfg: Dict[str, Any],
    *,
    build_patch_pairs_fn: Callable[[Dict[str, Any]], Tuple[np.ndarray, np.ndarray]],
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
) -> Tuple[
    DataLoader,
    DataLoader,
    Optional[DistributedSampler],
    Optional[DataLoader],
]:
    """Build train / test loaders from patches produced by ``build_patch_pairs_fn``.

    Parameters
    ----------
    cfg                  : experiment config dict.
    build_patch_pairs_fn : callable ``cfg -> (input_patches, target_patches)``.
    rank, world_size, distributed : DDP controls (see :func:`init_distributed`).

    Returns
    -------
    train_loader, test_loader, train_sampler, eval_train_loader
        ``eval_train_loader`` is ``None`` on non-zero ranks under DDP.
    """
    x, y = build_patch_pairs_fn(cfg)
    split = float(cfg["data"].get("test_ratio", cfg["data"].get("val_ratio", 0.1)))
    n_total = x.shape[0]
    n_test = max(1, int(round(n_total * split)))
    n_train = max(1, n_total - n_test)

    idx = np.arange(n_total)
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    rng.shuffle(idx)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    if test_idx.size == 0:
        test_idx = train_idx[:1]

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(y[test_idx])

    sampler_seed = int(cfg["experiment"]["seed"])
    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            TensorDataset(x_train, y_train),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=sampler_seed,
        )

    train_loader = _make_dataloader(x_train, y_train, cfg, shuffle=True, sampler=train_sampler)
    test_loader = _make_dataloader(x_test, y_test, cfg, shuffle=False)

    if distributed and rank != 0:
        eval_train_loader: Optional[DataLoader] = None
    else:
        eval_train_loader = _make_dataloader(x_train, y_train, cfg, shuffle=False)

    return train_loader, test_loader, train_sampler, eval_train_loader


def build_shot_split_loaders(
    cfg: Dict[str, Any],
    *,
    preprocess_fn: Callable[[Dict[str, Any]], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    patchify_fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Tuple[np.ndarray, np.ndarray]],
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
    test_set_dir: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DataLoader]]:
    """Build train / val loaders with shot-level (FFID) splitting.

    Splits are performed on unique FFID values in sequential order:
    the first ``train`` unique FFIDs go to training, the next ``val`` to
    validation, and the remaining ``test`` to the test set.
    All shot gathers belonging to the same FFID are kept together.

    Parameters
    ----------
    cfg                  : experiment config dict.
    preprocess_fn        : callable ``cfg -> (input_shots, target_shots, per_shot_ffid)``.
    patchify_fn          : callable ``(input_shots, target_shots, cfg) -> (input_patches, target_patches)``.
    rank, world_size, distributed : DDP controls.
    test_set_dir         : if provided, unpatchified test shots are saved here
                           as ``input_shots.npy``, ``target_shots.npy``, ``ffid.npy``.

    Returns
    -------
    train_loader, val_loader, train_sampler, eval_train_loader
        ``eval_train_loader`` is ``None`` on non-zero ranks under DDP.
    """
    input_shots, target_shots, per_shot_ffid = preprocess_fn(cfg)

    shot_split = cfg["data"]["shot_split"]
    n_train = int(shot_split["train"])
    n_val = int(shot_split["val"])
    n_test = int(shot_split["test"])

    unique_ffids = np.unique(per_shot_ffid)
    n_total_ffid = unique_ffids.size
    if n_train + n_val + n_test > n_total_ffid:
        raise ValueError(
            f"shot_split asks for {n_train}+{n_val}+{n_test} shots "
            f"but only {n_total_ffid} unique FFIDs available."
        )

    train_ffids = unique_ffids[:n_train]
    val_ffids = unique_ffids[n_train : n_train + n_val]
    test_ffids = unique_ffids[n_train + n_val : n_train + n_val + n_test]

    train_mask = np.isin(per_shot_ffid, train_ffids)
    val_mask = np.isin(per_shot_ffid, val_ffids)
    test_mask = np.isin(per_shot_ffid, test_ffids)

    train_x, train_y = patchify_fn(input_shots[train_mask], target_shots[train_mask], cfg)
    val_x, val_y = patchify_fn(input_shots[val_mask], target_shots[val_mask], cfg)
    test_x, test_y = patchify_fn(input_shots[test_mask], target_shots[test_mask], cfg)

    if test_set_dir is not None:
        out_dir = Path(test_set_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "input_shots.npy", input_shots[test_mask])
        np.save(out_dir / "target_shots.npy", target_shots[test_mask])
        np.save(out_dir / "ffid.npy", per_shot_ffid[test_mask])

    x_train = torch.from_numpy(train_x)
    y_train = torch.from_numpy(train_y)
    x_val = torch.from_numpy(val_x)
    y_val = torch.from_numpy(val_y)

    sampler_seed = int(cfg["experiment"]["seed"])
    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            TensorDataset(x_train, y_train),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=sampler_seed,
        )

    train_loader = _make_dataloader(x_train, y_train, cfg, shuffle=True, sampler=train_sampler)

    # Validation loader: each rank evaluates a distinct subset to avoid
    # redundant computation; metrics are aggregated in evaluate().
    val_sampler: Optional[DistributedSampler] = None
    if distributed:
        val_sampler = DistributedSampler(
            TensorDataset(x_val, y_val),
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=sampler_seed,
        )
    val_loader = _make_dataloader(x_val, y_val, cfg, shuffle=False, sampler=val_sampler)

    if distributed and rank != 0:
        eval_train_loader: Optional[DataLoader] = None
    else:
        eval_train_loader = _make_dataloader(x_train, y_train, cfg, shuffle=False)

    return train_loader, val_loader, train_sampler, eval_train_loader


def count_parameters(model: nn.Module) -> str:
    """Return a human-readable parameter count string.

    Example
    -------
    >>> count_parameters(model)
    'Total: 31.04M | Trainable: 31.04M'
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M"


# ----------------------------------------------------------------------
# First-break picking training / evaluation utilities
# ----------------------------------------------------------------------


def with_progress(
    iterable: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    total: Optional[int] = None,
) -> Iterable[Any]:
    """Wrap an iterable with tqdm when available."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)


def unpack_first_break_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        x, y, target_pick = batch
        return x, y, target_pick
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        return x, y, None
    raise ValueError("first-break loaders must yield (input, mask, target_pick).")


def train_one_epoch_first_break(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[Any] = None,
    grad_clip: Optional[float] = None,
    log_interval: int = 50,
    logger: Optional[Any] = None,
    progress_bar: bool = False,
    step_loss_logger: Optional[Any] = None,
) -> Dict[str, float]:
    """Run one first-break training epoch with ``(input, mask, target_pick)`` batches."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    epoch_start = time.perf_counter()
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    train_iter = with_progress(
        loader,
        enabled=progress_bar,
        desc=f"epoch {epoch + 1} train",
        total=total_batches,
    )
    for step, batch in enumerate(train_iter):
        step_start = time.perf_counter()
        x, y, _ = unpack_first_break_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().item())
        step_time_sec = time.perf_counter() - step_start
        epoch_elapsed_sec = time.perf_counter() - epoch_start
        total_loss += loss_value
        n_batches += 1
        mean_so_far = total_loss / max(n_batches, 1)
        lr = float(optimizer.param_groups[0]["lr"])

        if hasattr(train_iter, "set_postfix"):
            train_iter.set_postfix(
                loss=f"{loss_value:.4g}",
                avg_loss=f"{mean_so_far:.4g}",
                lr=f"{lr:.3g}",
            )
        if step_loss_logger is not None:
            if total_batches is not None:
                global_step = epoch * int(total_batches) + step + 1
            else:
                global_step = step_loss_logger.next_global_step
            step_loss_logger.log_step(
                epoch=epoch,
                step=step + 1,
                global_step=global_step,
                loss=loss_value,
                lr=lr,
                step_time_sec=step_time_sec,
                epoch_elapsed_sec=epoch_elapsed_sec,
            )
        if logger is not None and (step + 1) % max(1, int(log_interval)) == 0:
            logger.info(
                f"[epoch={epoch} step={step + 1}/{len(loader)}] "
                f"train_step_loss={loss_value:.6g}"
            )

    if step_loss_logger is not None:
        step_loss_logger.flush()
    if scheduler is not None:
        scheduler.step()
    mean_loss = total_loss / max(n_batches, 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ws = torch.distributed.get_world_size()
        if ws > 1:
            stat = torch.tensor(
                [float(total_loss), float(n_batches)],
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
            mean_loss = float(stat[0].item() / max(stat[1].item(), 1.0))
    return {"train": float(mean_loss)}


@torch.no_grad()
def evaluate_first_break(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: torch.nn.Module,
    metrics: Optional[Dict[str, Any]],
    device: torch.device,
    *,
    desc: str = "eval",
    progress_bar: bool = False,
    loss_key: str = "val",
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate first-break losses and metrics, skipping NaN metric batches."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metric_sums: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    metric_counts: Dict[str, int] = {k: 0 for k in (metrics or {})}
    eval_iter = with_progress(
        loader,
        enabled=progress_bar,
        desc=desc,
        total=len(loader) if hasattr(loader, "__len__") else None,
    )
    for batch in eval_iter:
        x, y, target_pick = unpack_first_break_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if target_pick is not None:
            target_pick = target_pick.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += float(loss.detach().item())
        n_batches += 1

        pred_for_metric = pred[0] if isinstance(pred, (tuple, list)) else pred
        for name, metric in (metrics or {}).items():
            value = float(metric(pred_for_metric, y, target_pick=target_pick))
            if np.isfinite(value):
                metric_sums[name] += value
                metric_counts[name] += 1

    losses = {loss_key: float(total_loss / max(n_batches, 1))}
    out_metrics = {
        name: float(metric_sums[name] / metric_counts[name])
        if metric_counts[name] > 0
        else float("nan")
        for name in metric_sums
    }
    return losses, out_metrics


def write_final_test_metrics(
    path: Path,
    *,
    best_epoch: int,
    test_loss: float,
    metrics: Dict[str, float],
    metric_names: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["split", "best_epoch", "loss", *metric_names]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        row = {
            "split": "test",
            "best_epoch": int(best_epoch),
            "loss": _safe_value(float(test_loss)),
        }
        for name in metric_names:
            row[name] = _safe_value(float(metrics.get(name, float("nan"))))
        writer.writerow(row)


def format_final_test_summary(
    *,
    best_epoch: int,
    test_loss: float,
    metrics: Dict[str, float],
    metric_names: Iterable[str],
) -> str:
    parts = [f"best_epoch={best_epoch}", f"loss={test_loss:.6g}"]
    for name in metric_names:
        value = float(metrics.get(name, float("nan")))
        parts.append(f"{name}={value:.6g}" if math.isfinite(value) else f"{name}=nan")
    return "Final test: " + ", ".join(parts)


def compute_length_stats(lengths: Sequence[int]) -> Tuple[int, float, float, float, int]:
    """Return ``(min, p10, median, p90, max)`` from a sequence of integer lengths."""
    if len(lengths) == 0:
        return 0, 0.0, 0.0, 0.0, 0
    arr = np.asarray(lengths, dtype=np.float64)
    p10, median, p90 = np.quantile(arr, [0.1, 0.5, 0.9])
    return int(arr.min()), float(p10), float(median), float(p90), int(arr.max())


def format_length_stats(lengths: Sequence[int]) -> str:
    """Format length statistics as a human-readable string."""
    mn, p10, median, p90, mx = compute_length_stats(lengths)
    if mx <= 0:
        return "n/a"
    return f"min={mn}, p10={p10:.1f}, median={median:.1f}, p90={p90:.1f}, max={mx}"
