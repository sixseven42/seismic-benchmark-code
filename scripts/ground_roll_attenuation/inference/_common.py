"""Shared inference pipeline for ground-roll attenuation models.

All standard noise-predicting models share the same preprocessing and
inference pipeline.  This module provides the common helpers so each
per-model inference script is ~30 lines.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# SEG-Y I/O with auto-padding
# ---------------------------------------------------------------------------

def read_sgy_flat(
    path: Path, traces_per_shot: int
) -> tuple[np.ndarray, int, int]:
    """Read a SEG-Y file, zero-pad traces to be divisible by ``traces_per_shot``.

    Returns
    -------
    shots   : ``(n_shots, traces_per_shot, n_time)`` float32.
    n_orig  : original (unpadded) number of traces.
    n_time  : number of time samples.
    """
    import segyio

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        n_traces = int(f.tracecount)
        n_samples = int(len(f.samples))
        traces = segyio.tools.collect(f.trace[:]).astype(np.float32, copy=False)

    padded_n = ((n_traces + traces_per_shot - 1) // traces_per_shot) * traces_per_shot
    pad_count = padded_n - n_traces
    if pad_count > 0:
        pad = np.zeros((pad_count, n_samples), dtype=traces.dtype)
        traces = np.concatenate([traces, pad], axis=0)
        print(f"  Padded {pad_count} zero-trace(s) to reach {padded_n} traces "
              f"({n_traces} → {padded_n}, {padded_n // traces_per_shot} shots)")

    n_shots = padded_n // traces_per_shot
    shots = traces.reshape(n_shots, traces_per_shot, n_samples)
    return shots, n_traces, n_samples


def save_sgy(path: Path, data: np.ndarray, src_path: Path) -> None:
    """Save a flat ``(n_traces, n_time)`` float32 array as SEG-Y IEEE float.

    Trace headers and bin metadata are copied from ``src_path`` when the
    trace counts match; otherwise minimal headers are generated.
    """
    import segyio

    if data.ndim != 2:
        raise ValueError(f"Expected 2D (n_traces, n_time), got shape {data.shape}.")
    n_traces, n_samples = data.shape
    data = np.ascontiguousarray(data, dtype=np.float32)

    spec = segyio.spec()
    spec.sorting = 1          # unknown / arbitrary sorting
    spec.format = 5           # IEEE float32
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.open(str(src_path), "r", ignore_geometry=True) as src:
        bin_interval = int(src.bin[segyio.BinField.Interval])
        copy_headers = (src.tracecount == n_traces)

    with segyio.create(str(path), spec) as dst:
        dst.bin[segyio.BinField.Interval] = bin_interval
        if copy_headers:
            with segyio.open(str(src_path), "r", ignore_geometry=True) as src:
                for tr in range(n_traces):
                    dst.header[tr] = src.header[tr]
                    dst.trace[tr] = data[tr]
        else:
            for tr in range(n_traces):
                dst.header[tr][segyio.TraceField.TRACE_SEQUENCE_LINE] = tr + 1
                dst.trace[tr] = data[tr]


# ---------------------------------------------------------------------------
# shared CLI
# ---------------------------------------------------------------------------

def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Register standard inference CLI arguments."""
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--input-sgy", type=str, required=True,
                        help="Path to noisy input SEG-Y file.")
    parser.add_argument("--output-sgy", type=str, required=True,
                        help="Path to save denoised SEG-Y output.")
    parser.add_argument("--traces-per-shot", type=int, default=None,
                        help="Override traces_per_shot from config.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda:0').")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Inference batch size.")


# ---------------------------------------------------------------------------
# shared pipeline
# ---------------------------------------------------------------------------

def run_inference(
    args: argparse.Namespace,
    default_config: str,
    build_model_fn: Callable[[dict], torch.nn.Module],
    forward_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    *,
    model_type_label: str = "",
) -> None:
    """Shared inference pipeline: read → normalize → patchify → model → save.

    Parameters
    ----------
    args             : parsed CLI arguments.
    default_config   : fallback config path (used when ``--config`` is not given).
    build_model_fn   : ``cfg['model'] -> nn.Module`` (usually ``build_model``).
    forward_fn       : ``(batch: Tensor) -> denoised_batch: Tensor``.
    model_type_label : human-readable label printed at startup.
    """
    # --- load config ----------------------------------------------------------
    from utils import load_config
    cfg = load_config(args.config if args.config else default_config)

    prep = cfg.get("preprocess", {})
    data_cfg = cfg.get("data", {})

    # --- resolve traces_per_shot ----------------------------------------------
    tps_from_cfg = None
    for key in ("segy_pair", "npy_pair", "mat_pair"):
        if key in data_cfg:
            tps_from_cfg = data_cfg[key].get("traces_per_shot")
            break
    traces_per_shot = args.traces_per_shot if args.traces_per_shot is not None else int(tps_from_cfg or 201)

    # --- device ---------------------------------------------------------------
    device = torch.device(
        args.device if args.device is not None
        else cfg["experiment"].get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    )
    batch_size = (
        args.batch_size if args.batch_size is not None
        else int(data_cfg.get("loader", {}).get("batch_size", 8))
    )

    # --- build model & load checkpoint ----------------------------------------
    model = build_model_fn(cfg["model"]).to(device)
    from utils.train_utils import load_checkpoint
    extras = load_checkpoint(args.checkpoint, model, map_location=device)
    label = model_type_label or str(cfg["model"]["type"])
    print(f"Loaded checkpoint epoch {extras.get('epoch', '?')} model={label}")

    # --- read noisy SEG-Y (with auto-padding) ---------------------------------
    input_path = Path(args.input_sgy)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input SEG-Y not found: {input_path}")

    print(f"Reading {input_path.name} with traces_per_shot={traces_per_shot} ...")
    noisy_shots, n_orig_traces, n_time = read_sgy_flat(input_path, traces_per_shot)
    print(f"  shape: {noisy_shots.shape}  (n_shots={noisy_shots.shape[0]}, "
          f"n_traces={noisy_shots.shape[1]}, n_time={n_time})")

    # --- preprocess: exactly match training -----------------------------------
    from tools.preprocessing import denormalize, normalize
    norm_mode = str(prep.get("normalize_mode", "max_abs"))
    norm_scope = str(prep.get("normalize_scope", "global"))
    print(f"Normalizing: mode={norm_mode}, per={norm_scope} ...")
    noisy_norm, stats = normalize(noisy_shots, mode=norm_mode, per=norm_scope)

    patch_trace = int(prep.get("patch_trace", 128))
    patch_time = int(prep.get("patch_time", 256))
    overlap = float(prep.get("patch_overlap", 0.5))
    print(f"Patching: size=({patch_trace}, {patch_time}), overlap={overlap}")

    # --- inference ------------------------------------------------------------
    from utils.inference_utils import inference_on_shots
    print("Running inference ...")
    t0 = time.time()

    _fw = forward_fn  # capture locally to avoid closure-over-loop issues

    pred_norm = inference_on_shots(
        model=model,
        input_shots=noisy_norm,
        patch_size=(patch_trace, patch_time),
        overlap=overlap,
        device=device,
        batch_size=batch_size,
        forward_fn=lambda batch: _fw(model, batch),
    )
    print(f"  done in {time.time() - t0:.1f}s, shape={pred_norm.shape}")

    # --- denormalize ----------------------------------------------------------
    pred_shots = denormalize(pred_norm, stats, mode=norm_mode, per=norm_scope)

    # --- crop padding & write output SEG-Y ------------------------------------
    n_shots_padded, tps, _nt = pred_shots.shape
    pred_flat = pred_shots.reshape(n_shots_padded * tps, _nt)
    pred_flat = pred_flat[:n_orig_traces]

    output_path = Path(args.output_sgy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {n_orig_traces} traces to {output_path} ...")
    save_sgy(output_path, pred_flat, input_path)
    print(f"Done. Output saved to {output_path}")
