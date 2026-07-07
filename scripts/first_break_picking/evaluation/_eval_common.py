"""Shared evaluation logic for first-break picking model inference on SEG-Y files.

Two modes:
  Single-file:  --input /path/to/seismic.sgy
  Batch:        --root /path/to/data_root  (discovers seismic.sgy + label.sgy)

Outputs per evaluation:
  pred_picks.txt  — per-trace pick indices
  prob_mask.sgy   — probability mask as IEEE float32 SEG-Y

Batch mode additionally writes batch_eval.xlsx with per-run metrics and a
summary sheet (mean / std / min / max).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root containing both model/ and utils/.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.first_break_picking import build_model                        # noqa: E402
from tools.patching import _gen_uniform_starts, unpatchify_uniform        # noqa: E402
from tools.preprocessing import first_pick_from_mask                     # noqa: E402
from tools.segy_read import (                                             # noqa: E402
    contiguous_ffid_blocks,
    group_coordinates_are_usable,
    read_group_coordinates,
    read_line_id_header,
)
from utils.train_utils import load_config                                # noqa: E402

try:
    import segyio
except ImportError as exc:
    raise ImportError("segyio is required for first-break SEG-Y evaluation.") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_eval_args(script_file: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first-break picking inference on SEG-Y file(s)."
    )
    # Shared
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (model architecture + preprocessing settings).",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for binarising the mask (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Number of patches per inference batch (default: 64).",
    )
    parser.add_argument(
        "--no-segment", action="store_true",
        help="Disable receiver-line segmentation; treat each FFID as one segment.",
    )
    parser.add_argument(
        "--no-seis-sgy", action="store_true",
        help="Skip writing seismic.sgy.",
    )

    # Single-file mode
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to single input SEG-Y file (single-file mode).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output files (single-file mode; default: <input_stem>_eval/).",
    )
    parser.add_argument(
        "--label-sgy", type=str, default=None,
        help="Optional label/mask SEG-Y for ground-truth comparison (single-file mode).",
    )

    # Batch mode
    parser.add_argument(
        "--root", type=str, default=None,
        help="Root directory to search for seismic.sgy + label.sgy pairs (batch mode).",
    )
    parser.add_argument(
        "--pattern", type=str, default="*",
        help="Glob pattern for subdirs under --root (default: *).",
    )
    parser.add_argument(
        "--data-name", type=str, default="seismic.sgy",
        help="Filename of the input SEG-Y data (default: seismic.sgy).",
    )
    parser.add_argument(
        "--label-name", type=str, default="label.sgy",
        help="Filename of the label SEG-Y mask (default: label.sgy).",
    )
    parser.add_argument(
        "--output-xlsx", type=str, default=None,
        help="Path for batch summary xlsx (default: <root>/batch_eval.xlsx).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _build_model_from_checkpoint(
    checkpoint_path: str,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model_cfg = dict(cfg.get("model", {}))
    if not model_cfg:
        raise ValueError("Config is missing 'model' section.")
    model = build_model(model_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    stripped = {}
    for k, v in state_dict.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        stripped[new_k] = v

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        warnings.warn(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        warnings.warn(f"Unexpected keys when loading checkpoint: {unexpected}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Receiver-line segmentation
# ---------------------------------------------------------------------------

def _line_id_segments(
    block_start: int,
    block_stop: int,
    line_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    values = line_ids[block_start:block_stop]
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            changes.astype(np.int64),
            np.asarray([block_stop - block_start], dtype=np.int64),
        )
    )
    starts = (block_start + offsets[:-1]).astype(np.int64)
    stops = (block_start + offsets[1:]).astype(np.int64)
    keep = stops > starts
    return starts[keep], stops[keep]


def _geometry_segments(
    block_start: int,
    block_stop: int,
    group_x: Optional[np.ndarray],
    group_y: Optional[np.ndarray],
    segment_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    if block_stop - block_start <= 1 or group_x is None or group_y is None:
        return (
            np.asarray([block_start], dtype=np.int64),
            np.asarray([block_stop], dtype=np.int64),
        )

    floor = float(segment_cfg.get("distance_floor",
                   segment_cfg.get("min_distance_threshold", 1000.0)))
    multiplier = float(segment_cfg.get("median_multiplier", 5.0))

    x = group_x[block_start:block_stop]
    y = group_y[block_start:block_stop]
    distances = np.hypot(np.diff(x), np.diff(y))
    finite = np.isfinite(distances)
    nonzero = distances[finite & (distances > 0)]
    if nonzero.size == 0:
        return (
            np.asarray([block_start], dtype=np.int64),
            np.asarray([block_stop], dtype=np.int64),
        )
    threshold = max(floor, multiplier * float(np.median(nonzero)))
    jumps = np.flatnonzero(finite & (distances > threshold)).astype(np.int64) + 1

    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            jumps,
            np.asarray([block_stop - block_start], dtype=np.int64),
        )
    )
    starts = (block_start + offsets[:-1]).astype(np.int64)
    stops = (block_start + offsets[1:]).astype(np.int64)
    keep = stops > starts
    return starts[keep], stops[keep]


def _receiver_line_segments(
    segy_file: Any,
    block_start: int,
    block_stop: int,
    segment_cfg: Mapping[str, Any],
    no_segment: bool,
    fname: str = "unknown.sgy",
) -> List[Tuple[int, int]]:
    if no_segment or block_stop - block_start <= 1:
        return [(int(block_start), int(block_stop))]

    line_id_header = segment_cfg.get("line_id_header",
                     segment_cfg.get("primary_header", "INLINE_3D"))
    if line_id_header is not None:
        line_id_header = str(line_id_header).strip()
        if line_id_header.lower() in ("", "none", "null", "false", "off", "disabled"):
            line_id_header = None

    infer_geometry = bool(segment_cfg.get("infer_line_from_geometry",
                           segment_cfg.get("fallback", True)))

    line_ids: Optional[np.ndarray] = None
    group_x: Optional[np.ndarray] = None
    group_y: Optional[np.ndarray] = None

    if line_id_header is not None:
        line_ids = read_line_id_header(segy_file, str(fname), header_name=line_id_header)

    if infer_geometry and line_ids is None:
        try:
            group_x, group_y = read_group_coordinates(segy_file, str(fname))
        except ValueError:
            pass
        else:
            if not group_coordinates_are_usable(group_x, group_y):
                group_x = None
                group_y = None

    if line_ids is not None:
        block_line_ids = line_ids[block_start:block_stop]
        if bool(np.any(block_line_ids != 0)):
            starts, stops = _line_id_segments(block_start, block_stop, line_ids)
            return list(zip(starts.tolist(), stops.tolist()))

    if group_x is not None and group_y is not None:
        starts, stops = _geometry_segments(block_start, block_stop, group_x, group_y, segment_cfg)
        return list(zip(starts.tolist(), stops.tolist()))

    return [(int(block_start), int(block_stop))]


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------

def _normalize_segment(
    segment: np.ndarray,
    normalize_mode: str,
    clip_percentile: Optional[float],
    eps: float,
) -> np.ndarray:
    mode = normalize_mode.lower()
    if mode in ("none", "off", "identity"):
        return segment.astype(np.float32, copy=False)

    x = np.nan_to_num(segment.astype(np.float32, copy=False), copy=False)

    if clip_percentile is not None:
        thresh = float(np.percentile(np.abs(x), float(clip_percentile)))
        if thresh > 0:
            x = np.clip(x, -thresh, thresh)

    if mode == "max_abs":
        denom = max(float(np.abs(x).max()), eps)
        return (x / denom).astype(np.float32, copy=False)
    elif mode == "mean_std":
        mean = float(np.mean(x))
        std = max(float(np.std(x)), eps)
        return ((x - mean) / std).astype(np.float32, copy=False)
    elif mode == "minmax":
        xmin = float(x.min())
        xmax = float(x.max())
        denom = max(xmax - xmin, eps)
        return ((x - xmin) / denom).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown normalize_mode: {mode!r}.")


@torch.no_grad()
def _run_inference_on_segment(
    segment: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_trace: int,
    patch_time: int,
    trace_stride: int,
    time_stride: int,
    batch_size: int,
) -> np.ndarray:
    n_traces, n_time = segment.shape

    pad_h = max(0, patch_trace - n_traces)
    pad_w = max(0, patch_time - n_time)
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(segment, ((0, pad_h), (0, pad_w)), mode="constant")
    else:
        padded = segment
    eff_traces, eff_time = padded.shape

    trace_starts = _gen_uniform_starts(eff_traces, patch_trace, trace_stride)
    time_starts = _gen_uniform_starts(eff_time, patch_time, time_stride)
    n_h, n_w = trace_starts.size, time_starts.size

    windows = sliding_window_view(padded, (patch_trace, patch_time), axis=(0, 1))
    grid = windows[trace_starts[:, None], time_starts[None, :], :, :]
    patches = np.ascontiguousarray(grid.reshape(-1, 1, patch_trace, patch_time))
    n_patches = patches.shape[0]

    prob_patches = np.zeros((n_patches, patch_trace, patch_time), dtype=np.float32)
    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        batch = torch.from_numpy(patches[start:end]).to(device, non_blocking=True)
        logits = model(batch)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 4 and probs.shape[1] == 1:
            probs = probs[:, 0, :, :]
        prob_patches[start:end] = probs.astype(np.float32, copy=False)

    prob_4d = prob_patches[:, None, :, :]
    info = {
        "shape": (1, eff_traces, eff_time),
        "was_2d": False,
        "patch_size": (patch_trace, patch_time),
        "trace_starts": trace_starts,
        "time_starts": time_starts,
        "n_shots": 1,
        "n_per_shot": int(n_h * n_w),
        "output_ndim": 4,
        "mode": "uniform",
    }
    full_prob = unpatchify_uniform(prob_4d, info)
    if full_prob.ndim == 3:
        full_prob = full_prob[0]

    if pad_h > 0 or pad_w > 0:
        full_prob = full_prob[:n_traces, :n_time]

    return full_prob.astype(np.float32, copy=False)


def _read_label_picks(label_path: str, threshold: float) -> Optional[np.ndarray]:
    try:
        with segyio.open(label_path, "r", ignore_geometry=True) as f:
            n_traces = int(f.tracecount)
            picks = np.full(n_traces, np.nan, dtype=np.float64)
            for tr in range(n_traces):
                trace = f.trace[tr].astype(np.float32, copy=False)
                pos = np.flatnonzero(trace >= threshold)
                if pos.size > 0:
                    picks[tr] = float(pos[0])
        return picks
    except Exception as e:
        warnings.warn(f"Failed to read label SEG-Y {label_path}: {e}")
        return None


def process_segy_file(
    sgy_path: str,
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 64,
    no_segment: bool = False,
    return_prob_mask: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    data_cfg = cfg.get("data", {})
    patch_cfg = data_cfg.get("patch", {})
    patch_trace = int(patch_cfg.get("trace", 256))
    patch_time = int(patch_cfg.get("time", 512))
    trace_stride = int(patch_cfg.get("trace_stride", max(1, patch_trace // 2)))
    time_stride = int(patch_cfg.get("time_stride", max(1, patch_time // 2)))
    segment_cfg = data_cfg.get("gather_segment", {})
    if segment_cfg is None:
        segment_cfg = {}
    prep = cfg.get("preprocess", {})
    normalize_mode = str(prep.get("normalize_mode", "max_abs"))
    clip_percentile = prep.get("clip_percentile")
    normalize_eps = float(prep.get("normalize_eps", 1.0e-6))

    with segyio.open(sgy_path, "r", ignore_geometry=True) as segy_file:
        n_traces = int(segy_file.tracecount)
        n_samples = int(len(segy_file.samples))
        ffids = np.asarray(
            segy_file.attributes(segyio.TraceField.FieldRecord)[:], dtype=np.int64
        )
        fname = Path(sgy_path).name
        _unique_ffids, block_starts, block_stops = contiguous_ffid_blocks(ffids, fname)

        picks = np.full((n_traces,), -1, dtype=np.int32)
        prob_mask = np.zeros((n_traces, n_samples), dtype=np.float32) if return_prob_mask else None
        seis_data = np.zeros((n_traces, n_samples), dtype=np.float32) if return_prob_mask else None

        for blk_start, blk_stop in zip(block_starts.tolist(), block_stops.tolist()):
            blk_start = int(blk_start)
            blk_stop = int(blk_stop)

            segments = _receiver_line_segments(
                segy_file, blk_start, blk_stop, segment_cfg, no_segment,
                fname=fname,
            )

            for seg_start, seg_stop in segments:
                seg_traces = segyio.tools.collect(
                    segy_file.trace[seg_start:seg_stop]
                ).astype(np.float32, copy=False)

                if seis_data is not None:
                    seis_data[seg_start:seg_stop] = seg_traces

                seg_norm = _normalize_segment(
                    seg_traces, normalize_mode, clip_percentile, normalize_eps,
                )

                prob = _run_inference_on_segment(
                    seg_norm, model, device,
                    patch_trace, patch_time,
                    trace_stride, time_stride,
                    batch_size,
                )

                if prob_mask is not None:
                    prob_mask[seg_start:seg_stop] = prob

                seg_picks, seg_valid = first_pick_from_mask(prob, threshold=threshold)
                seg_len = seg_stop - seg_start
                picks[seg_start:seg_stop] = np.where(
                    seg_valid[:seg_len],
                    seg_picks[:seg_len].astype(np.int32),
                    -1,
                )

    return picks, prob_mask, seis_data


# ---------------------------------------------------------------------------
# SEG-Y output
# ---------------------------------------------------------------------------

def _write_segy(path: str, data: np.ndarray, sample_interval_us: int = 1000) -> None:
    L, ns = data.shape
    data = np.ascontiguousarray(data, dtype=np.float32)

    spec = segyio.spec()
    spec.sorting = 1
    spec.format = 5        # IEEE float32
    spec.samples = range(ns)
    spec.tracecount = L

    with segyio.create(path, spec) as dst:
        dst.bin[segyio.BinField.Interval] = sample_interval_us
        for tr in range(L):
            dst.header[tr][segyio.TraceField.TRACE_SEQUENCE_LINE] = tr + 1
            dst.trace[tr] = data[tr]


# ---------------------------------------------------------------------------
# pred_picks.txt
# ---------------------------------------------------------------------------

def _save_picks_txt(
    path: str,
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    sup: Optional[np.ndarray],
) -> None:
    n = len(pred)
    with open(path, "w") as f:
        f.write("# trace_idx  pred_pick  gt_pick  supervised\n")
        for i in range(n):
            p_val = pred[i]
            p_str = f"{int(p_val)}" if p_val >= 0 else "nan"

            if gt is not None and sup is not None:
                g_val = gt[i]
                s_val = sup[i]
                g_str = f"{int(g_val)}" if np.isfinite(g_val) else "nan"
                s_str = f"{int(s_val)}"
            else:
                g_str = "nan"
                s_str = "0"

            f.write(f"{i}  {p_str}  {g_str}  {s_str}\n")

    n_valid = int((pred >= 0).sum())
    print(f"  pred_picks.txt saved to {path}  ({n_valid}/{n} traces with picks)")


def save_results(
    output_dir: str,
    picks: np.ndarray,
    prob_mask: Optional[np.ndarray],
    seis_data: Optional[np.ndarray],
    gt_picks: Optional[np.ndarray] = None,
    gt_supervised: Optional[np.ndarray] = None,
    sample_interval_us: int = 1000,
    skip_seis_sgy: bool = False,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    picks_txt = out / "pred_picks.txt"
    _save_picks_txt(str(picks_txt), picks, gt_picks, gt_supervised)

    if prob_mask is not None:
        mask_path = out / "prob_mask.sgy"
        _write_segy(str(mask_path), prob_mask, sample_interval_us)
        print(f"  prob_mask.sgy saved to {mask_path}  "
              f"({prob_mask.shape[0]} traces x {prob_mask.shape[1]} samples)")

    if seis_data is not None and not skip_seis_sgy:
        seis_path = out / "seismic.sgy"
        _write_segy(str(seis_path), seis_data, sample_interval_us)
        print(f"  seismic.sgy saved to {seis_path}  "
              f"({seis_data.shape[0]} traces x {seis_data.shape[1]} samples)")


# ---------------------------------------------------------------------------
# Metrics  (aligned with test/inference.py  compute_pick_metrics)
# ---------------------------------------------------------------------------

_HR_THRESHOLDS = [1, 3, 5, 7, 9]


def _compute_pick_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    sup: np.ndarray,
) -> Dict[str, float]:
    """Compute first-break picking metrics.

    Parameters
    ----------
    pred : (N,) float64 — predicted sample indices (nan where no pick).
    gt   : (N,) float64 — ground-truth sample indices (nan where no supervised label).
    sup  : (N,) float64 — 1.0 for supervised traces, 0.0 otherwise.

    Returns
    -------
    dict with keys RMSE, MAE, MBE, HR@1px, HR@3px, HR@5px, HR@7px, HR@9px, n_eval.
    """
    valid = sup > 0.5
    n_eval = int(valid.sum())
    if n_eval == 0:
        return {
            "RMSE": float("nan"), "MAE": float("nan"), "MBE": float("nan"),
            "HR@1px": float("nan"), "HR@3px": float("nan"), "HR@5px": float("nan"),
            "HR@7px": float("nan"), "HR@9px": float("nan"), "n_eval": 0,
        }

    gt_v = gt[valid]
    pred_v = pred[valid]
    pred_finite = np.isfinite(pred_v)
    if not pred_finite.any():
        return {
            "RMSE": float("nan"), "MAE": float("nan"), "MBE": float("nan"),
            "HR@1px": 0.0, "HR@3px": 0.0, "HR@5px": 0.0,
            "HR@7px": 0.0, "HR@9px": 0.0, "n_eval": n_eval,
        }

    gt_v = gt_v[pred_finite]
    pred_v = pred_v[pred_finite]
    err = pred_v - gt_v

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(err))

    abs_err = np.abs(err)
    hit_rates: Dict[str, float] = {}
    for t in _HR_THRESHOLDS:
        hit_rates[f"HR@{t}px"] = float(np.mean((abs_err <= t).astype(np.float64)))

    return {
        "RMSE": rmse, "MAE": mae, "MBE": mbe,
        **hit_rates, "n_eval": n_eval,
    }


# ---------------------------------------------------------------------------
# Batch discovery
# ---------------------------------------------------------------------------

def _discover_segy_pairs(
    root: str,
    pattern: str,
    data_name: str,
    label_name: str,
) -> List[Tuple[str, str, str]]:
    """Yield ``(run_name, data_sgy_path, label_sgy_path)`` for each subdir.

    A subdir is included when it contains both ``data_name`` and ``label_name``.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root_path}")

    pairs: List[Tuple[str, str, str]] = []
    for subdir in sorted(root_path.glob(pattern)):
        if not subdir.is_dir():
            continue
        data_path = subdir / data_name
        label_path = subdir / label_name
        if data_path.is_file() and label_path.is_file():
            pairs.append((subdir.name, str(data_path), str(label_path)))
    return pairs


# ---------------------------------------------------------------------------
# XLSX / CSV output
# ---------------------------------------------------------------------------

def _write_xlsx(rows: List[Dict[str, Any]], path: str) -> None:
    try:
        import openpyxl
    except ImportError:
        print("openpyxl not installed, falling back to CSV")
        _write_csv(rows, path.replace(".xlsx", ".csv"))
        return

    wb = openpyxl.Workbook()

    # Per-run sheet
    ws1 = wb.active
    ws1.title = "Per Run"
    headers = ["run", "n_eval",
               "RMSE", "MAE", "MBE",
               "HR@1px", "HR@3px", "HR@5px", "HR@7px", "HR@9px"]
    ws1.append(headers)
    for r in rows:
        ws1.append([r.get(h, "") for h in headers])

    # Summary sheet
    ws2 = wb.create_sheet("Summary")
    ws2.append(["metric", "mean", "std", "min", "max"])
    metric_keys = ["RMSE", "MAE", "MBE",
                   "HR@1px", "HR@3px", "HR@5px", "HR@7px", "HR@9px"]
    for key in metric_keys:
        vals = [r.get(key, float("nan")) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            ws2.append([key, round(np.mean(vals), 4), round(np.std(vals), 4),
                        round(np.min(vals), 4), round(np.max(vals), 4)])

    wb.save(path)
    print(f"Batch summary saved to {path}")


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = ["run", "n_eval",
            "RMSE", "MAE", "MBE",
            "HR@1px", "HR@3px", "HR@5px", "HR@7px", "HR@9px"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Batch summary saved to {path}")


# ---------------------------------------------------------------------------
# Single-file evaluation entry point
# ---------------------------------------------------------------------------

def _run_single_evaluation(args: argparse.Namespace, cfg: Mapping[str, Any],
                           device: torch.device, model: torch.nn.Module) -> None:
    if args.output_dir is None:
        input_stem = Path(args.input).stem
        output_dir = f"{input_stem}_eval"
    else:
        output_dir = args.output_dir

    gt_picks: Optional[np.ndarray] = None
    gt_supervised: Optional[np.ndarray] = None
    if args.label_sgy is not None:
        print(f"Reading label SEG-Y: {args.label_sgy}")
        gt_picks = _read_label_picks(args.label_sgy, threshold=0.5)
        if gt_picks is not None:
            gt_supervised = np.isfinite(gt_picks).astype(np.float64)

    sample_interval_us = 1000
    try:
        with segyio.open(args.input, "r", ignore_geometry=True) as f:
            sample_interval_us = int(f.bin[segyio.BinField.Interval])
    except Exception:
        pass

    print(f"Processing SEG-Y: {args.input}")
    picks, prob_mask, seis_data = process_segy_file(
        sgy_path=args.input,
        model=model,
        cfg=cfg,
        device=device,
        threshold=args.threshold,
        batch_size=args.batch_size,
        no_segment=args.no_segment,
        return_prob_mask=True,
    )

    save_results(
        output_dir=output_dir,
        picks=picks,
        prob_mask=prob_mask,
        seis_data=seis_data,
        gt_picks=gt_picks,
        gt_supervised=gt_supervised,
        sample_interval_us=sample_interval_us,
        skip_seis_sgy=args.no_seis_sgy,
    )

    # Print metrics if ground-truth is available
    if gt_picks is not None and gt_supervised is not None:
        pred_f64 = np.where(picks >= 0, picks.astype(np.float64), np.nan)
        metrics = _compute_pick_metrics(pred_f64, gt_picks, gt_supervised)
        print(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
              f"HR@1px={metrics['HR@1px']:.4f}  HR@5px={metrics['HR@5px']:.4f}  "
              f"n_eval={metrics['n_eval']}")


# ---------------------------------------------------------------------------
# Batch evaluation entry point
# ---------------------------------------------------------------------------

def _run_batch_evaluation(args: argparse.Namespace, cfg: Mapping[str, Any],
                          device: torch.device, model: torch.nn.Module) -> None:
    pairs = _discover_segy_pairs(
        root=args.root,
        pattern=args.pattern,
        data_name=args.data_name,
        label_name=args.label_name,
    )
    if not pairs:
        print(f"No {args.data_name}/{args.label_name} pairs found under {args.root}")
        return

    print(f"Found {len(pairs)} SEG-Y pairs in {args.root}:")
    for name, data_path, label_path in pairs:
        print(f"  {name}  data={data_path}  label={label_path}")

    output_xlsx = args.output_xlsx or str(Path(args.root) / "batch_eval.xlsx")
    all_metrics: List[Dict[str, Any]] = []

    for run_name, data_path, label_path in pairs:
        print(f"\n{'=' * 60}")
        print(f"[{run_name}]")

        # Output dir: <root>/<run_name>/
        output_dir = str(Path(args.root) / run_name)

        try:
            gt_picks = _read_label_picks(label_path, threshold=0.5)
            if gt_picks is None:
                print(f"  SKIP: could not read label {label_path}")
                continue
            gt_supervised = np.isfinite(gt_picks).astype(np.float64)

            sample_interval_us = 1000
            try:
                with segyio.open(data_path, "r", ignore_geometry=True) as f:
                    sample_interval_us = int(f.bin[segyio.BinField.Interval])
            except Exception:
                pass

            picks, prob_mask, seis_data = process_segy_file(
                sgy_path=data_path,
                model=model,
                cfg=cfg,
                device=device,
                threshold=args.threshold,
                batch_size=args.batch_size,
                no_segment=args.no_segment,
                return_prob_mask=True,
            )

            save_results(
                output_dir=output_dir,
                picks=picks,
                prob_mask=prob_mask,
                seis_data=seis_data,
                gt_picks=gt_picks,
                gt_supervised=gt_supervised,
                sample_interval_us=sample_interval_us,
                skip_seis_sgy=args.no_seis_sgy,
            )

            pred_f64 = np.where(picks >= 0, picks.astype(np.float64), np.nan)
            metrics = _compute_pick_metrics(pred_f64, gt_picks, gt_supervised)
            metrics["run"] = run_name
            all_metrics.append(metrics)

            print(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
                  f"HR@1px={metrics['HR@1px']:.4f}  HR@5px={metrics['HR@5px']:.4f}  "
                  f"n_eval={metrics['n_eval']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_metrics:
        print("\nNo successful evaluations.")
        return

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary ({len(all_metrics)} runs):")
    metric_keys = ["RMSE", "MAE", "MBE",
                   "HR@1px", "HR@3px", "HR@5px", "HR@7px", "HR@9px"]
    for key in metric_keys:
        vals = [m[key] for m in all_metrics if np.isfinite(m.get(key, float("nan")))]
        if vals:
            print(f"  {key:<10} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    _write_xlsx(all_metrics, output_xlsx)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evaluation(script_file: str) -> None:
    args = parse_eval_args(script_file)
    cfg = load_config(args.config)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = _build_model_from_checkpoint(args.checkpoint, cfg, device)

    if args.root is not None:
        _run_batch_evaluation(args, cfg, device, model)
    elif args.input is not None:
        _run_single_evaluation(args, cfg, device, model)
    else:
        print("ERROR: specify either --input (single-file) or --root (batch mode).")
        sys.exit(1)
