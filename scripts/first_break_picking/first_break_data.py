"""SEG-Y first-break-picking dataset built from data/mask pairs.

The label files in ``segy_with_masks`` are binary step masks: values before the
first break are 0 and values from the first break onward are 1. This module
keeps split boundaries at the FFID/shot level, then slices patches only inside
one continuous receiver-line segment within that FFID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tools.patching import _gen_uniform_starts
from tools.preprocessing import normalize
from tools.segy_read import (
    contiguous_ffid_blocks,
    group_coordinates_are_usable,
    read_group_coordinates,
    read_line_id_header,
)
from utils.datasets import as_path, cap_split_samples, split_block_indices
from utils.train_utils import compute_length_stats, format_length_stats

try:
    import segyio
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise ImportError("segyio is required for first-break SEG-Y datasets.") from exc


_SEGY_SUFFIXES = (".sgy", ".segy")
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SegyPairInfo:
    """Metadata for one data/mask SEG-Y pair."""

    name: str
    data_path: str
    label_path: str
    n_traces: int
    n_samples: int
    sample_interval_us: int
    n_ffids: int
    traces_per_ffid_min: int
    traces_per_ffid_max: int
    n_segments: int
    segment_source: str
    segment_length_min: int
    segment_length_p10: float
    segment_length_median: float
    segment_length_p90: float
    segment_length_max: int


@dataclass(frozen=True)
class PatchRef:
    """Lazy patch reference into one receiver-line segment inside an FFID."""

    pair_idx: int
    ffid: int
    ffid_start: int
    ffid_stop: int
    segment_idx: int
    segment_source: str
    segment_start: int
    segment_stop: int
    trace_start: int
    time_start: int


@dataclass(frozen=True)
class LabelSummary:
    """Read-only sampled label-format summary."""

    name: str
    checked_traces: int
    binary: bool
    step_fraction: float
    positive_trace_fraction: float


@dataclass(frozen=True)
class FirstBreakIndex:
    """Shared immutable index used by train/val/test datasets."""

    pairs: Tuple[SegyPairInfo, ...]
    patches_by_split: Mapping[str, Tuple[PatchRef, ...]]
    patch_shape: Tuple[int, int]
    label_summaries: Tuple[LabelSummary, ...]
    segment_lengths: Tuple[int, ...]
    pick_indices_by_pair: Tuple[np.ndarray, ...]


def _iter_data_files(data_dir: Path, files: Optional[Sequence[str]]) -> List[Path]:
    if files:
        return [as_path(data_dir, name) for name in files]
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SEGY_SUFFIXES
    )


def _resolve_label_path(data_path: Path, label_dir: Path) -> Path:
    candidates = [
        label_dir / f"{data_path.stem}_mask{data_path.suffix}",
        label_dir / f"{data_path.stem}_mask.sgy",
        label_dir / f"{data_path.stem}_mask.segy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(label_dir.glob(f"{data_path.stem}*_mask.*"))
    matches = [p for p in matches if p.suffix.lower() in _SEGY_SUFFIXES]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple mask files match {data_path.name}: "
            f"{[p.name for p in matches]}."
        )
    raise FileNotFoundError(
        f"No mask file found for {data_path.name} under {label_dir}."
    )


def _label_path_overrides(
    data_cfg: Mapping[str, Any],
    root: Path,
) -> Dict[Path, Path]:
    """Resolve optional explicit data-to-label SEG-Y path pairs."""
    raw_overrides = data_cfg.get("label_path_overrides", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("data.label_path_overrides must be a list when provided.")

    overrides: Dict[Path, Path] = {}
    for idx, raw in enumerate(raw_overrides):
        if not isinstance(raw, Mapping) or "data_path" not in raw or "label_path" not in raw:
            raise ValueError(
                f"data.label_path_overrides[{idx}] must contain data_path and label_path."
            )
        data_path = as_path(root, str(raw["data_path"])).resolve()
        label_path = as_path(root, str(raw["label_path"])).resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"Override data SEG-Y not found: {data_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Override label SEG-Y not found: {label_path}")
        if data_path in overrides:
            raise ValueError(f"Duplicate label override for data SEG-Y: {data_path}")
        overrides[data_path] = label_path
    return overrides




def _line_id_header_from_cfg(segment_cfg: Mapping[str, Any]) -> Optional[str]:
    if "primary_header" in segment_cfg:
        warnings.warn(
            "data.gather_segment.primary_header is deprecated; use "
            "data.gather_segment.line_id_header instead.",
            RuntimeWarning,
        )
    raw = segment_cfg.get("line_id_header", segment_cfg.get("primary_header", "INLINE_3D"))
    if raw is None:
        return None
    header = str(raw).strip()
    if header.lower() in ("", "none", "null", "false", "off", "disabled"):
        return None
    if header.upper() == "INLINE":
        return "INLINE_3D"
    return header


def _infer_line_from_geometry(segment_cfg: Mapping[str, Any]) -> bool:
    if "fallback" in segment_cfg:
        warnings.warn(
            "data.gather_segment.fallback is deprecated; use "
            "data.gather_segment.infer_line_from_geometry instead.",
            RuntimeWarning,
        )
    if "infer_line_from_geometry" in segment_cfg:
        return bool(segment_cfg["infer_line_from_geometry"])
    fallback = segment_cfg.get("fallback", True)
    if isinstance(fallback, str):
        return fallback.strip().lower() not in (
            "",
            "none",
            "null",
            "false",
            "off",
            "disabled",
        )
    return bool(fallback)



def _line_id_segments(
    *,
    block_start: int,
    block_stop: int,
    line_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split one FFID block whenever receiver-line id changes."""
    if block_stop <= block_start:
        raise ValueError("FFID block must be non-empty.")
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
    *,
    block_start: int,
    block_stop: int,
    group_x: Optional[np.ndarray],
    group_y: Optional[np.ndarray],
    segment_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Infer receiver-line segments from large GroupX/GroupY neighbor-distance jumps."""
    if block_stop <= block_start:
        raise ValueError("FFID block must be non-empty.")
    enabled = bool(segment_cfg.get("enabled", True))
    if not enabled or block_stop - block_start <= 1:
        return (
            np.asarray([block_start], dtype=np.int64),
            np.asarray([block_stop], dtype=np.int64),
            float("inf"),
        )
    if group_x is None or group_y is None:
        raise ValueError("GroupX/GroupY coordinates are required when gather_segment is enabled.")

    floor = float(
        segment_cfg.get(
            "distance_floor",
            segment_cfg.get("min_distance_threshold", 1000.0),
        )
    )
    multiplier = float(segment_cfg.get("median_multiplier", 5.0))
    if floor < 0:
        raise ValueError(f"gather_segment.distance_floor must be >= 0, got {floor}.")
    if multiplier <= 0:
        raise ValueError(f"gather_segment.median_multiplier must be > 0, got {multiplier}.")

    x = group_x[block_start:block_stop]
    y = group_y[block_start:block_stop]
    distances = np.hypot(np.diff(x), np.diff(y))
    finite = np.isfinite(distances)
    nonzero = distances[finite & (distances > 0)]
    if nonzero.size == 0:
        threshold = float("inf")
        jumps = np.empty((0,), dtype=np.int64)
    else:
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
    return starts[keep], stops[keep], threshold


def _receiver_line_segments(
    *,
    block_start: int,
    block_stop: int,
    line_ids: Optional[np.ndarray],
    line_id_header: Optional[str],
    group_x: Optional[np.ndarray],
    group_y: Optional[np.ndarray],
    segment_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Split one FFID into hardpicks-style shot x receiver-line gathers."""
    enabled = bool(segment_cfg.get("enabled", True))
    if not enabled or block_stop - block_start <= 1:
        return (
            np.asarray([block_start], dtype=np.int64),
            np.asarray([block_stop], dtype=np.int64),
            "ffid",
        )

    if line_ids is not None:
        block_line_ids = line_ids[block_start:block_stop]
        if bool(np.any(block_line_ids != 0)):
            starts, stops = _line_id_segments(
                block_start=block_start,
                block_stop=block_stop,
                line_ids=line_ids,
            )
            return starts, stops, f"line_id:{line_id_header or 'header'}"

    if group_x is not None and group_y is not None:
        starts, stops, _ = _geometry_segments(
            block_start=block_start,
            block_stop=block_stop,
            group_x=group_x,
            group_y=group_y,
            segment_cfg=segment_cfg,
        )
        return starts, stops, "geometry:GroupX/GroupY"

    return (
        np.asarray([block_start], dtype=np.int64),
        np.asarray([block_stop], dtype=np.int64),
        "ffid",
    )



def _sample_label_summary(
    label_file: Any,
    label_path: Path,
    *,
    max_traces: int,
    threshold: float,
) -> LabelSummary:
    n_traces = int(label_file.tracecount)
    n_samples = int(len(label_file.samples))
    if max_traces <= 0:
        return LabelSummary(label_path.name, 0, True, 1.0, 0.0)

    count = min(max_traces, n_traces)
    indices = np.unique(np.linspace(0, n_traces - 1, count, dtype=np.int64))
    labels = np.stack([label_file.trace[int(i)] for i in indices]).astype(np.float32, copy=False)
    binary = bool(np.all(np.isclose(labels, 0.0) | np.isclose(labels, 1.0)))

    mask = labels > threshold
    transitions = np.abs(np.diff(mask.astype(np.int8), axis=1)).sum(axis=1)
    positives = mask.any(axis=1)
    last_positive = np.where(positives, n_samples - 1 - np.argmax(mask[:, ::-1], axis=1), -1)
    positive_step = ((transitions == 1) | (transitions == 0)) & positives & (
        last_positive == n_samples - 1
    )
    valid_step = positive_step | ((transitions == 0) & ~positives)
    step_fraction = float(valid_step.mean()) if valid_step.size else 1.0
    positive_fraction = float(positives.mean()) if positives.size else 0.0
    return LabelSummary(
        name=label_path.name,
        checked_traces=int(indices.size),
        binary=binary,
        step_fraction=step_fraction,
        positive_trace_fraction=positive_fraction,
    )


def _read_first_break_pick_indices(
    label_file: Any,
    label_path: Path,
    *,
    threshold: float,
    chunk_traces: int,
) -> np.ndarray:
    """Read label SEG-Y once and return first positive sample per trace, or -1."""
    n_traces = int(label_file.tracecount)
    if chunk_traces <= 0:
        raise ValueError(
            f"label_pick_chunk_traces must be positive for {label_path.name}, got {chunk_traces}."
        )

    picks = np.full((n_traces,), -1, dtype=np.int32)
    for start in range(0, n_traces, chunk_traces):
        stop = min(start + chunk_traces, n_traces)
        labels = segyio.tools.collect(label_file.trace[start:stop]).astype(
            np.float32,
            copy=False,
        )
        positive = labels > threshold
        has_pick = positive.any(axis=1)
        if bool(has_pick.any()):
            chunk_picks = positive.argmax(axis=1).astype(np.int32, copy=False)
            picks[start:stop] = np.where(has_pick, chunk_picks, -1).astype(
                np.int32,
                copy=False,
            )
    return picks




def build_first_break_index(cfg: Mapping[str, Any]) -> FirstBreakIndex:
    """Build the full lazy patch index from a YAML config mapping."""
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"]).expanduser()
    data_dir = as_path(root, str(data_cfg.get("data_dir", "data")))
    label_dir = as_path(root, str(data_cfg.get("label_dir", "label")))
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    patch_cfg = data_cfg.get("patch", {})
    patch_trace = int(patch_cfg.get("trace", 256))
    patch_time = int(patch_cfg.get("time", 512))
    stride_trace = int(patch_cfg.get("trace_stride", max(1, patch_trace // 2)))
    stride_time = int(patch_cfg.get("time_stride", max(1, patch_time // 2)))
    label_threshold = float(data_cfg.get("label_threshold", 0.5))
    validate_labels = bool(data_cfg.get("validate_labels", True))
    max_label_checks = int(data_cfg.get("label_check_traces", 2048))
    label_pick_chunk_traces = int(data_cfg.get("label_pick_chunk_traces", 8192))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    segment_cfg = data_cfg.get("gather_segment", {})
    if segment_cfg is None:
        segment_cfg = {}
    if not isinstance(segment_cfg, Mapping):
        raise ValueError("data.gather_segment must be a mapping when provided.")
    segment_enabled = bool(segment_cfg.get("enabled", True))
    line_id_header = _line_id_header_from_cfg(segment_cfg)
    infer_line_from_geometry = _infer_line_from_geometry(segment_cfg)

    pair_infos: List[SegyPairInfo] = []
    label_summaries: List[LabelSummary] = []
    pick_indices_by_pair: List[np.ndarray] = []
    patches_by_split: Dict[str, List[PatchRef]] = {split: [] for split in _SPLITS}
    all_segment_lengths: List[int] = []

    data_files = _iter_data_files(data_dir, data_cfg.get("files"))
    if not data_files:
        raise FileNotFoundError(f"No SEG-Y files found under {data_dir}.")
    label_overrides = _label_path_overrides(data_cfg, root)

    for pair_idx, data_path in enumerate(data_files):
        if not data_path.exists():
            raise FileNotFoundError(f"Data SEG-Y not found: {data_path}")
        label_path = label_overrides.get(data_path.resolve())
        if label_path is None:
            label_path = _resolve_label_path(data_path, label_dir)

        with segyio.open(str(data_path), "r", ignore_geometry=True) as data_file, segyio.open(
            str(label_path), "r", ignore_geometry=True
        ) as label_file:
            n_traces = int(data_file.tracecount)
            n_samples = int(len(data_file.samples))
            sample_interval = int(data_file.bin[segyio.BinField.Interval])

            if int(label_file.tracecount) != n_traces:
                raise ValueError(
                    f"{data_path.name}/{label_path.name}: trace count mismatch "
                    f"{n_traces} vs {label_file.tracecount}."
                )
            if int(len(label_file.samples)) != n_samples:
                raise ValueError(
                    f"{data_path.name}/{label_path.name}: sample count mismatch "
                    f"{n_samples} vs {len(label_file.samples)}."
                )
            if int(label_file.bin[segyio.BinField.Interval]) != sample_interval:
                raise ValueError(
                    f"{data_path.name}/{label_path.name}: sample interval mismatch."
                )
            if not np.array_equal(np.asarray(data_file.samples), np.asarray(label_file.samples)):
                raise ValueError(f"{data_path.name}/{label_path.name}: sample axes differ.")

            data_ffids = np.asarray(
                data_file.attributes(segyio.TraceField.FieldRecord)[:], dtype=np.int64
            )
            label_ffids = np.asarray(
                label_file.attributes(segyio.TraceField.FieldRecord)[:], dtype=np.int64
            )
            if not np.array_equal(data_ffids, label_ffids):
                raise ValueError(f"{data_path.name}/{label_path.name}: FFID headers differ.")

            ffids, starts, stops = contiguous_ffid_blocks(data_ffids, data_path.name)
            counts = stops - starts
            line_ids: Optional[np.ndarray] = None
            group_x: Optional[np.ndarray] = None
            group_y: Optional[np.ndarray] = None
            if segment_enabled:
                if line_id_header is not None:
                    line_ids = read_line_id_header(
                        data_file,
                        data_path.name,
                        header_name=line_id_header,
                    )
                if infer_line_from_geometry:
                    try:
                        group_x, group_y = read_group_coordinates(data_file, data_path.name)
                    except ValueError as exc:
                        warnings.warn(str(exc), RuntimeWarning)
                    else:
                        if not group_coordinates_are_usable(group_x, group_y):
                            warnings.warn(
                                f"{data_path.name}: GroupX/GroupY headers have no usable "
                                "neighbor-distance variation; falling back to one segment per FFID.",
                                RuntimeWarning,
                            )
                            group_x = None
                            group_y = None
                if line_ids is None and group_x is None:
                    warnings.warn(
                        f"{data_path.name}: receiver-line segmentation could not use "
                        "a line-id header or inferred GroupX/GroupY geometry; "
                        "falling back to one segment per FFID.",
                        RuntimeWarning,
                    )

            if validate_labels:
                summary = _sample_label_summary(
                    label_file,
                    label_path,
                    max_traces=max_label_checks,
                    threshold=label_threshold,
                )
                label_summaries.append(summary)
                if not summary.binary:
                    raise ValueError(f"{label_path.name}: label sample contains values outside 0/1.")
                if summary.step_fraction < 0.99:
                    warnings.warn(
                        f"{label_path.name}: only {summary.step_fraction:.3f} of sampled "
                        "label traces look like first-break step masks.",
                        RuntimeWarning,
                    )

            pick_indices_by_pair.append(
                _read_first_break_pick_indices(
                    label_file,
                    label_path,
                    threshold=label_threshold,
                    chunk_traces=label_pick_chunk_traces,
                )
            )

        segments_by_block: List[List[Tuple[int, str, int, int]]] = []
        pair_segment_lengths: List[int] = []
        pair_segment_sources: List[str] = []
        for block_idx in range(int(ffids.size)):
            segment_starts, segment_stops, segment_source = _receiver_line_segments(
                block_start=int(starts[block_idx]),
                block_stop=int(stops[block_idx]),
                line_ids=line_ids,
                line_id_header=line_id_header,
                group_x=group_x,
                group_y=group_y,
                segment_cfg=segment_cfg,
            )
            block_segments: List[Tuple[int, str, int, int]] = []
            for segment_start, segment_stop in zip(segment_starts.tolist(), segment_stops.tolist()):
                segment_idx = len(pair_segment_lengths)
                segment_start = int(segment_start)
                segment_stop = int(segment_stop)
                block_segments.append((segment_idx, segment_source, segment_start, segment_stop))
                pair_segment_lengths.append(segment_stop - segment_start)
                pair_segment_sources.append(segment_source)
            segments_by_block.append(block_segments)
        all_segment_lengths.extend(pair_segment_lengths)
        unique_sources = sorted(set(pair_segment_sources))
        pair_segment_source = "+".join(unique_sources) if unique_sources else "n/a"
        (
            segment_length_min,
            segment_length_p10,
            segment_length_median,
            segment_length_p90,
            segment_length_max,
        ) = compute_length_stats(pair_segment_lengths)

        pair_infos.append(
            SegyPairInfo(
                name=data_path.stem,
                data_path=str(data_path),
                label_path=str(label_path),
                n_traces=n_traces,
                n_samples=n_samples,
                sample_interval_us=sample_interval,
                n_ffids=int(ffids.size),
                traces_per_ffid_min=int(counts.min()),
                traces_per_ffid_max=int(counts.max()),
                n_segments=len(pair_segment_lengths),
                segment_source=pair_segment_source,
                segment_length_min=segment_length_min,
                segment_length_p10=segment_length_p10,
                segment_length_median=segment_length_median,
                segment_length_p90=segment_length_p90,
                segment_length_max=segment_length_max,
            )
        )

        split_blocks = split_block_indices(
            int(ffids.size),
            data_cfg.get("split", {}),
            seed=seed,
            block_id_offset=pair_idx,
        )
        time_starts = _gen_uniform_starts(n_samples, patch_time, stride_time)
        for split, block_indices in split_blocks.items():
            for block_idx in block_indices.tolist():
                ffid_start = int(starts[block_idx])
                ffid_stop = int(stops[block_idx])
                for segment_idx, segment_source, segment_start, segment_stop in segments_by_block[int(block_idx)]:
                    segment_len = segment_stop - segment_start
                    trace_starts = _gen_uniform_starts(segment_len, patch_trace, stride_trace)
                    for trace_start in trace_starts.tolist():
                        for time_start in time_starts.tolist():
                            patches_by_split[split].append(
                                PatchRef(
                                    pair_idx=pair_idx,
                                    ffid=int(ffids[block_idx]),
                                    ffid_start=ffid_start,
                                    ffid_stop=ffid_stop,
                                    segment_idx=int(segment_idx),
                                    segment_source=segment_source,
                                    segment_start=int(segment_start),
                                    segment_stop=int(segment_stop),
                                    trace_start=int(trace_start),
                                    time_start=int(time_start),
                                )
                            )

    patches_by_split = cap_split_samples(
        patches_by_split,
        data_cfg.get("max_patches_per_split"),
        seed=seed,
    )

    frozen_patches = {
        split: tuple(patches_by_split.get(split, ()))
        for split in _SPLITS
    }
    for split, refs in frozen_patches.items():
        if not refs:
            raise ValueError(f"No patches generated for split {split!r}.")

    return FirstBreakIndex(
        pairs=tuple(pair_infos),
        patches_by_split=frozen_patches,
        patch_shape=(patch_trace, patch_time),
        label_summaries=tuple(label_summaries),
        segment_lengths=tuple(all_segment_lengths),
        pick_indices_by_pair=tuple(pick_indices_by_pair),
    )


def summarize_first_break_index(index: FirstBreakIndex) -> str:
    """Return a compact human-readable index summary."""
    total_ffids = sum(pair.n_ffids for pair in index.pairs)
    lines = [
        "First-break index:",
        f"  ffids={total_ffids}, receiver_line_segments={len(index.segment_lengths)}",
        f"  segment_length={format_length_stats(index.segment_lengths)}",
        "  patches: " + ", ".join(
            f"{split}={len(index.patches_by_split[split])}" for split in _SPLITS
        ),
        f"  patch_shape(trace,time)={index.patch_shape}",
    ]
    for pair in index.pairs:
        lines.append(
            "  pair "
            f"{pair.name}: traces={pair.n_traces}, samples={pair.n_samples}, "
            f"ffids={pair.n_ffids}, traces_per_ffid="
            f"{pair.traces_per_ffid_min}-{pair.traces_per_ffid_max}, "
            f"segments={pair.n_segments}, segment_source={pair.segment_source}, "
            f"segment_length="
            f"min={pair.segment_length_min}, p10={pair.segment_length_p10:.1f}, "
            f"median={pair.segment_length_median:.1f}, p90={pair.segment_length_p90:.1f}, "
            f"max={pair.segment_length_max}"
        )
    for summary in index.label_summaries:
        lines.append(
            "  label "
            f"{summary.name}: checked={summary.checked_traces}, "
            f"binary={summary.binary}, step_fraction={summary.step_fraction:.3f}, "
            f"positive_trace_fraction={summary.positive_trace_fraction:.3f}"
        )
    return "\n".join(lines)


class FirstBreakSegyPatchDataset(Dataset):
    """Lazy SEG-Y patch dataset returning ``(data, mask, target_pick)`` tensors."""

    def __init__(
        self,
        index: FirstBreakIndex,
        split: str,
        *,
        normalize_mode: str = "max_abs",
        normalize_scope: str = "gather",
        clip_percentile: Optional[float] = None,
        normalize_eps: float = 1.0e-6,
        label_threshold: float = 0.5,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {_SPLITS}.")
        self.index = index
        self.split = split
        self.pairs = index.pairs
        self.refs = index.patches_by_split[split]
        if len(index.pick_indices_by_pair) != len(index.pairs):
            raise ValueError(
                "FirstBreakIndex pick cache count does not match SEG-Y pair count: "
                f"{len(index.pick_indices_by_pair)} vs {len(index.pairs)}."
            )
        for pair, picks in zip(index.pairs, index.pick_indices_by_pair):
            if picks.ndim != 1 or int(picks.shape[0]) != pair.n_traces:
                raise ValueError(
                    f"{pair.name}: pick cache shape {tuple(picks.shape)} does not "
                    f"match trace count {pair.n_traces}."
                )
        self.patch_trace, self.patch_time = index.patch_shape
        self.normalize_mode = str(normalize_mode)
        self.normalize_scope = str(normalize_scope)
        self.clip_percentile = clip_percentile
        self.normalize_eps = float(normalize_eps)
        self.label_threshold = float(label_threshold)
        self._handles: Dict[int, Any] = {}
        self._segment_norm_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_segment_norm_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.refs)

    def describe_ref(self, idx: int) -> str:
        ref = self.refs[idx]
        pair = self.pairs[ref.pair_idx]
        return (
            f"file={pair.name}, FFID={ref.ffid}, segment={ref.segment_idx}, "
            f"source={ref.segment_source}, "
            f"segment_traces={ref.segment_start}:{ref.segment_stop}, "
            f"patch_trace_start={ref.trace_start}, time_start={ref.time_start}"
        )

    def _get_handle(self, pair_idx: int) -> Any:
        handle = self._handles.get(pair_idx)
        if handle is not None:
            return handle
        pair = self.pairs[pair_idx]
        data_file = segyio.open(pair.data_path, "r", ignore_geometry=True)
        self._handles[pair_idx] = data_file
        return data_file

    def close(self) -> None:
        for data_file in self._handles.values():
            data_file.close()
        self._handles.clear()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _patch_bounds(self, segy_file: Any, ref: PatchRef) -> Tuple[int, int, int, int]:
        abs_trace_start = ref.segment_start + ref.trace_start
        abs_trace_stop = min(abs_trace_start + self.patch_trace, ref.segment_stop)
        time_stop = min(ref.time_start + self.patch_time, len(segy_file.samples))
        valid_trace_count = max(0, abs_trace_stop - abs_trace_start)
        valid_time_count = max(0, time_stop - ref.time_start)
        return abs_trace_start, abs_trace_stop, valid_trace_count, valid_time_count

    def _read_patch(self, segy_file: Any, ref: PatchRef) -> np.ndarray:
        out = np.zeros((self.patch_trace, self.patch_time), dtype=np.float32)
        abs_trace_start, abs_trace_stop, _, valid_time_count = self._patch_bounds(segy_file, ref)
        if abs_trace_stop <= abs_trace_start or valid_time_count <= 0:
            return out

        traces = segyio.tools.collect(
            segy_file.trace[abs_trace_start:abs_trace_stop]
        ).astype(np.float32, copy=False)
        window = traces[:, ref.time_start:ref.time_start + valid_time_count]
        out[:window.shape[0], :window.shape[1]] = window
        return out

    def _read_segment(self, segy_file: Any, ref: PatchRef) -> np.ndarray:
        return segyio.tools.collect(
            segy_file.trace[ref.segment_start:ref.segment_stop]
        ).astype(np.float32, copy=False)

    def _get_segment_norm_stats(self, segy_file: Any, ref: PatchRef) -> Dict[str, Any]:
        key = (ref.pair_idx, ref.segment_start, ref.segment_stop)
        cached = self._segment_norm_cache.get(key)
        if cached is not None:
            return cached

        mode = self.normalize_mode.lower()
        if mode in ("none", "off", "identity"):
            stats: Dict[str, Any] = {}
        else:
            segment = np.nan_to_num(self._read_segment(segy_file, ref), copy=False)
            _, stats = normalize(
                segment,
                mode=mode,
                clip_percentile=self.clip_percentile,
                per="global",
            )
            if mode == "max_abs":
                stats["max_abs"] = max(float(np.asarray(stats["max_abs"])), self.normalize_eps)
            elif mode == "mean_std":
                stats["std"] = max(float(np.asarray(stats["std"])), self.normalize_eps)
            elif mode == "minmax":
                xmin = float(np.asarray(stats["min"]))
                xmax = float(np.asarray(stats["max"]))
                if xmax - xmin < self.normalize_eps:
                    stats["max"] = xmin + self.normalize_eps

        self._segment_norm_cache[key] = stats
        return stats

    def _normalize_patch(self, patch: np.ndarray, stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
        x = np.nan_to_num(patch.astype(np.float32, copy=False), copy=False)
        if self.clip_percentile is not None:
            if stats is not None and "clip_threshold" in stats:
                clip = float(np.asarray(stats["clip_threshold"]))
            else:
                clip = float(np.percentile(np.abs(x), float(self.clip_percentile)))
            if clip > 0:
                x = np.clip(x, -clip, clip)

        mode = self.normalize_mode.lower()
        if mode in ("none", "off", "identity"):
            return x
        if stats is not None:
            normalized, _ = normalize(
                x,
                mode=mode,
                per="global",
                override_stats=stats,
            )
            return normalized.astype(np.float32, copy=False)
        if mode == "max_abs":
            denom = float(np.max(np.abs(x)))
            return x / max(denom, self.normalize_eps)
        if mode == "mean_std":
            mean = float(np.mean(x))
            std = float(np.std(x))
            return (x - mean) / max(std, self.normalize_eps)
        raise ValueError(f"Unknown normalize_mode {self.normalize_mode!r}.")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.refs[idx]
        data_file = self._get_handle(ref.pair_idx)
        stats = None
        scope = self.normalize_scope.lower()
        if scope in ("gather", "segment"):
            stats = self._get_segment_norm_stats(data_file, ref)
        elif scope != "patch":
            raise ValueError(
                f"Unknown normalize_scope {self.normalize_scope!r}; "
                "expected 'gather', 'segment', or 'patch'."
            )
        data = self._normalize_patch(self._read_patch(data_file, ref), stats)
        _, _, valid_trace_count, valid_time_count = self._patch_bounds(data_file, ref)

        mask = np.full((self.patch_trace, self.patch_time), -1.0, dtype=np.float32)
        target_pick = np.full((self.patch_trace,), -1, dtype=np.int64)

        if valid_trace_count > 0 and valid_time_count > 0:
            abs_trace_start = ref.segment_start + ref.trace_start
            pair_picks = self.index.pick_indices_by_pair[ref.pair_idx]
            pick_slice = pair_picks[
                abs_trace_start:abs_trace_start + valid_trace_count
            ].astype(np.int64, copy=False)
            labeled_rows = pick_slice >= 0
            if bool(labeled_rows.any()):
                mask_window = mask[:valid_trace_count, :valid_time_count]
                time_indices = np.arange(
                    ref.time_start,
                    ref.time_start + valid_time_count,
                    dtype=np.int64,
                )
                mask_window[labeled_rows] = (
                    time_indices[None, :] >= pick_slice[labeled_rows, None]
                ).astype(np.float32, copy=False)

                in_patch = (
                    labeled_rows
                    & (pick_slice >= ref.time_start)
                    & (pick_slice < ref.time_start + valid_time_count)
                )
                target_pick_window = target_pick[:valid_trace_count]
                target_pick_window[in_patch] = (
                    pick_slice[in_patch] - ref.time_start
                ).astype(np.int64, copy=False)

        return (
            torch.from_numpy(data[None, :, :]),
            torch.from_numpy(mask[None, :, :]),
            torch.from_numpy(target_pick),
        )


def build_first_break_loaders(
    cfg: Mapping[str, Any],
    *,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Optional[DistributedSampler],
    Optional[DataLoader],
    FirstBreakIndex,
]:
    """Build DataLoaders and the shared first-break index."""
    index = build_first_break_index(cfg)
    data_cfg = cfg["data"]
    prep = cfg.get("preprocess", {})
    loader_cfg = data_cfg.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 16))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", True))

    dataset_kwargs = {
        "normalize_mode": str(prep.get("normalize_mode", "max_abs")),
        "normalize_scope": str(prep.get("normalize_scope", "gather")),
        "clip_percentile": prep.get("clip_percentile"),
        "normalize_eps": float(prep.get("normalize_eps", 1.0e-6)),
        "label_threshold": float(data_cfg.get("label_threshold", 0.5)),
    }
    train_ds = FirstBreakSegyPatchDataset(index, "train", **dataset_kwargs)
    val_ds = FirstBreakSegyPatchDataset(index, "val", **dataset_kwargs)
    test_ds = FirstBreakSegyPatchDataset(index, "test", **dataset_kwargs)

    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.get("experiment", {}).get("seed", 42)),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if distributed and rank != 0:
        eval_train_loader: Optional[DataLoader] = None
    else:
        eval_train_ds = FirstBreakSegyPatchDataset(index, "train", **dataset_kwargs)
        eval_train_loader = DataLoader(
            eval_train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader, train_sampler, eval_train_loader, index
