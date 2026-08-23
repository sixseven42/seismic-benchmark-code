#!/usr/bin/env python3
"""Pad each SEG-Y shot to a multiple of 201 traces.

The two input files are processed with identical trace selections. Padding
traces contain zeros and reuse the last trace header in the corresponding
shot, so FieldRecord remains constant inside every 201-trace gather.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import segyio


TRACES_PER_GATHER = 201


def _blocks(ffid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous ``(start, stop, padded_stop)`` FFID blocks."""
    if ffid.ndim != 1 or ffid.size == 0:
        raise ValueError("FieldRecord header is empty or not one-dimensional")
    changes = np.flatnonzero(ffid[1:] != ffid[:-1]) + 1
    starts = np.r_[0, changes]
    stops = np.r_[changes, ffid.size]
    out = []
    for start, stop in zip(starts, stops):
        n = int(stop - start)
        padded = int(((n + TRACES_PER_GATHER - 1) // TRACES_PER_GATHER) * TRACES_PER_GATHER)
        out.append((int(start), int(stop), padded))
    return out


def _read_layout(path: Path):
    with segyio.open(str(path), "r", ignore_geometry=True) as src:
        ffid = np.asarray(src.attributes(segyio.TraceField.FieldRecord)[:], dtype=np.int64)
        blocks = _blocks(ffid)
        return ffid, blocks, int(src.tracecount), int(len(src.samples))


def _pad_one(src_path: Path, dst_path: Path, blocks: list[tuple[int, int, int]]) -> int:
    """Write one file using a monotonic output cursor."""
    with segyio.open(str(src_path), "r", ignore_geometry=True) as src:
        spec = segyio.spec()
        spec.format = src.format
        spec.samples = src.samples
        spec.tracecount = sum(b[2] for b in blocks)
        with segyio.create(str(dst_path), spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            out_i = 0
            zero_trace = np.zeros(len(src.samples), dtype=np.float32)
            for start, stop, padded_stop in blocks:
                for src_i in range(start, stop):
                    dst.trace[out_i] = src.trace[src_i]
                    dst.header[out_i] = src.header[src_i]
                    out_i += 1
                last_header = src.header[stop - 1]
                for _ in range(padded_stop - stop + start):
                    dst.trace[out_i] = zero_trace
                    dst.header[out_i] = last_header
                    out_i += 1
            dst.flush()
        return out_i


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--suffix", default="_padded201")
    args = parser.parse_args()

    noisy_ffid, blocks, n_traces, n_samples = _read_layout(args.noisy)
    noise_ffid, noise_blocks, noise_n_traces, noise_n_samples = _read_layout(args.noise)
    if n_traces != noise_n_traces or n_samples != noise_n_samples:
        raise ValueError("Noisy/noise SEG-Y dimensions do not match")
    if not np.array_equal(noisy_ffid, noise_ffid):
        raise ValueError("Noisy/noise FieldRecord headers do not match")
    if blocks != noise_blocks:
        raise ValueError("Noisy/noise shot boundaries do not match")

    noisy_out = args.noisy.with_name(args.noisy.stem + args.suffix + args.noisy.suffix)
    noise_out = args.noise.with_name(args.noise.stem + args.suffix + args.noise.suffix)
    total_out = sum(b[2] for b in blocks)
    print(f"shots={len(blocks)}, traces={n_traces} -> {total_out}, samples={n_samples}")
    print(f"writing {noisy_out}")
    _pad_one(args.noisy, noisy_out, blocks)
    print(f"writing {noise_out}")
    _pad_one(args.noise, noise_out, blocks)
    print("done")


if __name__ == "__main__":
    main()
