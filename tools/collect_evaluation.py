"""Collect and package all ``evaluation/`` directories under a batch-evaluated
results tree, preserving the original directory structure.

Usage::

    # Copy all evaluation dirs into a flat destination (preserving relative paths)
    python tools/collect_evaluation.py \
        --root-dir /data/shared/benchmark/ground_roll/results \
        --output-dir results/evaluation_package

    # Create a tar.gz archive instead of copying
    python tools/collect_evaluation.py \
        --root-dir /data/shared/benchmark/ground_roll/results \
        --archive results/evaluation_package.tar.gz

    # Dry-run: list what would be collected without writing anything
    python tools/collect_evaluation.py \
        --root-dir /data/shared/benchmark/ground_roll/results \
        --dry-run

    # Collect only visualizations (skip each evaluation's npy/ directory)
    python tools/collect_evaluation.py \
        --root-dir /data/shared/benchmark/ground_roll/results_0822 \
        --archive results/ground_roll_viz_only.tar.gz \
        --exclude-npy
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path
from typing import List


def find_evaluation_dirs(root_dir: Path) -> List[Path]:
    """Recursively find all ``evaluation/`` directories under *root_dir*.

    Returns absolute paths sorted by parent directory name.
    """
    eval_dirs: List[Path] = []
    for path in sorted(root_dir.rglob("evaluation")):
        if path.is_dir():
            eval_dirs.append(path)
    return eval_dirs


def collect_to_dir(
    eval_dirs: List[Path],
    root_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    exclude_npy: bool = False,
) -> None:
    """Copy each ``evaluation/`` tree into *output_dir*, preserving relative paths.

    With *exclude_npy*, each evaluation's ``npy/`` subdirectory is skipped so
    only the visualizations are collected.

    Parameters
    ----------
    eval_dirs : list of Path
        Evaluation directories to collect.
    root_dir : Path
        Results root; relative paths are derived from it.
    output_dir : Path
        Destination directory.
    dry_run : bool
        Print what would be collected without writing anything.
    exclude_npy : bool
        Skip each evaluation's ``npy/`` subdirectory.

    Returns
    -------
    None
    """
    ignore = shutil.ignore_patterns("npy") if exclude_npy else None
    for src_eval in eval_dirs:
        rel = src_eval.parent.relative_to(root_dir)
        dst_parent = output_dir / rel
        dst_eval = output_dir / rel / "evaluation"
        if dry_run:
            size = _collect_size(src_eval, exclude_npy)
            print(f"  {rel}/evaluation/  ({_human_size(size)})")
            continue
        dst_parent.mkdir(parents=True, exist_ok=True)
        if dst_eval.exists():
            shutil.rmtree(dst_eval)
        shutil.copytree(src_eval, dst_eval, symlinks=False, ignore=ignore)
        print(f"  copied: {rel}/evaluation/")
    if dry_run:
        print(f"\n  Total: {len(eval_dirs)} evaluation dir(s) found (dry-run).")
    else:
        print(f"\nCollected {len(eval_dirs)} evaluation dir(s) to: {output_dir}")


def collect_to_archive(
    eval_dirs: List[Path],
    root_dir: Path,
    archive_path: Path,
    *,
    exclude_npy: bool = False,
) -> None:
    """Create a gzipped tar archive containing all ``evaluation/`` directories.

    With *exclude_npy*, each evaluation's ``npy/`` subdirectory is skipped so
    only the visualizations are archived.  Members are stored under relative
    paths derived from *root_dir*.

    Parameters
    ----------
    eval_dirs : list of Path
        Evaluation directories to archive.
    root_dir : Path
        Results root; relative paths are derived from it.
    archive_path : Path
        Destination .tar.gz path.
    exclude_npy : bool
        Skip each evaluation's ``npy/`` subdirectory.

    Returns
    -------
    None
    """

    def _filter(tarinfo: tarfile.TarInfo):
        if exclude_npy and tarinfo.name.endswith("/npy"):
            return None
        return tarinfo

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for src_eval in eval_dirs:
            rel = src_eval.relative_to(root_dir)
            arcname = str(rel)
            tar.add(str(src_eval), arcname=arcname, recursive=True, filter=_filter)
            size = _collect_size(src_eval, exclude_npy)
            print(f"  archived: {rel}/  ({_human_size(size)})")
    size = archive_path.stat().st_size
    print(f"\nCreated archive: {archive_path} ({_human_size(size)})")


def _collect_size(eval_dir: Path, exclude_npy: bool) -> int:
    """Return total bytes under *eval_dir*, optionally skipping its npy/ dir."""
    total = 0
    for f in eval_dir.rglob("*"):
        if not f.is_file():
            continue
        if exclude_npy and "npy" in f.relative_to(eval_dir).parts:
            continue
        total += f.stat().st_size
    return total


def _human_size(size_bytes: int) -> str:
    """Format a byte count for human-readable display."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect all evaluation/ directories from a batch-evaluated results tree."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/data/shared/benchmark/ground_roll/results"),
        help="Root of the batch-evaluated results tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (mutually exclusive with --archive).  "
             "Default: results/evaluation_package.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Create a .tar.gz archive instead of copying to a directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be collected without copying or archiving.",
    )
    parser.add_argument(
        "--exclude-npy",
        action="store_true",
        help="Skip each evaluation's npy/ subdirectory, collecting visualizations only.",
    )
    args = parser.parse_args()

    root_dir = args.root_dir.resolve()
    if not root_dir.is_dir():
        print(f"Root directory does not exist: {root_dir}")
        raise SystemExit(1)

    eval_dirs = find_evaluation_dirs(root_dir)
    if not eval_dirs:
        print("No evaluation/ directories found.")
        raise SystemExit(0)

    print(f"Found {len(eval_dirs)} evaluation directory(s) under: {root_dir}")

    if args.dry_run:
        collect_to_dir(eval_dirs, root_dir, Path(), dry_run=True, exclude_npy=args.exclude_npy)
        return

    if args.archive:
        collect_to_archive(
            eval_dirs, root_dir, args.archive.resolve(), exclude_npy=args.exclude_npy
        )
    else:
        output_dir = (args.output_dir or Path("results/evaluation_package")).resolve()
        collect_to_dir(eval_dirs, root_dir, output_dir, exclude_npy=args.exclude_npy)


if __name__ == "__main__":
    main()
