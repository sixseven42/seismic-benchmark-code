"""Backfill missing ``test_set/`` directories for Physics CNN experiments.

``train_denoise_physics.py`` historically did not persist the held-out test
shots, so ``batch_evaluate.py`` skips those result directories (its
``discover_results`` requires ``test_set/``).  This script regenerates the
missing ``test_set/{input_shots,target_shots,ffid}.npy`` files for Physics
directories under a results root, reusing the exact preprocessing and FFID
split semantics of the training pipeline:

* ``_preprocess_shots`` is imported from the training script (no duplication).
* Splits are applied on unique FFID values with ``ffid_split_masks``, the
  same helper used by ``build_shot_split_loaders``.
* ``target_shots.npy`` stores the ground-roll noise so ``batch_evaluate``
  recovers the clean signal as ``input - target``.

Usage::

    python scripts/ground_roll_attenuation/backfill_physics_test_set.py \
        --root-dir /data/shared/benchmark/ground_roll/results_0822

    # overwrite an existing test_set (default is to skip)
    python scripts/ground_roll_attenuation/backfill_physics_test_set.py \
        --root-dir /data/shared/benchmark/ground_roll/results_0822 --force
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import yaml

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.train_utils import ffid_split_masks  # noqa: E402

# Cache of preprocessed volumes keyed by the paired-data source signature.
# All Physics dirs under one root usually share the same SEG-Y pair, so this
# avoids re-reading and re-normalizing the ~8 GB of volumes for every seed.
_CACHE: Dict[Tuple[str, str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _data_pair_signature(cfg: Dict[str, Any]) -> Tuple[str, str, int, int]:
    """Return a stable key for the paired data source in *cfg*."""
    pair_cfg = None
    for key in ("segy_pair", "npy_pair", "mat_pair"):
        if key in cfg.get("data", {}):
            pair_cfg = cfg["data"][key]
            break
    if pair_cfg is None:
        raise ValueError("No paired data source found in config.")
    return (
        str(pair_cfg["input_path"]),
        str(pair_cfg["target_path"]),
        int(pair_cfg.get("traces_per_shot", 201)),
        int(pair_cfg.get("time_downsample", 1)),
    )


def _load_physics_preprocess() -> Callable[[Dict[str, Any]], Any]:
    """Return ``_preprocess_shots`` from the Physics training script."""
    train_script = (
        Path(__file__).resolve().parent / "train" / "train_denoise_physics.py"
    )
    spec = importlib.util.spec_from_file_location("train_denoise_physics", train_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._preprocess_shots


def _preprocess_volumes(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (noisy, gr_noise, per_shot_ffid), cached by data source."""
    sig = _data_pair_signature(cfg)
    if sig not in _CACHE:
        _preprocess_shots = _load_physics_preprocess()
        noisy, clean, gr, per_shot_ffid = _preprocess_shots(cfg)
        del clean  # not needed; drop 4 GB early
        _CACHE[sig] = (noisy, gr, per_shot_ffid)
    return _CACHE[sig]


def _parse_level_seed(dir_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract ``(level, seed)`` from a ``..._level{level}_seed{seed}`` name."""
    match = re.search(r"_level([\d.]+)_seed(\d+)$", dir_name)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def _verify_against_sibling(
    root_dir: Path, result_dir: Path, ffid: np.ndarray
) -> None:
    """Compare the regenerated FFIDs with a sibling model's test set."""
    level, seed = _parse_level_seed(result_dir.name)
    if level is None or seed is None:
        return
    for candidate in root_dir.glob(f"denoise_unet_*level{level}_seed{seed}"):
        sibling_ffid = candidate / "test_set" / "ffid.npy"
        if not sibling_ffid.is_file():
            continue
        ref = np.load(sibling_ffid, allow_pickle=False)
        if np.array_equal(ref, ffid):
            print(f"  [OK] ffid.npy matches sibling {candidate.name}")
        else:
            print(
                f"  [WARN] ffid.npy differs from sibling {candidate.name}: "
                f"len {ref.size} vs {ffid.size}"
            )
        return
    print("  [INFO] no sibling test_set found to cross-check ffid ordering")


def backfill_one(result_dir: Path, force: bool = False) -> None:
    """Regenerate ``test_set/`` for a single Physics result directory."""
    config_path = result_dir / "config.yaml"
    test_dir = result_dir / "test_set"
    ckpt_path = result_dir / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        print(f"  [SKIP] {result_dir.name}: missing checkpoints/best.pt (not an evaluable experiment)")
        return
    if test_dir.exists() and not force:
        print(f"  [SKIP] {result_dir.name}: test_set/ already exists (use --force to overwrite)")
        return
    if not config_path.is_file():
        print(f"  [SKIP] {result_dir.name}: missing config.yaml")
        return

    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle) or {}
    shot_split = cfg.get("data", {}).get("shot_split", {})
    if not shot_split:
        print(f"  [SKIP] {result_dir.name}: no data.shot_split in config")
        return

    noisy, gr, per_shot_ffid = _preprocess_volumes(cfg)
    n_train = int(shot_split["train"])
    n_val = int(shot_split["val"])
    n_test = int(shot_split["test"])
    _, _, test_mask = ffid_split_masks(per_shot_ffid, n_train, n_val, n_test)

    test_dir.mkdir(parents=True, exist_ok=True)
    np.save(test_dir / "input_shots.npy", noisy[test_mask])
    np.save(test_dir / "target_shots.npy", gr[test_mask])
    np.save(test_dir / "ffid.npy", per_shot_ffid[test_mask])
    print(
        f"  [OK] {result_dir.name}: saved {int(test_mask.sum())} shots "
        f"shape={noisy[test_mask].shape} -> {test_dir}"
    )
    _verify_against_sibling(result_dir.parent, result_dir, per_shot_ffid[test_mask])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill test_set/ for Physics CNN result directories."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/data/shared/benchmark/ground_roll/results_0822"),
        help="Root of the batch-evaluated results tree.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="denoise_physics_base*",
        help="Glob pattern for Physics result directories under --root-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing test_set/ instead of skipping it.",
    )
    args = parser.parse_args()

    root_dir = args.root_dir.resolve()
    if not root_dir.is_dir():
        print(f"Root directory does not exist: {root_dir}")
        raise SystemExit(1)

    dirs = sorted(root_dir.glob(args.pattern))
    if not dirs:
        print(f"No directories match {args.pattern!r} under {root_dir}")
        raise SystemExit(0)

    print(f"Found {len(dirs)} Physics result directory(s) under {root_dir}")
    for result_dir in dirs:
        backfill_one(result_dir, force=args.force)


if __name__ == "__main__":
    main()
