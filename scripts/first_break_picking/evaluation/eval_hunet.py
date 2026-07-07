"""Evaluate HU-Net for first-break picking on SEG-Y files.

Modes:
  Single-file  --input /path/to/seismic.sgy
  Batch        --root /path/to/data_root  (discovers seismic.sgy + label.sgy)

Output per evaluation: pred_picks.txt, prob_mask.sgy.
Batch mode additionally produces batch_eval.xlsx.

Examples:
    # Single file
    python scripts/first_break_picking/evaluation/eval_hunet.py \
        --config configs/first_break_picking/pick_hunet_seed42.yaml \
        --checkpoint results/.../checkpoints/best.pt \
        --input /path/to/seismic.sgy --label-sgy /path/to/label.sgy

    # Batch
    python scripts/first_break_picking/evaluation/eval_hunet.py \
        --config configs/first_break_picking/pick_hunet_seed42.yaml \
        --checkpoint results/.../checkpoints/best.pt \
        --root /path/to/data_root
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root containing both model/ and utils/.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _eval_common import run_evaluation  # noqa: E402

if __name__ == "__main__":
    run_evaluation(__file__)
