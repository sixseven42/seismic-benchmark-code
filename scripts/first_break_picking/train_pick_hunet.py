"""Train HU-Net for first-break picking mask segmentation.

HU-Net (holistically-nested U-Net) uses five side outputs + fused output
with deep supervision loss. The model returns ``(fused, sides)`` during training
and the ``hunet`` loss handles the weighted BCE across all outputs.

Example:
    python scripts/first_break_picking/train_pick_hunet.py \
        --config configs/first_break_picking/pick_hunet_seed42.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root containing both model/ and utils/.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_pick_common import run_training  # noqa: E402


if __name__ == "__main__":
    run_training(__file__)
