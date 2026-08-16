"""Train DSU-Net with the existing gather-segmentation/patch pipeline.

The historical four-dataset runs used Dongbei while it still lived under the
main ``segy_with_masks`` directory.  Dongbei is now stored separately, so this
entry supports a narrowly scoped label-path override while delegating all
indexing, FFID splitting, gather segmentation and patch extraction to the
unchanged :mod:`first_break_data` implementation.

Example:
    python scripts/first_break_picking/train_pick_dsu_net.py \
        --config configs/first_break_picking/pick_dsu_net_seed42.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping

_REPO_ROOT = next(
    (
        path
        for path in Path(__file__).resolve().parents
        if (path / "model").is_dir() and (path / "utils").is_dir()
    ),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Cannot find repo root containing both model/ and utils/.")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import first_break_data as _patch_data  # noqa: E402
import train_pick_common as _common  # noqa: E402
from model.first_break_picking import dsu_net as _dsu_net  # noqa: E402,F401


def _label_overrides(cfg: Mapping[str, Any]) -> Dict[Path, Path]:
    data_cfg = cfg["data"]
    root = Path(str(data_cfg["root"])).expanduser()
    raw_overrides = data_cfg.get("label_path_overrides", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("data.label_path_overrides must be a list when provided.")

    overrides: Dict[Path, Path] = {}
    for idx, raw in enumerate(raw_overrides):
        if not isinstance(raw, Mapping) or "data_path" not in raw or "label_path" not in raw:
            raise ValueError(
                f"data.label_path_overrides[{idx}] must contain data_path and label_path."
            )
        data_path = Path(str(raw["data_path"])).expanduser()
        label_path = Path(str(raw["label_path"])).expanduser()
        if not data_path.is_absolute():
            data_path = root / data_path
        if not label_path.is_absolute():
            label_path = root / label_path
        data_path = data_path.resolve()
        label_path = label_path.resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"Override data SEG-Y not found: {data_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Override label SEG-Y not found: {label_path}")
        overrides[data_path] = label_path
    return overrides


def _build_existing_patch_loaders(
    cfg: Mapping[str, Any],
    *,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
):
    overrides = _label_overrides(cfg)
    original_resolver = _patch_data._resolve_label_path

    def _resolve_label_path(data_path: Path, label_dir: Path) -> Path:
        override = overrides.get(data_path.resolve())
        if override is not None:
            return override
        return original_resolver(data_path, label_dir)

    _patch_data._resolve_label_path = _resolve_label_path
    try:
        return _patch_data.build_first_break_loaders(
            cfg,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )
    finally:
        _patch_data._resolve_label_path = original_resolver


if __name__ == "__main__":
    _common.build_first_break_loaders = _build_existing_patch_loaders
    _common.run_training(__file__)
