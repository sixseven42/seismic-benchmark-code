"""Train Kiraz trace-by-trace CNN on paired multiples attenuation data."""

from __future__ import annotations

import argparse

import train_denoise_unet as _base
from utils import default_config_relpath_for_train_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Kiraz trace-by-trace 1D CNN for multiples attenuation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_relpath_for_train_script(__file__),
        help="Path to denoise config (expects data.*_pair).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _base.parse_args = parse_args
    _base.main()
