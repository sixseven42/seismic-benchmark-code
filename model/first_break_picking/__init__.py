"""Models registered for the first-break-picking task.

This package mirrors the U-Net family used by the attenuation tasks, but
keeps the importable underscore path for first-break experiments.
"""

from ..registry import MODEL_REGISTRY, build_model, register_model

from . import atten_unet  # noqa: F401
from . import dncnn_seg  # noqa: F401
from . import dsu_net  # noqa: F401
from . import res_unet  # noqa: F401
from . import unet  # noqa: F401
from . import hunet  # noqa: F401
from . import stunet  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
