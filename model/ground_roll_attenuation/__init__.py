"""Models registered for the ground-roll-attenuation task.

Importing this sub-package executes every concrete model file so their
``@register_model`` decorators run and populate the shared registry exposed by
``model.registry``.
"""

from ..registry import MODEL_REGISTRY, build_model, register_model

from . import atten_unet  # noqa: F401
from . import ddpm  # noqa: F401
from . import dncnn  # noqa: F401
from . import enhanced_unet  # noqa: F401
from . import physics_unet  # noqa: F401
from . import physics_dnn  # noqa: F401
from . import pix2pix  # noqa: F401
from . import res_unet  # noqa: F401
from . import sanet  # noqa: F401
from . import dfb_cnn  # noqa: F401
from . import unet  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
