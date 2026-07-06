"""Models registered for the multiples-attenuation task.

Importing this sub-package executes every concrete model file so their
``@register_model`` decorators run and populate the shared registry exposed by
``model.registry``.
"""

from ..registry import MODEL_REGISTRY, build_model, register_model

from . import atten_unet  # noqa: F401
from . import dnndat  # noqa: F401
from . import dncnn  # noqa: F401
from . import kiraz_trace_cnn  # noqa: F401
from . import res_unet  # noqa: F401
from . import sagan  # noqa: F401
from . import unet  # noqa: F401

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "register_model",
]
