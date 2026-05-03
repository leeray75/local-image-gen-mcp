"""Model backend factory.

Import load_model() to instantiate the configured image generation backend.
"""

from .base import BaseImageModel, GenerationRequest, GenerationResult


def load_model() -> BaseImageModel:
    """Instantiate and return the configured model backend.

    Reads settings.image_model to determine which backend to load.
    To add a new backend:
    1. Create models/your_model.py subclassing BaseImageModel
    2. Add an import branch here
    3. Add the model name to Settings.image_model Literal in config/settings.py
    """
    from config.settings import settings

    if settings.image_model == "qwen_image_2512":
        from .qwen_image_2512 import QwenImage2512
        return QwenImage2512(torch_dtype_str=settings.torch_dtype)

    raise ValueError(
        f"Unknown image_model: '{settings.image_model}'. "
        "Update config/settings.py to add support for new models."
    )


__all__ = ["load_model", "BaseImageModel", "GenerationRequest", "GenerationResult"]