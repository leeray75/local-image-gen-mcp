"""Abstract base class for image generation backends.

To add a new model backend:
1. Create models/your_model.py
2. Subclass BaseImageModel
3. Implement all abstract methods
4. Add model name to Settings.image_model Literal
5. Add import to models/__init__.py load_model()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL.Image import Image as PILImage


@dataclass
class GenerationRequest:
    """Parameters for a single image generation request."""
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_inference_steps: int
    cfg_scale: float
    seed: int


@dataclass
class GenerationResult:
    """Result of a successful image generation."""
    image: PILImage          # PIL Image object
    seed: int                # actual seed used
    generation_time: float   # seconds


class BaseImageModel(ABC):
    """Abstract base class for image generation backends.

    To add a new model:
    1. Create models/your_model.py
    2. Subclass BaseImageModel
    3. Implement all abstract methods
    4. Add model name to Settings.image_model Literal
    5. Add import to models/__init__.py load_model()
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once at server startup."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory. Called at server shutdown."""
        ...

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image. Must be synchronous (called in thread executor)."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...