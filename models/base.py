"""Abstract base class for image generation backends.

To add a new model backend:
1. Create models/your_model.py
2. Subclass BaseImageModel
3. Implement all abstract methods
4. Add model name to Settings.image_model Literal
5. Add import to models/__init__.py load_model()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL.Image import Image as PILImage


@dataclass
class GenerationRequest:
    """Parameters for a single image generation request."""
    prompt: str                          # user prompt, before any backend wrapping
    negative_prompt: str | None          # None = no negative prompt
    width: int | None                    # None (edit only) = size from last reference
    height: int | None
    num_inference_steps: int
    cfg_scale: float
    seed: int
    reference_images: list[PILImage] = field(default_factory=list)  # order is significant
    transparent: bool = False


@dataclass
class GenerationResult:
    """Result of a successful image generation."""
    image: PILImage          # PIL Image object
    seed: int                # actual seed used
    generation_time: float   # seconds
    width: int               # actual output size
    height: int


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
    def quantization_info(self) -> dict:
        """Active weight quantization and measured weight memory (spec 008).

        Backends that do not quantize return an empty dict.
        """
        return {}

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def aspect_ratios(self) -> dict[str, tuple[int, int]]:
        """Mapping of aspect-ratio key to (width, height) in pixels."""
        ...

    @abstractmethod
    def sizes_for(self, resolution: int) -> dict[str, tuple[int, int]]:
        """Mapping of aspect-ratio key to (width, height) for a target side length."""
        ...

    @property
    @abstractmethod
    def default_inference_steps(self) -> int:
        """Backend-native default number of diffusion steps."""
        ...

    @property
    @abstractmethod
    def default_cfg_scale(self) -> float:
        """Backend-native default CFG scale."""
        ...

    @property
    @abstractmethod
    def default_negative_prompt(self) -> str | None:
        """Backend-native default negative prompt, or None."""
        ...

    @property
    def max_reference_images(self) -> int:
        """Maximum reference images for editing. 0 = editing unsupported."""
        return 0

    @property
    def supports_transparency(self) -> bool:
        """True if the backend can produce RGBA output."""
        return False

    @property
    def edit_output_resolution(self) -> int | None:
        """Pixel-area target (side length) for edits, or None if editing is unsupported."""
        return None

    @property
    def supports_editing(self) -> bool:
        """True if the backend accepts reference images."""
        return self.max_reference_images > 0
