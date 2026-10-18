"""Concrete implementation of BaseImageModel for Qwen-Image-2.1.

Uses diffusers QwenImage21Pipeline with the official Qwen/Qwen-Image-2.1
checkpoint. One model covers text-to-image, image editing (up to 10 reference
images) and native transparent (RGBA) generation.

Memory on GB10 unified memory (128GB):
  The weights are about 33GB in BF16; measured peaks are ~68GB for a 2K
  generate and ~40GB for an edit at 1024 (the default text-to-image size is
  1024 too since spec 007; its peak is estimated at or below ~40GB). This does
  NOT fit next to a running vLLM engine, so stop vLLM first. IMAGE_GPU_MEMORY_LIMIT caps accelerate's device placement via
  max_memory, but offloaded layers live in the same physical RAM on unified
  memory, so the cap does not reduce total memory use.

Meta-tensor rule:
  Pass device_map to from_pretrained() and NEVER call .to(device) afterwards:
  the pipeline uses meta-device init and .to() raises
  NotImplementedError: Cannot copy out of meta tensor.
"""

import logging
import math
import time

import torch
from diffusers import QwenImage21Pipeline

from .base import BaseImageModel, GenerationRequest, GenerationResult

log = logging.getLogger(__name__)


class QwenImage21(BaseImageModel):
    """Qwen-Image-2.1 backend (text-to-image, editing, RGBA)."""

    MODEL_ID = "Qwen/Qwen-Image-2.1"

    # Official 2K sizes from the model card (used as-is at resolution 2048)
    CARD_SIZES_2048: dict[str, tuple[int, int]] = {
        "1:1": (2048, 2048),
        "4:3": (2400, 1792),
        "3:4": (1792, 2400),
        "3:2": (2528, 1696),
        "2:3": (1696, 2528),
        "16:9": (2752, 1536),
        "9:16": (1536, 2752),
    }

    # Ratio keys as (w, h) factors, for the pipeline's calculate_dimensions rule
    RATIO_FACTORS: dict[str, tuple[int, int]] = {
        "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2),
        "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16),
    }

    # Phrasing from the model card for native RGBA generation
    TRANSPARENT_PROMPT = (
        "This is an RGBA image with transparency. {prompt} "
        "The image has alpha channel and the background is transparent."
    )

    def __init__(
        self,
        torch_dtype_str: str = "bfloat16",
        memory_limit_gb: int | None = None,
        edit_output_resolution: int = 1024,
        generate_output_resolution: int = 1024,
    ) -> None:
        self._pipeline: QwenImage21Pipeline | None = None
        self._memory_limit_gb = memory_limit_gb
        self._edit_output_resolution = edit_output_resolution
        self._generate_output_resolution = generate_output_resolution
        self._torch_dtype = (
            torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float32
        )
        # Used only for the Generator seed — device_map handles actual placement
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load model weights into GPU/unified memory.

        Uses device_map='balanced' with an optional max_memory cap so accelerate
        places weights during loading. Do NOT call .to(device) afterwards.
        """
        if self._pipeline is not None:
            log.warning("Model already loaded — skipping")
            return

        if self._device == "cpu":
            log.warning(
                "CUDA not available — loading on CPU. "
                "Generation will be extremely slow."
            )
            self._pipeline = QwenImage21Pipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self._torch_dtype,
            )
            log.info("Model loaded on CPU")
            return

        kwargs = {
            "torch_dtype": self._torch_dtype,
            "device_map": "balanced",
        }

        if self._memory_limit_gb is not None:
            kwargs["max_memory"] = {
                0: f"{self._memory_limit_gb}GiB",
                "cpu": "60GiB",  # allow offload to system RAM
            }
            log.info(
                "Loading %s with %dGB GPU memory cap (dtype=%s)...",
                self.MODEL_ID,
                self._memory_limit_gb,
                self._torch_dtype,
            )
            log.warning(
                "Memory cap active — layers exceeding %dGB are offloaded to CPU "
                "RAM and generation is slower. On unified memory the cap does "
                "not reduce total RAM use; stop vLLM before running this server.",
                self._memory_limit_gb,
            )
        else:
            log.info(
                "Loading %s on %s (dtype=%s, no memory cap)...",
                self.MODEL_ID,
                self._device,
                self._torch_dtype,
            )
            log.warning(
                "No memory cap — model will use ~33GB weights plus activations "
                "(~68GB peak for a 2K generate, less at 1K). Stop vLLM before starting image generation."
            )

        self._pipeline = QwenImage21Pipeline.from_pretrained(
            self.MODEL_ID,
            **kwargs,
        )

        if torch.cuda.is_available():
            allocated = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            reserved = round(torch.cuda.memory_reserved(0) / 1e9, 2)
            log.info(
                "Model loaded successfully. "
                "PyTorch CUDA — allocated: %.2fGB, reserved: %.2fGB",
                allocated,
                reserved,
            )
        else:
            log.info("Model loaded successfully on %s", self._device)

    def unload(self) -> None:
        """Release pipeline from memory and free GPU cache."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log.info("Model unloaded, GPU memory cleared")

    def _wrap_transparent(self, prompt: str) -> str:
        """Apply the model card's RGBA phrasing to a prompt."""
        p = prompt.strip()
        if not p.endswith((".", "!", "?")):
            p += "."
        return self.TRANSPARENT_PROMPT.format(prompt=p)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate or edit an image synchronously.

        Called via asyncio.to_thread() from the async MCP tool — blocks the
        calling thread during diffusion. Never call directly from async.

        Uses true_cfg_scale (NOT guidance_scale). Guidance is off at the default
        of 1.0, in which case negative_prompt has no effect.
        """
        if self._pipeline is None:
            raise RuntimeError(
                f"Model {self.MODEL_ID} is not loaded. Call load() first."
            )

        t_start = time.perf_counter()

        prompt = (
            self._wrap_transparent(request.prompt)
            if request.transparent
            else request.prompt
        )

        kwargs: dict = {
            "prompt": prompt,
            "num_inference_steps": request.num_inference_steps,
            "true_cfg_scale": request.cfg_scale,
            # Generator must use the same device as the model
            "generator": torch.Generator(device=self._device).manual_seed(
                request.seed
            ),
        }
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.reference_images:
            kwargs["image"] = list(request.reference_images)
            kwargs["output_resolution"] = self._edit_output_resolution
        if request.width is not None and request.height is not None:
            kwargs["width"] = request.width
            kwargs["height"] = request.height

        output = self._pipeline(**kwargs)

        # The VAE always returns RGBA; flatten unless transparency was requested
        img = output.images[0].convert("RGBA" if request.transparent else "RGB")
        generation_time = time.perf_counter() - t_start

        # Release temporary computation buffers (weights stay loaded)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info(
            "Generated %dx%d %s image in %.1fs (seed=%d, refs=%d)",
            img.width,
            img.height,
            img.mode,
            generation_time,
            request.seed,
            len(request.reference_images),
        )

        return GenerationResult(
            image=img,
            seed=request.seed,
            generation_time=round(generation_time, 2),
            width=img.width,
            height=img.height,
        )

    def is_loaded(self) -> bool:
        """Return True if the pipeline is loaded and ready."""
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return "Qwen-Image-2.1"

    def sizes_for(self, resolution: int) -> dict[str, tuple[int, int]]:
        """Size table for a target side length.

        2048 returns the model card's table; any other value applies the
        pipeline's calculate_dimensions rule with area = resolution^2 (the
        height comes from the unrounded width).
        """
        if resolution == 2048:
            return dict(self.CARD_SIZES_2048)
        sizes = {}
        for key, (rw, rh) in self.RATIO_FACTORS.items():
            ratio = rw / rh
            width = math.sqrt(resolution * resolution * ratio)
            height = width / ratio
            sizes[key] = (round(width / 32) * 32, round(height / 32) * 32)
        return sizes

    @property
    def aspect_ratios(self) -> dict[str, tuple[int, int]]:
        """Size table for the configured GENERATE_OUTPUT_RESOLUTION."""
        return self.sizes_for(self._generate_output_resolution)

    @property
    def default_inference_steps(self) -> int:
        return 40

    @property
    def default_cfg_scale(self) -> float:
        return 1.0

    @property
    def default_negative_prompt(self) -> str | None:
        return None

    @property
    def max_reference_images(self) -> int:
        return 10

    @property
    def edit_output_resolution(self) -> int:
        return self._edit_output_resolution

    @property
    def supports_transparency(self) -> bool:
        return True
