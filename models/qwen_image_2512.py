"""Concrete implementation of BaseImageModel for Qwen-Image-2512.

Uses diffusers DiffusionPipeline with the official Qwen/Qwen-Image-2512 checkpoint.

Fix for NotImplementedError: Cannot copy out of meta tensor:
  Qwen-Image-2512 uses accelerate's meta device initialization internally.
  Calling .to(device) after from_pretrained() attempts to copy meta tensors
  which have no data — this raises NotImplementedError.
  Solution: pass device_map="balanced" directly to from_pretrained() so
  accelerate handles device placement during weight loading, never after.
"""

import logging
import time
import torch
from diffusers import DiffusionPipeline

from .base import BaseImageModel, GenerationRequest, GenerationResult

log = logging.getLogger(__name__)

# Default negative prompt from official Qwen-Image-2512 model card.
# Chinese terms are intentional — the model was heavily trained on Chinese data
# and responds well to Chinese guidance even in English sessions.
DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    "low quality, blurry, deformed hands, extra fingers, watermark, "
    "text overlay, oversaturated, plastic skin, noise, compression artifacts"
)


class QwenImage2512(BaseImageModel):
    """Qwen-Image-2512 text-to-image diffusion backend."""

    MODEL_ID = "Qwen/Qwen-Image-2512"

    def __init__(self, torch_dtype_str: str = "bfloat16") -> None:
        self._pipeline: DiffusionPipeline | None = None
        self._torch_dtype = (
            torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float32
        )
        # Used only for the Generator seed — device_map handles actual placement
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load model weights into GPU memory.

        Uses device_map='balanced' so accelerate places weights during loading.
        Do NOT call .to(device) after from_pretrained() — meta tensors have no
        data and cannot be copied post-hoc (raises NotImplementedError).
        """
        if self._pipeline is not None:
            log.warning("Model already loaded — skipping")
            return

        log.info(
            "Loading %s on %s (dtype=%s)...",
            self.MODEL_ID,
            self._device,
            self._torch_dtype,
        )

        if self._device == "cpu":
            # CPU path: no device_map, direct load
            log.warning(
                "CUDA not available — loading on CPU. "
                "Generation will be extremely slow."
            )
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self._torch_dtype,
            )
        else:
            # GPU path: device_map handles placement during weight loading
            # NEVER call .to(device) after this — meta tensor error
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self._torch_dtype,
                device_map="cuda",
            )

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

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image synchronously.

        Called via asyncio.to_thread() from the async MCP tool — blocks
        the calling thread during diffusion. Never call directly from async.

        Uses true_cfg_scale per official Qwen-Image-2512 model card —
        NOT guidance_scale, which is a different parameter.
        """
        if self._pipeline is None:
            raise RuntimeError(
                f"Model {self.MODEL_ID} is not loaded. Call load() first."
            )

        t_start = time.perf_counter()

        # Generator must use the same device as the model
        generator = torch.Generator(device=self._device).manual_seed(request.seed)

        output = self._pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            true_cfg_scale=request.cfg_scale,   # official kwarg — NOT guidance_scale
            generator=generator,
        )

        image = output.images[0]
        generation_time = time.perf_counter() - t_start

        log.info(
            "Generated %dx%d image in %.1fs (seed=%d)",
            request.width,
            request.height,
            generation_time,
            request.seed,
        )

        return GenerationResult(
            image=image,
            seed=request.seed,
            generation_time=round(generation_time, 2),
        )

    def is_loaded(self) -> bool:
        """Return True if the pipeline is loaded and ready."""
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return "Qwen-Image-2512"