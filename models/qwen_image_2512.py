"""Concrete implementation of BaseImageModel for Qwen-Image-2512.

Uses diffusers DiffusionPipeline with the official Qwen/Qwen-Image-2512 checkpoint.
"""

import logging
import time
import torch
from typing import Any

from diffusers import DiffusionPipeline

from .base import BaseImageModel, GenerationRequest, GenerationResult

log = logging.getLogger(__name__)

# Default negative prompt from official Qwen-Image-2512 model card.
# Chinese terms are intentional — the model was heavily trained on Chinese data.
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
        self._torch_dtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float32
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load model weights into GPU/CPU memory."""
        if self._pipeline is not None:
            log.warning("Model already loaded — skipping")
            return

        log.info(
            "Loading %s on %s (dtype=%s)...",
            self.MODEL_ID, self._device, self._torch_dtype,
        )

        self._pipeline = DiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self._torch_dtype,
        ).to(self._device)

        if self._device == "cpu":
            log.warning(
                "Running on CPU — generation will be extremely slow. "
                "Use a CUDA-capable GPU for production."
            )

        log.info("Model loaded successfully")

    def unload(self) -> None:
        """Release model from memory and free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Model unloaded, GPU memory cleared")

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate an image synchronously.

        This method is designed to be called via asyncio.to_thread() from
        an async MCP tool — it blocks the calling thread during diffusion.
        """
        if self._pipeline is None:
            raise RuntimeError(
                f"Model {self.MODEL_ID} is not loaded. Call load() first."
            )

        t_start = time.perf_counter()

        generator = torch.Generator(device=self._device).manual_seed(request.seed)

        output = self._pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            true_cfg_scale=request.cfg_scale,
            generator=generator,
        )

        image = output.images[0]
        generation_time = time.perf_counter() - t_start

        log.info(
            "Generated %dx%d image in %.1fs (seed=%d)",
            request.width, request.height, generation_time, request.seed,
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