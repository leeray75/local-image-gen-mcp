"""Concrete implementation of BaseImageModel for Qwen-Image-2512.

Uses diffusers DiffusionPipeline with the official Qwen/Qwen-Image-2512 checkpoint.

Memory optimization on GB10 unified memory (128GB):
  Qwen-Image-2512 at BF16 with device_map="cuda" uses ~62GB unified memory,
  which conflicts with vLLM (~56GB). Together they consume ~118GB leaving
  only ~10GB for OS — causing system instability and swap thrashing.

  Solution: max_memory cap forces accelerate to offload layers exceeding the
  budget to CPU RAM, keeping GPU footprint bounded. Generation is slower but
  the system stays stable when both models run simultaneously.

  MEMORY_LIMIT_GB controls the trade-off:
    30GB -> ~15-20 min/image, system stable alongside vLLM
    40GB -> ~8-12 min/image, system stable alongside vLLM
    50GB -> ~4-5 min/image, marginal stability alongside vLLM
    None -> ~3-4 min/image, system unstable alongside vLLM (original behavior)

Fix for NotImplementedError: Cannot copy out of meta tensor:
  Qwen-Image-2512 uses accelerate's meta device initialization internally.
  Calling .to(device) after from_pretrained() attempts to copy meta tensors
  which have no data — this raises NotImplementedError.
  Solution: pass device_map="balanced" directly to from_pretrained() so
  accelerate handles device placement during weight loading, never after.
"""

import logging
import os
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

# GPU memory cap in GB — set via IMAGE_GPU_MEMORY_LIMIT env var.
# Controls how much unified memory the image model can use.
# Lower = more stable alongside vLLM, but slower generation.
# Set to None to disable cap (uses all available memory — original behavior).
#
# Recommended values for DGX Spark GB10 (128GB unified):
#   "30" -> safe alongside vLLM at any gpu-memory-utilization
#   "40" -> safe alongside vLLM at 0.6 utilization
#   "50" -> marginal alongside vLLM at 0.5 utilization
#   unset -> full speed, run exclusively (stop vLLM first)
_env_limit = os.getenv("IMAGE_GPU_MEMORY_LIMIT")
MEMORY_LIMIT_GB: int | None = int(_env_limit) if _env_limit else None


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
        """Load model weights into GPU/unified memory.

        Uses device_map='balanced' with optional max_memory cap so accelerate
        places weights during loading and respects the memory budget.

        Do NOT call .to(device) after from_pretrained() — meta tensors have no
        data and cannot be copied post-hoc (raises NotImplementedError).

        Memory behavior:
          - MEMORY_LIMIT_GB set: accelerate offloads excess layers to CPU RAM.
            Generation is slower but system stays stable alongside vLLM.
          - MEMORY_LIMIT_GB unset: all weights loaded onto GPU/unified memory.
            Fastest generation but ~62GB usage conflicts with vLLM.
        """
        if self._pipeline is not None:
            log.warning("Model already loaded — skipping")
            return

        if self._device == "cpu":
            log.warning(
                "CUDA not available — loading on CPU. "
                "Generation will be extremely slow."
            )
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self._torch_dtype,
            )
            log.info("Model loaded on CPU")
            return

        # Build kwargs — add max_memory cap if configured
        kwargs = {
            "torch_dtype": self._torch_dtype,
            "device_map": "balanced",
        }

        if MEMORY_LIMIT_GB is not None:
            # Cap GPU usage — offload remainder to CPU RAM
            # On GB10 unified memory, CPU RAM and GPU RAM are the same pool,
            # but accelerate still respects the cap for placement decisions.
            kwargs["max_memory"] = {
                0: f"{MEMORY_LIMIT_GB}GiB",
                "cpu": "60GiB",  # allow offload to system RAM
            }
            log.info(
                "Loading %s with %dGB GPU memory cap (dtype=%s)...",
                self.MODEL_ID,
                MEMORY_LIMIT_GB,
                self._torch_dtype,
            )
            log.warning(
                "Memory cap active — generation will be slower than uncapped mode. "
                "Layers exceeding %dGB will be offloaded to CPU RAM.",
                MEMORY_LIMIT_GB,
            )
        else:
            log.info(
                "Loading %s on %s (dtype=%s, no memory cap)...",
                self.MODEL_ID,
                self._device,
                self._torch_dtype,
            )
            log.warning(
                "No memory cap — model will use ~62GB unified memory. "
                "Stop vLLM before starting image generation to avoid instability. "
                "Set IMAGE_GPU_MEMORY_LIMIT env var to cap memory usage."
            )

        self._pipeline = DiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            **kwargs,
        )

        # Log actual memory usage after load
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

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an image synchronously.

        Called via asyncio.to_thread() from the async MCP tool — blocks
        the calling thread during diffusion. Never call directly from async.

        Uses true_cfg_scale per official Qwen-Image-2512 model card —
        NOT guidance_scale, which is a different parameter.

        Calls torch.cuda.empty_cache() after generation to release any
        temporary computation buffers back to the allocator.
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
            true_cfg_scale=request.cfg_scale,  # official kwarg — NOT guidance_scale
            generator=generator,
        )

        image = output.images[0]
        generation_time = time.perf_counter() - t_start

        # Release temporary computation buffers after generation
        # This does not free model weights but clears intermediate tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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