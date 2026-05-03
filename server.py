"""
Local Image Generation MCP Server
==================================
Model-agnostic MCP server for local text-to-image generation.

Architecture:
  - FastMCP with lifespan pattern — pipeline loaded ONCE at startup
  - Model backends in models/ — swap by changing IMAGE_MODEL env var
  - Thread-safe image storage in storage/
  - All config via environment variables (config/settings.py)

Transport: Streamable HTTP (production standard)
SDK:       mcp 1.27.0 — FastMCP

Tools exposed:
  - generate_image      Generate an image from a text prompt
  - list_aspect_ratios  List supported aspect ratios and dimensions
  - get_model_info      Return model metadata and current GPU status

Resources exposed:
  - image://{image_id}  Retrieve a previously generated image
  - images://list        List all generated images with metadata

Prompts exposed:
  - photorealistic       Enhance a prompt for photorealistic output
  - negative_defaults    Return the recommended negative prompt

Usage:
  uv run server.py
  # or
  python server.py

Environment variables:
  MCP_HOST              Host to bind to (default: 0.0.0.0)
  MCP_PORT              Port to bind to (default: 8500)
  MCP_PATH              MCP endpoint path (default: /mcp)
  IMAGE_MODEL           Backend model id (default: qwen_image_2512)
  IMAGE_OUTPUT_DIR      Directory to save generated images (default: /outputs)
  MAX_INFERENCE_STEPS   Diffusion steps (default: 50, range 20-100)
  DEFAULT_CFG_SCALE     Classifier-free guidance scale (default: 4.0)
  TORCH_DTYPE           torch dtype: bfloat16 or float32 (default: bfloat16)
  HF_TOKEN              HuggingFace API token (read from environment only)
"""

import asyncio
import io
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from mcp.server.fastmcp import Context, FastMCP, Image

from config.aspect_ratios import ASPECT_RATIOS, get_dimensions, valid_ratios
from config.settings import settings
from models import load_model
from models.base import BaseImageModel, GenerationRequest
from models.qwen_image_2512 import DEFAULT_NEGATIVE_PROMPT
from storage import store
from storage.image_store import StoredImage

log = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────────

@dataclass
class AppContext:
    """Application-level context shared across all requests."""
    model: BaseImageModel


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Server lifespan — runs ONCE at startup and ONCE at shutdown.

    The model pipeline is loaded here (blocking call run in thread pool)
    and shared across all tool calls via ctx.request_context.lifespan_context.
    """
    log.info("Loading image model: %s", settings.image_model)
    model = load_model()
    # load() is blocking (GPU memory allocation) — run in thread executor
    await asyncio.to_thread(model.load)
    log.info("Model loaded: %s (%s)", model.model_name, model.is_loaded())
    try:
        yield AppContext(model=model)
    finally:
        log.info("Unloading model")
        await asyncio.to_thread(model.unload)
        log.info("Model unloaded")


# ── Server ───────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "local-image-gen",
    instructions=(
        "Image generation server. "
        "Call generate_image with a text prompt to create images. "
        "Use list_aspect_ratios to see supported dimensions. "
        "Use get_model_info to check GPU status. "
        "Previously generated images can be retrieved via image://{image_id} resource URI."
    ),
    lifespan=app_lifespan,
    json_response=True,
)


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def generate_image(
    prompt: str,
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"] = "1:1",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = 50,
    cfg_scale: float = 4.0,
    seed: int | None = None,
    ctx: Context = None,
) -> Image:
    """
    Generate an image from a text prompt.

    Returns the image directly. The image is also stored and retrievable
    via the image://{image_id} resource URI.

    Args:
        prompt: Text description. English and Chinese both work well.
        aspect_ratio: Output dimensions. Use list_aspect_ratios to see options.
        negative_prompt: What to exclude. Default covers common AI artifacts.
        num_inference_steps: Quality/speed tradeoff. Range 20-100. Default 50.
        cfg_scale: Prompt adherence strength. Range 1.0-10.0. Default 4.0.
        seed: Reproducibility seed. None = random.
    """
    # Resolve seed
    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**32)

    # Get width/height from aspect ratio
    width, height = get_dimensions(aspect_ratio)

    # Clamp inference steps
    num_inference_steps = max(20, min(100, num_inference_steps))

    # Get model from lifespan context
    model = ctx.request_context.lifespan_context.model

    # Report initial progress
    await ctx.report_progress(0, num_inference_steps)

    log.info(
        "Generating image: aspect=%s (%dx%d), steps=%d, cfg=%.1f, seed=%d",
        aspect_ratio, width, height, num_inference_steps, cfg_scale, actual_seed,
    )

    # Build request
    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        seed=actual_seed,
    )

    # Run in thread — generate() is synchronous and blocks (GPU diffusion)
    result = await asyncio.to_thread(model.generate, request)

    # Report final progress
    await ctx.report_progress(num_inference_steps, num_inference_steps)

    log.info("Image generated in %.1fs", result.generation_time)

    # Save to store
    stored = store.save(result.image, {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "seed": actual_seed,
        "generation_time": result.generation_time,
        "model_name": model.model_name,
    })

    # Convert to bytes for Image return type
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    buf.seek(0)

    await ctx.info(
        f"Image generated: {stored.image_id} ({width}x{height}, seed={actual_seed}, "
        f"{result.generation_time}s)"
    )

    return Image(data=buf.read(), format="png")


@mcp.tool()
def list_aspect_ratios() -> dict:
    """
    List all supported aspect ratios and their pixel dimensions.

    Returns a dict mapping aspect ratio strings to their width/height.
    Use this to choose the right dimensions before calling generate_image.

    Common use cases:
      1:1  — social media posts, avatars
      16:9 — widescreen, presentations, desktop wallpapers
      9:16 — mobile/portrait, Instagram Stories, TikTok
      4:3  — traditional photo format, presentations
      3:2  — standard camera/DSLR aspect ratio
    """
    return {
        r: {"width": w, "height": h}
        for r, (w, h) in ASPECT_RATIOS.items()
    }


@mcp.tool()
def get_model_info(ctx: Context) -> dict:
    """
    Return model metadata and current GPU memory status.

    Useful for checking if the model is loaded and monitoring GPU usage.
    """
    model = ctx.request_context.lifespan_context.model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            ),
            "gpu_memory_allocated_gb": round(
                torch.cuda.memory_allocated(0) / 1e9, 2
            ),
        }

    return {
        "model_name": model.model_name,
        "backend": settings.image_model,
        "is_loaded": model.is_loaded(),
        "device": device,
        "torch_dtype": settings.torch_dtype,
        "default_inference_steps": settings.max_inference_steps,
        "default_cfg_scale": settings.default_cfg_scale,
        "supported_aspect_ratios": valid_ratios(),
        "images_generated": store.count(),
        "output_directory": settings.image_output_dir,
        **gpu_info,
    }


# ── Resources ────────────────────────────────────────────────────────────────

@mcp.resource("image://{image_id}")
def get_image_resource(image_id: str) -> Image:
    """Retrieve a previously generated image by ID."""
    img_bytes = store.get_image_bytes(image_id)
    return Image(data=img_bytes, format="png")


@mcp.resource("images://list")
def list_images() -> str:
    """List all generated images with metadata as JSON string."""
    images = store.list_all()
    return json.dumps([
        {
            "image_id": img.image_id,
            "prompt": img.prompt[:100],
            "aspect_ratio": img.aspect_ratio,
            "seed": img.seed,
            "model": img.model_name,
        }
        for img in images
    ], indent=2)


# ── Prompts ──────────────────────────────────────────────────────────────────

@mcp.prompt()
def photorealistic(subject: str, setting: str = "", style: str = "") -> str:
    """
    Enhance a prompt for photorealistic output with Qwen-Image-2512.

    Args:
        subject: What or who to photograph (e.g. "a golden retriever")
        setting: Where the scene takes place (e.g. "in a sunlit park")
        style: Additional style guidance (e.g. "close-up portrait, shallow DOF")
    """
    parts = [subject]
    if setting:
        parts.append(f"in {setting}")
    base = ", ".join(parts)

    return (
        f"{base}. "
        f"Ultra-realistic photography, shot on a professional camera. "
        f"Natural lighting with soft shadows. High resolution, sharp focus, "
        f"fine detail in textures and surfaces. "
        f"{style + '. ' if style else ''}"
        f"Photojournalistic composition. No AI-generated artifacts."
    )


@mcp.prompt()
def negative_defaults() -> str:
    """Return the recommended default negative prompt for Qwen-Image-2512."""
    return DEFAULT_NEGATIVE_PROMPT


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info(
        "Starting local-image-gen MCP server on %s:%d%s",
        settings.mcp_host, settings.mcp_port, settings.mcp_path,
    )
    log.info("Image output directory: %s", settings.image_output_dir)

    mcp.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        path=settings.mcp_path,
    )