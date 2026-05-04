"""
Local Image Generation MCP Server
==================================
Model-agnostic MCP server for local text-to-image generation.

Architecture:
  - Module-level singleton — model loaded ONCE at process start (no lifespan)
  - Model backends in models/ — swap by changing IMAGE_MODEL env var
  - Thread-safe image storage in storage/
  - All config via environment variables (config/settings.py)

Transport: Streamable HTTP (MCP 2025 production standard)
SDK:       mcp 1.27.0 — mcp.server.fastmcp.FastMCP

IMPORTANT — mcp.server.fastmcp vs fastmcp package:
  This uses the OFFICIAL SDK (mcp.server.fastmcp), NOT the standalone
  fastmcp package. In the official SDK, host/port are passed to the
  FastMCP() constructor, NOT to run(). Passing them to run() raises:
    TypeError: FastMCP.run() got an unexpected keyword argument 'host'

Usage:
  python server.py
  # or
  uv run server.py

Environment variables (all read from environment, never hardcoded):
  MCP_HOST              Host to bind to (default: 0.0.0.0)
  MCP_PORT              Port to bind to (default: 8500)
  MCP_PATH              URL path (default: /mcp)
  IMAGE_MODEL           Backend model id (default: qwen_image_2512)
  IMAGE_OUTPUT_DIR      Directory to save generated images (default: /outputs)
  MAX_INFERENCE_STEPS   Diffusion steps (default: 50, range 20-100)
  DEFAULT_CFG_SCALE     Classifier-free guidance scale (default: 4.0)
  TORCH_DTYPE           torch dtype: bfloat16 or float32 (default: bfloat16)
  HF_TOKEN              HuggingFace API token — read from ~/.bashrc, never written
"""

import asyncio
import io
import json
import logging
import time
from typing import Literal

import torch

from mcp.server.fastmcp import Context, FastMCP, Image

from config.aspect_ratios import ASPECT_RATIOS, get_dimensions, valid_ratios
from config.settings import settings
from models import load_model
from models.base import BaseImageModel, GenerationRequest
from models.qwen_image_2512 import DEFAULT_NEGATIVE_PROMPT
from storage import store

log = logging.getLogger(__name__)


# ── Model singleton — loaded once at module import ────────────────────────────
# Avoids the per-session lifespan issue in mcp.server.fastmcp 1.27.0
# ─────────────────────────────────────────────────────────────────────────────
_model: BaseImageModel | None = None

def get_model() -> BaseImageModel:
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded — server startup incomplete")
    return _model

def _startup_load_model() -> None:
    """Called once at process start, before the HTTP server accepts connections."""
    global _model
    log.info("Loading image model: %s", settings.image_model)
    _model = load_model()
    _model.load()
    log.info("Model loaded: %s", _model.model_name)


# ── Server — NO lifespan ──────────────────────────────────────────────────────
# host and port are passed to FastMCP() constructor — NOT to run().
# In mcp.server.fastmcp (official SDK), run() only accepts 'transport'.
# json_response=True:  required for Streamable HTTP per official SDK docs.
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "local-image-gen",
    instructions=(
        "Image generation server powered by a local diffusion model. "
        "Call generate_image with a text prompt to create images. "
        "Use list_aspect_ratios to see supported output dimensions. "
        "Use get_model_info to check GPU status and current configuration. "
        "Previously generated images can be retrieved via the image://{image_id} "
        "resource URI without re-generating them."
    ),
    json_response=True,
    host=settings.mcp_host,
    port=settings.mcp_port,
)


# ── Tools ─────────────────────────────────────────────────────────────────────

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
    Generate an image from a text prompt using the local diffusion model.

    Returns the image directly as PNG. The image is also persisted to disk
    and retrievable later via the image://{image_id} resource URI.

    Best practices for Qwen-Image-2512:
      - English and Chinese prompts both work well
      - Include lighting, camera, and texture details for photorealism
      - The default negative_prompt covers common AI artifacts — override if needed
      - Use seed for reproducibility; omit for random variation

    Args:
        prompt:               Text description of the image to generate.
        aspect_ratio:         Output dimensions. Use list_aspect_ratios for options.
        negative_prompt:      What to exclude. Default covers deformations, blur, AI look.
        num_inference_steps:  Quality/speed tradeoff. Range 20–100. Default 50.
        cfg_scale:            Prompt adherence. Range 1.0–10.0. Default 4.0.
        seed:                 Reproducibility seed. None = random.
    """
    # Resolve seed
    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**32)

    # Resolve pixel dimensions from aspect ratio
    width, height = get_dimensions(aspect_ratio)

    # Clamp inference steps to safe range
    num_inference_steps = max(20, min(100, num_inference_steps))

    # Access the loaded model from module-level singleton
    model = get_model()

    # Report generation start
    await ctx.report_progress(0, num_inference_steps)
    await ctx.info(
        f"Generating {aspect_ratio} ({width}x{height}) image, "
        f"steps={num_inference_steps}, cfg={cfg_scale}, seed={actual_seed}"
    )

    log.info(
        "generate_image: aspect=%s (%dx%d) steps=%d cfg=%.1f seed=%d",
        aspect_ratio, width, height, num_inference_steps, cfg_scale, actual_seed,
    )

    # Build the generation request
    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        seed=actual_seed,
    )

    # Run generation in thread — model.generate() is synchronous and blocks
    # Never call it directly in async context or you block the entire event loop
    result = await asyncio.to_thread(model.generate, request)

    # Report completion
    await ctx.report_progress(num_inference_steps, num_inference_steps)

    log.info("generate_image: completed in %.1fs", result.generation_time)

    # Persist to store (disk + in-memory index)
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

    await ctx.info(
        f"Image saved: id={stored.image_id} "
        f"({width}x{height}, seed={actual_seed}, {result.generation_time:.1f}s) — "
        f"retrieve via image://{stored.image_id}"
    )

    # Convert PIL image to bytes for MCP Image return
    # IMPORTANT: Image(data=...) expects raw bytes, NOT base64-encoded bytes
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    buf.seek(0)

    return Image(data=buf.read(), format="png")


@mcp.tool()
def list_aspect_ratios() -> dict:
    """
    List all supported aspect ratios and their pixel dimensions.

    Returns a mapping of ratio string to width/height dict.
    Use this before calling generate_image to choose the right format.

    Common use cases:
      1:1  — social media posts, avatars, thumbnails
      16:9 — widescreen, presentations, desktop wallpapers
      9:16 — mobile/portrait, Stories, Reels
      4:3  — traditional photo, slides
      3:2  — standard DSLR/camera format
      2:3  — portrait standard
      3:4  — portrait photo
    """
    return {
        ratio: {"width": w, "height": h}
        for ratio, (w, h) in ASPECT_RATIOS.items()
    }


@mcp.tool()
def get_model_info() -> dict:
    """
    Return model metadata and current GPU memory status.

    Useful for verifying the server is ready and checking available GPU memory
    before requesting large images or many inference steps.
    """
    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(props.total_memory / 1e9, 1),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
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


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("image://{image_id}")
def get_image_resource(image_id: str) -> Image:
    """
    Retrieve a previously generated image by its ID.

    Images are stored for the lifetime of the server process and persisted
    to disk at IMAGE_OUTPUT_DIR. Use the image_id returned by generate_image.
    """
    img_bytes = store.get_image_bytes(image_id)
    return Image(data=img_bytes, format="png")


@mcp.resource("images://list")
def list_images_resource() -> str:
    """
    List all generated images with metadata as a JSON string.

    Returns an array of objects with image_id, prompt preview, aspect_ratio,
    seed, and model name. Use image_id values with image://{image_id} to
    retrieve specific images.
    """
    images = store.list_all()
    return json.dumps(
        [
            {
                "image_id": img.image_id,
                "prompt": img.prompt[:100] + ("..." if len(img.prompt) > 100 else ""),
                "aspect_ratio": img.aspect_ratio,
                "width": img.width,
                "height": img.height,
                "seed": img.seed,
                "model": img.model_name,
                "generation_time_seconds": img.generation_time,
            }
            for img in images
        ],
        indent=2,
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def photorealistic(subject: str, setting: str = "", style: str = "") -> str:
    """
    Enhance a prompt for photorealistic output with Qwen-Image-2512.

    Qwen-Image-2512 excels at photorealism when prompts include lighting
    conditions, camera characteristics, and fine detail descriptors.

    Args:
        subject:  What or who to photograph (e.g. "a golden retriever")
        setting:  Where the scene takes place (e.g. "in a sunlit park")
        style:    Additional style guidance (e.g. "close-up, shallow DOF")
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
    """
    Return the recommended default negative prompt for Qwen-Image-2512.

    Includes Chinese terms intentionally — the model was trained heavily on
    Chinese data and responds well to Chinese guidance even in English sessions.
    Use as a starting point and customize for specific use cases.
    """
    return DEFAULT_NEGATIVE_PROMPT


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info(
        "Starting local-image-gen MCP server on %s:%d%s",
        settings.mcp_host,
        settings.mcp_port,
        settings.mcp_path,
    )
    log.info("Model backend: %s", settings.image_model)
    log.info("Image output directory: %s", settings.image_output_dir)
    log.info("Torch dtype: %s", settings.torch_dtype)

    # Load model ONCE before the HTTP server starts accepting connections
    _startup_load_model()

    # CORRECT: run() only takes 'transport' in mcp.server.fastmcp
    # host and port were already passed to the FastMCP() constructor above
    # DO NOT pass host= or port= here — raises TypeError in mcp 1.27.0
    mcp.run(transport="streamable-http")
