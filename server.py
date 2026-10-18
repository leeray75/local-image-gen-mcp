"""
Local Image Generation MCP Server
==================================
Model-agnostic MCP server for local text-to-image generation and editing.

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
  IMAGE_MODEL           Backend model id (default: qwen_image_2_1)
  IMAGE_OUTPUT_DIR      Directory to save generated images (default: /outputs)
  MAX_INFERENCE_STEPS   Optional override of the backend's default steps (clamped 20-100)
  DEFAULT_CFG_SCALE     Optional override of the backend's default CFG (clamped 1.0-10.0)
  IMAGE_GPU_MEMORY_LIMIT  Optional device-memory cap in GB (accelerate max_memory)
  MAX_REFERENCE_IMAGE_MB  Max decoded size of one edit reference (default: 20)
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

from config.settings import settings
from models import load_model
from models.base import BaseImageModel, GenerationRequest
from storage import store
from storage.references import resolve_references

log = logging.getLogger(__name__)

Ratio = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]

MIN_STEPS, MAX_STEPS = 20, 100
MIN_CFG, MAX_CFG = 1.0, 10.0


# ── Model singleton — loaded once at module import ────────────────────────────
# Avoids the per-session lifespan issue in mcp.server.fastmcp 1.27.0
# ─────────────────────────────────────────────────────────────────────────────
_model: BaseImageModel | None = None

# Only one pipeline call may run at a time (NFR-005-01)
_generation_lock = asyncio.Lock()

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
# streamable_http_path: MCP_PATH, also a constructor argument (FR-006-05).
# json_response=True:  required for Streamable HTTP per official SDK docs.
# ─────────────────────────────────────────────────────────────────────────────

# Transport-level worst case for edit_image: this many base64 references in one
# request. Matches the largest reference limit of any backend (Qwen-Image-2.1: 10).
# The server can't ask the model: it isn't loaded when FastMCP is constructed.
MAX_REQUEST_REFERENCES = 10

mcp = FastMCP(
    "local-image-gen",
    instructions=(
        "Image generation and editing server powered by a local diffusion model. "
        "Call generate_image with a text prompt to create images (set "
        "transparent=true for a PNG with a real alpha channel). "
        "Call edit_image with an instruction and 1+ reference images to edit or "
        "combine images (the result keeps the last reference's aspect ratio at "
        "about the server's edit_output_resolution, shown by get_model_info); "
        "references are image_ids returned by earlier calls "
        "(preferred, no bytes move) or base64 strings, so edits can be chained "
        "by image_id. "
        "Use list_aspect_ratios to see the output dimensions (about "
        "GENERATE_OUTPUT_RESOLUTION squared in area, default 1024). "
        "Use get_model_info to check GPU status, defaults and capabilities. "
        "Previously generated images can be retrieved via the image://{image_id} "
        "resource URI without re-generating them."
    ),
    json_response=True,
    host=settings.mcp_host,
    port=settings.mcp_port,
    streamable_http_path=settings.mcp_path,
    max_request_body_size=settings.effective_max_request_body_mb(MAX_REQUEST_REFERENCES) * 1024 * 1024,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_params(
    model: BaseImageModel,
    num_inference_steps: int | None,
    cfg_scale: float | None,
    negative_prompt: str | None,
) -> tuple[int, float, str | None, list[str]]:
    """Resolve steps, CFG and negative prompt (FR-005-06..08).

    Precedence: tool argument > env override (settings) > backend default.
    Steps are clamped to [20, 100] and CFG to [1.0, 10.0]. Returns
    (steps, cfg, negative_prompt, warnings).
    """
    steps = num_inference_steps
    if steps is None:
        steps = settings.max_inference_steps
    if steps is None:
        steps = model.default_inference_steps
    steps = max(MIN_STEPS, min(MAX_STEPS, int(steps)))

    cfg = cfg_scale
    if cfg is None:
        cfg = settings.default_cfg_scale
    if cfg is None:
        cfg = model.default_cfg_scale
    cfg = max(MIN_CFG, min(MAX_CFG, float(cfg)))

    neg = negative_prompt if negative_prompt else None  # "" == None
    if neg is None:
        neg = model.default_negative_prompt

    warnings: list[str] = []
    if neg and cfg <= 1.0:
        warnings.append(
            "negative_prompt is ignored unless cfg_scale > 1 "
            "(guidance is off at cfg_scale=1.0). Raise cfg_scale (e.g. 4.0) to "
            "use it; this roughly doubles the compute."
        )
    return steps, cfg, neg, warnings


def _check_transparency(model: BaseImageModel, transparent: bool) -> None:
    if transparent and not model.supports_transparency:
        raise ValueError(
            f"{model.model_name} does not support transparent output "
            "(transparent=true)."
        )


def _save_and_encode(pil_image, metadata: dict):
    """Blocking: persist to the store and PNG-encode. Call via asyncio.to_thread."""
    stored = store.save(pil_image, metadata)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return stored, buf.getvalue()


async def _run_generation(
    ctx: Context,
    model: BaseImageModel,
    request: GenerationRequest,
    metadata: dict,
) -> Image:
    """Run one pipeline call under the generation lock, store and return it."""
    steps = request.num_inference_steps
    await ctx.report_progress(0, steps)
    await ctx.info(
        f"{metadata['mode'].capitalize()}: aspect={metadata['aspect_ratio']} "
        f"steps={steps}, cfg={request.cfg_scale}, seed={request.seed}, "
        f"transparent={request.transparent}, "
        f"references={len(request.reference_images)}"
    )

    # Run generation in thread — model.generate() is synchronous and blocks.
    # Never call it directly in async context or you block the entire event loop.
    async with _generation_lock:
        result = await asyncio.to_thread(model.generate, request)

    await ctx.report_progress(steps, steps)
    log.info(
        "%s: completed in %.1fs (%dx%d)",
        metadata["mode"], result.generation_time, result.width, result.height,
    )

    # Disk write + PNG encode take seconds at large sizes: keep them off the event loop
    stored, png_bytes = await asyncio.to_thread(_save_and_encode, result.image, {
        **metadata,
        "width": result.width,
        "height": result.height,
        "seed": request.seed,
        "generation_time": result.generation_time,
        "model_name": model.model_name,
    })

    await ctx.info(
        f"Image saved: id={stored.image_id} "
        f"({result.width}x{result.height}, seed={request.seed}, "
        f"{result.generation_time:.1f}s"
        f"{', transparent' if request.transparent else ''}) — "
        f"retrieve via image://{stored.image_id}"
    )

    # IMPORTANT: Image(data=...) expects raw bytes, NOT base64-encoded bytes
    return Image(data=png_bytes, format="png")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def generate_image(
    prompt: str,
    aspect_ratio: Ratio = "1:1",
    transparent: bool = False,
    negative_prompt: str | None = None,
    num_inference_steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    ctx: Context = None,
) -> Image:
    """
    Generate an image from a text prompt using the local Qwen-Image-2.1 model.

    Returns the image directly as PNG (RGB, or RGBA when transparent=true). The
    image is also persisted and retrievable later via image://{image_id}; use
    get_latest_image_id to learn its ID, and pass that ID to edit_image to refine
    it. Output is about GENERATE_OUTPUT_RESOLUTION^2 pixels of area (default 1024,
    so 1:1 = 1024x1024; see list_aspect_ratios for the exact sizes).

    Guidance is OFF by default (cfg_scale=1.0, 40 steps). That is the model's
    tuned setting and the fastest. negative_prompt only takes effect when
    cfg_scale > 1 (roughly doubles the compute); otherwise it is ignored and a
    warning is sent.

    Results are large PNGs (several MB). To save one client-side, call
    get_image_base64 with its ID; prefer image IDs over base64 for any chained
    edits.

    Args:
        prompt:               Text description. English and Chinese both work.
        aspect_ratio:         One of 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3. Default 1:1.
        transparent:          true = RGBA output with a real alpha channel
                              (stickers, icons, cut-outs). false = opaque RGB.
        negative_prompt:      What to exclude. Default none. Needs cfg_scale > 1.
        num_inference_steps:  Range 20-100 (clamped). Default 40.
        cfg_scale:            Range 1.0-10.0 (clamped). Default 1.0 (guidance off).
        seed:                 Reproducibility seed. None = random.
    """
    model = get_model()
    _check_transparency(model, transparent)

    if aspect_ratio not in model.aspect_ratios:
        raise ValueError(
            f"Invalid aspect_ratio '{aspect_ratio}'. "
            f"Must be one of: {sorted(model.aspect_ratios)}"
        )
    width, height = model.aspect_ratios[aspect_ratio]

    steps, cfg, neg, warnings = _resolve_params(
        model, num_inference_steps, cfg_scale, negative_prompt
    )
    for w in warnings:
        await ctx.warning(w)

    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**32)

    log.info(
        "generate_image: aspect=%s (%dx%d) steps=%d cfg=%.1f seed=%d transparent=%s",
        aspect_ratio, width, height, steps, cfg, actual_seed, transparent,
    )

    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg,
        seed=actual_seed,
        reference_images=[],
        transparent=transparent,
    )
    return await _run_generation(ctx, model, request, {
        "prompt": prompt,
        "negative_prompt": neg or "",
        "aspect_ratio": aspect_ratio,
        "mode": "generate",
        "transparent": transparent,
        "reference_image_ids": [],
        "reference_count": 0,
    })


@mcp.tool()
async def edit_image(
    prompt: str,
    images: list[str],
    aspect_ratio: Ratio | None = None,
    transparent: bool = False,
    negative_prompt: str | None = None,
    num_inference_steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    ctx: Context = None,
) -> Image:
    """
    Edit or combine images with a text instruction (Qwen-Image-2.1).

    Returns the edited image directly as PNG (RGB, or RGBA when transparent=true).
    The result is stored and gets a NEW image_id (see get_latest_image_id), which
    you can pass as a reference to a further edit_image call to chain edits.

    References (images): an ordered list of 1-10 strings. Each entry is either
      - an image_id (UUID) from generate_image, edit_image or images://list
        (PREFERRED: no bytes move over the wire), or
      - a base64-encoded PNG, JPEG or WebP, optionally with a
        "data:image/png;base64," prefix. Each must be <= MAX_REFERENCE_IMAGE_MB
        (default 20 MB) once decoded. Uploaded images are NOT stored.
    Order matters: entry N is "Picture N" in your prompt, e.g. "Replace the
    background of Picture 1 with the scene from Picture 2". Mixing IDs and
    base64 is fine. Base64 references and the returned PNG are large payloads, so
    use image IDs for chained edits. The whole HTTP request (all base64
    references plus JSON overhead) must fit MAX_REQUEST_BODY_MB (default 268 MB);
    over that the server answers HTTP 413 before the tool runs. Prefer image IDs
    for large or chained images. All references are validated before any
    model work; an error names the offending reference by 1-based position.

    Output size: with aspect_ratio omitted, the result matches the aspect ratio
    of the LAST reference at about EDIT_OUTPUT_RESOLUTION^2 pixels of area (see
    edit_output_resolution in get_model_info; default 1024). With aspect_ratio
    set, the result uses that ratio's size at the same EDIT_OUTPUT_RESOLUTION
    (not the generate_image size table).

    Guidance is OFF by default (cfg_scale=1.0, 40 steps). negative_prompt only
    takes effect when cfg_scale > 1; otherwise it is ignored and a warning is
    sent. Editing at a high EDIT_OUTPUT_RESOLUTION with many references needs a lot of memory; the server
    runs one generation at a time.

    Args:
        prompt:               The edit instruction. May refer to "Picture 1..N".
        images:               Ordered list of 1-10 image_ids and/or base64 images.
        aspect_ratio:         Optional: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2 or 2:3.
                              Default (null) = follow the last reference.
        transparent:          true = RGBA output (e.g. extract a subject onto a
                              transparent background). false = opaque RGB.
        negative_prompt:      What to exclude. Default none. Needs cfg_scale > 1.
        num_inference_steps:  Range 20-100 (clamped). Default 40.
        cfg_scale:            Range 1.0-10.0 (clamped). Default 1.0 (guidance off).
        seed:                 Reproducibility seed. None = random.
    """
    model = get_model()
    if model.max_reference_images == 0:
        raise ValueError(f"{model.model_name} does not support image editing.")
    _check_transparency(model, transparent)

    if aspect_ratio is not None and aspect_ratio not in model.aspect_ratios:
        raise ValueError(
            f"Invalid aspect_ratio '{aspect_ratio}'. "
            f"Must be one of: {sorted(model.aspect_ratios)}"
        )

    # Validate and decode references BEFORE taking the lock or touching the model
    max_bytes = settings.max_reference_image_mb * 1024 * 1024
    ref_images, ref_ids = await asyncio.to_thread(
        resolve_references, images, store, model.max_reference_images, max_bytes
    )

    if aspect_ratio is not None:
        # F-5: size from the edit resolution, not the generate table
        width, height = model.sizes_for(model.edit_output_resolution)[aspect_ratio]
    else:
        width = height = None

    steps, cfg, neg, warnings = _resolve_params(
        model, num_inference_steps, cfg_scale, negative_prompt
    )
    for w in warnings:
        await ctx.warning(w)

    actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**32)

    log.info(
        "edit_image: refs=%d aspect=%s steps=%d cfg=%.1f seed=%d transparent=%s",
        len(ref_images), aspect_ratio or "input", steps, cfg, actual_seed,
        transparent,
    )

    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_inference_steps=steps,
        cfg_scale=cfg,
        seed=actual_seed,
        reference_images=ref_images,
        transparent=transparent,
    )
    return await _run_generation(ctx, model, request, {
        "prompt": prompt,
        "negative_prompt": neg or "",
        "aspect_ratio": aspect_ratio or "input",
        "mode": "edit",
        "transparent": transparent,
        "reference_image_ids": ref_ids,
        "reference_count": len(ref_images),
    })


@mcp.tool()
def list_aspect_ratios() -> dict:
    """
    List all supported aspect ratios and their pixel dimensions.

    Returns a mapping of ratio string to width/height dict for the active model
    for generate_image (about GENERATE_OUTPUT_RESOLUTION^2 pixels of area,
    default 1024, e.g. 1:1 = 1024x1024). edit_image with an aspect_ratio uses
    the same ratios at EDIT_OUTPUT_RESOLUTION (see get_model_info).
    Use this before calling generate_image or edit_image to choose the format.

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
        for ratio, (w, h) in get_model().aspect_ratios.items()
    }


@mcp.tool()
def get_model_info() -> dict:
    """
    Return model metadata, capabilities, resolved defaults and GPU memory status.

    Reports max_reference_images, supports_editing, supports_transparency,
    edit_output_resolution (the edit size target as a side length in pixels;
    null if editing is unsupported), generate_output_resolution (the same for
    generate_image), max_output_resolution (the configured ceiling for both)
    and the default steps/CFG that apply when
    you pass null (after env overrides). Also reports quantize_transformer and
    quantize_text_encoder ("none", "int8", "fp8" or "nvfp4", the effective modes after "auto" is resolved; configured_* appears when it differs) and, once loaded,
    weight_memory_gb (per-component resident weight size as stored).
    Useful for verifying the server is ready before requesting large images.
    """
    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps, cfg, neg, _ = _resolve_params(model, None, None, None)

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
        "default_inference_steps": steps,
        "default_cfg_scale": cfg,
        "has_default_negative_prompt": neg is not None,
        "max_reference_images": model.max_reference_images,
        "supports_editing": model.supports_editing,
        "supports_transparency": model.supports_transparency,
        "edit_output_resolution": model.edit_output_resolution,
        "generate_output_resolution": settings.generate_output_resolution,
        "max_output_resolution": settings.max_output_resolution,
        "supported_aspect_ratios": sorted(model.aspect_ratios),
        "images_generated": store.count(),
        "output_directory": settings.image_output_dir,
        **model.quantization_info,
        **gpu_info,
    }


@mcp.tool()
async def get_image_base64(image_id: str) -> str:
    """
    Return a previously generated image as a base64-encoded PNG string.

    Use this when you need to save the image to a client-side filesystem.
    The returned string is plain base64 — decode it to binary before writing
    to a .png file. Writing the raw string to a .png file will NOT work.

    Correct client-side usage (Python):
        import base64
        data = get_image_base64(image_id)
        with open('output.png', 'wb') as f:
            f.write(base64.b64decode(data))

    Args:
        image_id: UUID returned by generate_image or from images://list resource.

    Returns:
        Base64-encoded PNG string (plain text, not a data URI).
    """
    import base64 as b64
    def _read_and_encode() -> str:
        return b64.b64encode(store.get_image_bytes(image_id)).decode("utf-8")

    return await asyncio.to_thread(_read_and_encode)


@mcp.tool()
def get_latest_image_id() -> dict:
    """
    Return the image_id of the most recently generated or edited image.

    Use this immediately after generate_image or edit_image to get the image_id without
    having to parse the images://list resource. Combines with get_image_base64
    for a two-step save workflow:
        1. id = get_latest_image_id()["image_id"]
        2. b64 = get_image_base64(id)
        3. decode and write to .png on client filesystem

    Returns:
        dict with keys:
          image_id:     UUID of the most recent image
          prompt:       First 100 chars of the prompt used
          aspect_ratio: e.g. "16:9"
          width:        Pixel width
          height:       Pixel height
          seed:         Seed used for generation
          model_name:   Model that generated it
          mode:         "generate" or "edit"
          transparent:  True if the image is RGBA

    Raises:
        RuntimeError if no images have been generated yet this session.
    """
    images = store.list_all()
    if not images:
        raise RuntimeError(
            "No images have been generated in this session. "
            "Call generate_image first."
        )
    latest = images[0]  # list_all() returns newest first
    return {
        "image_id": latest.image_id,
        "prompt": latest.prompt[:100],
        "aspect_ratio": latest.aspect_ratio,
        "width": latest.width,
        "height": latest.height,
        "seed": latest.seed,
        "model_name": latest.model_name,
        "mode": latest.mode,
        "transparent": latest.transparent,
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
    seed, model name, mode (generate/edit) and transparent flag. Use image_id values with image://{image_id} to
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
                "mode": img.mode,
                "transparent": img.transparent,
            }
            for img in images
        ],
        indent=2,
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def photorealistic(subject: str, setting: str = "", style: str = "") -> str:
    """
    Enhance a prompt for photorealistic output.

    Photorealism improves when prompts include lighting conditions, camera
    characteristics, and fine detail descriptors.

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
def transparent_asset(subject: str, style: str = "") -> str:
    """
    Build a prompt for a standalone object on a transparent background.

    Use the result with generate_image(transparent=true).

    Args:
        subject:  The object to render (e.g. "a red ceramic coffee mug")
        style:    Optional style (e.g. "flat vector sticker, bold outline")
    """
    return (
        f"{subject}, a single standalone object, centered, clean edges, "
        f"no background elements"
        f"{', ' + style if style else ''}."
    )


@mcp.prompt()
def extract_subject(subject: str = "the main subject") -> str:
    """
    Build an edit instruction that cuts a subject out onto a transparent background.

    Use the result with edit_image(images=[photo], transparent=true).

    Args:
        subject:  What to extract from Picture 1 (default "the main subject")
    """
    return (
        f"Extract {subject} from Picture 1 and place it on a fully transparent "
        f"background. Keep its edges, colors and details exactly; remove "
        f"everything else."
    )


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
