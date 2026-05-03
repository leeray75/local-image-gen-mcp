# local-image-gen-mcp — Implementation Plan
# For: Qwen3.6-27B via Cline Code
# SDK: mcp 1.27.0 (mcp.server.fastmcp), Python 3.11+

---

## CONTEXT

You are implementing a production MCP server that wraps Qwen-Image-2512 text-to-image
diffusion model. The scaffolding exists. Existing files were copied into the tree:
- `server.py` — partial, monolithic, missing architecture
- `README.md` — partial
- `Dockerfile` — in `docker/`
- `docker-compose.yml` — in `docker/`
- `pyproject.toml` — root

HF_TOKEN is in ~/.bashrc — do NOT add it to any file. Read it from environment only.
All env vars are read via `os.getenv()` or Pydantic `BaseSettings`.

Current tree (already exists — do not re-scaffold):
```
local-image-gen-mcp/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── server.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   └── qwen_image_2512.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── aspect_ratios.py
├── storage/
│   ├── __init__.py
│   └── image_store.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.override.yml
└── tests/
    ├── __init__.py
    ├── test_tools.py
    └── test_models.py
```

---

## MCP SDK REFERENCE (mcp 1.27.0)

CRITICAL — use exactly these imports. Do not guess or use older patterns.

```python
# Server creation
from mcp.server.fastmcp import FastMCP, Context, Image

# Lifespan pattern (official SDK 1.27 pattern)
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

@dataclass
class AppContext:
    pipeline: Any  # your loaded resource

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # runs ONCE at server startup
    resource = await load_something()
    try:
        yield AppContext(pipeline=resource)
    finally:
        # runs ONCE at server shutdown
        del resource

mcp = FastMCP(
    "server-name",
    lifespan=app_lifespan,
    json_response=True,       # required for Streamable HTTP scalability
)

# Access lifespan context inside a tool
@mcp.tool()
def my_tool(ctx: Context) -> str:
    pipeline = ctx.request_context.lifespan_context.pipeline
    return pipeline.do_something()

# Image return type — FastMCP handles base64 encoding automatically
from mcp.server.fastmcp import Image
@mcp.tool()
def my_tool() -> Image:
    img_bytes = generate_image_bytes()
    return Image(data=img_bytes, format="png")

# Run with Streamable HTTP (production transport)
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8500, path="/mcp")
```

Tool decorator options:
```python
@mcp.tool()                           # basic
@mcp.tool(description="override")    # override docstring
```

Resource pattern:
```python
@mcp.resource("image://{image_id}")
def get_image(image_id: str) -> str:
    return base64_string
```

Prompt pattern:
```python
@mcp.prompt()
def my_prompt(subject: str) -> str:
    return f"Enhanced: {subject}"
```

Progress reporting inside async tools:
```python
@mcp.tool()
async def long_tool(ctx: Context) -> str:
    await ctx.report_progress(0, 50)
    # ... work ...
    await ctx.report_progress(50, 50)
    return "done"
```

Logging inside tools:
```python
@mcp.tool()
async def my_tool(ctx: Context) -> str:
    await ctx.info("Starting generation")
    await ctx.warning("Low memory")
    return "result"
```

---

## IMPLEMENTATION TASKS

Complete each task in order. Each task is self-contained.
Read the existing file before writing. Preserve useful content from the partial `server.py`.

---

### TASK 1 — `config/aspect_ratios.py`

Write a module containing ONLY the aspect ratio data.
No classes. Just a plain dict and a helper function.

```python
# Exact pixel dimensions from official Qwen-Image-2512 model card
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1":  (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928,  1664),
    "4:3":  (1472, 1104),
    "3:4":  (1104, 1472),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
}

def get_dimensions(ratio: str) -> tuple[int, int]:
    """Return (width, height) for a ratio string. Raises ValueError if invalid."""
    ...

def valid_ratios() -> list[str]:
    """Return sorted list of valid ratio strings."""
    ...
```

---

### TASK 2 — `config/settings.py`

Use Pydantic v2 `BaseSettings`. Read ALL config from environment.
Do NOT hardcode any values. Do NOT reference .env file path (handled by docker).

```python
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # MCP server
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8500
    mcp_path: str = "/mcp"

    # Model selection — controls which backend in models/ is loaded
    image_model: Literal["qwen_image_2512"] = "qwen_image_2512"

    # Storage
    image_output_dir: str = "/outputs"

    # Diffusion defaults
    max_inference_steps: int = 50
    default_cfg_scale: float = 4.0
    torch_dtype: Literal["bfloat16", "float32"] = "bfloat16"

    # HuggingFace — from ~/.bashrc, never hardcoded
    hf_token: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Singleton
settings = Settings()
```

---

### TASK 3 — `models/base.py`

Abstract base class. This is the interface contract all model backends must satisfy.
Using `abc.ABC` and `abc.abstractmethod`.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL.Image import Image as PILImage

@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_inference_steps: int
    cfg_scale: float
    seed: int

@dataclass
class GenerationResult:
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
        """Load model weights into memory. Called once at startup."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory. Called at shutdown."""
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
```

---

### TASK 4 — `models/qwen_image_2512.py`

Concrete implementation of BaseImageModel for Qwen-Image-2512.

Key implementation notes:
- Use `DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=...)`
- `true_cfg_scale` is the correct kwarg (not `guidance_scale`) per official model card
- `torch.Generator(device=device).manual_seed(seed)` for reproducibility
- `generate()` must be synchronous — it will be called via `asyncio.to_thread()`
- Default negative prompt must include the official Chinese terms from model card

```python
DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    "low quality, blurry, deformed hands, extra fingers, watermark, "
    "text overlay, oversaturated, plastic skin, noise, compression artifacts"
)
```

Class structure:
```python
import torch
import time
from diffusers import DiffusionPipeline
from .base import BaseImageModel, GenerationRequest, GenerationResult

class QwenImage2512(BaseImageModel):
    MODEL_ID = "Qwen/Qwen-Image-2512"
    
    def __init__(self, torch_dtype_str: str = "bfloat16"):
        self._pipeline = None
        self._torch_dtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float32
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def is_loaded(self) -> bool: ...
    
    @property
    def model_name(self) -> str:
        return "Qwen-Image-2512"
```

---

### TASK 5 — `models/__init__.py`

Factory function that returns the correct backend based on settings.

```python
from config.settings import settings
from .base import BaseImageModel

def load_model() -> BaseImageModel:
    """Instantiate and return the configured model backend."""
    if settings.image_model == "qwen_image_2512":
        from .qwen_image_2512 import QwenImage2512
        return QwenImage2512(torch_dtype_str=settings.torch_dtype)
    raise ValueError(f"Unknown image_model: {settings.image_model}")
```

---

### TASK 6 — `storage/image_store.py`

Thread-safe in-memory store + disk persistence for generated images.
Images are stored in memory for fast retrieval and saved to disk for persistence.

```python
import uuid
import threading
from pathlib import Path
from dataclasses import dataclass, field
from PIL.Image import Image as PILImage
from config.settings import settings

@dataclass
class StoredImage:
    image_id: str
    prompt: str
    negative_prompt: str
    aspect_ratio: str
    width: int
    height: int
    seed: int
    generation_time: float
    model_name: str
    disk_path: Path

class ImageStore:
    """Thread-safe in-memory image store with disk persistence."""
    
    def __init__(self, output_dir: Path):
        self._store: dict[str, StoredImage] = {}
        self._lock = threading.Lock()
        self._output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, image: PILImage, metadata: dict) -> StoredImage:
        """Save image to disk and store metadata. Returns StoredImage."""
        ...
    
    def get(self, image_id: str) -> StoredImage:
        """Retrieve StoredImage by ID. Raises KeyError if not found."""
        ...
    
    def get_image_bytes(self, image_id: str) -> bytes:
        """Read image bytes from disk."""
        ...
    
    def list_all(self) -> list[StoredImage]:
        """Return all stored images sorted by newest first."""
        ...
    
    def count(self) -> int:
        ...

# Singleton
store = ImageStore(Path(settings.image_output_dir))
```

---

### TASK 7 — `storage/__init__.py`

```python
from .image_store import ImageStore, StoredImage, store
__all__ = ["ImageStore", "StoredImage", "store"]
```

---

### TASK 8 — `server.py` (REPLACE ENTIRELY)

This is the main deliverable. Replace the existing server.py completely.

Architecture:
- FastMCP with `lifespan=app_lifespan` — pipeline loaded ONCE at startup via lifespan
- `json_response=True` — required for Streamable HTTP production scalability
- Tools access pipeline via `ctx.request_context.lifespan_context`
- Long-running `generate()` called via `asyncio.to_thread()` — never blocks event loop
- Progress reported via `ctx.report_progress()` during generation
- Returns `mcp.server.fastmcp.Image` type for image data (SDK handles base64)

```python
# Imports
import asyncio
import base64
import io
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    model: BaseImageModel

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    log.info("Loading image model: %s", settings.image_model)
    model = load_model()
    # load() is blocking (GPU memory allocation) — run in thread
    await asyncio.to_thread(model.load)
    log.info("Model loaded: %s", model.model_name)
    try:
        yield AppContext(model=model)
    finally:
        log.info("Unloading model")
        await asyncio.to_thread(model.unload)

# ── Server ───────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "local-image-gen",
    instructions=(
        "Image generation server. "
        "Call generate_image with a text prompt to create images. "
        "Use list_aspect_ratios to see supported dimensions. "
        "Use get_model_info to check GPU status. "
        "Previously generated images can be retrieved via image://{image_id} resource."
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
    
    Returns the image directly. The image_id is also stored and retrievable
    via the image://{image_id} resource URI.
    
    Args:
        prompt: Text description. English and Chinese both work well.
        aspect_ratio: Output dimensions. Use list_aspect_ratios to see options.
        negative_prompt: What to exclude. Default covers common AI artifacts.
        num_inference_steps: Quality/speed tradeoff. Range 20-100. Default 50.
        cfg_scale: Prompt adherence strength. Range 1.0-10.0. Default 4.0.
        seed: Reproducibility seed. None = random.
    """
    ...

@mcp.tool()
def list_aspect_ratios() -> dict:
    """List all supported aspect ratios and their pixel dimensions."""
    return {r: {"width": w, "height": h} for r, (w, h) in ASPECT_RATIOS.items()}

@mcp.tool()
def get_model_info(ctx: Context) -> dict:
    """Return model metadata and current GPU memory status."""
    ...

# ── Resources ────────────────────────────────────────────────────────────────

@mcp.resource("image://{image_id}")
def get_image_resource(image_id: str) -> Image:
    """Retrieve a previously generated image by ID."""
    img_bytes = store.get_image_bytes(image_id)
    return Image(data=img_bytes, format="png")

@mcp.resource("images://list")
def list_images() -> str:
    """List all generated images with metadata as JSON string."""
    import json
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
    """Enhance a prompt for photorealistic output with Qwen-Image-2512."""
    ...

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
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        path=settings.mcp_path,
    )
```

CRITICAL implementation notes for `generate_image`:
1. Resolve seed: `actual_seed = seed if seed is not None else int(time.time() * 1000) % (2**32)`
2. Get width/height: `width, height = get_dimensions(aspect_ratio)`
3. Clamp steps: `num_inference_steps = max(20, min(100, num_inference_steps))`
4. Get model from lifespan: `model = ctx.request_context.lifespan_context.model`
5. Report progress start: `await ctx.report_progress(0, num_inference_steps)`
6. Build request: `GenerationRequest(prompt=prompt, negative_prompt=negative_prompt, ...)`
7. Run in thread: `result = await asyncio.to_thread(model.generate, request)`
8. Report progress end: `await ctx.report_progress(num_inference_steps, num_inference_steps)`
9. Save to store: `stored = store.save(result.image, {...metadata...})`
10. Convert to bytes: `buf = io.BytesIO(); result.image.save(buf, format="PNG"); buf.seek(0)`
11. Return: `return Image(data=buf.read(), format="png")`

---

### TASK 9 — `pyproject.toml` (UPDATE)

Replace with correct dependencies. `pydantic-settings` is separate from `pydantic`.

```toml
[project]
name = "local-image-gen-mcp"
version = "1.0.0"
description = "Model-agnostic MCP server for local text-to-image generation"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.27.0",
    "pydantic>=2.11.0",
    "pydantic-settings>=2.7.0",
    "diffusers>=0.33.0",
    "torch>=2.7.0",
    "transformers>=4.52.0",
    "accelerate>=1.7.0",
    "safetensors>=0.5.3",
    "pillow>=11.2.1",
    "huggingface-hub>=0.32.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.25.0",
    "httpx>=0.28.0",
]
```

---

### TASK 10 — `docker/Dockerfile` (UPDATE)

Replace with correct multi-stage Dockerfile. Non-root user. Healthcheck on /mcp.

```dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY server.py .
COPY models/ ./models/
COPY config/ ./config/
COPY storage/ ./storage/

# Install dependencies
RUN uv pip install --system --no-cache \
    "mcp[cli]>=1.27.0" \
    "pydantic>=2.11.0" \
    "pydantic-settings>=2.7.0" \
    "diffusers>=0.33.0" \
    "torch>=2.7.0" \
    "transformers>=4.52.0" \
    "accelerate>=1.7.0" \
    "safetensors>=0.5.3" \
    "pillow>=11.2.1" \
    "huggingface-hub>=0.32.0"

RUN mkdir -p /outputs && chown appuser:appuser /outputs
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8500

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8500/mcp || exit 1

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8500
ENV MCP_PATH=/mcp
ENV IMAGE_OUTPUT_DIR=/outputs
ENV MAX_INFERENCE_STEPS=50
ENV DEFAULT_CFG_SCALE=4.0
ENV TORCH_DTYPE=bfloat16
ENV IMAGE_MODEL=qwen_image_2512

CMD ["python", "server.py"]
```

---

### TASK 11 — `docker/docker-compose.yml` (UPDATE)

Standalone compose. No external network dependency.
HF_TOKEN read from host environment (already in ~/.bashrc).

```yaml
services:
  local-image-gen-mcp:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: local-image-gen-mcp
    ports:
      - "8500:8500"
    volumes:
      - ~/.cache/huggingface:/home/appuser/.cache/huggingface
      - image-outputs:/outputs
    environment:
      HF_TOKEN: ${HF_TOKEN}      # from ~/.bashrc — never hardcoded
      IMAGE_MODEL: qwen_image_2512
      MCP_HOST: "0.0.0.0"
      MCP_PORT: "8500"
      IMAGE_OUTPUT_DIR: /outputs
      MAX_INFERENCE_STEPS: "50"
      DEFAULT_CFG_SCALE: "4.0"
      TORCH_DTYPE: bfloat16
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ipc: host
    shm_size: '8gb'
    ulimits:
      memlock:
        soft: -1
        hard: -1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8500/mcp"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 180s
    restart: unless-stopped

volumes:
  image-outputs:
```

---

### TASK 12 — `docker/docker-compose.override.yml` (WRITE)

Joins the existing `ai-bridge` network from the main DGX Spark stack.
Used when running alongside vLLM + LiteLLM + Langfuse.

```yaml
# Override for DGX Spark stack integration
# Usage: docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
services:
  local-image-gen-mcp:
    networks:
      - ai-bridge

networks:
  ai-bridge:
    external: true
```

---

### TASK 13 — `tests/test_models.py`

Unit tests for the model layer. Mock the DiffusionPipeline so tests run without GPU.

```python
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage
from models.base import GenerationRequest
from models.qwen_image_2512 import QwenImage2512

@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.return_value.images = [PILImage.new("RGB", (64, 64), color="red")]
    return mock

def test_is_loaded_before_load():
    model = QwenImage2512()
    assert not model.is_loaded()

def test_model_name():
    model = QwenImage2512()
    assert "Qwen" in model.model_name

@patch("models.qwen_image_2512.DiffusionPipeline")
def test_load_and_generate(MockPipeline, mock_pipeline):
    MockPipeline.from_pretrained.return_value = mock_pipeline
    model = QwenImage2512()
    model.load()
    assert model.is_loaded()
    request = GenerationRequest(
        prompt="a red square",
        negative_prompt="",
        width=64,
        height=64,
        num_inference_steps=1,
        cfg_scale=4.0,
        seed=42,
    )
    result = model.generate(request)
    assert result.seed == 42
    assert result.image is not None
    assert result.generation_time > 0
```

---

### TASK 14 — `tests/test_tools.py`

Integration tests using MCP in-memory transport (no HTTP server needed).
Use `mcp.server.fastmcp` test utilities.

```python
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from PIL import Image as PILImage

# Test that tools are registered
def test_tool_registration():
    from server import mcp
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "generate_image" in tool_names
    assert "list_aspect_ratios" in tool_names
    assert "get_model_info" in tool_names

def test_list_aspect_ratios_returns_all():
    from config.aspect_ratios import ASPECT_RATIOS
    from server import mcp
    # Direct call — list_aspect_ratios has no ctx dependency
    from server import list_aspect_ratios
    result = list_aspect_ratios()
    assert set(result.keys()) == set(ASPECT_RATIOS.keys())
    for ratio, dims in result.items():
        assert "width" in dims
        assert "height" in dims

def test_aspect_ratio_dimensions():
    from config.aspect_ratios import get_dimensions
    w, h = get_dimensions("16:9")
    assert w == 1664
    assert h == 928
    with pytest.raises(ValueError):
        get_dimensions("invalid")
```

---

### TASK 15 — `README.md` (REPLACE ENTIRELY)

Write a complete README covering:
1. What this is (one paragraph)
2. Tree structure (copy from plan)
3. Quick start — Docker (primary path)
4. Quick start — local with uv
5. MCP client configuration (Claude Code / Cline JSON snippet)
6. All tools, resources, and prompts table
7. Environment variables table
8. Adding a new model backend (3 steps)
9. Memory requirements warning

---

## IMPLEMENTATION ORDER

Execute tasks in this exact sequence. Do not skip ahead.
Each task depends on the previous ones being complete.

```
1  → config/aspect_ratios.py     (no dependencies)
2  → config/settings.py          (no dependencies)
3  → models/base.py              (no dependencies)
4  → models/qwen_image_2512.py   (depends on: 3)
5  → models/__init__.py          (depends on: 2, 3, 4)
6  → storage/image_store.py      (depends on: 2)
7  → storage/__init__.py         (depends on: 6)
8  → server.py                   (depends on: 1, 2, 3, 4, 5, 6, 7)
9  → pyproject.toml              (depends on: 8)
10 → docker/Dockerfile           (depends on: 9)
11 → docker/docker-compose.yml   (depends on: 10)
12 → docker/docker-compose.override.yml (depends on: 11)
13 → tests/test_models.py        (depends on: 3, 4)
14 → tests/test_tools.py         (depends on: 8)
15 → README.md                   (depends on: all)
```

---

## VERIFICATION

After all tasks complete, run:

```bash
# Verify imports resolve correctly
python -c "from server import mcp; print('OK')"

# Verify tools registered
python -c "from server import mcp; print([t.name for t in mcp._tool_manager.list_tools()])"

# Run tests (no GPU required — mocked)
uv run pytest tests/ -v

# Test MCP endpoint (requires docker running)
curl -s http://localhost:8500/mcp
```

---

## KNOWN GOTCHAS

1. `true_cfg_scale` — NOT `guidance_scale`. Qwen-Image-2512 uses this kwarg specifically.
2. `asyncio.to_thread()` — REQUIRED for `model.generate()`. It is synchronous and blocks.
   Never call it directly in an async tool or you will block the entire event loop.
3. `pydantic-settings` — separate pip package from `pydantic`. Both required.
4. HF_TOKEN — NEVER write to any file. Read from `os.getenv("HF_TOKEN")` only.
5. `json_response=True` on FastMCP — required for Streamable HTTP production scalability per SDK docs.
6. `ctx.request_context.lifespan_context` — this is how to access lifespan data inside tools.
   `ctx.lifespan_context` does NOT work — use the full path.
7. Lifespan runs ONCE at server start, not per request. The pipeline is shared across all tool calls.
8. `Image(data=bytes, format="png")` — `data` must be bytes, not base64 string.
   `buf.read()` returns bytes. `base64.b64encode(buf.read())` returns bytes that are base64
   — do NOT base64 encode before passing to Image().
9. `docker/Dockerfile` build context is `..` (parent dir) because it needs to COPY models/, config/, storage/.
10. The `docker-compose.override.yml` uses `external: true` for ai-bridge.
    If the main stack isn't running, this will fail. Use standalone `docker-compose.yml` for isolated testing.