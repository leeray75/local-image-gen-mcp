# local-image-gen-mcp

Model-agnostic MCP server for local text-to-image generation. Currently wraps [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512), the strongest open-source text-to-image model, with a pluggable backend architecture for adding new models.

Exposes image generation as MCP tools, resources, and prompts for use with Claude Code, Cline, and any MCP-compatible agent.

---

## Architecture

```
local-image-gen-mcp/
├── server.py                 # FastMCP server with lifespan pattern
├── pyproject.toml            # Dependencies (uv/pip)
├── .env.example              # Environment variable template
├── .gitignore                # Git exclusions
├── config/
│   ├── settings.py           # Pydantic BaseSettings — all config from env vars
│   └── aspect_ratios.py      # Aspect ratio data + helper functions
├── models/
│   ├── base.py               # Abstract BaseImageModel interface
│   ├── qwen_image_2512.py    # Qwen-Image-2512 implementation
│   └── __init__.py           # load_model() factory function
├── storage/
│   ├── image_store.py        # Thread-safe in-memory + disk image store
│   └── __init__.py
├── docker/
│   ├── Dockerfile            # CUDA 12.8, non-root user, healthcheck
│   ├── docker-compose.yml    # Standalone compose (GPU required)
│   └── docker-compose.override.yml  # Joins ai-bridge network
└── tests/
    ├── test_models.py        # Unit tests (mocked, no GPU needed)
    └── test_tools.py         # Integration tests
```

## Quick Start — Docker (Recommended)

### Prerequisites

- NVIDIA GPU with CUDA-compatible drivers
- Docker + NVIDIA Container Toolkit
- `HF_TOKEN` in your environment (e.g., `~/.bashrc`)

### Build and run

```bash
cd local-image-gen-mcp/docker

# Standalone (no external network)
docker compose up -d --build

# With DGX Spark stack (joins ai-bridge network)
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build
```

The server starts on `http://localhost:8500/mcp`. First startup downloads ~15GB of model weights.

### Verify

```bash
# Check container health
docker ps | grep local-image-gen

# Check MCP endpoint
curl -s http://localhost:8500/mcp
```

## Quick Start — Local with uv

```bash
cd local-image-gen-mcp

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# Set HF_TOKEN (already in ~/.bashrc for most users)
export HF_TOKEN=$(grep HF_TOKEN ~/.bashrc | cut -d'"' -f2)

# Run
python server.py
```

## MCP Client Configuration

### Cline / Claude Code

Add to your MCP settings (e.g., `.mcp.json` or client config):

```json
{
  "mcpServers": {
    "local-image-gen": {
      "command": "http",
      "url": "http://localhost:8500/mcp"
    }
  }
}
```

## Tools, Resources & Prompts

### Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate an image from a text prompt. Returns the image directly. |
| `list_aspect_ratios` | List supported aspect ratios and pixel dimensions. |
| `get_model_info` | Return model metadata and GPU memory status. |

### `generate_image` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *(required)* | Text description. English and Chinese both work. |
| `aspect_ratio` | `str` | `"1:1"` | One of: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3` |
| `negative_prompt` | `str` | *(built-in)* | What to exclude. Default covers common AI artifacts. |
| `num_inference_steps` | `int` | `50` | Quality/speed tradeoff. Range 20-100. |
| `cfg_scale` | `float` | `4.0` | Prompt adherence. Range 1.0-10.0. |
| `seed` | `int \| null` | `null` | Reproducibility seed. `null` = random. |

### Resources

| Resource URI | Description |
|-------------|-------------|
| `image://{image_id}` | Retrieve a previously generated image as PNG bytes. |
| `images://list` | List all generated images with metadata (JSON). |

### Prompts

| Prompt | Description |
|--------|-------------|
| `photorealistic(subject, setting, style)` | Enhance a prompt for photorealistic output. |
| `negative_defaults()` | Return the recommended default negative prompt. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HOST` | `0.0.0.0` | Host to bind to |
| `MCP_PORT` | `8500` | Port to bind to |
| `MCP_PATH` | `/mcp` | MCP endpoint path |
| `IMAGE_MODEL` | `qwen_image_2512` | Backend model identifier |
| `IMAGE_OUTPUT_DIR` | `/outputs` | Directory for generated images |
| `MAX_INFERENCE_STEPS` | `50` | Default diffusion steps (20-100) |
| `DEFAULT_CFG_SCALE` | `4.0` | Default guidance scale (1.0-10.0) |
| `TORCH_DTYPE` | `bfloat16` | Torch dtype: `bfloat16` or `float32` |
| `HF_TOKEN` | *(required)* | HuggingFace API token (from environment only) |

## Adding a New Model Backend

1. Create `models/your_model.py` subclassing `BaseImageModel`
2. Implement `load()`, `unload()`, `generate()`, `is_loaded()`, and `model_name`
3. Add an import branch in `models/__init__.py` `load_model()`
4. Add the model name to `Settings.image_model` Literal in `config/settings.py`

Example:
```python
# models/flux_schnell.py
from .base import BaseImageModel, GenerationRequest, GenerationResult

class FluxSchnell(BaseImageModel):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def is_loaded(self) -> bool: ...
    @property
    def model_name(self) -> str:
        return "Flux-Schnell"
```

Then in `models/__init__.py`:
```python
if settings.image_model == "flux_schnell":
    from .flux_schnell import FluxSchnell
    return FluxSchnell()
```

## Memory Requirements

| Component | Approximate VRAM |
|-----------|-----------------|
| Qwen-Image-2512 (bfloat16) | ~12-16 GB |
| Generation (1:1, 50 steps) | ~2-4 GB additional |
| Generation (16:9, 50 steps) | ~4-6 GB additional |

**Minimum recommended:** 24 GB VRAM (RTX 3090/4090, A100 24GB)
**Recommended:** 40+ GB VRAM (A100 80GB, multi-GPU)

For GPUs with less VRAM, set `TORCH_DTYPE=float32` and reduce `MAX_INFERENCE_STEPS` to 20-30.

## Running Tests

```bash
# Unit tests (no GPU required — mocked)
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_models.py -v
```

## License

Same as the wrapped model's license. Check the model card on HuggingFace for details.