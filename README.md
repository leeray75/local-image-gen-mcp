# local-image-gen-mcp

Model-agnostic MCP server for local text-to-image generation. Currently wraps [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), one model for text-to-image (about 1K by default, 2K opt-in), image editing (up to 10 reference images) and transparent (RGBA) output, with a pluggable backend architecture for adding new models.

> **License:** Qwen-Image-2.1 is released under the Qwen Research License, which is **non-commercial use only**. Check the [model card](https://huggingface.co/Qwen/Qwen-Image-2.1) before using the output commercially.

Exposes image generation as MCP tools, resources, and prompts for use with Claude Code, Cline, and any MCP-compatible agent.

---

## Architecture

```
local-image-gen-mcp/
├── server.py                 # FastMCP server; model loaded once at startup (no lifespan)
├── pyproject.toml            # Dependencies (uv/pip)
├── pytest.ini                # pytest config (asyncio_mode = auto)
├── .env.example              # Environment variable template
├── AGENTS.md                 # Instructions for AI coding agents (CLAUDE.md imports it)
├── CHANGELOG.md
├── config/
│   └── settings.py           # Pydantic BaseSettings — all config from env vars
├── models/
│   ├── base.py               # Abstract BaseImageModel interface
│   ├── qwen_image_2_1.py     # Qwen-Image-2.1 implementation (size table, defaults)
│   └── __init__.py           # load_model() factory function
├── storage/
│   ├── image_store.py        # Thread-safe in-memory + disk image store
│   ├── references.py         # edit_image reference decoding/validation
│   └── __init__.py
├── docker/
│   ├── Dockerfile            # NVIDIA NGC container (pytorch:26.04), healthcheck
│   ├── docker-compose.yml    # Standalone compose (GPU required)
│   └── docker-compose.override.yml  # Joins ai-bridge network
├── docs/specs/               # Spec-driven development docs (start at SPECS.md)
└── tests/
    ├── conftest.py           # Sets IMAGE_OUTPUT_DIR before imports
    ├── test_models.py        # Backend unit tests (mocked, no GPU needed)
    ├── test_references.py    # Reference decoding tests
    └── test_tools.py         # Tool / prompt / settings tests (fake model)
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

The server starts on `http://localhost:8500/mcp`. The first startup downloads ~33 GB of model weights into `~/.cache/huggingface` (mounted from the host, so later starts don't download again). Each start still loads the model into memory (~3¼ min) before the server accepts connections. With the default `int8` quantization the server **runs alongside vLLM** (~18.5 GB of weights). BF16 (`QUANTIZE_*=none`) or 2K output needs vLLM stopped (see [Memory Requirements](#memory-requirements)).

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

# Run (the default /outputs only exists in the container)
IMAGE_OUTPUT_DIR=./outputs python server.py
```

## MCP Client Configuration

The server speaks MCP over Streamable HTTP at `http://localhost:8500/mcp`.

### Claude Code

```bash
claude mcp add --transport http local-image-gen http://localhost:8500/mcp
```

or in `.mcp.json`:

```json
{
  "mcpServers": {
    "local-image-gen": {
      "type": "http",
      "url": "http://localhost:8500/mcp"
    }
  }
}
```

### Cline and other clients

Configure a Streamable HTTP server with the URL `http://localhost:8500/mcp`. In Cline's `cline_mcp_settings.json`, that is `"type": "streamableHttp"` with that `url`.

## Tools, Resources & Prompts

### Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate an image from a text prompt (about 1K by default; see `GENERATE_OUTPUT_RESOLUTION`). Returns the image directly. |
| `edit_image` | Edit or combine 1-10 reference images with an instruction. Returns the image directly. |
| `list_aspect_ratios` | List supported aspect ratios and pixel dimensions. |
| `get_model_info` | Model metadata, capabilities, resolved defaults and GPU memory status. |
| `get_image_base64` | Return a stored image as a base64 PNG string. |
| `get_latest_image_id` | ID and metadata of the most recent image. |

### `generate_image` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *(required)* | Text description. English and Chinese both work. |
| `aspect_ratio` | `str` | `"1:1"` | One of: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3` (about `GENERATE_OUTPUT_RESOLUTION`² of area; default 1024, so 1:1 = 1024x1024; `list_aspect_ratios` shows the exact sizes) |
| `transparent` | `bool` | `false` | `true` = RGBA PNG with a real alpha channel |
| `negative_prompt` | `str \| null` | `null` | What to exclude. Only takes effect when `cfg_scale > 1`. |
| `num_inference_steps` | `int \| null` | `null` (40) | Range 20-100 (clamped). |
| `cfg_scale` | `float \| null` | `null` (1.0) | Range 1.0-10.0 (clamped). 1.0 = guidance off (the model's tuned default). |
| `seed` | `int \| null` | `null` | Reproducibility seed. `null` = random. |

### `edit_image` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *(required)* | Edit instruction. Refer to references as "Picture 1..N" by position. |
| `images` | `list[str]` | *(required)* | 1-10 ordered references: store `image_id`s (preferred) and/or base64 PNG/JPEG/WebP (optional `data:` prefix), each at most `MAX_REFERENCE_IMAGE_MB` decoded. |
| `aspect_ratio` | `str \| null` | `null` | `null` = aspect ratio of the last reference at about `EDIT_OUTPUT_RESOLUTION`² pixels of area (default 1024; shown by `get_model_info`). A ratio gives that ratio's size at the same `EDIT_OUTPUT_RESOLUTION` (not the `generate_image` table). |
| `transparent` | `bool` | `false` | e.g. extract a subject onto a transparent background |
| `negative_prompt`, `num_inference_steps`, `cfg_scale`, `seed` | | | as for `generate_image` |

Every result gets a new `image_id`, so edits can be chained without moving bytes. Uploaded base64 references are not stored. Only one generation runs at a time.

Examples:

```text
generate_image(prompt="a red fox in snow", aspect_ratio="16:9")
generate_image(prompt="a flat vector sticker of a coffee mug", transparent=true)
edit_image(prompt="Make it dusk with warm lamp light", images=["<image_id>"])
edit_image(prompt="Extract the main subject from Picture 1 ...", images=["<base64>"], transparent=true)
```

### Resources

| Resource URI | Description |
|-------------|-------------|
| `image://{image_id}` | Retrieve a previously generated image as PNG bytes. |
| `images://list` | List all generated images with metadata (JSON, includes `mode` and `transparent`). |

### Prompts

| Prompt | Description |
|--------|-------------|
| `photorealistic(subject, setting, style)` | Enhance a prompt for photorealistic output. |
| `transparent_asset(subject, style)` | Prompt for a standalone object; use with `generate_image(transparent=true)`. |
| `extract_subject(subject)` | Edit instruction to cut a subject out onto a transparent background; use with `edit_image(transparent=true)`. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HOST` | `0.0.0.0` | Host to bind to |
| `MCP_PORT` | `8500` | Port to bind to |
| `MCP_PATH` | `/mcp` | MCP endpoint path |
| `IMAGE_MODEL` | `qwen_image_2_1` | Backend model identifier (only accepted value) |
| `IMAGE_OUTPUT_DIR` | `/outputs` | Directory for generated images |
| `MAX_INFERENCE_STEPS` | *(unset)* | Optional override of the default steps (backend default 40; clamped 20-100) |
| `DEFAULT_CFG_SCALE` | *(unset)* | Optional override of the default CFG (backend default 1.0; clamped 1.0-10.0) |
| `IMAGE_GPU_MEMORY_LIMIT` | *(unset; compose sets 40)* | Device-memory cap in GB (accelerate `max_memory`) |
| `MAX_REFERENCE_IMAGE_MB` | `20` | Max decoded size of one `edit_image` reference |
| `MAX_REQUEST_BODY_MB` | *(unset = computed, 268)* | Total HTTP request-body limit in MB; the MCP SDK's own default of 4 MB would reject base64 references above ~3 MB with HTTP 413. Computed as `ceil(10 × MAX_REFERENCE_IMAGE_MB × 4/3) + 1`; must be ≥ 4. Prefer `image_id`s over base64 for large or chained images |
| `AI_BRIDGE_NETWORK` | `ai-litellm-proxy_ai-bridge` | Name of the external Docker network joined by `docker-compose.override.yml` (read by compose, not by the server) |
| `EDIT_OUTPUT_RESOLUTION` | `1024` | Edit output size: about this many pixels per side in area (multiple of 32, 512-2048; empty = default). `get_model_info` reports the value in effect |
| `GENERATE_OUTPUT_RESOLUTION` | `1024` | Text-to-image output size: about this many pixels per side in area (multiple of 32, 512-2048; empty = default). 2048 uses the model card's 2K table. `get_model_info` reports the value in effect |
| `MAX_OUTPUT_RESOLUTION` | `1024` | Ceiling for `GENERATE_OUTPUT_RESOLUTION` and `EDIT_OUTPUT_RESOLUTION` (multiple of 32, 512-2048; empty = default). The server refuses to start if either is above it. To opt into 2K, raise **both** the resolution and this ceiling |
| `QUANTIZE_TRANSFORMER` | `auto` | Load-time weight quantization of the image model (DiT): `auto` (= `int8` on CUDA, `none` on CPU), `none` (BF16), `int8`, `fp8` (torchao weight-only) or `nvfp4` (NVIDIA ModelOpt; **unsupported on GB10, see below**). Empty = `auto`. Set `none` for BF16. An explicit quantized mode needs CUDA; an unsupported mode fails startup and never falls back to BF16 ([spec 008](docs/specs/008-load-time-quantization/spec.md)) |
| `QUANTIZE_TEXT_ENCODER` | `auto` | The same for the Qwen3-VL text encoder: `auto`, `none`, `int8`, `fp8` or `nvfp4` (**unsupported on GB10, see below**; it also uses a torchao prototype API, `torchao.prototype.mx_formats`). The VAE is never quantized |
| `TORCH_DTYPE` | `bfloat16` | Torch dtype: `bfloat16` or `float32` |
| `HF_TOKEN` | *(required)* | HuggingFace API token (from environment only) |

## Adding a New Model Backend

1. Create `models/your_model.py` subclassing `BaseImageModel`
2. Implement `load()`, `unload()`, `generate()`, `is_loaded()`, `model_name`, `aspect_ratios`, `default_inference_steps`, `default_cfg_scale` and `default_negative_prompt` (override `max_reference_images` / `supports_transparency` if supported)
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
    # ...plus aspect_ratios, default_inference_steps, default_cfg_scale, default_negative_prompt
```

Then in `models/__init__.py`:
```python
if settings.image_model == "flux_schnell":
    from .flux_schnell import FluxSchnell
    return FluxSchnell()
```

## Memory Requirements

Measured on the DGX Spark (GB10) on 2026-10-18.

**Default (`int8`, spec 008), running next to vLLM at `gpu-memory-utilization 0.6`:** 18.5 GB of weights. A 1K generate takes ~62 s and a 1024 edit ~72 s. At least ~10 GB of host memory stayed free while loading and ~15 GB while generating. See the [weight quantization](#weight-quantization-spec-008) results table.

**BF16 (`QUANTIZE_TRANSFORMER=none`, `QUANTIZE_TEXT_ENCODER=none`), with vLLM stopped.** Memory is the image server's approximate share: the system-wide peak minus the other running services.

| | Memory | Time |
|---|---|---|
| Idle, model loaded (bfloat16 weights) | ~30–32 GB | — |
| `generate_image` at the default `GENERATE_OUTPUT_RESOLUTION=1024` | est. ≤ ~40 GB peak (not yet measured) | est. ~60 s |
| 2K `generate_image` (`GENERATE_OUTPUT_RESOLUTION=2048`, 1:1, 40 steps) | ~68 GB peak | ~256 s |
| 2K `generate_image`, `transparent=true` | ~68 GB peak | ~256 s |
| `edit_image`, 1 reference, default `EDIT_OUTPUT_RESOLUTION=1024` | ~40 GB peak | ~61 s |
| `edit_image` at `EDIT_OUTPUT_RESOLUTION=2048` (not recommended) | ~76–81 GB peak | ~340–372 s |

The GB10 has 128 GB shared by CPU and GPU. With vLLM at `gpu-memory-utilization 0.6`, about 37 GB is available. That's enough for the `int8` default, but not for BF16 or 2K output, which need the vLLM engine **stopped**. The default's tightest moment is startup, so avoid starting the image server while vLLM is under heavy load. To switch for BF16 or 2K work:

```bash
docker stop qwen3-8-27b-nvfp4-engine && (cd docker && docker compose up -d)   # image generation
docker stop local-image-gen-mcp && docker start qwen3-8-27b-nvfp4-engine        # back to vLLM
```

 `IMAGE_GPU_MEMORY_LIMIT` only limits where accelerate places layers; offloaded layers live in the same physical RAM on unified memory, so the cap does **not** let the server fit alongside vLLM. At the default 1024, both edits and text-to-image need less memory than a 2K generate. More references, or a higher `EDIT_OUTPUT_RESOLUTION` or `GENERATE_OUTPUT_RESOLUTION`, need more. The BF16 1K text-to-image numbers are estimates, because the BF16 1K vs 2K A/B test is still deferred ([spec 007](docs/specs/007-1k-default-resolution/spec.md)). The `int8` 1K numbers are measured (spec 008).

**Edit quality:** edits at `output_resolution=2048` showed over-sharpening and ignored instructions ([spec 005](docs/specs/005-qwen-image-2-1/spec.md), finding F-1). The A/B test in [spec 006](docs/specs/006-open-items-closeout/spec.md) found 1024 clean, so `EDIT_OUTPUT_RESOLUTION` defaults to 1024. Raise it up to 2048 at your own risk.

**Opting into 2K:** set `GENERATE_OUTPUT_RESOLUTION=2048` and `MAX_OUTPUT_RESOLUTION=2048` (and `EDIT_OUTPUT_RESOLUTION=2048` for edits, not recommended). Raising a resolution without the ceiling makes the server exit at startup with an error naming the setting. Stop vLLM first: a 2K generate peaks at ~68 GB.

### Weight quantization (spec 008)

`QUANTIZE_TRANSFORMER` and `QUANTIZE_TEXT_ENCODER` each take `auto`, `none`, `int8`, `fp8` or `nvfp4`. The server quantizes the official Qwen weights while loading, so nothing extra is downloaded. **The default is `auto`: `int8` on CUDA, `none` (BF16) on CPU** (chosen from the GPU results below). To get BF16 on a GPU, set both settings to `none`; that needs about 40 GB for a 1K generate, so stop vLLM first. An explicit `int8`, `fp8` or `nvfp4` on CPU fails at startup. `get_model_info` reports the effective modes (plus `configured_quantize_*` when `auto` was resolved) and the per-component weight size as stored (`weight_memory_gb`), and the startup log prints the same.

- Layers kept in BF16: the text encoder's vision tower and `lm_head`, and the image model's input/output projections, timestep, modulation and norm layers (module names verified in [research R-5](docs/specs/008-load-time-quantization/research.md)).
- `int8` and `fp8` are weight-only (torchao); `nvfp4` on the image model is weight-only through NVIDIA ModelOpt. Weight-only quantization saves memory and may be *slower* than BF16.
- **Warning:** NVFP4 is **unsupported on GB10 with the current library versions** (spec 008 F-7): it does not reduce resident memory (the ModelOpt transformer stayed at its BF16 size, 14.2 GB) and the torchao prototype text-encoder path used far more than expected while loading. The value is still accepted, with a startup warning, so it can be re-tested after library updates. Prefer `int8` or `fp8`.
- Also: `nvfp4` on the text encoder uses a torchao **prototype** API. NVFP4 on the GB10 (sm_121) was not pursued further (D-8).
- **Staged loading:** with any effective mode other than `none` (which is the default on CUDA, since `auto` resolves to `int8`), the image model and then the text encoder load one at a time, with `gc.collect()`, `torch.cuda.empty_cache()` and a memory log line (CUDA allocated/reserved and host `MemAvailable`) after each step; then the pipeline is built around them. `IMAGE_GPU_MEMORY_LIMIT` is not applied in this path. With `none`/`none` loading is unchanged.
- Compose sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default (override with the `PYTORCH_CUDA_ALLOC_CONF` variable) to reduce CUDA caching-allocator retention while loading.
- If a mode cannot load, startup fails with an error naming both settings. There is no silent fallback. On CPU, `auto` resolves to `none`; an explicit `int8`, `fp8` or `nvfp4` fails.

Measured on the DGX Spark (GB10) on 2026-10-18, running next to vLLM (`gpu-memory-utilization 0.6`), 1K outputs, seed 42, 40 steps. G1 is a photo, G2 typography, G3 a transparent sticker, E1 a 1024 edit.

| Image model / text encoder | Weights allocated | Load | Min host available (load / generate) | G1 / G2 / G3 / E1 time | Quality |
|---|---|---|---|---|---|
| `int8` / `int8` (**default on CUDA**) | 18.54 GB (transformer 7.27, text encoder 10.60, VAE 0.68) | 194 s | 10.2 / 14.6 GB | 77.5 / 62 / 62 / 71.6 s | lettering exact, sticker 61.1% transparent, edit ≈ BF16 |
| `fp8` / `fp8` | 18.53 GB (7.26 / 10.59 / 0.68) | 191 s | 9.8 / 15.5 GB | 176 / 86 / 87 / 121 s | lettering exact, sticker 60.9% transparent, edit ≈ BF16 |
| `none` / `none` (BF16) | 32.4 GB | ~195 s | does not fit next to vLLM | 1024 edit 61 s in spec 006 (vLLM stopped) | reference |
| `nvfp4` | transformer 14.23 GB (no saving) | did not complete (memory watchdog) | n/a | n/a | unsupported on GB10 with current libraries (F-7); dropped (D-8) |

`int8` was chosen as the default (D-10): the same memory as `fp8`, about BF16 speed, exact lettering and an edit nearly identical to BF16. The 1K BF16 generate peak was never measured (it does not fit next to vLLM).

### Rebuilds and compile caches

- **Rebuilds are fast after code-only changes.** The Dockerfile installs dependencies before copying the app source, so a change to `server.py`, `models/`, `config/` or `storage/` reuses the cached `pip install` layer.
- **Compile caches persist.** Compose mounts the named volumes `image-gen-cuda-cache` (`/root/.nv`, the CUDA JIT cache) and `image-gen-torch-cache` (`/root/.cache/torch`, PyTorch kernel cache), so GPU kernels compiled on earlier runs survive `docker compose down`/`up` and `up --build`. The HuggingFace cache bind mount is separate and untouched.
- **Measured effect:** about 6 s off the first generation after a recreate. Startup is dominated by the ~3¼ min weight load, which no cache can skip ([spec 006 research R-5](docs/specs/006-open-items-closeout/research.md)).
- **Updates recompile automatically.** The CUDA cache is invalidated by an NVIDIA driver upgrade; the PyTorch kernel caches are keyed by code and toolkit version.
- **Force a full recompile:** remove the volumes (`docker volume rm docker_image-gen-cuda-cache docker_image-gen-torch-cache`; the prefix is the compose project name), or set `CUDA_CACHE_DISABLE=1` for one run.

## Running Tests

```bash
# Unit tests (no GPU required — the pipeline is mocked)
uv run pytest tests/ -v

# Or inside the project image (no GPU, no weights); build it once first:
docker build -f docker/Dockerfile -t local-image-gen-mcp:qwen21-test .
docker run --rm --entrypoint bash -v "$PWD":/app -w /app local-image-gen-mcp:qwen21-test \
  -c 'pip install -q pytest pytest-asyncio pyyaml && IMAGE_OUTPUT_DIR=/tmp/out python -m pytest tests -v'

# Specific test file
uv run pytest tests/test_models.py -v
```

## License

The wrapped model, Qwen-Image-2.1, uses the Qwen Research License (non-commercial use only). Check the model card on HuggingFace for details.