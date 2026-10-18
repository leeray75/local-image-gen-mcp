# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`get_image_base64(image_id)` tool** — returns a stored image as a plain base64 PNG string so agents can save images to the client filesystem; docstring documents decode-before-write ([spec 003](docs/specs/003-image-retrieval-tools/spec.md))
- **`get_latest_image_id()` tool** — returns the id and metadata of the most recently generated image without reading the `images://list` resource ([spec 003](docs/specs/003-image-retrieval-tools/spec.md))
- **`IMAGE_GPU_MEMORY_LIMIT` env var** — caps the model's device memory (GB) via accelerate `max_memory`, offloading excess layers to CPU so the image model can coexist with vLLM on GB10 unified memory. Unset = uncapped with a startup warning ([spec 004](docs/specs/004-memory-bounded-loading/spec.md))
- Startup logging of memory mode and post-load PyTorch CUDA allocated/reserved memory
- **Spec-driven development docs** in `docs/specs/` — project constitution plus spec, research, data model, contracts, plan and tasks for features 001–004, with requirement traceability ([index](docs/specs/SPECS.md))
- `AGENTS.md` ([agents.md](https://agents.md) open format) with setup/test commands, architecture, code style, pitfalls, spec workflow, security and PR guidelines for all coding agents; `CLAUDE.md` imports it via `@AGENTS.md`

### Changed

- **Model loading uses `device_map="balanced"`** instead of `device_map="cuda"` (v0.1.1) — required for `max_memory` to be honored; still avoids post-load `.to()` (meta tensor fix preserved)
- `docker-compose.yml` sets `IMAGE_GPU_MEMORY_LIMIT: "40"` by default

### Removed

- `deploy.resources.limits.memory: 40g` from `docker-compose.yml` (added in `a42846d`) — a cgroup limit does not change model placement and only OOM-kills the container; replaced by `IMAGE_GPU_MEMORY_LIMIT`

### Known issues

Tracked as open tasks in `docs/specs/`:

- `tests/test_tools.py` is empty — no tests for tools, aspect ratios or the image store
- `MCP_PATH` is not passed to `FastMCP` (works only because `/mcp` is the SDK default)
- `cfg_scale` is not clamped to its documented 1.0–10.0 range
- Concurrent `generate_image` calls are not serialized on the shared pipeline
- `IMAGE_GPU_MEMORY_LIMIT` is read via `os.getenv` rather than `Settings`, and is missing from `.env.example` and the README

## [0.1.1] — 2026-05-04

### Fixed

- **Meta tensor `NotImplementedError` on model load** — Qwen-Image-2512 uses `accelerate` meta device initialization; calling `.to(device)` after `from_pretrained()` crashes. Now passes `device_map="cuda"` directly to `from_pretrained()` so device placement happens during weight loading.
- **Per-session lifespan causing model reload on every connection** — MCP SDK 1.27.0 runs lifespan per-session with `json_response=True`. Replaced lifespan pattern with module-level singleton (`_startup_load_model()`) loaded once before `mcp.run()`.
- **`mcp.run()` TypeError in MCP SDK 1.27.0** — `run()` only accepts `transport` argument; `host`/`port` are passed to `FastMCP()` constructor. Removed invalid arguments from `mcp.run()`.
- **Docker healthcheck always failing** — `/mcp` endpoint returns HTTP 406 without MCP headers, causing `curl -f` to fail. Replaced with Python socket connect check to verify server is listening.

### Changed

- **Docker base image** switched from `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` to `nvcr.io/nvidia/pytorch:26.04-py3` for Blackwell GPU support
- **Docker container** now runs as root (required by NVIDIA NGC container)
- **Removed `uv` dependency** in Dockerfile; using direct `pip install` with NVIDIA-optimized torch/CUDA
- **Added CPU fallback path** in model loading when CUDA is unavailable
- **`images://list` resource** now includes `width`, `height`, and `generation_time_seconds` fields
- **`get_model_info()`** now includes `gpu_memory_reserved_gb` field
- **`torch.cuda.synchronize()`** added to `unload()` for proper GPU memory cleanup
- **Docker `restart` policy** changed from `unless-stopped` to `no`

### Added

- `HF_HOME` environment variable in docker-compose for explicit HuggingFace cache path
- `entrypoint` in docker-compose explicitly set to `["python", "server.py"]`
- Improved docstrings throughout with device placement and usage notes

### Removed

- Non-root user (`appuser`) from Dockerfile (NVIDIA NGC containers run as root)
- `ctx: Context` parameter from `get_model_info()` (no longer needed with singleton pattern)

## [0.1.0] — 2026-05-03

### Added

- **Model-agnostic architecture** with `BaseImageModel` abstract interface for pluggable backends
- **Qwen-Image-2512 backend** — official diffusion pipeline via `diffusers` library
- **FastMCP lifespan pattern** — pipeline loaded once at server startup, shared across all requests
- **Thread-safe `ImageStore`** — in-memory metadata with disk persistence (PNG format)
- **Pydantic `BaseSettings`** — all configuration via environment variables, validated at startup
- **Aspect ratio module** — 7 official aspect ratios with pixel dimensions from Qwen model card
- **MCP tools**:
  - `generate_image` — text-to-image with progress reporting
  - `list_aspect_ratios` — supported dimensions reference
  - `get_model_info` — model metadata and GPU memory status
- **MCP resources**:
  - `image://{image_id}` — retrieve previously generated images
  - `images://list` — list all generated images with metadata
- **MCP prompts**:
  - `photorealistic(subject, setting, style)` — prompt enhancement template
  - `negative_defaults()` — recommended negative prompt
- **Docker support**:
  - Multi-stage Dockerfile (CUDA 12.8, non-root user, healthcheck)
  - `docker-compose.yml` for standalone deployment
  - `docker-compose.override.yml` for DGX Spark stack integration (`ai-bridge` network)
- **Test suite**:
  - Unit tests for model layer (mocked, no GPU required)
  - Integration tests for config, storage, and model factory

### Changed

- Refactored monolithic `server.py` into modular architecture (`config/`, `models/`, `storage/`)
- Switched from lazy-loading to lifespan-based model loading (official MCP SDK 1.27 pattern)
- Replaced global dict image store with thread-safe `ImageStore` class
- Added `asyncio.to_thread()` for blocking diffusion calls — never blocks event loop
- Added `json_response=True` on FastMCP for Streamable HTTP production scalability
- Added `ctx.report_progress()` for generation progress feedback to MCP clients

### Technical Details

- Python 3.11+
- MCP SDK 1.27.0 (`mcp.server.fastmcp`)
- Pydantic v2 + `pydantic-settings`
- `diffusers` >= 0.33.0
- `torch` >= 2.7.0 with CUDA support
- Streamable HTTP transport (production standard)