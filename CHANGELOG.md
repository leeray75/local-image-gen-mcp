# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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