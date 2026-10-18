# AGENTS.md

Instructions for AI coding agents working in this repository. Human-oriented docs are in `README.md`. This file follows the [AGENTS.md](https://agents.md) open format, stewarded by the Agentic AI Foundation (Linux Foundation). If an `AGENTS.md` exists in a subdirectory, the closest one to the file you're editing wins. Explicit user instructions override this file.

## Project overview

An MCP server (Streamable HTTP on `:8500/mcp`) that wraps a local text-to-image diffusion model and exposes it as MCP tools, resources and prompts. The model is currently Qwen-Image-2512, run via `diffusers`. It runs on an NVIDIA DGX Spark: aarch64, GB10 Blackwell GPU, and 128 GB of memory shared by the CPU and GPU. A vLLM stack runs on the same machine, and that drives most of the memory-related design.

- Python ≥ 3.11
- MCP SDK `mcp[cli]` ≥ 1.27, using **`mcp.server.fastmcp`** (the official SDK, not the standalone `fastmcp` package)
- `pydantic` v2 + `pydantic-settings`, `diffusers`, `transformers`, `accelerate`, `torch`

## Setup and commands

```bash
# Local environment
uv venv && source .venv/bin/activate && uv pip install -e .

# Run locally. The default /outputs only exists in the container; storage creates the directory at import time.
IMAGE_OUTPUT_DIR=./outputs python server.py

# Tests: the DiffusionPipeline is mocked, so no GPU is needed
uv run pytest tests/ -v
uv run pytest tests/test_models.py::test_load_and_generate -v      # single test

# Quick import / tool-registration check
IMAGE_OUTPUT_DIR=./outputs python -c "from server import mcp; print([t.name for t in mcp._tool_manager.list_tools()])"

# Docker (run from docker/)
docker compose up -d --build
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build   # joins the external ai-bridge network
docker logs -f local-image-gen-mcp
```

No linter or formatter is configured. Before you finish a change, run the test suite and the tool-registration check.

## Architecture

**Request flow:**
1. A `server.py` tool builds a `GenerationRequest`.
2. `await asyncio.to_thread(model.generate, request)` runs it. `generate` is synchronous and takes minutes.
3. The result comes back as a `GenerationResult`.
4. `storage.store.save()` writes a PNG to disk and adds the metadata to an in-memory index.
5. The tool returns an MCP `Image`.

| Path | Role |
|------|------|
| `server.py` | FastMCP app with all tools, resources and prompts, plus the entrypoint |
| `models/base.py` | `BaseImageModel` ABC (`load`, `unload`, `generate`, `is_loaded`, `model_name`) and the request/result dataclasses |
| `models/__init__.py` | `load_model()` factory that switches on `settings.image_model` |
| `models/qwen_image_2512.py` | Qwen backend and `DEFAULT_NEGATIVE_PROMPT` |
| `config/settings.py` | `settings` singleton. All config comes from the environment and `.env`. |
| `config/aspect_ratios.py` | Ratio → pixel-size table |
| `storage/image_store.py` | Thread-safe `store` singleton, created at import time |

**Key invariants:**
- **The model loads once per process, with no lifespan.** In this SDK version a FastMCP lifespan runs once per session, so the model would reload on every connection. Instead, `_startup_load_model()` runs in `__main__` before `mcp.run()`, and tools call `get_model()`. Importing `server` does not load the model.
- **Host and port go to the `FastMCP()` constructor.** `mcp.run()` accepts only `transport`. Passing `host` or `port` raises a `TypeError`.
- **Adding a backend takes three edits:** a new `BaseImageModel` subclass, a branch in `load_model()`, and the new ID added to the `Literal` on `Settings.image_model`. Tool code must not depend on a particular model.
- **Two copies must stay in sync:** the `aspect_ratio` `Literal` in `generate_image` and the keys of `ASPECT_RATIOS`.
- **The image index lives only in memory.** PNGs survive a restart on the volume, but `image://`, `images://list` and `get_latest_image_id` only know about images from the current process.

## Code style

Match the surrounding code:
- Every module has a docstring.
- Use type hints throughout, written as `X | None` rather than `Optional`.
- Use dataclasses for plain records.
- Use one `log = logging.getLogger(__name__)` per module, with `%`-style log arguments (not f-strings).
- `server.py` groups code under `# ── Section ───` banner comments. Put new tools with the other tools, before `# ── Resources`.
- Tool docstrings are the only documentation the calling agent sees. Each one must state parameter ranges, the return format, and the intended workflow.
- Never call a blocking function (diffusion, weight loading, heavy I/O) directly from an `async` tool. Use `asyncio.to_thread`.

## Model and runtime pitfalls

- Pass `device_map` to `from_pretrained()`. **Never call `.to(device)` afterwards**: the pipeline uses meta-tensor init, and `.to()` raises `NotImplementedError: Cannot copy out of meta tensor`. On CUDA the code uses `device_map="balanced"`, because accelerate ignores `max_memory` with `"cuda"`.
- `IMAGE_GPU_MEMORY_LIMIT` (in GB) caps device memory through `max_memory` and offloads the rest to the CPU. Uncapped, the model uses about 62 GB and destabilizes the host while vLLM is running. Compose sets it to 40. Don't replace it with a Docker memory limit, which only gets the container OOM-killed.
- Qwen's CFG kwarg is `true_cfg_scale`, not `guidance_scale`.
- `DEFAULT_NEGATIVE_PROMPT` contains Chinese on purpose. Don't "fix" it.
- `Image(data=...)` takes raw PNG bytes. Don't base64-encode them first.

## Testing instructions

- Tests must run without a GPU. Patch `models.qwen_image_2512.DiffusionPipeline`, the way `tests/test_models.py` does.
- Tests that import `server` or `storage` need `IMAGE_OUTPUT_DIR` pointing at a writable temporary directory (use `tmp_path` / `monkeypatch`).
- `tests/test_tools.py` is currently empty. When you touch tools, resources or the store, add tests there.
- GPU-only behavior (generation determinism, memory cap, healthcheck) is checked by hand, using the acceptance scenarios in `docs/specs/`.

## Spec-driven workflow

This project uses spec-driven development. `docs/specs/SPECS.md` indexes:
- `constitution.md`, the project principles
- one folder per feature, `NNN-name/`, containing `spec.md`, `research.md`, `plan.md` and `tasks.md`, with requirement IDs (`FR-001-07`) and traceability tables

For a new feature or a change in behavior:
1. Update or create the `spec.md` **before** the code.
2. Check SDK and library behavior against the installed version, and record what you find in `research.md`. Three of the four v0.1.1 bugs came from SDK assumptions nobody checked.
3. Update `plan.md` and `tasks.md`, then write the code.
4. Tick the task boxes and update the traceability table, then add a `CHANGELOG.md` entry under `[Unreleased]` ([Keep a Changelog](https://keepachangelog.com/) format).

`ai-workspace/plans/` and `ai-workspace/reports/` hold planning documents and change summaries. Add a report there when you finish a significant body of work.

## Security considerations

- `HF_TOKEN` comes from the host environment only. Never write it to a file, a Dockerfile, compose YAML, a log line, or tool output.
- The server has no authentication. It is meant to be reached only on localhost or the `ai-bridge` Docker network. Don't add features that assume it is exposed publicly.
- **Host safety (DGX Spark):**
  - Don't run `apt`, driver, firmware or `sudo` commands as part of work in this repo.
  - Never install a generic `nvidia-driver-*` package.
  - Keep `restart: no` in compose. Auto-restarting GPU containers can make a bad boot worse after a driver or kernel update.

## Deployment notes

- The base image is `nvcr.io/nvidia/pytorch:26.04-py3`, which supports Blackwell on aarch64. The image runs as root. Python deps are installed with `PIP_CONSTRAINT="" pip install`, and torch is **not** installed, so NVIDIA's build is kept.
- Dependency versions appear in both `docker/Dockerfile` and `pyproject.toml`. Update both.
- The healthcheck is a raw TCP connect, because `/mcp` returns 406 without MCP headers, so `curl -f` always fails. The start period is 180 s. The first start downloads about 15 GB into the mounted `~/.cache/huggingface`.

## Commits and pull requests

- Use Conventional Commit messages, as in the existing history: `feat: …`, `fix: …`, `docs: …`.
- Keep feature, fix and docs changes in separate commits where you can.
- The main branch is `master`. Work happens on branches such as `bugfix/<topic>` and `feature/<topic>`.
- A PR description should list the spec requirement IDs it addresses and any acceptance scenarios that were checked by hand on the GPU.
