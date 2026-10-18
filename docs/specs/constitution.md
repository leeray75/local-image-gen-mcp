# Project Constitution

These principles govern every feature in this repository. A spec or plan that conflicts with one of them must either change, or amend this document explicitly with a rationale.

## I. Local-first, single-host inference

The server generates images on the local GPU. It never calls a remote inference API. Model weights are fetched from Hugging Face once and cached on the host (`~/.cache/huggingface`).

## II. Model-agnostic core

Server, storage and configuration code MUST NOT depend on any specific diffusion model. All model-specific behavior (pipeline class, kwargs, device placement, default negative prompt, valid resolutions) lives behind `BaseImageModel` in `models/`. Selecting a backend is a configuration change (`IMAGE_MODEL`), not a code change in `server.py`.

## III. Configuration from the environment only

All runtime configuration comes from environment variables, validated at startup. Defaults may exist for local development. Secrets — specifically `HF_TOKEN` — MUST NOT be written to any tracked file, image layer, log line, or tool output.

## IV. Verify the SDK, don't assume it

The MCP Python SDK (`mcp.server.fastmcp`, not the standalone `fastmcp` package) changes between versions. Any reliance on SDK lifecycle or signature behavior (lifespan scope, `run()` arguments, `Image` encoding, transport options) MUST be confirmed against the pinned version during research and recorded in `research.md` before design. *(Added after v0.1.1: three of the four v0.1.1 bugs came from unverified SDK assumptions.)*

## V. Never block the event loop

Diffusion and weight loading are synchronous and take seconds to minutes. Any blocking call made from an async MCP handler MUST be run off the event loop (`asyncio.to_thread`).

## VI. Respect the shared host

This server shares a DGX Spark (GB10, 128 GB unified CPU/GPU memory) with a vLLM stack. Features MUST keep the image model's memory footprint bounded and configurable. Containers MUST NOT auto-restart — a crash loop on a GPU box can compound a bad boot after a driver or kernel update.

## VII. Testable without a GPU

Unit tests MUST run on a machine with no CUDA device by mocking the diffusion pipeline. GPU-dependent behavior is verified by a documented manual check.

## VIII. Self-describing tools

MCP tool docstrings are the agent-facing contract. They MUST state parameter ranges, return format (raw bytes vs. base64, etc.) and the intended workflow, because the calling agent has no other documentation.

## Technology constraints

| Area | Constraint |
|------|------------|
| Language | Python ≥ 3.11 |
| MCP SDK | `mcp[cli]` ≥ 1.27.0, `mcp.server.fastmcp` |
| Transport | Streamable HTTP |
| Inference | `diffusers`, `transformers`, `accelerate`, `torch` (NVIDIA NGC build in containers) |
| Config | `pydantic` v2 + `pydantic-settings` |
| Container base | `nvcr.io/nvidia/pytorch` (Blackwell/aarch64 support) |
