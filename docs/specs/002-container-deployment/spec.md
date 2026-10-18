# Feature 002 — Container Deployment on DGX Spark

| | |
|---|---|
| **Status** | Implemented (v0.1.0, corrected in v0.1.1) |
| **Constitution** | I, III, VI |
| **Depends on** | [001](../001-image-generation-server/spec.md) |
| **Next** | [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

The server needs to run reproducibly on the DGX Spark. That host is aarch64 with a GB10 Blackwell GPU and 128 GB of unified memory, and it already runs a vLLM / LiteLLM / Langfuse stack on the Docker network `ai-bridge`. Generic PyTorch images don't support this GPU. Monitoring has to know when the server is actually up. The first boot downloads the model weights (about 15 GB was the original estimate; Qwen-Image-2.1 is about 33 GB, see 005) and shouldn't repeat that on every rebuild.

## Goal

A container image and compose files that start the server with GPU access. They should either run standalone or join the existing AI stack, and report their health correctly.

## Non-goals

- Kubernetes or Swarm manifests.
- Multi-replica deployment.
- Publishing images to a registry.

## User stories

**US-1 (P1)** As an operator, I can run `docker compose up -d --build` and get a server on `localhost:8500/mcp` with GPU access.

**US-2 (P1)** As an operator, I can run the same service attached to `ai-bridge`, so other containers in the stack can reach it by name.

**US-3 (P1)** As an operator, `docker ps` shows `healthy` once the server is listening, and doesn't show it before.

**US-4 (P2)** As an operator, weights and generated images survive container rebuilds.

## Functional requirements

- **FR-002-01** The image MUST be based on an NVIDIA NGC PyTorch image that supports Blackwell on aarch64 (`nvcr.io/nvidia/pytorch:26.04-py3`). It MUST use the base image's torch/CUDA rather than installing torch from PyPI.
- **FR-002-02** The image MUST include only `server.py`, `config/`, `models/`, `storage/` and `pyproject.toml`. The build context is the repo root.
- **FR-002-03** Compose MUST reserve all NVIDIA GPUs and set `ipc: host`, `shm_size: 8gb`, and unlimited `memlock`.
- **FR-002-04** The Hugging Face cache MUST be bind-mounted from the host (`~/.cache/huggingface`), with `HF_HOME` pointing at it. Generated images MUST go to a named volume mounted at `/outputs`.
- **FR-002-05** `HF_TOKEN` MUST be passed through from the host environment. It MUST NOT be baked into the image or written into a compose file.
- **FR-002-06** The healthcheck MUST succeed only once the server's TCP listener is accepting connections. It MUST NOT depend on the HTTP status of `/mcp` (see research note below). Start period: 180 s.
- **FR-002-07** The restart policy MUST be `no` (Constitution VI).
- **FR-002-08** An override compose file MUST attach the service to the external network `ai-bridge`. The base file MUST work without that network existing.
- **FR-002-09** All feature-001 settings MUST be passed as environment variables, with defaults matching the server defaults.

## Research notes

- **The `/mcp` endpoint returns 406** to a plain GET with no MCP `Accept` headers. So `curl -f http://localhost:8500/mcp` always fails, and a container would never be marked healthy. *(Found late, in v0.1.1.)* A TCP connect check is enough, because the listener only opens after the model has loaded (feature 001, FR-001-14).
- **Running as root:** the NGC container expects root. The v0.1.0 non-root `appuser` was removed, and the HF cache path became `/root/.cache/huggingface`.
- **`PIP_CONSTRAINT=""`:** NGC images ship a pip constraints file. Clearing it lets the project's pins install. `pillow` is pinned to `11.1.0` for compatibility with the base image.

## Acceptance scenarios

- **AS-1** With a warm HF cache, `docker compose up -d --build` reaches `healthy` and `curl -s localhost:8500/mcp` returns a response (406 is fine).
- **AS-2** While the model is still loading, `docker ps` shows `starting`, not `healthy`.
- **AS-3** After `docker compose down && up`, earlier PNGs are still on the `image-outputs` volume and no weights are downloaded again.
- **AS-4** With the `ai-bridge` override applied, another container on `ai-bridge` can reach `http://local-image-gen-mcp:8500/mcp`.
- **AS-5** If the process exits, the container stays stopped.
- **AS-6** `docker history` / `docker inspect` of the image shows no `HF_TOKEN` value.

## Traceability

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| FR-002-01, 02 | `docker/Dockerfile` | ✅ |
| FR-002-03, 04, 05, 09 | `docker/docker-compose.yml` | ✅ |
| FR-002-06 | Python socket check in the Dockerfile and compose | ✅ |
| FR-002-07 | `restart: no` | ✅ |
| FR-002-08 | `docker/docker-compose.override.yml` | ✅ |
| AS-1 | manual (006 GPU window, 2026-10-18) | ✅ healthy 212 s after start (cold) |
| AS-2 | manual | ✅ status `starting` during the model load |
| AS-3 | manual | ✅ after `compose down && up`, all 16 PNGs were still on `image-outputs`; HF cache unchanged at 31 GB; no download log lines |
| AS-4 | manual | ✅ HTTP 200 `initialize` from a throwaway container on `ai-litellm-proxy_ai-bridge`. **Required fix F-3 (006 FR-006-14):** the override referenced a network `ai-bridge` that doesn't exist. |
| AS-5 | manual | ✅ after `kill 1`: `exited`, exit 0, 0 restarts, policy `no` |
| AS-6 | manual | ✅ 0 token matches in `docker history --no-trunc` / `docker image inspect` |

> **F-3 (found 2026-10-18 in 006):** the LiteLLM stack creates its network as `ai-litellm-proxy_ai-bridge`, so `docker-compose.override.yml` now sets `name: ${AI_BRIDGE_NETWORK:-ai-litellm-proxy_ai-bridge}`.

> Dependency versions are duplicated between `docker/Dockerfile` and `pyproject.toml`. Changing one requires changing the other.
