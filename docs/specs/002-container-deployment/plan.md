# Feature 002 — Implementation Plan

**Inputs:** [spec.md](spec.md)

## Constitution check

| Principle | Compliance |
|-----------|------------|
| III Env-only config / secrets | `HF_TOKEN: ${HF_TOKEN}` comes from the host shell. Nothing is set in the Dockerfile. |
| VI Shared host | `restart: no`. The memory cap is set via env (feature 004). |

## Layout

```
docker/
├── Dockerfile                    # build context = ..
├── docker-compose.yml            # standalone
└── docker-compose.override.yml   # + external ai-bridge network
```

## Decisions

### D-1 Base image

`nvcr.io/nvidia/pytorch:26.04-py3`. The generic `pytorch/pytorch:*-cuda12.8` image used in v0.1.0 doesn't target GB10/aarch64. We install the Python deps with `pip` straight into the system interpreter, with `PIP_CONSTRAINT=""`, and do **not** list `torch`, so NVIDIA's build is kept. We don't use `uv` in the image.

### D-2 Healthcheck

```
python3 -c "import socket; s=socket.socket(); s.connect(('localhost', 8500)); s.close()"
```

The check is the same in the Dockerfile `HEALTHCHECK` and in compose `healthcheck`: interval 30 s, timeout 10 s, 5 retries, start period 180 s. It uses Python rather than curl because the base image is guaranteed to have Python and `/mcp` returns 406.

### D-3 Volumes

| Mount | Kind | Why |
|-------|------|-----|
| `~/.cache/huggingface:/root/.cache/huggingface` | bind | Shares the host cache across rebuilds and with other HF tools |
| `image-outputs:/outputs` | named volume | Output survives `down`/`up` |

### D-4 Entrypoint

Compose sets `entrypoint: ["python", "server.py"]` explicitly, to override whatever entrypoint the NGC base image defines. The Dockerfile `CMD` stays as the default.

### D-5 Network integration

`docker-compose.override.yml` adds `networks: [ai-bridge]` with `external: true`. Run both files with `-f` when the stack is up. The base file stays standalone, so isolated testing doesn't need the stack.

## Commands

```bash
cd docker
docker compose up -d --build
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build
docker logs -f local-image-gen-mcp
```

## Risks

| Risk | Mitigation |
|------|------------|
| First boot downloads the weights (about 33 GB for Qwen-Image-2.1) and can exceed the start period | The start period only delays marking the container unhealthy, and the restart policy is `no`, so nothing kills it. Pre-warm the cache if needed. |
| NGC base image update changes the torch ABI | Pin the NGC tag. Rebuild deliberately. |
| Deps drift between the Dockerfile and `pyproject.toml` | Review both in every dependency PR |
