# Feature 004 — Memory-Bounded Model Loading

| | |
|---|---|
| **Status** | Implemented, uncommitted (`models/qwen_image_2512.py`, `docker/docker-compose.yml` working tree) |
| **Constitution** | VI |
| **Depends on** | [001](../001-image-generation-server/spec.md), [002](../002-container-deployment/spec.md) |
| **Next** | [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

The GB10's 128 GB is a single pool shared by the CPU and the GPU. With all weights on the device in BF16, Qwen-Image-2512 uses about 62 GB. vLLM, which is resident on the same host, uses about 56 GB. Together that is about 118 GB, which leaves around 10 GB for the OS and causes swap thrashing and system instability.

Commit `a42846d` tried to fix this with a Docker `deploy.resources.limits.memory: 40g`. A cgroup limit doesn't make the model *place* less data on the device. It only gets the container OOM-killed when it goes over the limit.

## Goal

Let the operator cap how much device memory the image model takes. Weights that don't fit spill to CPU offload, which makes generation slower but keeps the host stable while vLLM is running.

## Non-goals

- Changing the model to a quantized format (that would be a separate backend or feature).
- Coordinating dynamically with vLLM (for example, reading its utilization).
- Unloading the model while idle.

## User stories

**US-1 (P1)** As an operator, I can set one environment variable to cap the image model's device memory, so it can run alongside vLLM without destabilizing the host.

**US-2 (P2)** As an operator, if I leave the variable unset I get the original full-speed behavior, plus a clear warning about the risk.

**US-3 (P3)** As an operator, I can see from the startup logs which mode is active and how much memory was allocated.

## Functional requirements

- **FR-004-01** On CUDA, the backend MUST load with `device_map="balanced"` passed to `from_pretrained`. This keeps the meta-tensor rule from 001 R-6 and lets accelerate honor `max_memory`.
- **FR-004-02** When `IMAGE_GPU_MEMORY_LIMIT=<N>` (an integer number of GB) is set, the backend MUST pass `max_memory={0: "<N>GiB", "cpu": "60GiB"}`, so layers over the cap are offloaded to CPU.
- **FR-004-03** When the variable is unset or empty, no `max_memory` MUST be passed. The backend logs a warning that the model will use about 62 GB and that vLLM should be stopped first.
- **FR-004-04** At startup, the backend MUST log the mode (capped with N, or uncapped). After loading, it MUST log the PyTorch CUDA allocated and reserved GB.
- **FR-004-05** `IMAGE_GPU_MEMORY_LIMIT` SHOULD be part of `Settings` (validated, and listed in `.env.example` and the README environment table), consistent with Constitution III. *(Open: currently read via `os.getenv` at module import in `models/qwen_image_2512.py`.)*
- **FR-004-06** The default deployment (compose) MUST set `IMAGE_GPU_MEMORY_LIMIT=40` and MUST NOT use a container memory limit as the mechanism.
- **FR-004-07** The CPU-only path (no CUDA) MUST be unaffected: no `device_map`, no `max_memory`.

> **Correction (2026-10-18, measured during 005):** on GB10 unified memory, layers that `max_memory` offloads to "CPU" still occupy the same physical RAM. The cap bounds GPU allocator placement, **not** the process's total memory, so it does not let the model coexist with vLLM by itself. With vLLM at `gpu-memory-utilization 0.6` (~73 GB), only ~28 GiB was free. The coexistence claims below are unverified, and probably wrong for total-memory pressure. See [005 NFR-005-02](../005-qwen-image-2-1/spec.md).

## Guidance values

> **Superseded by 005:** the table below was estimated for Qwen-Image-2512 and its coexistence column is unverified (see the correction above). Qwen-Image-2.1 needs ~33 GB of weights plus activations; run it with vLLM stopped.

These come from the code comments and are estimates, not benchmarks. Re-measure them and update this table (see tasks T005).

| Cap | Estimated time / image (1:1, 50 steps) | Coexists with vLLM |
|-----|-----------------------------------|--------------------|
| 30 GB | ~15–20 min | yes, at any vLLM utilization |
| 40 GB | ~8–12 min | yes, at vLLM `gpu-memory-utilization` 0.6 |
| 50 GB | ~4–5 min | marginal, at 0.5 |
| unset | ~3–4 min | no; stop vLLM first |

## Acceptance scenarios

- **AS-1** With `IMAGE_GPU_MEMORY_LIMIT=40` and vLLM running, the server loads, generates a 1:1 image, and the host stays responsive with no swap storm. `get_model_info` reports `gpu_memory_allocated_gb` ≤ about 40.
- **AS-2** With the variable unset, the startup log contains the "No memory cap" warning.
- **AS-3** With the variable set, the startup log contains "Memory cap active" with the value.
- **AS-4** A unit test with a mocked `DiffusionPipeline` and `IMAGE_GPU_MEMORY_LIMIT=30` asserts that `from_pretrained` got `device_map="balanced"` and `max_memory[0] == "30GiB"`.
- **AS-5** On CPU, `from_pretrained` gets no `device_map`.

## Traceability

| Requirement | Implementation | Test | Status |
|-------------|----------------|------|--------|
| FR-004-01–04, 07 | `QwenImage21.load`, `memory_limit_gb` | `test_models::test_load_*` | ✅ |
| FR-004-05 | `Settings.image_gpu_memory_limit` (005 FR-005-24) | `test_settings_memory_limit_reaches_backend` | ✅ |
| FR-004-06 | `docker-compose.yml` env; `limits.memory` removed | — | ✅ |
| AS-1 | manual on DGX Spark | — | Obsolete (006 FR-006-09): vLLM can't run at the same time (005 M-1); the measured footprint is recorded in 005/006 |

> The v0.1.1 CHANGELOG entry and `reports/bugfix-v0.1.1-summary.md` say `device_map="cuda"`. That was accurate for v0.1.1. The `[Unreleased]` CHANGELOG entry records the change to `"balanced"`.
