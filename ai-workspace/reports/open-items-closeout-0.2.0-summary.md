# Open Items Close-out and 0.2.0 Release (Feature 006) — Summary Report

**Date:** 2026-10-18
**Branch:** `feature/qwen-image-2-1-edit-rgba-upgrade` (continued, per decision D-4)
**Spec:** [`docs/specs/006-open-items-closeout/`](../../docs/specs/006-open-items-closeout/spec.md)
**Status:** Implemented, reviewed and GPU-verified. All tasks in features 001–006 are ticked. Committed and pushed on the branch.

---

## Overview

This feature closes every open item left after the Qwen-Image-2.1 upgrade, except **F-2**: the memory footprint and running alongside vLLM, which the operator excluded. It also cuts the **0.2.0** release in `pyproject.toml` and the CHANGELOG. The `v0.2.0` tag is created when the branch merges.

The GPU window found **two new bugs**, F-3 and F-4. Both are fixed and verified on hardware.

---

## Process

| Stage | Outcome |
|-------|---------|
| Planning | 8 open items found. Several "open" checkboxes turned out to be stale, with the work already done in 005. |
| Decisions | D-1: A/B test, then an env setting for the edit resolution · D-2: version 0.2.0, tagged at merge · D-3: every pending GPU check in one window · D-4: leave the pushed git history alone · D-5: persist compile caches (O-9, from the operator's question) · D-6: request-size setting sized to fit (F-4) |
| Implementation | A Sonnet implementer: T001–T009, then T009b/T009c (F-3, F-4), then T012 (the 1024 default), then doc fixes |
| Review | A separate Sonnet reviewer, **4 rounds** (limit 5). R1 approved with 2 minor findings. R2 approved after the F-3/F-4 work. R3 asked for changes (1 major bookkeeping finding, 4 minor). R4 was a final consistency check: every recorded number matched the evidence, and 3 minor stale statuses were fixed by the orchestrator. |
| GPU window | vLLM was stopped for the whole implementation and GPU window, then restarted and confirmed healthy. The image server was left stopped. |

---

## Results by Item

| # | Item | Resolution | Evidence |
|---|------|------------|----------|
| O-1 / F-1 | 2K edit artifacts | **`EDIT_OUTPUT_RESOLUTION`, default 1024** | A/B test: 2048 had artifacts, **ignored the dusk instruction** and **failed to extract** (0% transparent pixels). 1024 was clean on all 3 cases (85.5% transparent on the extraction), **~5.5× faster** (61 s vs 340 s) and used **~40 GB less** peak RAM. Thumbnails are in `docs/specs/006-…/ab/`. |
| O-2 | Dockerfile layer order | App `COPY` lines moved after `pip install` | Rebuilds show the pip step as `CACHED` |
| O-3 | `MCP_PATH` ignored | Passed as `streamable_http_path` | Unit test |
| O-4 | Version mismatch | **0.2.0**, with CHANGELOG `[0.2.0] — 2026-10-18` | — |
| O-5 | Manual GPU acceptance | **All passed** | 001 AS-2: byte-identical seed output · 001 AS-5: 0.02 s response during a generation · 002 AS-1–AS-6 · 003 save workflow |
| O-6 | 003 tests | Added | 99 tests in total |
| O-7 | 001 contract | Historical banner, current tool set, strikethroughs, links | — |
| O-8 | Stale checkboxes | Ticked with notes. 004 T005 marked obsolete. | — |
| O-9 | Compile caches lost on container recreate | Named volumes `image-gen-cuda-cache` and `image-gen-torch-cache` | The cache survived a recreate. It saves **~6 s** on the first generation and nothing on startup, which is dominated by the ~3¼ min weight load. |
| **O-10 / F-3** | *(new)* The `ai-bridge` override never worked, because the network is really `ai-litellm-proxy_ai-bridge` | `name: ${AI_BRIDGE_NETWORK:-ai-litellm-proxy_ai-bridge}` | 002 AS-4: HTTP 200 from a container on the bridge |
| **O-11 / F-4** | *(new)* The MCP SDK rejects request bodies over 4 MiB, so base64 references above ~3 MB got HTTP 413 | `MAX_REQUEST_BODY_MB` (computed default of 268 MB), plus an HTTP-level test | AS-8: a 12.7 MB base64 2K reference was edited in 61.3 s |

---

## Measurements (DGX Spark GB10, vLLM stopped)

| Metric | Value |
|--------|-------|
| Container start → ready | 194 s cold, 197 s warm (healthy at 212 s) |
| 20-step 1:1 generate | 136.9 s cold first run, 131.8 s second, 131.0 s warm first run |
| 40-step 2K generate | ~257 s |
| Edit at 1024 (new default) | ~61 s, ~49 GB peak host RAM (system-wide) |
| Edit at 2048 | ~340 s, ~90 GB peak host RAM |
| Compile caches | CUDA 195 MB, torch 104 KB |

---

## Process Notes

- **The hidden token-exposure risk was handled, not avoided.** During implementation, `docker compose config` printed `HF_TOKEN` into an agent's tool output. It was never written to a file: secret scans found 0 matches in the repo and in the image. The later agent prompts banned `compose config`. **Rotating the token is recommended**, because it appeared in the session transcript.
- **Runbook bug.** My first GPU-check script hung: command substitution waited on a background memory sampler, so the GPU sat idle for about 15 minutes until the operator noticed. It was fixed and rerun. No results were lost.
- **Unit tests missed F-4**, because they call tool functions directly. The new HTTP-level test goes through the real ASGI stack.

---

## Open Items

| Item | Status |
|------|--------|
| **F-2:** the image server (~68 GB peak for a 2K generate) can't run alongside vLLM at 0.6 | Excluded from this feature by the operator. Options: keep switching between the two, or a quantized variant. |
| `v0.2.0` tag | Create it when the branch merges into `master` |
| Commit and push | Done on `feature/qwen-image-2-1-edit-rgba-upgrade` |
| Pushed commit `e2dfbd9` doesn't start on its own (deletions landed in the wrong commit) | Left as is, by operator decision D-4 |

---

## Host State

- **vLLM `qwen3-8-27b-nvfp4-engine`:** running and healthy.
- **`local-image-gen-mcp`:** stopped (exited during 002 AS-5).
- **`docker-local-image-gen-mcp` image:** the 0.2.0 build with F-3 and F-4. Its default is still 2048, because the 1024 default was applied after the last rebuild (see below).
- **Volumes:** `docker_image-outputs`, `docker_image-gen-cuda-cache`, `docker_image-gen-torch-cache`.

> **Rebuild needed before the next use:** the 1024 default was applied after the last image build. Run `docker compose build` (only the cached code layers change) before starting the server, or set `EDIT_OUTPUT_RESOLUTION=1024` explicitly.
