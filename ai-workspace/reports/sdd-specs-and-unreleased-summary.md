# SDD Specs & Unreleased Changes — Summary Report

> **Historical snapshot.** This report describes the repo as of its date. For current state see `CHANGELOG.md`, `docs/specs/SPECS.md` and later reports.

**Date:** 2026-10-18
**Branch:** `bugfix/meta-tensor-and-lifespan-fixes`
**Last committed:** `a42846d` — add memory limit 40g to docker compose
**Status:** All changes below are uncommitted

---

## Overview

This report covers three bodies of uncommitted work on the branch:

1. **Two new MCP tools** for client-side image retrieval (from `plans/add-tools.md`)
2. **Memory-bounded model loading** so Qwen-Image-2512 can run alongside vLLM on the GB10's 128 GB unified memory
3. **Spec-driven development (SDD) documentation**, written after the fact from the plans and reports in this folder, plus a `CLAUDE.md` for the repo

It supersedes `uncommitted-changes-report.md`. That report covered only the tools change in `server.py`, before the memory-cap and documentation work.

---

## 1. Image Retrieval Tools

**File:** `server.py` (+69 lines, inserted after `get_model_info`, before `# ── Resources`)
**Spec:** `docs/specs/003-image-retrieval-tools/`

| Tool | Returns | Purpose |
|------|---------|---------|
| `get_latest_image_id()` | `{image_id, prompt[:100], aspect_ratio, width, height, seed, model_name}` | Gives the agent the newest image ID without reading resources. Raises `RuntimeError` if nothing has been generated yet. |
| `get_image_base64(image_id)` | Plain base64 PNG string | Lets an agent save the image on the client. The docstring warns that the string must be decoded before writing. |

**Why:** Many MCP clients show agents only tools, not resources, and the server's `/outputs` is inside the container. Before this, agents had no reliable way to save a generated image into a project.

**Constraint met:** no existing tools, resources, prompts, imports or startup code were changed.

---

## 2. Memory-Bounded Model Loading

**Files:** `models/qwen_image_2512.py` (+119 / −19), `docker/docker-compose.yml`
**Spec:** `docs/specs/004-memory-bounded-loading/`

**Problem:** Uncapped BF16 loading uses about 62 GB. vLLM uses about 56 GB. Together that is about 118 GB of the 128 GB unified pool, which caused swap thrashing and instability. The `limits.memory: 40g` added in `a42846d` didn't help: a cgroup limit doesn't change where accelerate places weights, so it only gets the container OOM-killed.

**Changes:**

| Change | Detail |
|--------|--------|
| `device_map="cuda"` → `"balanced"` | Needed for accelerate to honor `max_memory`. Still passed to `from_pretrained()`, so the meta-tensor fix from v0.1.1 is kept. |
| `IMAGE_GPU_MEMORY_LIMIT` env var | When set, `max_memory={0: "<N>GiB", "cpu": "60GiB"}`. When unset, the model loads uncapped and a warning is logged. |
| Startup logging | Logs the memory mode and the post-load CUDA allocated/reserved GB |
| Compose | Adds `IMAGE_GPU_MEMORY_LIMIT: "40"` and removes `deploy.resources.limits.memory: 40g` |

**Trade-off (estimated, not measured):**

| Cap | Time per image |
|-----|----------------|
| 30 GB | ~15–20 min |
| 40 GB | ~8–12 min |
| 50 GB | ~4–5 min |
| unset | ~3–4 min |

---

## 3. Documentation

### `docs/specs/` — spec-driven development artifacts (new)

| Document | Content |
|----------|---------|
| `SPECS.md` | Index: the SDD pipeline, feature table, conventions, and gaps across features |
| `constitution.md` | Eight project principles (local-first, model-agnostic, env-only config, verify the SDK, non-blocking, respect the shared host, GPU-free tests, self-describing tools) |
| `001-image-generation-server/` | `spec.md`, `research.md`, `data-model.md`, `contracts/mcp-interface.md`, `plan.md`, `tasks.md` |
| `002-container-deployment/` | `spec.md`, `plan.md`, `tasks.md` |
| `003-image-retrieval-tools/` | `spec.md`, `plan.md`, `tasks.md` |
| `004-memory-bounded-loading/` | `spec.md`, `plan.md`, `tasks.md` |

**Sources:** `plans/implementation.md`, `plans/bug-fix.md`, `plans/add-tools.md`, `reports/implementation-summary.md`, `reports/bugfix-v0.1.1-summary.md`, `CHANGELOG.md`, and the current code.

**Lessons recorded:** The four v0.1.1 bugs appear in `001/research.md` and `002/spec.md` as findings marked *found late*, at the point where a research phase would have caught them:
- the lifespan runs per session
- `run()` accepts only `transport`
- meta tensors break `.to()`
- `/mcp` returns 406 to a plain GET

Three of the four came from unverified SDK assumptions in the original plan. That led to Constitution principle IV, "Verify the SDK, don't assume it".

### Other documentation

- **`CLAUDE.md` (new):** commands, architecture, Qwen and Docker pitfalls, and a pointer to the SDD workflow.
- **`CHANGELOG.md`:** new `[Unreleased]` section with Added, Changed, Removed and Known issues.

---

## Spec Gaps Found

Writing the traceability tables turned up these places where the code doesn't meet its spec. Each one is an open task in the matching `tasks.md`.

| Gap | Severity | Task |
|-----|----------|------|
| `tests/test_tools.py` is empty. No coverage for tools, aspect ratios, the image store, or the two new tools. | High | 001 T014, 003 T004 |
| Concurrent `generate_image` calls share one pipeline with no lock | Medium | 001 T018 |
| `MCP_PATH` is logged but never passed to `FastMCP` | Low | 001 T017 |
| `cfg_scale` isn't clamped to 1.0–10.0 | Low | 001 T016 |
| `IMAGE_GPU_MEMORY_LIMIT` bypasses `Settings` and is missing from `.env.example`, the README and the Dockerfile `ENV` | Low | 004 T003 |
| The `ImageStore` docstring claims the index is rebuilt from disk; it isn't | Low | 001 T019 |
| No GPU acceptance run has been recorded for any feature | Medium | 001 T020, 002 T005, 003 T007, 004 T005 |

---

## Testing Recommendations

1. **Unit tests:** `uv run pytest tests/ -v`. These only cover the model layer until T014 is done.
2. **Tool registration:**
   ```bash
   python3 -c "from server import mcp; print([t.name for t in mcp._tool_manager.list_tools()])"
   ```
   The list should contain all seven tools.
3. **Memory cap:** run with vLLM up and `IMAGE_GPU_MEMORY_LIMIT=40`. Check that the log shows "Memory cap active", that `get_model_info` reports `gpu_memory_allocated_gb` ≤ about 40, and that the host stays responsive during generation.
4. **Save workflow:** from an MCP client, call `generate_image`, then `get_latest_image_id`, then `get_image_base64`. Decode the result, write it as a `.png`, and open it.
5. **Timing:** measure the time per image at each cap and replace the estimates in `004/spec.md`.

---

## Suggested Commits

```bash
git add models/qwen_image_2512.py docker/docker-compose.yml
git commit -m "feat: cap model memory via IMAGE_GPU_MEMORY_LIMIT for vLLM coexistence"

git add server.py ai-workspace/plans/add-tools.md
git commit -m "feat: add get_image_base64 and get_latest_image_id tools"

git add docs/ CLAUDE.md CHANGELOG.md ai-workspace/reports/
git commit -m "docs: add SDD specs, CLAUDE.md, changelog and reports"
```
