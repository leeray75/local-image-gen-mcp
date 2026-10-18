# Specifications — local-image-gen-mcp

Spec-driven development (SDD) artifacts for this project. Every feature moves through the same pipeline, and each stage produces a document that the next stage consumes:

```
constitution.md          (project-wide principles — applies to every feature)
      │
      ▼
spec.md        WHAT and WHY — user stories, requirements, acceptance criteria. No implementation detail.
      │
      ▼
research.md    Facts that must be verified before design (SDK behavior, model card, platform limits).
      │
      ▼
data-model.md  Entities and their fields / invariants.
contracts/     External interface contracts (MCP tools, resources, prompts, env vars).
      │
      ▼
plan.md        HOW — architecture, technical decisions, file layout, risks.
      │
      ▼
tasks.md       Ordered, dependency-annotated implementation tasks, each traced to requirements.
      │
      ▼
code + tests   Every requirement ID must map to code and (where testable) a test.
```

> **Provenance.** These documents were written retroactively (2026-10-18) from the plans and reports in `ai-workspace/`, `CHANGELOG.md`, and the code on branch `bugfix/meta-tensor-and-lifespan-fixes`. They describe what *should* have been specified before implementation. Findings that were discovered the hard way during v0.1.1 (per-session lifespan, `run()` signature, meta tensors, 406 healthcheck) are recorded in the research documents where a research phase would have caught them, and the requirements are written to the corrected behavior.

## Features

| ID | Feature | Release | Status | Source material |
|----|---------|---------|--------|-----------------|
| [001](001-image-generation-server/spec.md) | Image generation MCP server (core tools, resources, prompts, pluggable backends, storage) | 0.1.0 → 0.1.1 | Implemented | `ai-workspace/plans/implementation.md`, `plans/bug-fix.md`, `reports/implementation-summary.md`, `reports/bugfix-v0.1.1-summary.md` |
| [002](002-container-deployment/spec.md) | Container deployment on DGX Spark (Dockerfile, compose, healthcheck, ai-bridge) | 0.1.0 → 0.1.1 | Implemented | `plans/implementation.md` (Tasks 10–12), `reports/bugfix-v0.1.1-summary.md` |
| [003](003-image-retrieval-tools/spec.md) | Client-side image retrieval tools (`get_image_base64`, `get_latest_image_id`) | Unreleased | Implemented, uncommitted | `plans/add-tools.md`, `reports/uncommitted-changes-report.md` |
| [004](004-memory-bounded-loading/spec.md) | Memory-bounded model loading for coexistence with vLLM | Unreleased | Implemented, uncommitted | Uncommitted diff in `models/qwen_image_2512.py`, `docker/docker-compose.yml` |
| [005](005-qwen-image-2-1/spec.md) | Upgrade to Qwen-Image-2.1 (native 2K text-to-image, `edit_image`, RGBA, serialization, settings cleanup) | Unreleased (breaking) | Implemented, GPU-verified (edit-quality finding F-1 open) | Branch `feature/qwen-image-2-1` |

## Document index

| Document | Purpose |
|----------|---------|
| [constitution.md](constitution.md) | Non-negotiable project principles. Specs and plans must not violate these. |
| [001 spec](001-image-generation-server/spec.md) · [research](001-image-generation-server/research.md) · [data model](001-image-generation-server/data-model.md) · [contracts](001-image-generation-server/contracts/mcp-interface.md) · [plan](001-image-generation-server/plan.md) · [tasks](001-image-generation-server/tasks.md) | Core server |
| [002 spec](002-container-deployment/spec.md) · [plan](002-container-deployment/plan.md) · [tasks](002-container-deployment/tasks.md) | Deployment |
| [003 spec](003-image-retrieval-tools/spec.md) · [plan](003-image-retrieval-tools/plan.md) · [tasks](003-image-retrieval-tools/tasks.md) | Retrieval tools |
| [004 spec](004-memory-bounded-loading/spec.md) · [plan](004-memory-bounded-loading/plan.md) · [tasks](004-memory-bounded-loading/tasks.md) | Memory cap |
| [005 spec](005-qwen-image-2-1/spec.md) · [research](005-qwen-image-2-1/research.md) · [data model](005-qwen-image-2-1/data-model.md) · [contracts](005-qwen-image-2-1/contracts/mcp-interface.md) · [plan](005-qwen-image-2-1/plan.md) · [tasks](005-qwen-image-2-1/tasks.md) | Qwen-Image-2.1 upgrade |

## Conventions

- **Requirement IDs** are scoped per feature: `FR-001-07` = feature 001, functional requirement 7. Non-functional requirements use `NFR-`, acceptance scenarios use `AS-`.
- **Requirement language:** MUST / MUST NOT / SHOULD / MAY (RFC 2119).
- **Status markers** in traceability tables: ✅ met · ⚠️ partially met · ❌ not met · ⏳ not yet verified on hardware.
- **Adding a feature:** copy the folder layout of an existing feature as `NNN-short-name/`, write `spec.md` first and get it reviewed before writing `plan.md`. Add a row to the table above.
- **Changing a feature:** update `spec.md` first, then `plan.md`/`tasks.md`, then code. Record the change in `CHANGELOG.md`.

## Open gaps across features

Collected from the traceability tables in each spec. These are the places where the current code does not yet meet its spec.

| Gap | Feature | Reference |
|-----|---------|-----------|
| ~~`tests/test_tools.py` is empty~~ — closed by 005 (NFR-005-03) | 001, 003 | FR-001-17, FR-003-06 |
| `MCP_PATH` is read into settings and logged but never passed to `FastMCP`; it only works because `/mcp` is the SDK default | 001 | FR-001-13 |
| ~~`cfg_scale` not clamped~~ — closed by 005 (FR-005-07) | 001 | FR-001-04 |
| ~~`ImageStore` docstring claimed a rebuild from disk~~ — closed by 005 (T005) | 001 | FR-001-10 |
| 2K edits show over-sharpening artifacts; suspected `output_resolution=2048` | 005 | F-1, T022 |
| Dockerfile copies code before `pip install`, so code-only rebuilds reinstall all deps | 002 | T023 (005) |
| ~~Concurrent calls not serialized~~ — closed by 005 (NFR-005-01) | 001 | NFR-001-03 |
| ~~`IMAGE_GPU_MEMORY_LIMIT` bypassed `Settings`~~ — closed by 005 (FR-005-24) | 004 | FR-004-05 |
