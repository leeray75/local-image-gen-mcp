# Feature 006 — Tasks

## Phase 1 — Code (implementer)

- [x] **T001** *(done: `config/settings.py`; `test_edit_output_resolution_*`)* `config/settings.py`: `edit_output_resolution` with its validators. → FR-006-01, AS-2
- [x] **T002** *(done: `models/__init__.py`, `models/base.py`, `models/qwen_image_2_1.py`)* Backend, base and factory: pass `edit_output_resolution` through and use it for edits. → FR-006-01, AS-1
- [x] **T003** *(done: `server.py`)* `server.py`: `streamable_http_path`, the `get_model_info` field, and the `edit_image` docstring and instructions wording. → FR-006-02, FR-006-05, AS-3, AS-4
- [x] **T004** *(done: `docker/Dockerfile`; pip step `CACHED` after a `server.py` change)* `docker/Dockerfile`: move the `COPY` lines below `pip install`. → FR-006-04
- [x] **T004b** *(done: `docker/docker-compose.yml`: two named volumes (see traceability FR-006-11); also passes `EDIT_OUTPUT_RESOLUTION` through)* `docker/docker-compose.yml`: compile-cache volume(s). → FR-006-11
- [x] **T005** *(done: `tests/test_models.py`, `tests/test_tools.py`, `tests/test_deploy.py`; full suite run in Docker)* Tests for AS-1 to AS-6, including the 003 AS-1 to AS-4 coverage. Run the full suite in Docker. → FR-006-07, NFR-006-02

## Phase 2 — Docs and release (implementer)

- [x] **T006** *(done: `.env.example`, `README.md`, `AGENTS.md`)* `.env.example`, README and AGENTS.md: `EDIT_OUTPUT_RESOLUTION`, the edit size wording, the rebuild note, and the compile-cache explanation. → FR-006-02, FR-006-12
- [x] **T007** *(done: `pyproject.toml`, `CHANGELOG.md`, 005 contract)* Version 0.2.0: `pyproject.toml`, CHANGELOG `[0.2.0] — 2026-10-18` with a new empty `[Unreleased]`, and the 005 contract note. → FR-006-06
- [x] **T008** *(done: 001 contract)* 001 contract update. → FR-006-08
- [x] **T009** *(done: 001/003/004 tasks.md, 001/003 traceability, SPECS.md)* Bookkeeping: the stale checkboxes, 004 T005 marked obsolete, and the SPECS.md 006 row and gaps. → FR-006-09

## Phase 2b — GPU-window findings (implementer)

- [x] **T009b** *(done: `tests/test_deploy.py`, README, AGENTS.md, `.env.example`, CHANGELOG)* F-3: add the `test_deploy.py` assertion and the docs for `AI_BRIDGE_NETWORK`. The override file is already fixed. → FR-006-14
- [x] **T009c** *(done: `config/settings.py`, `server.py`, `tests/test_tools.py`, docs)* F-4: `MAX_REQUEST_BODY_MB` setting, `FastMCP(max_request_body_size=…)`, unit and HTTP-level tests, docs. → FR-006-15–17

## Phase 3 — Review (orchestrator)

- [x] **T010** Independent review with up to 5 implementer↔reviewer rounds. *(3 review rounds plus a final doc check; findings fixed.)*

## Phase 4 — GPU window (orchestrator, vLLM stopped)

- [x] **T011** Run plan.md runbook steps 1–11. → FR-006-04 (cache check), FR-006-10, FR-006-13 (cold vs warm) *(Done 2026-10-18; results in the traceability table and research R-1/R-5.)*
- [x] **T012** *(done: default 1024 in `config/settings.py` and `models/qwen_image_2_1.py`; docs and tests updated)* Apply the A/B decision to the default and docs, and rerun the unit tests. → FR-006-03
- [x] **T013** Restart vLLM and confirm it is healthy. → NFR-006-03 *(vLLM `qwen3-8-27b-nvfp4-engine` healthy again 2026-10-18.)*

## Phase 5 — Close-out (orchestrator)

- [x] **T014** Record the results in the traceability tables (001, 002, 003, 006), research R-1, `ab/` thumbnails and the CHANGELOG. Set the spec status. *(Done.)*
- [x] **T015** Write a summary report in `ai-workspace/reports/`. *(`open-items-closeout-0.2.0-summary.md`)*
