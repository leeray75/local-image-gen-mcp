# Feature 008 — Tasks

## Phase 1 — Research and code (implementer)

- [x] **T001** Verify the exact quantization constructor calls for each mode and component against the installed versions, and the module names to skip (on the meta device, without a GPU). Record both in research.md as R-5. → FR-008-02, FR-008-05
- [x] **T002** `config/settings.py`: the two settings. → FR-008-01, AS-4
- [x] **T003** Backend: `_build_quantization_config`, the `from_pretrained` wiring, the CPU guard, error wrapping, logging and `quantization_info`. → FR-008-02 to 06, AS-1 to 3, AS-5
- [x] **T004** `models/base.py`, `models/__init__.py`, `server.py` (`get_model_info`), and compose. → FR-008-07, FR-008-08
- [x] **T005** Tests (NFR-008-02). Run the full suite in Docker, under the memory-pressure rule.

## Phase 2 — Docs (implementer)

- [x] **T006** README, AGENTS.md, `.env.example` and CHANGELOG `[Unreleased]`. The results table is a placeholder. → FR-008-08, FR-008-09
- [x] **T007** SPECS.md row; F-2 still open, with 008 noted as "in progress". → bookkeeping

## Phase 2b — F-6 mitigations (implementer, D-5 Phase A)

- [x] **T007b** Sequential component loading with memory freed between components, plus per-component memory logging. → FR-008-10, FR-008-12, AS-5b
- [x] **T007c** `PYTORCH_CUDA_ALLOC_CONF` in compose, plus docs. → FR-008-11, AS-5c
- [x] **T007d** Tests and the full suite in Docker, under the memory rule. Review round 3 follows.

## Phase 2c — NVFP4 status (implementer, D-9)

- [x] **T007e** FR-008-14: the `nvfp4` startup warning, docs, and a test.

## Phase 2d — Default (implementer, D-10/D-11)

- [x] **T007f** FR-008-15: the `auto` value and default, resolution, logging and `get_model_info`, tests (AS-5d), and docs (README results table with the measured fp8 and int8 rows, the round-4 README notes, AGENTS.md, `.env.example`, CHANGELOG).

## Phase 3 — Review (orchestrator)

- [x] **T008** Independent review, with up to 5 implementer↔reviewer rounds. *(Approved in round 2.)*

## Phase 4 — GPU (orchestrator, alongside vLLM, under the watchdog)

- [x] **T009** Build the image, then run the configurations in the AS-7 order. Stop and report if the watchdog fires. → AS-6, AS-7, NFR-008-03 *(Run 1, 2026-10-18: `nvfp4`/`nvfp4` tripped the watchdog during loading. F-6 recorded. Retry after T007b–d.)*
- [x] **T009b** *(Not needed: Phase A plus 8-bit fit next to vLLM, so pre-quantizing wasn't triggered.)* *(Contingent on D-5 Phase B.)* Spec addendum for FR-008-13, implementation, then a pre-quantize run in a vLLM-down window of up to 30 min, with notice (D-6).
- [x] **T010** Present the evidence to the operator, and record the default decision. → AS-8 *(int8/int8, with `auto`: D-10, D-11.)*

## Phase 5 — Close-out (orchestrator)

- [x] **T011** Apply the default, results table, thumbnails, research R-6, traceability, CHANGELOG, and close F-2 if it's resolved.
- [x] **T012** Write a summary report in `ai-workspace/reports/`. *(`load-time-quantization-summary.md`)*
