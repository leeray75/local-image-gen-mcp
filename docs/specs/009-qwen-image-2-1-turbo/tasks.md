# Feature 009 — Tasks

## Phase 1 — Code (implementer)

- [x] **T001** Read the diffusers v0.41.0 `QwenImage21Pipeline.__call__` and record how `num_inference_steps` interacts with the configured `sample_sigmas`. Decide whether to omit the argument or pass 8. Note it in research R-1. → FR-009-03
- [x] **T002** Settings, backend variants, and the factory. → FR-009-01, AS-1
- [x] **T003** Backend-driven step limits, the Turbo warning, and `get_model_info`. → FR-009-02, FR-009-04, AS-2 to AS-4
- [x] **T004** Docstrings and server instructions. → FR-009-05
- [x] **T005** Tests, then the full suite in Docker under the memory-pressure rule. → NFR-009-02
- [x] **T006** Docs: README, AGENTS.md, `.env.example`, compose check, CHANGELOG, SPECS.md row. → FR-009-06, FR-009-07

## Phase 2 — Review (orchestrator)

- [x] **T007** Independent review, up to 5 implementer↔reviewer rounds *(Approved in round 3.)*

## Phase 3 — GPU (orchestrator, alongside vLLM, under the watchdog)

- [x] **T008** Runbook steps 1–6 in plan.md → AS-5, AS-6, NFR-009-03 *(2026-10-18; research R-4.)*
- [x] **T009** The operator's default decision → AS-7 *(Turbo is the default, D-5.)*

## Phase 4 — Close-out

- [x] **T010** Results, thumbnails, traceability, CHANGELOG, and the default if it changes *(Done.)*
- [x] **T011** Summary report in `ai-workspace/reports/` *(`qwen-image-2-1-turbo-summary.md`)*
