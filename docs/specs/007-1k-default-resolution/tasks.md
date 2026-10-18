# Feature 007 — Tasks

## Phase 1 — Code (implementer)

- [x] **T001** `config/settings.py`: the two new settings, the shared validators, and the ceiling `model_validator`. → FR-007-01, FR-007-02, AS-3
- [x] **T002** Backend, base and factory: `sizes_for`, `CARD_SIZES_2048`, and `generate_output_resolution` passed through. → FR-007-03, FR-007-04, AS-1, AS-2
- [x] **T003** `server.py`: the F-5 fix in `edit_image`, the `get_model_info` fields, the docstrings and the instructions. → FR-007-05 to 07, AS-4, AS-5
- [x] **T004** `docker/docker-compose.yml`: pass the variables through. → FR-007-08
- [x] **T005** Tests (NFR-007-01). Update the tests that assume a 2K default, then run the full suite in Docker.

## Phase 2 — Docs (implementer)

- [x] **T006** `.env.example`, README and AGENTS.md. → FR-007-08
- [x] **T007** CHANGELOG `[Unreleased]`, marked breaking. → FR-007-09
- [x] **T008** Spec bookkeeping and the SPECS.md row and gaps. → FR-007-10

## Phase 3 — Review (orchestrator)

- [x] **T009** Independent review, with up to 5 implementer↔reviewer rounds. *(Approved in round 1. The one MINOR finding was applied by the orchestrator.)*

## Phase 4 — GPU window (orchestrator, vLLM stopped) — **deferred (D-5)**; T010–T011 stay open for a later window

- [ ] **T010** Runbook steps 1–7 in plan.md. → AS-6, AS-8, AS-3 on hardware, NFR-007-03
- [ ] **T011** Operator decision on the default, from the evidence. → AS-7

## Phase 5 — Close-out (orchestrator)

- [ ] **T012** Thumbnails, research R-5, traceability, CHANGELOG measurements, and the spec status. *(Spec status done. The measurements are deferred with T010.)*
- [x] **T013** Write a summary report in `ai-workspace/reports/`. *(`1k-default-resolution-summary.md`)*
