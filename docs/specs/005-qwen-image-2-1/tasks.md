# Feature 005 — Tasks

Run the tasks in order. `[P]` marks a task that can run in parallel with the others in the same phase. Each task names the requirements it covers.

## Phase 0 — Environment

- [x] **T001** Update the deps in `pyproject.toml` and `docker/Dockerfile` (`diffusers>=0.41.0`, `transformers>=5.17`). Build `local-image-gen-mcp:qwen21-test`. Check that `python -c "from diffusers import QwenImage21Pipeline"` works in the image. Record any extra pins in research.md R-2. → FR-005-25

## Phase 1 — Core types and config

- [x] **T002 [P]** `models/base.py`: the new request/result fields and the `BaseImageModel` properties (data model). → FR-005-03
- [x] **T003 [P]** `config/settings.py`: the field changes and the empty-string → `None` validator. → FR-005-02, FR-005-24

## Phase 2 — Backend and storage

- [x] **T004** `models/qwen_image_2_1.py` (D-1). Delete `models/qwen_image_2512.py`. Update the `models/__init__.py` factory. *(depends T002, T003)* → FR-005-01–05, 15–17
- [x] **T005 [P]** `storage/image_store.py`: the new fields, `load_image()`, and the docstring fix. *(depends T003)* → FR-005-18
- [x] **T006 [P]** `storage/references.py` (D-3) and the `storage/__init__.py` exports. *(depends T005)* → FR-005-12, 13, 20

## Phase 3 — Server

- [x] **T007** `server.py`:
  - remove the `config.aspect_ratios` and 2512 imports, and delete `config/aspect_ratios.py`
  - add the `Ratio` alias, `_generation_lock`, `_resolve_params` and `_run_generation`

  *(depends T004–T006)* → FR-005-06–09, NFR-005-01
- [x] **T008** `generate_image`: the new signature and the `transparent` flag. → FR-005-10, 16, 17
- [x] **T009** `edit_image`. → FR-005-11–15, 18
- [x] **T010 [P]** `list_aspect_ratios`, `get_model_info`, `images://list`, `get_latest_image_id`, and the server `instructions`. → FR-005-19, 21, 22
- [x] **T011 [P]** Prompts: remove `negative_defaults`, add `transparent_asset` and `extract_subject`, and make `photorealistic` model-neutral. → FR-005-23
- [x] Docstrings for the T008/T009 tools must meet NFR-005-04 and NFR-005-05.

## Phase 4 — Deployment and docs

- [x] **T012** Deployment and docs:
  - `docker/Dockerfile` and `docker/docker-compose.yml`: `ENV` / environment changes, removing the step and CFG defaults
  - `.env.example`
  - `README.md`: the model, license note (non-commercial), tools table with `edit_image`, prompts, the env table, memory, and new examples
  - `AGENTS.md`: pitfalls (guidance off by default, the VAE always returns RGBA, `output_resolution`, reference order, the 32-hex UUID trap); remove the 2512-only notes
  - in README and AGENTS.md, describe memory as NFR-005-02 requires: ~33 GB of weights plus activations, run with vLLM stopped, and the cap does not reduce total RAM on unified memory. Fix the old "safe alongside vLLM" wording in `models/` comments.

  → FR-005-26
- [x] **T013** `CHANGELOG.md` `[Unreleased]`: mark it **breaking**, and list Added/Changed/Removed per the contract. Spec updates:
  - `docs/specs/SPECS.md`: add the 005 row, and update the gaps (001 FR-001-04, NFR-001-03, T019 and 004 FR-004-05 are now closed by 005)
  - add "Superseded by 005" notes to 001 R-5 and the 001 spec traceability table, and to the 004 guidance table

## Phase 5 — Tests

- [x] **T014 [P]** `tests/conftest.py` and `tests/test_models.py`. → NFR-005-03; AS-1, 2, 5, 6, 13
- [x] **T015 [P]** `tests/test_references.py`. → AS-7, 8
- [x] **T016 [P]** `tests/test_tools.py`. → AS-3, 4, 9, 10, 11, 12
- [x] **T017** Run the full suite with the Docker command in plan.md. It must pass with 0 failures. Paste the summary line into the implementation report.

## Phase 6 — Review loop and close-out

- [x] **T018** Independent review against spec.md, the contract and the constitution, with up to 5 implementer↔reviewer rounds.
- [x] **T019** Fill in the traceability table in spec.md and tick the boxes in this file. Set the spec status to "Implemented (GPU smoke test pending)".
- [x] **T020** *(only after the operator gives the go-ahead)* GPU smoke test AS-14 and AS-15. Record the timings and memory in the CHANGELOG and the report. *(Done 2026-10-18. Edit-quality finding F-1 is open.)*
- [x] **T022** *(done in 006: A/B run, default 1024)* Follow-up for F-1: A/B the edit at `output_resolution=1024` vs 2048, and adjust `EDIT_OUTPUT_RESOLUTION`.
- [x] **T023** *(done in 006 FR-006-04: app COPY after pip install; pip step CACHED on rebuild)* Follow-up: in the Dockerfile, copy the app code *after* `pip install` so code-only rebuilds reuse the dependency layer.
- [x] **T021** Write a summary report in `ai-workspace/reports/`.
