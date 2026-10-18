# Feature 001 — Tasks

Run the tasks in order. `[P]` marks a task that can run in parallel with the others at the same level. Each task names the requirements it satisfies and how it's checked.

## Phase 1 — Foundations

- [x] **T001 [P]** `config/aspect_ratios.py`: the `ASPECT_RATIOS` table from R-5, plus `get_dimensions()` (raises `ValueError`) and `valid_ratios()`. → FR-001-02
- [x] **T002 [P]** `config/settings.py`: `Settings(BaseSettings)` with the fields from the data model and a module singleton. → FR-001-18, FR-001-17 (Literal)
- [x] **T003 [P]** `models/base.py`: the `GenerationRequest` and `GenerationResult` dataclasses and the `BaseImageModel` ABC. → FR-001-17

## Phase 2 — Backend and storage

- [x] **T004** `models/qwen_image_2512.py`: `QwenImage2512` and `DEFAULT_NEGATIVE_PROMPT`. Use `true_cfg_scale` and a seeded Generator. Pass `device_map` to `from_pretrained` with no `.to()`. Add a CPU fallback and call `empty_cache` after generation. *(depends T003)* → FR-001-05, 06, 19
- [x] **T005** `models/__init__.py`: the `load_model()` factory with a lazy import, raising `ValueError` for unknown backends. *(depends T002, T004)* → FR-001-17
- [x] **T006 [P]** `storage/image_store.py` and `storage/__init__.py`: `StoredImage`, a thread-safe `ImageStore`, and the `store` singleton. *(depends T002)* → FR-001-10, 12; NFR-001-03 (index)

## Phase 3 — Server

- [x] **T007** `server.py` skeleton: create `FastMCP("local-image-gen", instructions=…, json_response=True, host=…, port=…)` with **no lifespan**. Add the `_model` singleton, `get_model()`, and `_startup_load_model()`. In the entrypoint, set up logging, load the model, then call `mcp.run(transport="streamable-http")`. *(depends T005, T006)* → FR-001-13, 14, 15
- [x] **T008** `generate_image` tool: resolve the seed, map the ratio, clamp the steps, report progress and info logs, run `to_thread` generation, save to the store, and return an `Image`. *(depends T007)* → FR-001-01–07, NFR-001-01
- [x] **T009 [P]** `list_aspect_ratios` and `get_model_info` tools. *(depends T007)* → FR-001-08, 09
- [x] **T010 [P]** `image://{image_id}` and `images://list` resources. *(depends T007)* → FR-001-11, 12
- [x] **T011 [P]** `photorealistic` and `negative_defaults` prompts. *(depends T007)* → FR-001-16

## Phase 4 — Packaging and tests

- [x] **T012** `pyproject.toml`: runtime deps (including `pydantic-settings`) and dev deps (`pytest`, `pytest-asyncio`, `httpx`). → R-8
- [x] **T013 [P]** `tests/test_models.py`: lifecycle, name, negative prompt, generate-before-load error, and dtype selection, all with a mocked pipeline. → NFR-001-05
- [ ] **T014 [P]** `tests/test_tools.py` *(file exists but is empty)*:
  - tool registration (all five tools, both resources, both prompts)
  - `list_aspect_ratios` keys equal `ASPECT_RATIOS`
  - the `generate_image` `Literal` equals `ASPECT_RATIOS` keys
  - `get_dimensions("16:9") == (1664, 928)`; invalid input raises
  - `ImageStore` save/get/list ordering and the not-found error, using `tmp_path`
  - `get_model()` raises before load

  → FR-001-02, 08, 10–12, 15, 17
- [x] **T015** `README.md` and `CHANGELOG.md`.

## Phase 5 — Closing spec gaps (open)

- [ ] **T016** Clamp `cfg_scale` to [1.0, 10.0] in `generate_image`, and add a test. → FR-001-04
- [ ] **T017** Pass `settings.mcp_path` to the `FastMCP` constructor (confirm the keyword, R-3). → FR-001-13
- [ ] **T018** Serialize generation with a module-level `asyncio.Lock` around the `to_thread` call. → NFR-001-03
- [ ] **T019** Correct the `ImageStore` class docstring (no rebuild from disk). → FR-001-10
- [ ] **T020** GPU acceptance run of AS-1, AS-2, AS-5 and AS-6, with the results recorded in the CHANGELOG. → ⏳ items in the traceability table
