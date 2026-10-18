# Feature 003 — Tasks

- [x] **T001** Add `get_image_base64` to `server.py` after `get_model_info`, with the docstring workflow and decode example. → FR-003-03, 04, 05
- [x] **T002** Add `get_latest_image_id` to `server.py` immediately after it. → FR-003-01, 02, 05
- [x] **T003** Check that the diff touches nothing else. → FR-003-06
- [x] **T004** *(done in 006: `test_latest_image_id_returns_newest`, `test_latest_image_id_empty_store`, `test_image_base64_is_byte_identical`, `test_image_base64_unknown_id`, plus registration in 005)* Tests in `tests/test_tools.py`:
  - registration (AS-5)
  - with a `tmp_path` `ImageStore` monkeypatched into `server.store`: latest ordering (AS-1), empty error (AS-2), base64 round-trip equals the file bytes (AS-3), not-found (AS-4)
- [x] **T005** *(README tools table already lists both; 001 contract updated in 006)* Document both tools in the README tools table and the 001 contract file.
- [x] **T006** Add a CHANGELOG `Unreleased → Added` entry.
- [x] **T007** Rebuild the container and run a manual end-to-end check from an MCP client: generate, get the latest ID, get the base64, decode, write, and open the file. *(Done 2026-10-18 in 006: a valid 4.8 MB PNG.)*
- [x] **T008** *(committed in `9567d08`)* Commit `server.py` and `ai-workspace/plans/add-tools.md`.
