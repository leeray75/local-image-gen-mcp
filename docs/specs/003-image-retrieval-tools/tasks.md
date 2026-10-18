# Feature 003 — Tasks

- [x] **T001** Add `get_image_base64` to `server.py` after `get_model_info`, with the docstring workflow and decode example. → FR-003-03, 04, 05
- [x] **T002** Add `get_latest_image_id` to `server.py` immediately after it. → FR-003-01, 02, 05
- [x] **T003** Check that the diff touches nothing else. → FR-003-06
- [ ] **T004** Tests in `tests/test_tools.py`:
  - registration (AS-5)
  - with a `tmp_path` `ImageStore` monkeypatched into `server.store`: latest ordering (AS-1), empty error (AS-2), base64 round-trip equals the file bytes (AS-3), not-found (AS-4)
- [ ] **T005** Document both tools in the README tools table and the 001 contract file.
- [x] **T006** Add a CHANGELOG `Unreleased → Added` entry.
- [ ] **T007** Rebuild the container and run a manual end-to-end check from an MCP client: generate, get the latest ID, get the base64, decode, write, and open the file.
- [ ] **T008** Commit `server.py` and `ai-workspace/plans/add-tools.md`.
