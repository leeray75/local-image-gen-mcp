# Feature 004 — Tasks

- [x] **T001** In `QwenImage2512.load()` (CUDA path): pass `device_map="balanced"`, and add `max_memory` when `IMAGE_GPU_MEMORY_LIMIT` is set. Add the mode logs and the post-load memory log. → FR-004-01–04, 07
- [x] **T002** In `docker/docker-compose.yml`: set `IMAGE_GPU_MEMORY_LIMIT: "40"` and remove `limits.memory: 40g`. → FR-004-06
- [ ] **T003** Move the setting into `Settings` and pass it through `load_model()`. Add it to `.env.example`, the README environment table, and the Dockerfile `ENV` defaults. → FR-004-05
- [ ] **T004** Unit tests in `tests/test_models.py`: check the `from_pretrained` kwargs for the capped, uncapped, and CPU paths. *(Easier after T003.)* → AS-4, AS-5
- [ ] **T005** Measure on the DGX Spark: time per 1:1 image at 50 steps for caps 30/40/50/unset, with vLLM running. Update the guidance table in spec.md. → AS-1
- [x] **T006** Update CHANGELOG (`Unreleased`: `device_map` is now `"balanced"`, the memory cap is added, and the container memory limit is removed).
- [ ] **T007** Commit.
