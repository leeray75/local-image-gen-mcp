# Feature 007 — Implementation Plan

**Inputs:** [spec.md](spec.md), [research.md](research.md)
**Branch:** `feature/1k-default-resolution` (from `master` at v0.2.0)

## Constitution check

| Principle | Compliance |
|-----------|------------|
| II Model-agnostic | The size formula and the card table live on the backend (`sizes_for`). The server only picks which resolution to ask for. |
| III Env-only config | Both new settings are validated in `Settings`, with a startup-failing ceiling |
| VI Shared host | The default footprint drops from ~68 GB to an expected ≤ ~40 GB. vLLM is stopped only for the GPU window. |
| VII GPU-free tests | All logic is unit-tested with the mocked pipeline and fake model |
| VIII Self-describing tools | The docstrings describe the configured sizes, not 2K |

## Changes

| File | Change |
|------|--------|
| `config/settings.py` | `generate_output_resolution: int = 1024` and `max_output_resolution: int = 1024`. Reuse the existing `""` → default and multiple-of-32-in-[512, 2048] validators: extend their field lists, without duplicating code. Add `@model_validator(mode="after")` `_check_ceiling` per FR-007-02. |
| `models/base.py` | `sizes_for(resolution) -> dict[str, tuple[int, int]]` (abstract). Optional properties `generate_output_resolution` and `max_output_resolution` aren't needed, because the server reads Settings. |
| `models/qwen_image_2_1.py` | Rename `ASPECT_RATIOS` to `CARD_SIZES_2048`, the official table. Add `sizes_for(resolution)`, which returns a copy of the card table at 2048 and otherwise applies the R-1 formula to the ratio keys. Add `__init__(…, generate_output_resolution=1024)`. `aspect_ratios` returns `sizes_for(self._generate_output_resolution)`. |
| `models/__init__.py` | Pass `generate_output_resolution=settings.generate_output_resolution` |
| `server.py` | `generate_image` stays as it is (it uses `model.aspect_ratios`). In `edit_image`, when `aspect_ratio` is set, take the size from `model.sizes_for(model.edit_output_resolution)` (F-5). Add `generate_output_resolution` and `max_output_resolution` (from settings) to `get_model_info`. Reword the docstrings and instructions (FR-007-07). |
| `docker/docker-compose.yml` | `GENERATE_OUTPUT_RESOLUTION: ${GENERATE_OUTPUT_RESOLUTION:-}` and `MAX_OUTPUT_RESOLUTION: ${MAX_OUTPUT_RESOLUTION:-}` |
| `.env.example`, `README.md`, `AGENTS.md`, `CHANGELOG.md` | FR-007-08 and FR-007-09 |
| `tests/` | NFR-007-01. Update the existing tests that assert the 2K table as the default, for example a 2048×2048 t2i or the `test_aspect_ratios_table` checks. |
| `docs/specs/**` | FR-007-10, plus the 007 traceability table |

Note: the `Ratio` `Literal` in `server.py` stays the same. The keys don't change, and the existing test that compares the Literal with the backend keys still holds.

## GPU window runbook (deferred by D-5; run in a later window the operator chooses)

1. `docker stop qwen3-8-27b-nvfp4-engine`, after checking there are no in-flight requests.
2. `docker compose build`, then `up` with the `ai-bridge` override and default settings (1K). Wait until it's healthy.
3. Run G1, G2 and G3 at 1024 with seed 42 and 40 steps, recording time and sampled peak host RAM.
4. `docker compose down`, then `up` with `GENERATE_OUTPUT_RESOLUTION=2048 MAX_OUTPUT_RESOLUTION=2048`. This also confirms that the ceiling allows 2K only when it's raised.
5. Run G1, G2 and G3 at 2048.
6. Check that `GENERATE_OUTPUT_RESOLUTION=2048` **without** raising the ceiling makes the container exit at startup with the FR-007-02 message (AS-3 on real hardware).
7. Stop the image server, `docker start qwen3-8-27b-nvfp4-engine`, and wait until it's healthy.
8. Commit the thumbnails to `docs/specs/007-1k-default-resolution/ab/` and record the results in research.md as R-5.
9. Present the evidence to the operator for AS-7, then finish the traceability, CHANGELOG and report.

## Risks

| Risk | Mitigation |
|------|------------|
| 1K text rendering is noticeably worse than 2K | AS-7 puts the call to the operator, with evidence. 2K stays one config change away. |
| Existing users relied on 2K output | CHANGELOG marks it as breaking. Opting in is documented. |
| The ceiling makes a misconfigured server fail to start, which could surprise an operator | The error message tells them exactly what to change. README documents it. |
