# Feature 009 — Implementation Plan

**Inputs:** [spec.md](spec.md), [research.md](research.md)
**Branch:** `feature/turbo`, from `feature/quantization`. Rebase onto `master` after 0.3.0 and 008 merge.

## Constitution check

| Principle | Compliance |
|-----------|------------|
| II Model-agnostic | Step rules move from server constants into the backend. Turbo is selected through the existing `IMAGE_MODEL` backend mechanism. |
| IV Verify the SDK | `sample_sigmas` support was confirmed in diffusers v0.41.0 (R-1) |
| VI Shared host | Same int8 footprint as 008. The GPU run is under the watchdog, alongside vLLM. |
| VII GPU-free tests | All selection and step logic is unit-tested |

## Changes

| File | Change |
|------|--------|
| `config/settings.py` | `image_model: Literal["qwen_image_2_1", "qwen_image_2_1_turbo"]` |
| `models/base.py` | Add the properties `min_inference_steps` and `max_inference_steps` (defaults 20 and 100) and `model_id` (default `""`) |
| `models/qwen_image_2_1.py` | `__init__(…, variant="base")` with a `VARIANTS` table: `base` → (`Qwen/Qwen-Image-2.1`, "Qwen-Image-2.1", default 40, range 20–100); `turbo` → (`Qwen/Qwen-Image-2.1-Turbo`, "Qwen-Image-2.1-Turbo", 8, 8–8). `MODEL_ID` becomes an instance attribute, and every `from_pretrained` uses `self._model_id`. Turbo passes no `sigmas`. Whether to omit `num_inference_steps` on Turbo or pass 8 (the pipeline ignores it) is decided in T001, after reading the v0.41.0 `__call__`. |
| `models/__init__.py` | Map `qwen_image_2_1_turbo` to `QwenImage21(variant="turbo", …)` |
| `server.py` | Replace `MIN_STEPS` / `MAX_STEPS` with `model.min_inference_steps` / `max_inference_steps` in `_resolve_params`. Warn when the requested or overridden steps differ from Turbo's fixed 8. `get_model_info` adds `model_id`, `min_inference_steps` and `max_inference_steps`. Update the docstrings (FR-009-05). |
| `docker/docker-compose.yml` | Check that `IMAGE_MODEL` is passed through, and make it `${IMAGE_MODEL:-qwen_image_2_1}` if it isn't |
| Docs and tests | FR-009-06, FR-009-07, NFR-009-02 |

## GPU runbook (orchestrator, after review approval; vLLM stays up)

1. Pre-flight: `free -m` (expect ~37 GB available after stopping the int8 base server), vLLM healthy and idle.
2. Stop the running base int8 server (`docker compose down`). It's needed for the comparison anyway; restart it afterwards (step 6).
3. Start the watchdog, then `IMAGE_MODEL=qwen_image_2_1_turbo` with the default quantization (`auto`, so int8), using the override file. The first start downloads ~32 GB, so the load takes longer. Record the per-step memory logs and the minimum available memory.
4. Run G1/G2/G3 and E1 with the 008 client (`gpu008.py`), and record the results.
5. Make thumbnails and compare them side by side with the base int8 results in `008/ab/`.
6. `docker compose down`, then start the server again with **the operator's chosen default** (base unless the operator picks Turbo), so the server is left running as the operator prefers.
7. Record the results in research R-4, `ab/`, the traceability table and the CHANGELOG. Present to the operator for AS-7.

## Risks

| Risk | Mitigation |
|------|------------|
| Turbo quality or instruction-following is weaker | The A/B test decides, and the base stays the default unless the operator chooses otherwise |
| Edits with Turbo at 1024 behave differently from the base | E1 is in the A/B test |
| Downloading 32 GB alongside vLLM uses disk I/O and page cache | Page cache can be reclaimed, and the watchdog guards memory |
| Removing the server-wide step constants changes base behavior | AS-3 locks in the base behavior |
