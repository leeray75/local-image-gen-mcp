# Qwen-Image-2.1-Turbo (Feature 009) — Summary Report

**Date:** 2026-10-18
**Branch:** `feature/turbo` (stacked on `feature/quantization`, which is stacked on `release/0.3.0`)
**Spec:** [`docs/specs/009-qwen-image-2-1-turbo/`](../../docs/specs/009-qwen-image-2-1-turbo/spec.md)
**Status:** Implemented, reviewed, and GPU-verified next to vLLM. **Turbo is the default.**

---

## Outcome

| | Base int8 (008) | **Turbo int8 (new default)** |
|---|---|---|
| 1K generate | 62 s | **~14 s** |
| 1024 edit | 72 s | **~17 s** |
| Weights | 18.5 GB | 18.5 GB |
| Min host available while loading / generating (next to vLLM) | 10.2 / 14.6 GB (vLLM at 0.6) | **13.8 / 17.7 GB** (vLLM at 0.54) |
| Quality (lettering, transparency, edit) | ✅ | ✅ equal on all four checks |

That is about **4.3–4.6× faster** with the same memory and no visible quality loss in these cases. The base model is still selectable with `IMAGE_MODEL=qwen_image_2_1`.

## What was built

- **`IMAGE_MODEL=qwen_image_2_1_turbo`.** The same backend class gains a `variant` setting that selects the checkpoint, model name and step rules. The checkpoint ID reaches every loader.
- **Step limits come from the backend.** The base model uses 40 steps by default within 20–100. Turbo is fixed at 8, taken from the `sample_sigmas` stored in its checkpoint. A request for any other step count is clamped to 8 and the client gets a warning. The server-wide `MIN_STEPS`/`MAX_STEPS` constants are removed.
- **Steps on Turbo (T001), checked against the diffusers v0.41.0 source:** the backend still passes `num_inference_steps=8`, and never passes `sigmas` or changes the scheduler. The checkpoint's schedule always wins. If a checkpoint ever lacked a stored schedule, the pipeline would fall back to 8 steps rather than 40.
- **`get_model_info`** now reports `model_id`, `model_name` and the step range. Docstrings no longer claim 40 steps for every model.
- **Compose** used to hard-code `IMAGE_MODEL`. It now passes it through as `${IMAGE_MODEL:-qwen_image_2_1_turbo}`. The Dockerfile, `.env.example` and docs follow the new default.
- **Tests:** 191 passing, up from 174.

## Process

- **Spec approval.** The planning decisions (D-1 to D-4) were made by the orchestrator, because the operator said "do whatever you think is best". The approval gate stayed in place, and the operator approved the spec.
- **Implementer (Sonnet) and reviewer (Sonnet): 3 rounds.**
  - R1: approved, with 2 minor test gaps.
  - R2: the fixes were confirmed.
  - R3: the default switch was approved, with 1 minor doc cell, which the orchestrator fixed.
- **GPU run:** the base server was stopped while the A/B ran, then Turbo was started. vLLM was never stopped. The watchdog guarded memory the whole time and never fired. The first start downloaded ~32 GB in about 7 minutes, including the load.
- **Default decision.** The operator chose Turbo (D-5) and asked that it stay running. It was already running, so no restart was needed.

## Related host change: vLLM KV right-sizing

This was done in the same session, with the operator's approval (option 2). Two narrow allow rules were added to `~/.claude/settings.json`: one for editing that single compose file, and one for that exact recreate command.

- In `ai-litellm-proxy/docker-compose.qwen3.8.yml`, `--gpu-memory-utilization` went from 0.6 to **0.54**, with an explanatory comment.
- vLLM shrank from 73.2 GB to 67.2 GB, and its maximum concurrency at a 262,144-token context is **4.48×**, still at least 4.
- **The change is not committed.** That repo is on an unrelated branch, so committing it is left to the operator.

## Open items

| Item | Status |
|------|--------|
| Release `0.3.0` | Waiting for the operator to merge the `release/0.3.0` PR. Then tag `v0.3.0` and rebase the stacked branches (`feature/quantization`, then `feature/turbo`) onto `master`. |
| Commit to `ai-litellm-proxy` | Left to the operator |
| 007 AS-6 to AS-8 (BF16 1K vs 2K) | Still deferred. Lower priority now that int8 Turbo is the default. |
| Disk | The HF cache holds 62 GB of blobs (base and Turbo). If the base model isn't needed, its ~31 GB could be removed later; it would download again on demand. |

## Host state

- **Turbo int8 image server: running and healthy**, next to vLLM. Its configuration equals the new default.
- vLLM `qwen3-8-27b-nvfp4-engine`: healthy at 0.54.
- The `docker-local-image-gen-mcp` image was rebuilt with the final code. The running container was not restarted.
