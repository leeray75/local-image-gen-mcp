# Feature 008 — Implementation Plan

**Inputs:** [spec.md](spec.md), [research.md](research.md)
**Branch:** `feature/quantization`, branched from `release/0.3.0`. Rebase onto `master` once 0.3.0 has merged.

## Constitution check

| Principle | Compliance |
|-----------|------------|
| I Local-first | Only the official Qwen weights are used, quantized in-process. Nothing third-party is downloaded. |
| II Model-agnostic | The mode → config mapping lives in the Qwen backend, and the server only reports it. Settings are generic ("transformer" and "text encoder" are the pipeline's component names). |
| III Env-only config | Two validated `Settings` fields |
| IV Verify the SDK | The exact constructor calls are verified against the installed versions and recorded in research.md |
| VI Shared host | GPU runs happen alongside vLLM under a memory watchdog. Memory pressure means stop and report. |
| VII GPU-free tests | Config building and error handling are unit-tested, with library classes patched |

## Changes

| File | Change |
|------|--------|
| `config/settings.py` | `quantize_transformer` and `quantize_text_encoder`: `Literal["none","int8","fp8","nvfp4"] = "none"`, with `""` → `"none"` (reusing the existing empty-string pattern) |
| `models/qwen_image_2_1.py` | `__init__(…, quantize_transformer="none", quantize_text_encoder="none")`. A new pure helper, `_build_quantization_config() -> PipelineQuantizationConfig | None`, imports the libraries lazily inside the helper, so the module still imports where ModelOpt isn't installed. Pass `quantization_config=` in the CUDA `from_pretrained` call. Wrap that load call so a backend error is re-raised as `RuntimeError(f"Failed to load {component} with {mode} quantization: {err}")` (FR-008-04); the error can't always be pinned to one component, so if it's ambiguous, name both modes. On CPU, any value other than `none` raises before loading. Add `modules_to_not_convert` lists per component (verified names, FR-008-05). Log the post-load modes and memory (FR-008-06), and expose `quantization_info` → dict for `get_model_info`. |
| `models/base.py` | Optional `quantization_info` property, returning `{}` by default |
| `models/__init__.py` | Pass the two settings through |
| `server.py` | `get_model_info` merges `model.quantization_info` (FR-008-07) |
| `docker/docker-compose.yml` | `QUANTIZE_TRANSFORMER: ${QUANTIZE_TRANSFORMER:-}` and `QUANTIZE_TEXT_ENCODER: ${QUANTIZE_TEXT_ENCODER:-}` |
| Docs | FR-008-08 and FR-008-09; the results table gets a placeholder until T-GPU |
| `tests/` | NFR-008-02 |

### Verifying module names (FR-008-05), without a GPU

Instantiate each component **on the meta device** from its config. That needs no weights and no memory beyond a few MB of config downloads, and the HF cache already has the configs. Then list the module names, and pick the vision tower, the embedding and LM-head names (text encoder), and the DiT input, output, timestep, modulation and norm names (image model). Record them in research.md as R-5.

## GPU runbook (orchestrator, after review approval; vLLM stays up)

**Watchdog** (start before every server start; a background process):

```bash
while sleep 1; do a=$(free -m | awk '/Mem:/{print $7}')
  echo "$(date +%T) $a" >> watchdog.log
  [ "$a" -lt 8192 ] && { docker kill local-image-gen-mcp; echo "WATCHDOG FIRED at ${a} MB" >> watchdog.log; break; }
done
```

For each configuration in the AS-7 order:
1. `docker compose up -d` with the override and `QUANTIZE_TRANSFORMER` / `QUANTIZE_TEXT_ENCODER` set. Record the load time, the log lines with the modes and memory, and the minimum available memory during the load.
2. Run 1K G1/G2/G3 (seed 42, 40 steps) and E1 at 1024 (the 006 source as base64), recording time and peak RAM.
3. `docker compose down`.

If the watchdog fires: stop, keep the logs, record the configuration as "does not fit", skip the heavier configurations, and **report to the operator** (standing rule).

Afterwards: commit the thumbnails, fill in research R-6 with the results, present the evidence for the operator's AS-8 decision, then apply the default and the docs.

## Risks

| Risk | Mitigation |
|------|------------|
| A mode fails on GB10 (sm_121 kernels, prototype API) | FR-008-04 fails startup loudly. Record it and continue with the other configurations. |
| Load-time peak higher than expected | The watchdog, plus running the lowest-memory configuration first |
| Weight-only quantization slower than BF16 | Record the timings. Memory is the goal. |
| A quality regression in the text encoder | Separate settings (D-2). The operator decides per component. |
