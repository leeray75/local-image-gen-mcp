# Feature 003 — Client-side Image Retrieval Tools

| | |
|---|---|
| **Status** | Implemented, uncommitted (in `server.py` working tree) |
| **Constitution** | VIII |
| **Depends on** | [001](../001-image-generation-server/spec.md) (ImageStore, `image://` resource) |
| **Next** | [plan.md](plan.md) → [tasks.md](tasks.md) |

## Problem

`generate_image` returns the image as MCP image content, which the client *displays*. An agent that needs to *save* the image to the user's project (for example `assets/hero.png`) has no easy way to do it:

- The server's `/outputs` is inside a container, so the client can't see it.
- Many clients expose only **tools** to the agent, not resources, so `image://{id}` and `images://list` can't be reached.
- The image ID only appears in a log message, which agents often don't see.
- Agents that do get base64 content tend to write the base64 *text* straight into a `.png` file, which produces a corrupt file.

## Goal

Add two tools that give an agent a reliable, tools-only workflow: generate, find out what was generated, fetch the bytes, then decode and write the file on the client.

## Non-goals

- Writing files on the client's filesystem from the server.
- Changing `generate_image`'s return type.
- Pagination or search across the image list.

## User stories

**US-1 (P1)** As an agent, right after `generate_image`, I can get the new image's ID and key metadata with one tool call, without reading resources.

**US-2 (P1)** As an agent, I can get any stored image as a base64 string, along with instructions that make clear I must decode it before writing it.

## Functional requirements

- **FR-003-01** The server MUST expose a `get_latest_image_id()` tool that returns, for the most recently generated image: `image_id`, `prompt` (first 100 characters), `aspect_ratio`, `width`, `height`, `seed`, `model_name`.
- **FR-003-02** When no image has been generated in the current process, `get_latest_image_id` MUST fail with an error telling the caller to call `generate_image` first.
- **FR-003-03** The server MUST expose a `get_image_base64(image_id: str)` tool that returns the stored PNG as a plain base64 string: standard alphabet, no `data:` URI prefix, no line breaks.
- **FR-003-04** `get_image_base64` MUST return the same not-found and missing-file errors as the `image://{id}` resource (FR-001-12).
- **FR-003-05** Both tool docstrings MUST describe the two-step save workflow. The `get_image_base64` docstring MUST state that the result has to be base64-decoded before writing, and include a working Python example (Constitution VIII).
- **FR-003-06** Existing tools, resources, prompts, imports and startup code MUST NOT change.

## Acceptance scenarios

- **AS-1** Given images A then B were generated, `get_latest_image_id()["image_id"]` is B.
- **AS-2** Given a fresh server, `get_latest_image_id()` raises an error containing "Call generate_image first".
- **AS-3** For a stored ID, `base64.b64decode(get_image_base64(id))` is byte-identical to the file `IMAGE_OUTPUT_DIR/<id>.png`, and it opens as a PNG.
- **AS-4** `get_image_base64("nope")` raises a not-found error.
- **AS-5** Tool registration lists `get_image_base64` and `get_latest_image_id` in addition to the five existing tools.

## Traceability

| Requirement | Implementation | Test | Status |
|-------------|----------------|------|--------|
| FR-003-01, 02 | `server.py` `get_latest_image_id` | — | ✅ (no test) |
| FR-003-03, 04 | `server.py` `get_image_base64` → `store.get_image_bytes` | — | ✅ (no test) |
| FR-003-05 | docstrings | review | ✅ |
| FR-003-06 | the diff only adds the two tools | review of `git diff server.py` | ✅ |
| AS-1–5 | — | `tests/test_tools.py` (empty) | ❌ |

## Notes

- Base64 makes the payload about 33% larger. A 2K PNG (2048² since 005) is about 3–10 MB, more as base64, which counts against the client's context window. Agents should call this tool only when they need to save the file.
- "Most recent" means newest by insertion in the in-process index. After a restart, earlier images can't be reached through these tools (FR-001-10).
