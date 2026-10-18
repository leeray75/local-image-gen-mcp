# Feature 003 — Implementation Plan

**Inputs:** [spec.md](spec.md), [001 data model](../001-image-generation-server/data-model.md)

## Approach

Add two synchronous tools to `server.py`. Place them **after** `get_model_info` and **before** the `# ── Resources` section. Both are thin wrappers over the existing `ImageStore` API. No new modules and no changes to the store.

| Tool | Store call | Notes |
|------|-----------|-------|
| `get_latest_image_id()` | `store.list_all()[0]` | `list_all()` is already newest first. An empty list raises `RuntimeError`. |
| `get_image_base64(image_id)` | `store.get_image_bytes(image_id)` | `base64.b64encode(...).decode("utf-8")`. Errors from the store propagate unchanged (FR-003-04). |

## Decisions

- **D-1 Encode the bytes on disk, don't re-encode the PIL image.** The result is then guaranteed to be identical to the stored file (AS-3).
- **D-2 Import `base64` inside the function.** This keeps the "don't touch existing imports" constraint (FR-003-06). It is a deliberate exception to normal style. Move the import to the top of the file the next time `server.py`'s imports are edited.
- **D-3 Return a plain string, not `Image` content.** The point of the tool is to give the agent text it can decode. Returning `Image` would just be `image://` again, and clients would render it instead of passing it to the agent.
- **D-4 A separate `get_latest_image_id` rather than changing `generate_image` to return the ID.** Changing the return type of `generate_image` would be a breaking contract change (see the 001 contracts compatibility rules).

## Contract additions

These are appended to [001 contracts](../001-image-generation-server/contracts/mcp-interface.md):

```
get_latest_image_id() -> { image_id, prompt[:100], aspect_ratio, width, height, seed, model_name }
  error: RuntimeError "No images have been generated in this session. Call generate_image first."

get_image_base64(image_id: string) -> string   # plain base64 PNG, no data: prefix
  error: KeyError (not found) | FileNotFoundError (file removed)
```

## Verification

```bash
python3 -c "from server import mcp; t=[x.name for x in mcp._tool_manager.list_tools()]; assert {'get_image_base64','get_latest_image_id'} <= set(t); print('OK')"
cd docker && docker compose build --no-cache && docker compose up -d
```
