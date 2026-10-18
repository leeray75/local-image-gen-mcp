"""Tests for server tools, prompts, resources and settings.

The model is a fake BaseImageModel, so no GPU or weights are needed.
"""

import asyncio
import base64
import io
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image as PILImage

import server
from config.settings import Settings
from models.base import BaseImageModel, GenerationRequest, GenerationResult
from models.qwen_image_2_1 import QwenImage21
from storage.image_store import ImageStore


class FakeModel(BaseImageModel):
    """Records requests; returns a solid image of the requested (or 40x30) size."""

    RATIOS = {"1:1": (64, 64), "16:9": (96, 54), "9:16": (54, 96), "4:3": (64, 48),
              "3:4": (48, 64), "3:2": (60, 40), "2:3": (40, 60)}

    @property
    def edit_output_resolution(self): return 1024 if self._editing else None

    def __init__(self, editing=True, transparency=True, delay=0.0):
        self.requests: list[GenerationRequest] = []
        self._editing, self._transparency, self._delay = editing, transparency, delay
        self.active = 0
        self.max_active = 0

    def load(self): ...
    def unload(self): ...
    def is_loaded(self): return True

    @property
    def model_name(self): return "Fake"
    @property
    def aspect_ratios(self): return dict(self.RATIOS)

    def sizes_for(self, resolution):
        # distinct from RATIOS so tests can tell which table a tool used
        return {k: (w * resolution // 64, h * resolution // 64) for k, (w, h) in self.RATIOS.items()}
    @property
    def default_inference_steps(self): return 40
    @property
    def default_cfg_scale(self): return 1.0
    @property
    def default_negative_prompt(self): return None
    @property
    def max_reference_images(self): return 10 if self._editing else 0
    @property
    def supports_transparency(self): return self._transparency

    def generate(self, request):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                time.sleep(self._delay)
            self.requests.append(request)
            w = request.width or 40
            h = request.height or 30
            mode = "RGBA" if request.transparent else "RGB"
            return GenerationResult(PILImage.new(mode, (w, h)), request.seed, 0.01, w, h)
        finally:
            self.active -= 1


@pytest.fixture
def fake(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "store", ImageStore(tmp_path))
    # fresh lock per test (the loop differs between tests)
    monkeypatch.setattr(server, "_generation_lock", asyncio.Lock())
    return model


@pytest.fixture
def ctx():
    c = MagicMock()
    c.info, c.warning, c.report_progress = AsyncMock(), AsyncMock(), AsyncMock()
    return c


def b64png(size=(8, 6)):
    buf = io.BytesIO()
    PILImage.new("RGB", size, "blue").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def png_mode(img) -> str:
    return PILImage.open(io.BytesIO(img.data)).mode


# ── Registration (AS-11) and ratio sync (AS-10) ───────────────────────────────

def test_tool_registration():
    names = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert {"generate_image", "edit_image", "list_aspect_ratios", "get_model_info",
            "get_image_base64", "get_latest_image_id"} <= names


def test_prompt_registration():
    names = {p.name for p in server.mcp._prompt_manager.list_prompts()}
    assert {"photorealistic", "transparent_asset", "extract_subject"} <= names
    assert "negative_defaults" not in names


def test_tool_functions_are_directly_callable():
    assert asyncio.iscoroutinefunction(server.generate_image)
    assert asyncio.iscoroutinefunction(server.edit_image)


def test_ratio_literal_matches_backend_table_keys():
    from typing import get_args
    assert set(get_args(server.Ratio)) == set(QwenImage21.CARD_SIZES_2048)
    schema = server.mcp._tool_manager.get_tool("generate_image").parameters
    assert set(schema["properties"]["aspect_ratio"]["enum"]) == set(QwenImage21.CARD_SIZES_2048)


def test_list_aspect_ratios_equals_backend(fake):
    out = server.list_aspect_ratios()
    assert out == {r: {"width": w, "height": h} for r, (w, h) in fake.RATIOS.items()}


def test_tool_docstrings_cover_contract():
    gen = server.generate_image.__doc__
    edit = server.edit_image.__doc__
    for doc in (gen, edit):
        assert "cfg_scale > 1" in doc and "transparent" in doc
        assert "base64" in doc and "image ID" in doc
    assert "GENERATE_OUTPUT_RESOLUTION" in gen and "default 1024" in gen
    assert "2K" not in gen and "2K" not in edit and "2K" not in server.list_aspect_ratios.__doc__
    assert "default 1024" in edit
    assert "Picture" in edit and "1-10" in edit


# ── _resolve_params (AS-3, AS-4) ──────────────────────────────────────────────

def test_resolve_defaults():
    steps, cfg, neg, warn = server._resolve_params(FakeModel(), None, None, None)
    assert (steps, cfg, neg, warn) == (40, 1.0, None, [])


def test_resolve_clamps():
    steps, cfg, _, _ = server._resolve_params(FakeModel(), 5, 50, None)
    assert (steps, cfg) == (20, 10.0)
    steps, cfg, _, _ = server._resolve_params(FakeModel(), 500, 0.1, None)
    assert (steps, cfg) == (100, 1.0)


def test_resolve_env_overrides(monkeypatch):
    monkeypatch.setattr(server.settings, "max_inference_steps", 30)
    monkeypatch.setattr(server.settings, "default_cfg_scale", 3.0)
    assert server._resolve_params(FakeModel(), None, None, None)[:2] == (30, 3.0)
    # explicit args beat env
    assert server._resolve_params(FakeModel(), 60, 2.0, None)[:2] == (60, 2.0)
    # env values are clamped too
    monkeypatch.setattr(server.settings, "max_inference_steps", 1000)
    assert server._resolve_params(FakeModel(), None, None, None)[0] == 100


def test_resolve_negative_warning_and_empty_string():
    _, _, neg, warn = server._resolve_params(FakeModel(), None, None, "blurry")
    assert neg == "blurry" and len(warn) == 1 and "cfg_scale > 1" in warn[0]
    _, _, neg, warn = server._resolve_params(FakeModel(), None, 4.0, "blurry")
    assert neg == "blurry" and warn == []
    _, _, neg, warn = server._resolve_params(FakeModel(), None, None, "")
    assert neg is None and warn == []


# ── generate_image (AS-1, AS-2, AS-3, AS-4) ───────────────────────────────────

async def test_generate_defaults_and_metadata(fake, ctx):
    img = await server.generate_image("a cat", ctx=ctx, seed=7)
    req = fake.requests[0]
    assert (req.width, req.height) == (64, 64)
    assert (req.num_inference_steps, req.cfg_scale, req.negative_prompt) == (40, 1.0, None)
    assert req.reference_images == [] and req.transparent is False
    assert png_mode(img) == "RGB"
    s = server.store.list_all()[0]
    assert (s.mode, s.transparent, s.reference_image_ids, s.reference_count) == \
        ("generate", False, [], 0)
    assert (s.aspect_ratio, s.width, s.height, s.seed, s.negative_prompt) == \
        ("1:1", 64, 64, 7, "")
    assert s.model_name == "Fake"


async def test_generate_transparent_16x9(fake, ctx):
    img = await server.generate_image("a sticker", aspect_ratio="16:9",
                                      transparent=True, ctx=ctx)
    req = fake.requests[0]
    assert (req.width, req.height) == (96, 54) and req.transparent is True
    assert png_mode(img) == "RGBA"
    assert server.store.list_all()[0].transparent is True


async def test_generate_clamps(fake, ctx):
    await server.generate_image("x", num_inference_steps=5, cfg_scale=50, ctx=ctx)
    assert (fake.requests[0].num_inference_steps, fake.requests[0].cfg_scale) == (20, 10.0)


async def test_generate_negative_prompt_warns_but_succeeds(fake, ctx):
    await server.generate_image("x", negative_prompt="blurry", ctx=ctx)
    ctx.warning.assert_awaited_once()
    assert "ignored" in ctx.warning.await_args.args[0]
    assert fake.requests[0].negative_prompt == "blurry"
    assert server.store.list_all()[0].negative_prompt == "blurry"


async def test_generate_rejects_transparent_when_unsupported(fake, ctx):
    fake._transparency = False
    with pytest.raises(ValueError, match="transparent"):
        await server.generate_image("x", transparent=True, ctx=ctx)
    assert fake.requests == []


async def test_generate_rejects_ratio_missing_from_backend(fake, ctx):
    del fake.RATIOS["3:2"]
    fake.RATIOS = dict(fake.RATIOS)
    with pytest.raises(ValueError, match="aspect_ratio"):
        await server.generate_image("x", aspect_ratio="3:2", ctx=ctx)


# ── edit_image (AS-5, AS-6, AS-7) ─────────────────────────────────────────────

async def test_edit_mixed_references(fake, ctx):
    await server.generate_image("base", ctx=ctx)
    a = server.store.list_all()[0].image_id
    img = await server.edit_image("combine", [a, b64png((8, 6))], ctx=ctx)
    req = fake.requests[-1]
    assert [r.size for r in req.reference_images] == [(64, 64), (8, 6)]
    assert req.width is None and req.height is None
    s = server.store.list_all()[0]
    assert (s.mode, s.aspect_ratio) == ("edit", "input")
    assert s.reference_image_ids == [a] and s.reference_count == 2
    assert (s.width, s.height) == (40, 30)
    assert png_mode(img) == "RGB"
    assert server.store.count() == 2          # uploads are not stored


async def test_edit_with_aspect_ratio_override(fake, ctx):
    await server.edit_image("x", [b64png()], aspect_ratio="1:1", ctx=ctx)
    req = fake.requests[0]
    assert (req.width, req.height) == (1024, 1024)  # edit table (F-5), not generate table
    assert server.store.list_all()[0].aspect_ratio == "1:1"


async def test_edit_transparent(fake, ctx):
    img = await server.edit_image("x", [b64png()], transparent=True, ctx=ctx)
    assert png_mode(img) == "RGBA" and fake.requests[0].transparent


async def test_edit_data_uri(fake, ctx):
    await server.edit_image("x", ["data:image/png;base64," + b64png()], ctx=ctx)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("images, match", [
    ([], "1–10"),
    (["AAAA"] * 11, "1–10"),
    (["00000000-0000-4000-8000-000000000000"], "reference 1: .*not found"),
    (["%%%not-base64"], "reference 1: invalid base64"),
    ([base64.b64encode(b"GIF89a" + b"\0" * 20).decode()], "reference 1"),
])
async def test_edit_bad_references_never_call_model(fake, ctx, images, match):
    with pytest.raises(ValueError, match=match):
        await server.edit_image("x", images, ctx=ctx)
    assert fake.requests == []
    ctx.report_progress.assert_not_awaited()


async def test_edit_gif_and_oversize_rejected(fake, ctx, monkeypatch):
    buf = io.BytesIO()
    PILImage.new("P", (4, 4)).save(buf, format="GIF")
    with pytest.raises(ValueError, match="unsupported format GIF"):
        await server.edit_image("x", [base64.b64encode(buf.getvalue()).decode()], ctx=ctx)
    monkeypatch.setattr(server.settings, "max_reference_image_mb", 0)
    with pytest.raises(ValueError, match="exceeds MAX_REFERENCE_IMAGE_MB"):
        await server.edit_image("x", [b64png()], ctx=ctx)
    assert fake.requests == []


async def test_edit_rejected_when_backend_cannot_edit(fake, ctx):
    fake._editing = False
    with pytest.raises(ValueError, match="does not support image editing"):
        await server.edit_image("x", [b64png()], ctx=ctx)


# ── Serialization (AS-9) ──────────────────────────────────────────────────────

async def test_generation_calls_are_serialized(fake, ctx):
    fake._delay = 0.05
    await asyncio.gather(
        server.generate_image("a", ctx=ctx),
        server.generate_image("b", ctx=ctx),
        server.edit_image("c", [b64png()], ctx=ctx),
    )
    assert len(fake.requests) == 3
    assert fake.max_active == 1


# ── Discovery, retrieval ──────────────────────────────────────────────────────

def test_get_model_info(fake):
    info = server.get_model_info()
    assert info["max_reference_images"] == 10
    assert info["supports_editing"] is True and info["supports_transparency"] is True
    assert (info["default_inference_steps"], info["default_cfg_scale"]) == (40, 1.0)
    assert info["has_default_negative_prompt"] is False
    assert info["supported_aspect_ratios"] == sorted(fake.RATIOS)
    assert info["edit_output_resolution"] == 1024  # 006 AS-3
    assert info["generate_output_resolution"] == 1024  # 007 AS-5
    assert info["max_output_resolution"] == 1024


def test_get_model_info_resolved_with_env(fake, monkeypatch):
    monkeypatch.setattr(server.settings, "max_inference_steps", 25)
    monkeypatch.setattr(server.settings, "default_cfg_scale", 4.0)
    info = server.get_model_info()
    assert (info["default_inference_steps"], info["default_cfg_scale"]) == (25, 4.0)


async def test_latest_id_and_list_include_mode_and_transparent(fake, ctx):
    with pytest.raises(RuntimeError):
        server.get_latest_image_id()
    await server.generate_image("p", ctx=ctx)
    await server.edit_image("q", [b64png()], transparent=True, ctx=ctx)
    latest = server.get_latest_image_id()
    assert (latest["mode"], latest["transparent"]) == ("edit", True)
    listing = json.loads(server.list_images_resource())
    assert [(e["mode"], e["transparent"]) for e in listing] == \
        [("edit", True), ("generate", False)]
    b64 = await server.get_image_base64(latest["image_id"])
    assert base64.b64decode(b64)[:4] == b"\x89PNG"


async def test_save_and_encode_run_off_the_event_loop(fake, ctx, monkeypatch):
    called = []
    real = asyncio.to_thread

    async def spy(fn, *a, **kw):
        called.append(getattr(fn, "__name__", str(fn)))
        return await real(fn, *a, **kw)

    monkeypatch.setattr(server.asyncio, "to_thread", spy)
    await server.generate_image("p", ctx=ctx)
    assert "_save_and_encode" in called and "generate" in called
    called.clear()
    await server.get_image_base64(server.store.list_all()[0].image_id)
    assert called  # file read + base64 went through to_thread


# ── Prompts ───────────────────────────────────────────────────────────────────

def test_prompts_text():
    assert server.transparent_asset("a mug") == (
        "a mug, a single standalone object, centered, clean edges, "
        "no background elements.")
    assert server.transparent_asset("a mug", "flat vector").endswith(
        "no background elements, flat vector.")
    assert server.extract_subject().startswith("Extract the main subject from Picture 1")
    assert "2512" not in server.photorealistic.__doc__
    assert "Qwen" not in server.photorealistic("x")


# ── Settings (AS-12 and friends) ──────────────────────────────────────────────

def test_settings_reject_old_model(monkeypatch):
    monkeypatch.setenv("IMAGE_MODEL", "qwen_image_2512")
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_settings_defaults(monkeypatch):
    for v in ("IMAGE_MODEL", "MAX_INFERENCE_STEPS", "DEFAULT_CFG_SCALE",
              "IMAGE_GPU_MEMORY_LIMIT", "MAX_REFERENCE_IMAGE_MB"):
        monkeypatch.delenv(v, raising=False)
    s = Settings(_env_file=None)
    assert s.image_model == "qwen_image_2_1"
    assert s.max_inference_steps is None and s.default_cfg_scale is None
    assert s.image_gpu_memory_limit is None and s.max_reference_image_mb == 20


def test_settings_empty_strings_become_none(monkeypatch):
    monkeypatch.setenv("IMAGE_GPU_MEMORY_LIMIT", "")
    monkeypatch.setenv("MAX_INFERENCE_STEPS", "")
    monkeypatch.setenv("DEFAULT_CFG_SCALE", "")
    s = Settings(_env_file=None)
    assert s.image_gpu_memory_limit is None and s.max_inference_steps is None
    assert s.default_cfg_scale is None


def test_settings_memory_limit_reaches_backend(monkeypatch):  # AS-13
    from unittest.mock import patch
    monkeypatch.setenv("IMAGE_GPU_MEMORY_LIMIT", "30")
    import models
    import config.settings as cs
    new = Settings(_env_file=None)
    assert new.image_gpu_memory_limit == 30
    monkeypatch.setattr(cs, "settings", new)
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as P, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=0), \
         patch("torch.cuda.memory_reserved", return_value=0):
        models.load_model().load()
    kwargs = P.from_pretrained.call_args.kwargs
    assert kwargs["max_memory"][0] == "30GiB" and kwargs["device_map"] == "balanced"


# ── 006: settings, transport path, 003 AS-1..AS-4 ─────────────────────────────

def test_edit_output_resolution_default_and_valid(monkeypatch):
    monkeypatch.delenv("EDIT_OUTPUT_RESOLUTION", raising=False)
    assert Settings(_env_file=None).edit_output_resolution == 1024
    monkeypatch.setenv("MAX_OUTPUT_RESOLUTION", "2048")
    monkeypatch.setenv("EDIT_OUTPUT_RESOLUTION", "2048")
    assert Settings(_env_file=None).edit_output_resolution == 2048
    monkeypatch.setenv("EDIT_OUTPUT_RESOLUTION", "512")
    assert Settings(_env_file=None).edit_output_resolution == 512


@pytest.mark.parametrize("value", ["1000", "256", "4096"])
def test_edit_output_resolution_invalid(monkeypatch, value):  # 006 AS-2
    monkeypatch.setenv("EDIT_OUTPUT_RESOLUTION", value)
    with pytest.raises(Exception):
        Settings(_env_file=None)


def test_edit_output_resolution_empty_is_default(monkeypatch):  # 006 AS-2
    monkeypatch.setenv("EDIT_OUTPUT_RESOLUTION", "")
    assert Settings(_env_file=None).edit_output_resolution == 1024


def test_edit_output_resolution_reaches_backend(monkeypatch):
    import models
    import config.settings as cs
    monkeypatch.setenv("EDIT_OUTPUT_RESOLUTION", "1024")
    monkeypatch.setattr(cs, "settings", Settings(_env_file=None))
    assert models.load_model().edit_output_resolution == 1024


def test_streamable_http_path_follows_settings():  # 006 AS-4
    assert server.mcp.settings.streamable_http_path == server.settings.mcp_path


def test_latest_image_id_returns_newest(fake):  # 003 AS-1
    a = server.store.save(PILImage.new("RGB", (4, 4)), {"prompt": "first"})
    b = server.store.save(PILImage.new("RGB", (4, 4)), {"prompt": "second"})
    assert server.get_latest_image_id()["image_id"] == b.image_id != a.image_id


def test_latest_image_id_empty_store(fake):  # 003 AS-2
    with pytest.raises(RuntimeError, match="Call generate_image first"):
        server.get_latest_image_id()


async def test_image_base64_is_byte_identical(fake, ctx):  # 003 AS-3
    await server.generate_image("p", ctx=ctx)
    stored = server.store.get(server.get_latest_image_id()["image_id"])
    b64 = await server.get_image_base64(stored.image_id)
    assert base64.b64decode(b64) == stored.disk_path.read_bytes()


async def test_image_base64_unknown_id(fake):  # 003 AS-4
    with pytest.raises(KeyError, match="not found"):
        await server.get_image_base64("00000000-0000-0000-0000-000000000000")


def test_get_model_raises_before_load(monkeypatch):  # 001 T014
    monkeypatch.setattr(server, "_model", None)
    with pytest.raises(RuntimeError, match="not loaded"):
        server.get_model()


# ── 006 F-4: request body limit ───────────────────────────────────────────────

def test_request_body_default_is_computed(monkeypatch):
    for v in ("MAX_REQUEST_BODY_MB", "MAX_REFERENCE_IMAGE_MB"):
        monkeypatch.delenv(v, raising=False)
    s = Settings(_env_file=None)
    assert s.max_request_body_mb is None
    assert s.effective_max_request_body_mb(10) == 268
    monkeypatch.setenv("MAX_REFERENCE_IMAGE_MB", "30")
    assert Settings(_env_file=None).effective_max_request_body_mb(10) == 401


def test_request_body_explicit_override(monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BODY_MB", "50")
    assert Settings(_env_file=None).effective_max_request_body_mb(10) == 50


def test_request_body_validation(monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BODY_MB", "3")
    with pytest.raises(Exception):
        Settings(_env_file=None)
    monkeypatch.setenv("MAX_REQUEST_BODY_MB", "")
    s = Settings(_env_file=None)
    assert s.max_request_body_mb is None and s.effective_max_request_body_mb(10) == 268


def test_fastmcp_receives_body_limit_in_bytes():
    expected = server.settings.effective_max_request_body_mb(
        server.MAX_REQUEST_REFERENCES) * 1024 * 1024
    assert server.mcp.settings.max_request_body_size == expected
    assert expected > 4 * 1024 * 1024


async def test_body_over_4mib_is_not_rejected_with_413():
    """HTTP level (F-4): direct tool calls bypass the transport's size check."""
    import httpx
    app = server.mcp.streamable_http_app()
    body = b"x" * (5 * 1024 * 1024)
    headers = {"content-type": "application/json",
               "accept": "application/json, text/event-stream"}
    # ASGITransport doesn't run lifespans; start the session manager by hand
    # (its run() may be entered only once per process, so keep this the only user).
    async with server.mcp.session_manager.run(), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8500"
    ) as client:
        r = await client.post(server.settings.mcp_path, content=body, headers=headers)
        assert r.status_code != 413
        # control: past the limit the SDK does answer 413
        too_big = b"x" * (server.mcp.settings.max_request_body_size + 1)
        r = await client.post(server.settings.mcp_path, content=too_big, headers=headers)
        assert r.status_code == 413


# ── 007: generate resolution, ceiling, F-5 ────────────────────────────────────

def test_resolution_settings_defaults(monkeypatch):
    for v in ("GENERATE_OUTPUT_RESOLUTION", "MAX_OUTPUT_RESOLUTION", "EDIT_OUTPUT_RESOLUTION"):
        monkeypatch.delenv(v, raising=False)
    s = Settings(_env_file=None)
    assert (s.generate_output_resolution, s.max_output_resolution) == (1024, 1024)


@pytest.mark.parametrize("var", ["GENERATE_OUTPUT_RESOLUTION", "MAX_OUTPUT_RESOLUTION"])
@pytest.mark.parametrize("value", ["1000", "256", "4096"])
def test_new_resolution_settings_invalid(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(Exception):
        Settings(_env_file=None)


@pytest.mark.parametrize("var", ["GENERATE_OUTPUT_RESOLUTION", "MAX_OUTPUT_RESOLUTION"])
def test_new_resolution_settings_empty_is_default(monkeypatch, var):
    monkeypatch.setenv(var, "")
    s = Settings(_env_file=None)
    assert (s.generate_output_resolution, s.max_output_resolution) == (1024, 1024)


@pytest.mark.parametrize("var", ["GENERATE_OUTPUT_RESOLUTION", "EDIT_OUTPUT_RESOLUTION"])
def test_ceiling_rejects_resolution_above_max(monkeypatch, var):  # 007 AS-3
    monkeypatch.setenv(var, "2048")
    with pytest.raises(ValueError) as exc:
        Settings(_env_file=None)
    assert (f"{var}=2048 exceeds MAX_OUTPUT_RESOLUTION=1024; "
            "raise MAX_OUTPUT_RESOLUTION to allow it") in str(exc.value)


def test_ceiling_raised_allows_2048(monkeypatch):
    monkeypatch.setenv("GENERATE_OUTPUT_RESOLUTION", "2048")
    monkeypatch.setenv("EDIT_OUTPUT_RESOLUTION", "2048")
    monkeypatch.setenv("MAX_OUTPUT_RESOLUTION", "2048")
    s = Settings(_env_file=None)
    assert (s.generate_output_resolution, s.edit_output_resolution) == (2048, 2048)


def test_generate_output_resolution_reaches_backend(monkeypatch):
    import models
    import config.settings as cs
    monkeypatch.setenv("GENERATE_OUTPUT_RESOLUTION", "1536")
    monkeypatch.setenv("MAX_OUTPUT_RESOLUTION", "2048")
    monkeypatch.setattr(cs, "settings", Settings(_env_file=None))
    assert models.load_model().aspect_ratios["1:1"] == (1536, 1536)


class _RealSizesModel(QwenImage21):
    """Real QwenImage21 size tables with a recording generate(); no pipeline."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        w, h = request.width or 40, request.height or 30
        return GenerationResult(PILImage.new("RGB", (w, h)), request.seed, 0.01, w, h)


async def test_generate_uses_1024_table_by_default(monkeypatch, tmp_path, ctx):  # 007 AS-1
    model = _RealSizesModel()
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "store", ImageStore(tmp_path))
    monkeypatch.setattr(server, "_generation_lock", asyncio.Lock())
    await server.generate_image("x", aspect_ratio="16:9", ctx=ctx)
    assert (model.requests[0].width, model.requests[0].height) == (1376, 768)


async def test_generate_uses_card_table_at_2048(monkeypatch, tmp_path, ctx):  # 007 AS-2
    model = _RealSizesModel(generate_output_resolution=2048)
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "store", ImageStore(tmp_path))
    monkeypatch.setattr(server, "_generation_lock", asyncio.Lock())
    await server.generate_image("x", aspect_ratio="16:9", ctx=ctx)
    assert (model.requests[0].width, model.requests[0].height) == (2752, 1536)


async def test_edit_aspect_ratio_uses_edit_table_not_generate_table(monkeypatch, tmp_path, ctx):  # F-5, AS-4
    model = _RealSizesModel(generate_output_resolution=2048, edit_output_resolution=1024)
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "store", ImageStore(tmp_path))
    monkeypatch.setattr(server, "_generation_lock", asyncio.Lock())
    await server.edit_image("x", [b64png()], aspect_ratio="1:1", ctx=ctx)
    await server.edit_image("x", [b64png()], aspect_ratio="16:9", ctx=ctx)
    assert [(r.width, r.height) for r in model.requests] == [(1024, 1024), (1376, 768)]


def test_list_aspect_ratios_real_backend_is_1024_table(monkeypatch):  # 007 AS-5
    monkeypatch.setattr(server, "_model", _RealSizesModel())
    out = server.list_aspect_ratios()
    assert out["1:1"] == {"width": 1024, "height": 1024}
    assert out["16:9"] == {"width": 1376, "height": 768}
