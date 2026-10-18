"""Unit tests for reference decoding and resolution (AS-7, AS-8)."""

import base64
import io
import uuid

import pytest
from PIL import Image as PILImage

from storage.image_store import ImageStore
from storage.references import UUID_RE, decode_base64_image, resolve_references

MB = 1024 * 1024


def b64(fmt="PNG", size=(8, 6), mode="RGB", color="red") -> str:
    buf = io.BytesIO()
    PILImage.new(mode, size, color).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def store(tmp_path):
    return ImageStore(tmp_path)


def _stored_id(store, color="green", size=(5, 5)):
    return store.save(PILImage.new("RGB", size, color), {}).image_id


def test_uuid_regex_only_matches_canonical_form():
    assert UUID_RE.match(str(uuid.uuid4()))
    assert not UUID_RE.match(uuid.uuid4().hex)  # 32-hex is base64, not an ID


def test_32_hex_string_is_treated_as_base64(store):
    with pytest.raises(ValueError, match=r"reference 1: .*(base64|image)"):
        resolve_references([uuid.uuid4().hex], store, 10, 20 * MB)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_decode_allowed_formats(fmt):
    img = decode_base64_image(b64(fmt), 20 * MB)
    assert img.size == (8, 6)


def test_decode_data_uri_prefix_and_whitespace():  # AS-8
    raw = b64()
    img = decode_base64_image("data:image/png;base64," + raw[:20] + "\n" + raw[20:], 20 * MB)
    assert img.size == (8, 6)


def test_decode_preserves_alpha():
    img = decode_base64_image(b64("PNG", mode="RGBA", color=(1, 2, 3, 4)), 20 * MB)
    assert img.mode == "RGBA"


def test_decode_invalid_base64():
    with pytest.raises(ValueError, match="invalid base64"):
        decode_base64_image("not*base64!!", 20 * MB)


def test_decode_not_an_image():
    with pytest.raises(ValueError, match="not a valid image"):
        decode_base64_image(base64.b64encode(b"hello world").decode(), 20 * MB)


def test_decode_gif_rejected():
    with pytest.raises(ValueError, match="unsupported format GIF"):
        decode_base64_image(b64("GIF", mode="P"), 20 * MB)


def test_decode_oversize():
    with pytest.raises(ValueError, match="exceeds MAX_REFERENCE_IMAGE_MB"):
        decode_base64_image(b64(size=(200, 200)), 100)


def test_decode_truncated_image():
    buf = io.BytesIO()
    PILImage.effect_noise((256, 256), 100).convert("RGB").save(buf, format="PNG")
    raw = buf.getvalue()
    truncated = base64.b64encode(raw[: len(raw) // 2]).decode()
    with pytest.raises(ValueError):
        decode_base64_image(truncated, 20 * MB)


def test_decode_decompression_bomb(monkeypatch):
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="decompression limit"):
        decode_base64_image(b64(size=(100, 100)), 20 * MB)


def test_resolve_order_and_ids(store):  # AS-5 / AS-7 order
    a = _stored_id(store, "green", (5, 5))
    imgs, ids = resolve_references([b64(size=(8, 6)), a, b64(size=(3, 3))], store, 10, 20 * MB)
    assert [i.size for i in imgs] == [(8, 6), (5, 5), (3, 3)]
    assert ids == [a]


def test_resolve_uppercase_uuid_resolves(store):
    a = _stored_id(store)
    imgs, ids = resolve_references([a.upper()], store, 10, 20 * MB)
    assert ids == [a] and imgs[0].mode == "RGB"


def test_decode_between_1x_and_2x_pixel_limit(monkeypatch):
    # Pillow only warns in this band; we must reject (100x100 = 10000 pixels)
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 5000)
    with pytest.raises(ValueError, match="decompression limit"):
        decode_base64_image(b64(size=(100, 100)), 20 * MB)


def test_resolve_zero_references(store):
    with pytest.raises(ValueError, match="1–10"):
        resolve_references([], store, 10, 20 * MB)


def test_resolve_too_many_references(store):
    with pytest.raises(ValueError, match="1–10"):
        resolve_references([b64()] * 11, store, 10, 20 * MB)


def test_resolve_unknown_id_names_position(store):
    with pytest.raises(ValueError, match=r"reference 2: image '.*' not found"):
        resolve_references([b64(), str(uuid.uuid4())], store, 10, 20 * MB)


def test_resolve_missing_file(store):
    a = _stored_id(store)
    store.get(a).disk_path.unlink()
    with pytest.raises(ValueError, match="reference 1: .*missing"):
        resolve_references([a], store, 10, 20 * MB)


def test_resolve_bad_base64_names_position(store):
    with pytest.raises(ValueError, match="reference 3: invalid base64"):
        resolve_references([b64(), b64(), "%%%"], store, 10, 20 * MB)


def test_resolve_gif_names_position(store):
    with pytest.raises(ValueError, match="reference 1: unsupported format GIF"):
        resolve_references([b64("GIF", mode="P")], store, 10, 20 * MB)


def test_resolve_oversize(store):
    with pytest.raises(ValueError, match="reference 1: decoded size"):
        resolve_references([b64(size=(400, 400))], store, 10, 100)
