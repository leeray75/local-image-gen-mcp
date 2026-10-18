"""Unit tests for the model layer.

All tests mock QwenImage21Pipeline so they run without a GPU or model weights.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from models.base import GenerationRequest
from models.qwen_image_2_1 import QwenImage21


def _req(**over) -> GenerationRequest:
    base = dict(
        prompt="a red square",
        negative_prompt=None,
        width=64,
        height=64,
        num_inference_steps=40,
        cfg_scale=1.0,
        seed=42,
    )
    base.update(over)
    return GenerationRequest(**base)


@pytest.fixture
def mock_pipeline():
    """A mock pipeline returning a 4-channel image like the real VAE output."""
    mock = MagicMock()
    mock.return_value.images = [PILImage.new("RGBA", (64, 48), color=(255, 0, 0, 128))]
    return mock


@pytest.fixture
def loaded(mock_pipeline):
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        model = QwenImage21()
        model.load()
        yield model, mock_pipeline


def test_is_loaded_before_load():
    assert not QwenImage21().is_loaded()


def test_model_name_and_properties():
    m = QwenImage21()
    assert m.model_name == "Qwen-Image-2.1"
    assert m.default_inference_steps == 40
    assert m.default_cfg_scale == 1.0
    assert m.default_negative_prompt is None
    assert m.max_reference_images == 10
    assert m.supports_transparency is True
    assert m.supports_editing is True


CARD_2048 = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
    "3:2": (2528, 1696), "2:3": (1696, 2528),
    "16:9": (2752, 1536), "9:16": (1536, 2752),
}
TABLE_1024 = {
    "1:1": (1024, 1024), "16:9": (1376, 768), "9:16": (768, 1376),
    "4:3": (1184, 896), "3:4": (896, 1184), "3:2": (1248, 832), "2:3": (832, 1248),
}


def test_aspect_ratios_default_is_1024_table_and_copy():  # 007 FR-007-03/04
    m = QwenImage21()
    assert m.aspect_ratios == TABLE_1024
    m.aspect_ratios["1:1"] = (1, 1)  # mutating the copy must not leak
    assert m.aspect_ratios["1:1"] == (1024, 1024)


def test_sizes_for_2048_is_card_table_and_copy():
    m = QwenImage21()
    assert m.sizes_for(2048) == CARD_2048
    m.sizes_for(2048)["1:1"] = (1, 1)
    assert m.sizes_for(2048) == CARD_2048


def test_sizes_for_formula_at_1536():
    sizes = QwenImage21().sizes_for(1536)
    assert set(sizes) == set(CARD_2048)
    assert sizes["1:1"] == (1536, 1536)
    # w = round(sqrt(1536^2 * 16/9) / 32) * 32 = 2048; h from the unrounded width
    assert sizes["16:9"] == (2048, 1152)
    assert sizes["9:16"] == (1152, 2048)
    assert all(w % 32 == 0 and h % 32 == 0 for w, h in sizes.values())


def test_aspect_ratios_follow_generate_output_resolution():
    assert QwenImage21(generate_output_resolution=2048).aspect_ratios == CARD_2048
    assert QwenImage21(generate_output_resolution=512).aspect_ratios["1:1"] == (512, 512)


def test_torch_dtype_selection():
    assert QwenImage21(torch_dtype_str="bfloat16")._torch_dtype == torch.bfloat16
    assert QwenImage21(torch_dtype_str="float32")._torch_dtype == torch.float32


@patch("models.qwen_image_2_1.QwenImage21Pipeline")
def test_unload(MockPipeline, mock_pipeline):
    MockPipeline.from_pretrained.return_value = mock_pipeline
    model = QwenImage21()
    model.load()
    assert model.is_loaded()
    model.unload()
    assert not model.is_loaded()


def test_generate_raises_when_not_loaded():
    with pytest.raises(RuntimeError, match="not loaded"):
        QwenImage21().generate(_req())


# ── load kwargs (AS-13) ───────────────────────────────────────────────────────

@patch("models.qwen_image_2_1.QwenImage21Pipeline")
def test_load_cpu_has_no_device_map(MockPipeline):
    with patch("torch.cuda.is_available", return_value=False):
        QwenImage21(memory_limit_gb=30).load()
    kwargs = MockPipeline.from_pretrained.call_args.kwargs
    assert "device_map" not in kwargs
    assert "max_memory" not in kwargs


@patch("models.qwen_image_2_1.QwenImage21Pipeline")
def test_load_cuda_capped(MockPipeline):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=0), \
         patch("torch.cuda.memory_reserved", return_value=0):
        QwenImage21(memory_limit_gb=30).load()
    args, kwargs = MockPipeline.from_pretrained.call_args
    assert args[0] == "Qwen/Qwen-Image-2.1"
    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"] == {0: "30GiB", "cpu": "60GiB"}


@patch("models.qwen_image_2_1.QwenImage21Pipeline")
def test_load_cuda_uncapped(MockPipeline):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=0), \
         patch("torch.cuda.memory_reserved", return_value=0):
        QwenImage21().load()
    kwargs = MockPipeline.from_pretrained.call_args.kwargs
    assert kwargs["device_map"] == "balanced"
    assert "max_memory" not in kwargs


def test_load_never_calls_to(mock_pipeline):
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        QwenImage21().load()
    mock_pipeline.to.assert_not_called()


# ── generate kwargs ───────────────────────────────────────────────────────────

def test_generate_t2i_kwargs(loaded):
    model, pipe = loaded
    result = model.generate(_req(width=2048, height=2048))
    kwargs = pipe.call_args.kwargs
    assert kwargs["prompt"] == "a red square"
    assert kwargs["width"] == 2048 and kwargs["height"] == 2048
    assert kwargs["num_inference_steps"] == 40
    assert kwargs["true_cfg_scale"] == 1.0
    assert "guidance_scale" not in kwargs
    assert "negative_prompt" not in kwargs      # omitted when None
    assert "image" not in kwargs
    assert "output_resolution" not in kwargs
    assert "use_kv_cache" not in kwargs
    assert isinstance(kwargs["generator"], torch.Generator)
    assert result.seed == 42


def test_generate_rgb_conversion(loaded):
    model, _ = loaded
    result = model.generate(_req())
    assert result.image.mode == "RGB"
    assert (result.width, result.height) == (64, 48)


def test_generate_negative_prompt_passed_when_set(loaded):
    model, pipe = loaded
    model.generate(_req(negative_prompt="blurry", cfg_scale=4.0))
    kwargs = pipe.call_args.kwargs
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["true_cfg_scale"] == 4.0


def test_generate_transparent_wraps_prompt_and_returns_rgba(loaded):
    model, pipe = loaded
    result = model.generate(_req(prompt="  a sticker  ", transparent=True))
    assert pipe.call_args.kwargs["prompt"] == (
        "This is an RGBA image with transparency. a sticker. "
        "The image has alpha channel and the background is transparent."
    )
    assert result.image.mode == "RGBA"


def test_transparent_prompt_keeps_terminal_punctuation(loaded):
    model, pipe = loaded
    model.generate(_req(prompt="A cat!", transparent=True))
    assert "A cat! The image has alpha" in pipe.call_args.kwargs["prompt"]


def test_generate_edit_kwargs(loaded):
    model, pipe = loaded
    a = PILImage.new("RGB", (10, 10), "red")
    b = PILImage.new("RGBA", (20, 10), (0, 0, 255, 0))
    model.generate(_req(width=None, height=None, reference_images=[a, b]))
    kwargs = pipe.call_args.kwargs
    assert kwargs["image"] == [a, b] and kwargs["image"][0] is a
    assert kwargs["output_resolution"] == 1024
    assert "width" not in kwargs and "height" not in kwargs


def test_generate_edit_with_explicit_size(loaded):
    model, pipe = loaded
    a = PILImage.new("RGB", (10, 10), "red")
    model.generate(_req(width=2048, height=2048, reference_images=[a]))
    kwargs = pipe.call_args.kwargs
    assert kwargs["width"] == 2048 and kwargs["height"] == 2048
    assert kwargs["output_resolution"] == 1024


def test_edit_output_resolution_reaches_pipeline_for_edits_only():  # 006 AS-1
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline:
        pipe = MagicMock()
        pipe.return_value.images = [PILImage.new("RGBA", (64, 48))]
        MockPipeline.from_pretrained.return_value = pipe
        model = QwenImage21(edit_output_resolution=1024)
        model.load()
        assert model.edit_output_resolution == 1024
        a = PILImage.new("RGB", (10, 10), "red")
        model.generate(_req(width=None, height=None, reference_images=[a]))
        assert pipe.call_args.kwargs["output_resolution"] == 1024
        model.generate(_req())
        assert "output_resolution" not in pipe.call_args.kwargs
