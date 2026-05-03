"""Unit tests for the model layer.

All tests mock the DiffusionPipeline so they run without a GPU.
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage
from models.base import GenerationRequest
from models.qwen_image_2512 import QwenImage2512, DEFAULT_NEGATIVE_PROMPT


@pytest.fixture
def mock_pipeline():
    """Create a mock DiffusionPipeline that returns a dummy image."""
    mock = MagicMock()
    mock.return_value.images = [PILImage.new("RGB", (64, 64), color="red")]
    return mock


def test_is_loaded_before_load():
    """Model should report not loaded before load() is called."""
    model = QwenImage2512()
    assert not model.is_loaded()


def test_model_name():
    """Model name should contain 'Qwen'."""
    model = QwenImage2512()
    assert "Qwen" in model.model_name


def test_default_negative_prompt_not_empty():
    """Default negative prompt must be non-empty."""
    assert len(DEFAULT_NEGATIVE_PROMPT) > 10
    assert "low quality" in DEFAULT_NEGATIVE_PROMPT


@patch("models.qwen_image_2512.DiffusionPipeline")
def test_load_and_generate(MockPipeline, mock_pipeline):
    """Full lifecycle: load → generate → verify result."""
    MockPipeline.from_pretrained.return_value = mock_pipeline
    model = QwenImage2512()

    # Load
    model.load()
    assert model.is_loaded()

    # Generate
    request = GenerationRequest(
        prompt="a red square",
        negative_prompt="",
        width=64,
        height=64,
        num_inference_steps=1,
        cfg_scale=4.0,
        seed=42,
    )
    result = model.generate(request)

    assert result.seed == 42
    assert result.image is not None
    assert result.generation_time >= 0


@patch("models.qwen_image_2512.DiffusionPipeline")
def test_unload(MockPipeline, mock_pipeline):
    """After unload(), model should report not loaded."""
    MockPipeline.from_pretrained.return_value = mock_pipeline
    model = QwenImage2512()
    model.load()
    assert model.is_loaded()
    model.unload()
    assert not model.is_loaded()


@patch("models.qwen_image_2512.DiffusionPipeline")
def test_generate_raises_when_not_loaded(MockPipeline):
    """generate() must raise RuntimeError if model is not loaded."""
    model = QwenImage2512()
    request = GenerationRequest(
        prompt="test",
        negative_prompt="",
        width=64,
        height=64,
        num_inference_steps=1,
        cfg_scale=4.0,
        seed=0,
    )
    with pytest.raises(RuntimeError, match="not loaded"):
        model.generate(request)


def test_torch_dtype_selection():
    """Verify torch dtype is set correctly from string."""
    model_bf16 = QwenImage2512(torch_dtype_str="bfloat16")
    model_fp32 = QwenImage2512(torch_dtype_str="float32")

    import torch
    assert model_bf16._torch_dtype == torch.bfloat16
    assert model_fp32._torch_dtype == torch.float32