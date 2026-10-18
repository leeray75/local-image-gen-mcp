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
        QwenImage21(memory_limit_gb=30, quantize_transformer="none", quantize_text_encoder="none").load()
    args, kwargs = MockPipeline.from_pretrained.call_args
    assert args[0] == "Qwen/Qwen-Image-2.1"
    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"] == {0: "30GiB", "cpu": "60GiB"}


@patch("models.qwen_image_2_1.QwenImage21Pipeline")
def test_load_cuda_uncapped(MockPipeline):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=0), \
         patch("torch.cuda.memory_reserved", return_value=0):
        QwenImage21(quantize_transformer="none", quantize_text_encoder="none").load()
    kwargs = MockPipeline.from_pretrained.call_args.kwargs
    assert kwargs["device_map"] == "balanced"
    assert "max_memory" not in kwargs


def test_load_never_calls_to(mock_pipeline):
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        QwenImage21(quantize_transformer="none", quantize_text_encoder="none").load()
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


# ── 008: load-time weight quantization ────────────────────────────────────────

def _load_cuda(model_kwargs=None, pipeline=None):
    """Load a QwenImage21 on a faked CUDA device; returns (model, MockPipeline).

    The staged loaders are patched too and hang off the returned mock:
    MockPipeline.order is a manager recording every call in order
    (transformer, text_encoder, free, pipeline).
    """
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline, \
         patch("models.qwen_image_2_1.QwenImage21Transformer2DModel") as MockTr, \
         patch("models.qwen_image_2_1.Qwen3VLForConditionalGeneration") as MockTe, \
         patch("models.qwen_image_2_1.gc.collect") as mock_gc, \
         patch("torch.cuda.empty_cache") as mock_empty, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=0), \
         patch("torch.cuda.memory_reserved", return_value=0):
        order = MagicMock()
        order.attach_mock(MockTr.from_pretrained, "transformer")
        order.attach_mock(MockTe.from_pretrained, "text_encoder")
        order.attach_mock(mock_gc, "gc")
        order.attach_mock(mock_empty, "empty_cache")
        order.attach_mock(MockPipeline.from_pretrained, "pipeline")
        MockPipeline.from_pretrained.return_value = pipeline or MagicMock()
        MockPipeline.order = order
        MockPipeline.tr, MockPipeline.te = MockTr, MockTe
        model = QwenImage21(**(model_kwargs or {}))
        model.load()
    return model, MockPipeline


def test_quantization_default_passes_no_config():  # 008 AS-1
    m = QwenImage21()
    assert m._build_quantization_config() is None
    _, MockPipeline = _load_cuda({"quantize_transformer": "none", "quantize_text_encoder": "none"})
    assert "quantization_config" not in MockPipeline.from_pretrained.call_args.kwargs
    # AS-5b: the none/none path is the single pipeline call, as before
    MockPipeline.tr.from_pretrained.assert_not_called()
    MockPipeline.te.from_pretrained.assert_not_called()
    kwargs = MockPipeline.from_pretrained.call_args.kwargs
    assert kwargs["device_map"] == "balanced" and "transformer" not in kwargs


def test_quantization_mapping_fp8_int8():  # 008 AS-2
    import diffusers
    import transformers
    from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig

    cfg = QwenImage21(
        quantize_transformer="fp8", quantize_text_encoder="int8"
    )._build_quantization_config()
    assert set(cfg.quant_mapping) == {"transformer", "text_encoder"}  # never the VAE
    tr, te = cfg.quant_mapping["transformer"], cfg.quant_mapping["text_encoder"]
    assert type(tr) is diffusers.TorchAoConfig
    assert isinstance(tr.quant_type, Float8WeightOnlyConfig)
    assert type(te) is transformers.TorchAoConfig
    assert isinstance(te.quant_type, Int8WeightOnlyConfig)


def test_quantization_mapping_nvfp4_transformer_only():  # 008 AS-3
    import diffusers

    cfg = QwenImage21(quantize_transformer="nvfp4")._build_quantization_config()
    assert set(cfg.quant_mapping) == {"transformer"}  # text encoder stays BF16
    tr = cfg.quant_mapping["transformer"]
    assert type(tr) is diffusers.NVIDIAModelOptConfig
    assert tr.quant_type == "NVFP4"
    assert tr.weight_only is True


def test_quantization_mapping_nvfp4_text_encoder_uses_torchao_prototype():
    import transformers
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig

    cfg = QwenImage21(quantize_text_encoder="nvfp4")._build_quantization_config()
    assert set(cfg.quant_mapping) == {"text_encoder"}
    te = cfg.quant_mapping["text_encoder"]
    assert type(te) is transformers.TorchAoConfig
    assert isinstance(te.quant_type, NVFP4WeightOnlyConfig)


@pytest.mark.parametrize("mode", ["int8", "fp8", "nvfp4"])
def test_quantization_skip_modules_and_fresh_lists(mode):  # 008 FR-008-05
    m = QwenImage21(quantize_transformer=mode, quantize_text_encoder=mode)
    cfg = m._build_quantization_config().quant_mapping
    assert cfg["transformer"].modules_to_not_convert == [
        "img_in", "txt_in", "time_text_embed", "modulation", "norm_out", "proj_out",
    ]
    assert cfg["text_encoder"].modules_to_not_convert == ["model.visual", "lm_head"]
    # the quantizers extend these lists in place, so they must not be shared
    cfg["transformer"].modules_to_not_convert.append("x")
    again = m._build_quantization_config().quant_mapping["transformer"]
    assert "x" not in again.modules_to_not_convert
    assert "x" not in m.TRANSFORMER_SKIP_MODULES


@pytest.mark.parametrize("cuda", [True, False])
def test_quantization_build_error_names_modes(cuda):  # 008 FR-008-04
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline, \
         patch("torch.cuda.is_available", return_value=cuda), \
         patch.object(QwenImage21, "_torchao_config", side_effect=ImportError("no torchao")):
        model = QwenImage21(quantize_transformer="fp8", quantize_text_encoder="nvfp4")
        with pytest.raises(RuntimeError) as exc:
            model.load()
    msg = str(exc.value)
    if cuda:
        assert "transformer=fp8" in msg and "text_encoder=nvfp4" in msg
        assert "no torchao" in msg and isinstance(exc.value.__cause__, ImportError)
    else:  # the CPU guard wins over a missing library
        assert "needs CUDA" in msg
    MockPipeline.from_pretrained.assert_not_called()


def test_quantization_modelopt_construction_error_names_modes():
    with patch("models.qwen_image_2_1.QwenImage21Pipeline"), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("diffusers.NVIDIAModelOptConfig", side_effect=ImportError("no modelopt")):
        model = QwenImage21(quantize_transformer="nvfp4", quantize_text_encoder="none")
        with pytest.raises(RuntimeError, match="transformer=nvfp4.*text_encoder=none.*no modelopt"):
            model.load()


def test_staged_load_order_and_arguments():  # 008 AS-5b, FR-008-10
    import diffusers
    import transformers
    from unittest.mock import call

    pipe = MagicMock()
    _, P = _load_cuda(
        {"quantize_transformer": "fp8", "quantize_text_encoder": "int8",
         "memory_limit_gb": 40},
        pipeline=pipe,
    )
    names = [c[0] for c in P.order.mock_calls if c[0] in
             ("transformer", "text_encoder", "gc", "empty_cache", "pipeline")]
    assert names == [
        "transformer", "gc", "empty_cache",
        "text_encoder", "gc", "empty_cache",
        "pipeline",
    ]
    tr_kw = P.tr.from_pretrained.call_args.kwargs
    assert P.tr.from_pretrained.call_args.args == ("Qwen/Qwen-Image-2.1",)
    assert tr_kw["subfolder"] == "transformer"
    assert type(tr_kw["quantization_config"]) is diffusers.TorchAoConfig
    assert tr_kw["device_map"] == {"": 0} and tr_kw["dtype"] == torch.bfloat16
    te_kw = P.te.from_pretrained.call_args.kwargs
    assert te_kw["subfolder"] == "text_encoder"
    assert type(te_kw["quantization_config"]) is transformers.TorchAoConfig
    assert te_kw["device_map"] == {"": 0}
    pk = P.from_pretrained.call_args.kwargs
    assert pk["transformer"] is P.tr.from_pretrained.return_value
    assert pk["text_encoder"] is P.te.from_pretrained.return_value
    assert "quantization_config" not in pk and "max_memory" not in pk
    assert pk["device_map"] == "cuda"


def test_staged_load_none_component_loads_bf16():  # 008 FR-008-10
    _, P = _load_cuda({"quantize_transformer": "nvfp4", "quantize_text_encoder": "none"})
    assert "quantization_config" in P.tr.from_pretrained.call_args.kwargs
    te_kw = P.te.from_pretrained.call_args.kwargs
    assert "quantization_config" not in te_kw  # BF16, same sequence
    assert te_kw["device_map"] == {"": 0}
    _, P = _load_cuda({"quantize_transformer": "none", "quantize_text_encoder": "nvfp4"})
    assert "quantization_config" not in P.tr.from_pretrained.call_args.kwargs
    assert "quantization_config" in P.te.from_pretrained.call_args.kwargs


def test_staged_load_logs_memory_after_each_step(caplog):  # 008 FR-008-12
    import logging
    with caplog.at_level(logging.INFO, logger="models.qwen_image_2_1"):
        with patch.object(QwenImage21, "_host_available_gb", return_value=33.3):
            _load_cuda({"quantize_transformer": "int8", "quantize_text_encoder": "none"})
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Memory after")]
    assert [l.split(" — ")[0] for l in lines] == [
        "Memory after start",
        "Memory after transformer (int8)",
        "Memory after text_encoder (none)",
        "Memory after pipeline (remaining components)",
    ]
    assert all("allocated" in l and "reserved" in l and "host available: 33.3 GB" in l
               for l in lines)


def test_host_available_gb_reads_meminfo(tmp_path):
    from unittest.mock import mock_open
    data = "MemTotal: 100 kB\nMemAvailable:   2097152 kB\n"
    with patch("builtins.open", mock_open(read_data=data)):
        assert QwenImage21._host_available_gb() == 2.0
    with patch("builtins.open", side_effect=OSError):
        assert QwenImage21._host_available_gb() is None


@pytest.mark.parametrize("failing,expect", [
    ("transformer", "while loading transformer (nvfp4)"),
    ("text_encoder", "while loading text_encoder (fp8)"),
    ("pipeline", "while loading pipeline"),
])
def test_staged_load_error_names_component_and_modes(failing, expect):  # 008 AS-5
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as P, \
         patch("models.qwen_image_2_1.QwenImage21Transformer2DModel") as Tr, \
         patch("models.qwen_image_2_1.Qwen3VLForConditionalGeneration") as Te, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=0), \
         patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.empty_cache"):
        {"transformer": Tr, "text_encoder": Te, "pipeline": P}[failing] \
            .from_pretrained.side_effect = ValueError("kernel not supported")
        model = QwenImage21(quantize_transformer="nvfp4", quantize_text_encoder="fp8")
        with pytest.raises(RuntimeError) as exc:
            model.load()
    msg = str(exc.value)
    assert expect in msg
    assert "transformer=nvfp4" in msg and "text_encoder=fp8" in msg
    assert "kernel not supported" in msg
    assert isinstance(exc.value.__cause__, ValueError)
    assert not model.is_loaded()


@pytest.mark.parametrize("field", ["quantize_transformer", "quantize_text_encoder"])
@patch("models.qwen_image_2_1.QwenImage21Pipeline")
def test_quantization_on_cpu_fails(MockPipeline, field):  # 008 FR-008-03
    with patch("torch.cuda.is_available", return_value=False):
        model = QwenImage21(**{field: "int8"})
        with pytest.raises(RuntimeError, match="needs CUDA"):
            model.load()
    MockPipeline.from_pretrained.assert_not_called()
    assert not model.is_loaded()


def test_quantization_cpu_with_none_still_loads():
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline, \
         patch("torch.cuda.is_available", return_value=False):
        QwenImage21().load()
    MockPipeline.from_pretrained.assert_called_once()


def test_quantization_invalid_mode_rejected():
    with pytest.raises(ValueError, match="quantize_text_encoder"):
        QwenImage21(quantize_text_encoder="bf16")


def test_load_error_without_quantization_is_not_wrapped():
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline, \
         patch("torch.cuda.is_available", return_value=True):
        MockPipeline.from_pretrained.side_effect = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            QwenImage21(quantize_transformer="none", quantize_text_encoder="none").load()


def test_quantization_library_imported_lazily():
    import importlib
    src = importlib.util.find_spec("models.qwen_image_2_1").origin
    text = open(src).read()
    head = text.split("class QwenImage21")[0]
    for lib in ("torchao", "modelopt", "NVIDIAModelOptConfig"):
        assert lib not in head, f"{lib} must be imported inside the helper"


def test_quantization_info_and_weight_measurement():  # 008 FR-008-06/07
    pipe = MagicMock()
    pipe.transformer = torch.nn.Linear(100, 100, bias=False)  # 10000 x 4 B
    pipe.text_encoder = torch.nn.Linear(10, 10, bias=False).to(torch.bfloat16)
    pipe.vae = torch.nn.Linear(1, 1, bias=False)
    model, _ = _load_cuda({"quantize_transformer": "int8", "quantize_text_encoder": "none"}, pipeline=pipe)
    info = model.quantization_info
    assert info["quantize_transformer"] == "int8"
    assert info["quantize_text_encoder"] == "none"
    assert info["weight_memory_gb"] == {"transformer": 0.0, "text_encoder": 0.0, "vae": 0.0}
    assert QwenImage21._module_gb(pipe.transformer) == round(40000 / 1e9, 2)
    model.unload()
    assert model.quantization_info["weight_memory_gb"] is None


def test_tensor_nbytes_counts_inner_tensors_of_subclasses():
    class Packed:  # stands in for a torchao tensor subclass
        def __init__(self):
            self.qdata = torch.zeros(10, dtype=torch.int8)
            self.scale = torch.zeros(2, dtype=torch.float32)

        def __tensor_flatten__(self):
            return ["qdata", "scale"], None

    assert QwenImage21._tensor_nbytes(Packed()) == 10 + 8
    assert QwenImage21._tensor_nbytes(torch.zeros(3, dtype=torch.bfloat16)) == 6


def test_quantization_info_default_on_base_class():
    from models.base import BaseImageModel
    assert BaseImageModel.quantization_info.fget(object()) == {}


@pytest.mark.parametrize("tr,te,expected", [
    ("nvfp4", "none", True), ("none", "nvfp4", True), ("nvfp4", "nvfp4", True),
    ("fp8", "int8", False), ("none", "none", False),
])
def test_nvfp4_logs_startup_warning(caplog, tr, te, expected):  # 008 FR-008-14
    import logging
    with caplog.at_level(logging.WARNING, logger="models.qwen_image_2_1"):
        model, _ = _load_cuda({"quantize_transformer": tr, "quantize_text_encoder": te})
    warned = [r for r in caplog.records
              if "NVFP4" in r.getMessage() and "F-7" in r.getMessage()
              and "NOT reduce resident memory" in r.getMessage()]
    assert bool(warned) is expected
    assert model.quantization_info["quantize_transformer"] == tr  # value still accepted


# ── 008 AS-5d: the auto default ───────────────────────────────────────────────

def test_auto_resolves_to_int8_on_cuda_with_staged_load(caplog):  # AS-5d
    import logging
    import torchao.quantization as tq
    with caplog.at_level(logging.INFO, logger="models.qwen_image_2_1"):
        model, P = _load_cuda({})  # defaults: auto / auto
    assert model.quantization_info["quantize_transformer"] == "int8"
    assert model.quantization_info["quantize_text_encoder"] == "int8"
    assert model.quantization_info["configured_quantize_transformer"] == "auto"
    for loader in (P.tr, P.te):
        qc = loader.from_pretrained.call_args.kwargs["quantization_config"]
        assert isinstance(qc.quant_type, tq.Int8WeightOnlyConfig)
    assert "device_map" in P.from_pretrained.call_args.kwargs
    assert P.from_pretrained.call_args.kwargs["device_map"] == "cuda"  # staged path
    assert any("transformer=int8 (configured auto)" in r.getMessage() for r in caplog.records)


def test_auto_resolves_to_none_on_cpu_single_call(caplog):  # AS-5d
    import logging
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as P, \
         patch("models.qwen_image_2_1.QwenImage21Transformer2DModel") as Tr, \
         patch("torch.cuda.is_available", return_value=False), \
         caplog.at_level(logging.INFO, logger="models.qwen_image_2_1"):
        model = QwenImage21()
        model.load()
    info = model.quantization_info
    assert (info["quantize_transformer"], info["quantize_text_encoder"]) == ("none", "none")
    P.from_pretrained.assert_called_once()
    Tr.from_pretrained.assert_not_called()
    assert "transformer" not in P.from_pretrained.call_args.kwargs
    assert any("transformer=none (configured auto)" in r.getMessage() for r in caplog.records)


def test_auto_with_explicit_int8_on_cpu_still_fails():  # AS-5d, FR-008-03
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as P, \
         patch("torch.cuda.is_available", return_value=False):
        model = QwenImage21(quantize_transformer="int8")  # text encoder auto -> none
        with pytest.raises(RuntimeError, match="needs CUDA"):
            model.load()
    P.from_pretrained.assert_not_called()


def test_explicit_none_on_cuda_is_bf16_single_call():
    _, P = _load_cuda({"quantize_transformer": "none", "quantize_text_encoder": "none"})
    P.tr.from_pretrained.assert_not_called()
    assert P.from_pretrained.call_args.kwargs["device_map"] == "balanced"


# ── Turbo variant (spec 009) ──────────────────────────────────────────────────

def test_base_variant_properties():  # AS-3 (backend side)
    m = QwenImage21()
    assert m.model_id == "Qwen/Qwen-Image-2.1"
    assert (m.default_inference_steps, m.min_inference_steps, m.max_inference_steps) == (40, 20, 100)


def test_turbo_variant_properties():  # AS-1
    m = QwenImage21(variant="turbo")
    assert m.model_id == "Qwen/Qwen-Image-2.1-Turbo"
    assert m.model_name == "Qwen-Image-2.1-Turbo"
    assert (m.default_inference_steps, m.min_inference_steps, m.max_inference_steps) == (8, 8, 8)
    assert m.default_cfg_scale == 1.0 and m.default_negative_prompt is None
    assert m.max_reference_images == 10 and m.supports_transparency is True


def test_invalid_variant_rejected():
    with pytest.raises(ValueError, match="variant"):
        QwenImage21(variant="fast")


@patch("models.qwen_image_2_1.QwenImage21Pipeline")
def test_turbo_loads_turbo_checkpoint(MockPipeline):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=0), \
         patch("torch.cuda.memory_reserved", return_value=0):
        QwenImage21(variant="turbo", quantize_transformer="none", quantize_text_encoder="none").load()
    assert MockPipeline.from_pretrained.call_args.args[0] == "Qwen/Qwen-Image-2.1-Turbo"


def test_turbo_generate_kwargs_use_checkpoint_schedule(mock_pipeline):
    """No sigmas and no scheduler override: the checkpoint's sample_sigmas apply (R-1)."""
    with patch("models.qwen_image_2_1.QwenImage21Pipeline") as MockPipeline:
        MockPipeline.from_pretrained.return_value = mock_pipeline
        model = QwenImage21(variant="turbo")
        model.load()
    model.generate(_req(num_inference_steps=8))
    kwargs = mock_pipeline.call_args.kwargs
    assert kwargs["num_inference_steps"] == 8
    assert "sigmas" not in kwargs and "timesteps" not in kwargs
    assert kwargs["true_cfg_scale"] == 1.0
    mock_pipeline.scheduler.assert_not_called()
