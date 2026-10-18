"""Concrete implementation of BaseImageModel for Qwen-Image-2.1.

Uses diffusers QwenImage21Pipeline with the official Qwen/Qwen-Image-2.1
checkpoint. One model covers text-to-image, image editing (up to 10 reference
images) and native transparent (RGBA) generation.

Memory on GB10 unified memory (128GB):
  The weights are about 33GB in BF16; measured peaks are ~68GB for a 2K
  generate and ~40GB for an edit at 1024 (the default text-to-image size is
  1024 too since spec 007; its peak is estimated at or below ~40GB). In BF16 this
  does NOT fit next to a running vLLM engine, so stop vLLM first. The default int8
  quantization (spec 008, `auto` on CUDA) cuts the weights to ~18.5GB, which does
  fit next to vLLM. IMAGE_GPU_MEMORY_LIMIT caps accelerate's device placement via
  max_memory, but offloaded layers live in the same physical RAM on unified
  memory, so the cap does not reduce total memory use.

Quantization (spec 008):
  QUANTIZE_TRANSFORMER / QUANTIZE_TEXT_ENCODER (none|int8|fp8|nvfp4) quantize
  the official weights at load time, one setting per component, through a
  diffusers PipelineQuantizationConfig. Default auto = int8 on CUDA, none
  (BF16) on CPU; set none explicitly for BF16 on CUDA. The VAE is never
  quantized. See _build_quantization_config() and research.md R-5. With any
  quantization on, components load one at a time with memory freed between
  them (spec 008 F-6, research R-6); with none/none loading is unchanged.

Meta-tensor rule:
  Pass device_map to from_pretrained() and NEVER call .to(device) afterwards:
  the pipeline uses meta-device init and .to() raises
  NotImplementedError: Cannot copy out of meta tensor.
"""

import gc
import logging
import math
import time

import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from transformers import Qwen3VLForConditionalGeneration

from .base import BaseImageModel, GenerationRequest, GenerationResult

log = logging.getLogger(__name__)


class QwenImage21(BaseImageModel):
    """Qwen-Image-2.1 backend (text-to-image, editing, RGBA)."""

    # variant -> (checkpoint, model_name, default steps, min steps, max steps).
    # Turbo's checkpoint carries `sample_sigmas` (8 values), which the pipeline
    # uses in place of num_inference_steps (spec 009 research R-1), so its range
    # is fixed at 8.
    VARIANTS: dict[str, tuple[str, str, int, int, int]] = {
        "base": ("Qwen/Qwen-Image-2.1", "Qwen-Image-2.1", 40, 20, 100),
        "turbo": ("Qwen/Qwen-Image-2.1-Turbo", "Qwen-Image-2.1-Turbo", 8, 8, 8),
    }

    # Official 2K sizes from the model card (used as-is at resolution 2048)
    CARD_SIZES_2048: dict[str, tuple[int, int]] = {
        "1:1": (2048, 2048),
        "4:3": (2400, 1792),
        "3:4": (1792, 2400),
        "3:2": (2528, 1696),
        "2:3": (1696, 2528),
        "16:9": (2752, 1536),
        "9:16": (1536, 2752),
    }

    # Ratio keys as (w, h) factors, for the pipeline's calculate_dimensions rule
    RATIO_FACTORS: dict[str, tuple[int, int]] = {
        "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2),
        "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16),
    }

    # Quantization modes (spec 008) and the layers kept in BF16 (research R-5).
    QUANT_MODES = ("auto", "none", "int8", "fp8", "nvfp4")
    # What "auto" resolves to on CUDA (spec 008 D-10); on CPU it resolves to "none"
    AUTO_CUDA_MODE = "int8"
    # diffusers torchao skips by substring "name.", ModelOpt by "*name*"
    TRANSFORMER_SKIP_MODULES = (
        "img_in", "txt_in", "time_text_embed", "modulation", "norm_out", "proj_out",
    )
    # transformers has no default skips once a list is given, so lm_head is explicit
    TEXT_ENCODER_SKIP_MODULES = ("model.visual", "lm_head")

    # Phrasing from the model card for native RGBA generation
    TRANSPARENT_PROMPT = (
        "This is an RGBA image with transparency. {prompt} "
        "The image has alpha channel and the background is transparent."
    )

    def __init__(
        self,
        torch_dtype_str: str = "bfloat16",
        memory_limit_gb: int | None = None,
        edit_output_resolution: int = 1024,
        generate_output_resolution: int = 1024,
        quantize_transformer: str = "auto",
        quantize_text_encoder: str = "auto",
        variant: str = "base",
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(
                f"variant={variant!r} is invalid; must be one of {tuple(self.VARIANTS)}"
            )
        (self._model_id, self._model_name, self._default_steps,
         self._min_steps, self._max_steps) = self.VARIANTS[variant]
        for name, mode in (
            ("quantize_transformer", quantize_transformer),
            ("quantize_text_encoder", quantize_text_encoder),
        ):
            if mode not in self.QUANT_MODES:
                raise ValueError(
                    f"{name}={mode!r} is invalid; must be one of {self.QUANT_MODES}"
                )
        self._configured_transformer = quantize_transformer
        self._configured_text_encoder = quantize_text_encoder
        # "auto" resolves here, from the same CUDA check that sets self._device
        auto = self.AUTO_CUDA_MODE if torch.cuda.is_available() else "none"
        self._quantize_transformer = auto if quantize_transformer == "auto" else quantize_transformer
        self._quantize_text_encoder = auto if quantize_text_encoder == "auto" else quantize_text_encoder
        self._weight_memory_gb: dict[str, float] | None = None
        self._pipeline: QwenImage21Pipeline | None = None
        self._memory_limit_gb = memory_limit_gb
        self._edit_output_resolution = edit_output_resolution
        self._generate_output_resolution = generate_output_resolution
        self._torch_dtype = (
            torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float32
        )
        # Used only for the Generator seed — device_map handles actual placement
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Staged loading (spec 008 FR-008-10, FR-008-12) ────────────────────────

    @staticmethod
    def _host_available_gb() -> float | None:
        """Host MemAvailable from /proc/meminfo in GB, or None if unreadable."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return round(int(line.split()[1]) / 1048576, 2)
        except (OSError, ValueError, IndexError):
            pass
        return None

    def _log_memory(self, stage: str) -> None:
        """Log CUDA allocated/reserved and host MemAvailable after a load step."""
        if torch.cuda.is_available():
            allocated = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            reserved = round(torch.cuda.memory_reserved(0) / 1e9, 2)
        else:
            allocated = reserved = None
        log.info(
            "Memory after %s — CUDA allocated: %s GB, reserved: %s GB, "
            "host available: %s GB",
            stage, allocated, reserved, self._host_available_gb(),
        )

    def _free_memory(self) -> None:
        """Drop Python garbage and return cached CUDA blocks (on unified memory,
        host RAM) before the next component loads."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_staged(self, quantization_config):
        """Load transformer, then text encoder, then the pipeline (FR-008-10).

        Each component is loaded on its own so that loading buffers of one are
        released before the next starts (F-6). Components set to "none" take the
        same path in BF16. Placement: device_map={"": 0} per component and
        "cuda" for the rest of the pipeline; never .to() afterwards (research R-6).
        """
        mapping = quantization_config.quant_mapping
        modes = {
            "transformer": self._quantize_transformer,
            "text_encoder": self._quantize_text_encoder,
        }
        device_map = {"": 0}
        loaded = {}
        stage = "start"

        def fail(err: Exception) -> RuntimeError:
            return RuntimeError(
                f"Failed to load {self._model_name} with quantization "
                f"(transformer={self._quantize_transformer}, "
                f"text_encoder={self._quantize_text_encoder}) "
                f"while loading {stage}: {type(err).__name__}: {err}"
            )

        self._log_memory("start")
        try:
            for name, cls in (
                ("transformer", QwenImage21Transformer2DModel),
                ("text_encoder", Qwen3VLForConditionalGeneration),
            ):
                stage = f"{name} ({modes[name]})"
                kw = {"subfolder": name, "dtype": self._torch_dtype, "device_map": device_map}
                if name in mapping:
                    kw["quantization_config"] = mapping[name]
                loaded[name] = cls.from_pretrained(self._model_id, **kw)
                self._free_memory()
                self._log_memory(stage)

            stage = "pipeline (remaining components)"
            pipeline = QwenImage21Pipeline.from_pretrained(
                self._model_id,
                transformer=loaded["transformer"],
                text_encoder=loaded["text_encoder"],
                dtype=self._torch_dtype,
                device_map="cuda",
            )
            self._log_memory(stage)
            return pipeline
        except Exception as err:
            raise fail(err) from err

    def _build_quantization_config(self):
        """Build the pipeline quantization config, or None when nothing is quantized.

        One entry per quantized component in a diffusers PipelineQuantizationConfig;
        components set to "none" are left out and stay BF16. The VAE is never
        quantized. The quantization libraries are imported here, lazily, so
        importing this module needs neither torchao nor ModelOpt. The calls were
        verified against the installed versions (research R-5).
        """
        if self._quantize_transformer == "none" and self._quantize_text_encoder == "none":
            return None

        from diffusers import NVIDIAModelOptConfig, PipelineQuantizationConfig
        from diffusers import TorchAoConfig as DiffusersTorchAoConfig
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        # Fresh lists each time: the diffusers quantizers extend them in place
        mapping = {}

        mode = self._quantize_transformer
        if mode == "nvfp4":
            mapping["transformer"] = NVIDIAModelOptConfig(
                quant_type="NVFP4",
                weight_only=True,
                modules_to_not_convert=list(self.TRANSFORMER_SKIP_MODULES),
            )
        elif mode != "none":
            mapping["transformer"] = DiffusersTorchAoConfig(
                self._torchao_config(mode),
                modules_to_not_convert=list(self.TRANSFORMER_SKIP_MODULES),
            )

        mode = self._quantize_text_encoder
        if mode != "none":
            mapping["text_encoder"] = TransformersTorchAoConfig(
                self._torchao_config(mode),
                modules_to_not_convert=list(self.TEXT_ENCODER_SKIP_MODULES),
            )

        return PipelineQuantizationConfig(quant_mapping=mapping)

    @staticmethod
    def _torchao_config(mode: str):
        """The torchao weight-only config for a mode (lazy import)."""
        if mode == "int8":
            from torchao.quantization import Int8WeightOnlyConfig
            return Int8WeightOnlyConfig()
        if mode == "fp8":
            from torchao.quantization import Float8WeightOnlyConfig
            return Float8WeightOnlyConfig()
        if mode == "nvfp4":
            # torchao prototype API (may change between releases)
            from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
            return NVFP4WeightOnlyConfig()
        raise ValueError(f"No torchao config for mode {mode!r}")

    @staticmethod
    def _tensor_nbytes(t: torch.Tensor) -> int:
        """Bytes a tensor really stores, looking inside torchao tensor subclasses."""
        if hasattr(t, "__tensor_flatten__"):
            names, _ = t.__tensor_flatten__()
            return sum(QwenImage21._tensor_nbytes(getattr(t, n)) for n in names)
        return t.numel() * t.element_size()

    @classmethod
    def _module_gb(cls, module) -> float:
        """Stored size of a module's parameters and buffers, in GB."""
        seen: set[int] = set()
        total = 0
        for t in [*module.parameters(), *module.buffers()]:
            if id(t) not in seen:
                seen.add(id(t))
                total += cls._tensor_nbytes(t)
        return round(total / 1e9, 2)

    def _measure_weights(self) -> None:
        """Record and log the per-component weight memory as stored (FR-008-06)."""
        sizes = {}
        for name in ("transformer", "text_encoder", "vae"):
            try:
                sizes[name] = self._module_gb(getattr(self._pipeline, name))
            except Exception as err:  # measurement must never fail a load
                log.warning("Could not measure %s weight size: %s", name, err)
        self._weight_memory_gb = sizes or None
        log.info(
            "Quantization: transformer=%s, text_encoder=%s, vae=none. "
            "Weights as stored (GB): %s",
            self._quantize_transformer,
            self._quantize_text_encoder,
            sizes,
        )

    @property
    def quantization_info(self) -> dict:
        """Active modes and measured per-component weight memory (FR-008-07)."""
        info = {
            "quantize_transformer": self._quantize_transformer,
            "quantize_text_encoder": self._quantize_text_encoder,
            "weight_memory_gb": self._weight_memory_gb,
        }
        # The configured value is shown only where it differs (i.e. "auto")
        if self._configured_transformer != self._quantize_transformer:
            info["configured_quantize_transformer"] = self._configured_transformer
        if self._configured_text_encoder != self._quantize_text_encoder:
            info["configured_quantize_text_encoder"] = self._configured_text_encoder
        return info

    def load(self) -> None:
        """Load model weights into GPU/unified memory.

        Uses device_map='balanced' with an optional max_memory cap so accelerate
        places weights during loading. Do NOT call .to(device) afterwards.
        """
        if self._pipeline is not None:
            log.warning("Model already loaded — skipping")
            return

        log.info(
            "Quantization modes: transformer=%s (configured %s), "
            "text_encoder=%s (configured %s)",
            self._quantize_transformer, self._configured_transformer,
            self._quantize_text_encoder, self._configured_text_encoder,
        )
        quantizing = (
            self._quantize_transformer != "none"
            or self._quantize_text_encoder != "none"
        )
        # CPU guard first, so a missing quantization library cannot pre-empt it
        if self._device == "cpu" and quantizing:
            raise RuntimeError(
                "Weight quantization (QUANTIZE_TRANSFORMER="
                f"{self._quantize_transformer}, QUANTIZE_TEXT_ENCODER="
                f"{self._quantize_text_encoder}) needs CUDA; set both to "
                "'none' to load on CPU"
            )
        if "nvfp4" in (self._quantize_transformer, self._quantize_text_encoder):
            log.warning(
                "NVFP4 is selected (transformer=%s, text_encoder=%s). With the "
                "installed diffusers, ModelOpt and torchao versions on GB10, NVFP4 "
                "does NOT reduce resident memory (spec 008 F-7) and may use more "
                "than BF16 while loading. Prefer int8 or fp8.",
                self._quantize_transformer,
                self._quantize_text_encoder,
            )
        try:
            quantization_config = self._build_quantization_config()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load {self._model_name} with quantization "
                f"(transformer={self._quantize_transformer}, "
                f"text_encoder={self._quantize_text_encoder}): "
                f"{type(err).__name__}: {err}"
            ) from err

        if self._device == "cpu":
            log.warning(
                "CUDA not available — loading on CPU. "
                "Generation will be extremely slow."
            )
            self._pipeline = QwenImage21Pipeline.from_pretrained(
                self._model_id,
                torch_dtype=self._torch_dtype,
            )
            log.info("Model loaded on CPU")
            return

        kwargs = {
            "torch_dtype": self._torch_dtype,
            "device_map": "balanced",
        }

        if self._memory_limit_gb is not None:
            kwargs["max_memory"] = {
                0: f"{self._memory_limit_gb}GiB",
                "cpu": "60GiB",  # allow offload to system RAM
            }
            log.info(
                "Loading %s with %dGB GPU memory cap (dtype=%s)...",
                self._model_id,
                self._memory_limit_gb,
                self._torch_dtype,
            )
            log.warning(
                "Memory cap active — layers exceeding %dGB are offloaded to CPU "
                "RAM and generation is slower. On unified memory the cap does "
                "not reduce total RAM use; stop vLLM before running this server.",
                self._memory_limit_gb,
            )
        else:
            log.info(
                "Loading %s on %s (dtype=%s, no memory cap)...",
                self._model_id,
                self._device,
                self._torch_dtype,
            )
            log.warning(
                "No memory cap — model will use ~33GB weights plus activations "
                "(~68GB peak for a 2K generate, less at 1K). Stop vLLM before starting image generation."
            )

        if quantization_config is not None:
            log.info(
                "Quantizing at load: transformer=%s, text_encoder=%s "
                "(components load one at a time)",
                self._quantize_transformer,
                self._quantize_text_encoder,
            )
            if self._memory_limit_gb is not None:
                log.warning(
                    "IMAGE_GPU_MEMORY_LIMIT is not applied when quantizing: "
                    "each component loads onto GPU 0 (quantized weights are small)."
                )
            self._pipeline = self._load_staged(quantization_config)
        else:
            self._pipeline = QwenImage21Pipeline.from_pretrained(
                self._model_id,
                **kwargs,
            )

        self._measure_weights()

        if torch.cuda.is_available():
            allocated = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            reserved = round(torch.cuda.memory_reserved(0) / 1e9, 2)
            log.info(
                "Model loaded successfully. "
                "PyTorch CUDA — allocated: %.2fGB, reserved: %.2fGB",
                allocated,
                reserved,
            )
        else:
            log.info("Model loaded successfully on %s", self._device)

    def unload(self) -> None:
        """Release pipeline from memory and free GPU cache."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._weight_memory_gb = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log.info("Model unloaded, GPU memory cleared")

    def _wrap_transparent(self, prompt: str) -> str:
        """Apply the model card's RGBA phrasing to a prompt."""
        p = prompt.strip()
        if not p.endswith((".", "!", "?")):
            p += "."
        return self.TRANSPARENT_PROMPT.format(prompt=p)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate or edit an image synchronously.

        Called via asyncio.to_thread() from the async MCP tool — blocks the
        calling thread during diffusion. Never call directly from async.

        Uses true_cfg_scale (NOT guidance_scale). Guidance is off at the default
        of 1.0, in which case negative_prompt has no effect.
        """
        if self._pipeline is None:
            raise RuntimeError(
                f"Model {self._model_id} is not loaded. Call load() first."
            )

        t_start = time.perf_counter()

        prompt = (
            self._wrap_transparent(request.prompt)
            if request.transparent
            else request.prompt
        )

        kwargs: dict = {
            "prompt": prompt,
            "num_inference_steps": request.num_inference_steps,
            "true_cfg_scale": request.cfg_scale,
            # Generator must use the same device as the model
            "generator": torch.Generator(device=self._device).manual_seed(
                request.seed
            ),
        }
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.reference_images:
            kwargs["image"] = list(request.reference_images)
            kwargs["output_resolution"] = self._edit_output_resolution
        if request.width is not None and request.height is not None:
            kwargs["width"] = request.width
            kwargs["height"] = request.height

        output = self._pipeline(**kwargs)

        # The VAE always returns RGBA; flatten unless transparency was requested
        img = output.images[0].convert("RGBA" if request.transparent else "RGB")
        generation_time = time.perf_counter() - t_start

        # Release temporary computation buffers (weights stay loaded)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info(
            "Generated %dx%d %s image in %.1fs (seed=%d, refs=%d)",
            img.width,
            img.height,
            img.mode,
            generation_time,
            request.seed,
            len(request.reference_images),
        )

        return GenerationResult(
            image=img,
            seed=request.seed,
            generation_time=round(generation_time, 2),
            width=img.width,
            height=img.height,
        )

    def is_loaded(self) -> bool:
        """Return True if the pipeline is loaded and ready."""
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self._model_name

    @property
    def model_id(self) -> str:
        """The Hugging Face checkpoint this backend loads."""
        return self._model_id

    def sizes_for(self, resolution: int) -> dict[str, tuple[int, int]]:
        """Size table for a target side length.

        2048 returns the model card's table; any other value applies the
        pipeline's calculate_dimensions rule with area = resolution^2 (the
        height comes from the unrounded width).
        """
        if resolution == 2048:
            return dict(self.CARD_SIZES_2048)
        sizes = {}
        for key, (rw, rh) in self.RATIO_FACTORS.items():
            ratio = rw / rh
            width = math.sqrt(resolution * resolution * ratio)
            height = width / ratio
            sizes[key] = (round(width / 32) * 32, round(height / 32) * 32)
        return sizes

    @property
    def aspect_ratios(self) -> dict[str, tuple[int, int]]:
        """Size table for the configured GENERATE_OUTPUT_RESOLUTION."""
        return self.sizes_for(self._generate_output_resolution)

    @property
    def default_inference_steps(self) -> int:
        return self._default_steps

    @property
    def min_inference_steps(self) -> int:
        return self._min_steps

    @property
    def max_inference_steps(self) -> int:
        return self._max_steps

    @property
    def default_cfg_scale(self) -> float:
        return 1.0

    @property
    def default_negative_prompt(self) -> str | None:
        return None

    @property
    def max_reference_images(self) -> int:
        return 10

    @property
    def edit_output_resolution(self) -> int:
        return self._edit_output_resolution

    @property
    def supports_transparency(self) -> bool:
        return True
