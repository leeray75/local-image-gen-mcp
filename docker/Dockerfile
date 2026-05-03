FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY server.py .
COPY models/ ./models/
COPY config/ ./config/
COPY storage/ ./storage/

# Install dependencies
RUN uv pip install --system --no-cache \
    "mcp[cli]>=1.27.0" \
    "pydantic>=2.11.0" \
    "pydantic-settings>=2.7.0" \
    "diffusers>=0.33.0" \
    "torch>=2.7.0" \
    "transformers>=4.52.0" \
    "accelerate>=1.7.0" \
    "safetensors>=0.5.3" \
    "pillow>=11.2.1" \
    "huggingface-hub>=0.32.0"

# Create output directory
RUN mkdir -p /outputs && chown appuser:appuser /outputs
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8500

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8500/mcp || exit 1

# Default environment variables
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8500
ENV MCP_PATH=/mcp
ENV IMAGE_OUTPUT_DIR=/outputs
ENV MAX_INFERENCE_STEPS=50
ENV DEFAULT_CFG_SCALE=4.0
ENV TORCH_DTYPE=bfloat16
ENV IMAGE_MODEL=qwen_image_2512

CMD ["python", "server.py"]