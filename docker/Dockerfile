# Upgraded to the latest version to properly support your Blackwell GPU
FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /app

COPY pyproject.toml .
COPY server.py .
COPY models/ ./models/
COPY config/ ./config/
COPY storage/ ./storage/

# Install dependencies directly to system Python to use NVIDIA's optimized torch/CUDA
RUN PIP_CONSTRAINT="" pip install --no-cache-dir \
    "mcp[cli]>=1.27.0" \
    "pydantic>=2.11.0" \
    "pydantic-settings>=2.7.0" \
    "diffusers>=0.33.0" \
    "transformers>=4.52.0" \
    "accelerate>=1.7.0" \
    "safetensors>=0.5.3" \
    "pillow==11.1.0" \
    "huggingface-hub>=0.32.0"

RUN mkdir -p /outputs

EXPOSE 8500

# Healthcheck on TCP port — the /mcp endpoint returns 406 without MCP headers
# so curl -f always fails. TCP connect confirms the server is actually listening.
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD python3 -c "import socket; s=socket.socket(); s.connect(('localhost', 8500)); s.close()"

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8500
ENV MCP_PATH=/mcp
ENV IMAGE_OUTPUT_DIR=/outputs
ENV MAX_INFERENCE_STEPS=50
ENV DEFAULT_CFG_SCALE=4.0
ENV TORCH_DTYPE=bfloat16
ENV IMAGE_MODEL=qwen_image_2512

CMD ["python", "server.py"]