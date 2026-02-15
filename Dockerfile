FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
    runpod \
    Pillow \
    accelerate \
    transformers \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface_hub && \
    pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers

# Download the model at build time (cached in Docker layer)
# Using official Qwen-Image-Edit-2511 via diffusers (handles NSFW without filters)
RUN python3 -c "from diffusers import QwenImageEditPlusPipeline; \
    QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2511', torch_dtype='auto')" || true

WORKDIR /app
COPY handler.py /app/handler.py

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
