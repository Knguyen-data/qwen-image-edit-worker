FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/models/cache
ENV MODEL_CACHE_DIR=/models

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git wget curl libgl1-mesa-glx libglib2.0-0 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install -r requirements.txt && \
    pip3 install git+https://github.com/huggingface/diffusers

# Copy worker code
COPY handler.py /app/handler.py
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Step 1: Download base Qwen-Image-Edit-2511 configs only (scheduler, tokenizer, etc.)
# We skip the transformer weights — we'll use Phr00t's NSFW merge instead
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen-Image-Edit-2511', \
    local_dir='/models/qwen-base', \
    local_dir_use_symlinks=False, \
    ignore_patterns=['transformer/*.safetensors', 'transformer/*.bin']); \
print('Base configs downloaded (no transformer weights)')"

# Step 2: Download Phr00t v23 NSFW checkpoint (28.4GB)
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('Phr00t/Qwen-Image-Edit-Rapid-AIO', \
    filename='v23/Qwen-Rapid-AIO-NSFW-v23.safetensors', \
    local_dir='/models/phr00t', \
    local_dir_use_symlinks=False); \
print('Phr00t v23 NSFW downloaded')"

# Step 3: Convert ComfyUI checkpoint → diffusers transformer format
RUN python3 /app/scripts/convert_checkpoint.py \
    /models/phr00t/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors \
    /models/qwen-nsfw \
    /models/qwen-base

# Step 4: Copy base configs into the NSFW model dir
RUN cp -r /models/qwen-base/scheduler /models/qwen-nsfw/ 2>/dev/null || true && \
    cp -r /models/qwen-base/tokenizer /models/qwen-nsfw/ 2>/dev/null || true && \
    cp -r /models/qwen-base/tokenizer_2 /models/qwen-nsfw/ 2>/dev/null || true && \
    cp -r /models/qwen-base/text_encoder /models/qwen-nsfw/ 2>/dev/null || true && \
    cp -r /models/qwen-base/text_encoder_2 /models/qwen-nsfw/ 2>/dev/null || true && \
    cp /models/qwen-base/model_index.json /models/qwen-nsfw/ 2>/dev/null || true && \
    echo "Configs copied to /models/qwen-nsfw"

# Step 5: If conversion didn't extract VAE, copy from base
RUN if [ ! -f /models/qwen-nsfw/vae/diffusion_pytorch_model.safetensors ]; then \
        cp -r /models/qwen-base/vae /models/qwen-nsfw/ 2>/dev/null || true; \
        echo "VAE copied from base"; \
    else \
        echo "VAE extracted from Phr00t checkpoint"; \
    fi

# Cleanup: remove raw checkpoint and base transformer to save space
RUN rm -rf /models/phr00t /models/qwen-base/transformer && \
    echo "Cleanup done"

# Verify
RUN ls -la /models/qwen-nsfw/ && \
    ls -la /models/qwen-nsfw/transformer/ && \
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" || exit 1

EXPOSE 8000

CMD ["python3", "-u", "/app/handler.py"]
