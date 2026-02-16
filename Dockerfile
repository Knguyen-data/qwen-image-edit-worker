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

# Pre-download model into Docker image (~57GB)
# This makes cold starts instant — no network volume needed
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen-Image-Edit-2511', local_dir='/models/qwen-image-edit-2511', local_dir_use_symlinks=False); \
print('Qwen-Image-Edit-2511 downloaded to /models/qwen-image-edit-2511')"

# Verify model files exist
RUN ls -la /models/qwen-image-edit-2511/ && \
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" || exit 1

# RunPod serverless listens on 8000
EXPOSE 8000

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
