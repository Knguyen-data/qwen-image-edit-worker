FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/runpod-volume/huggingface

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

# Pre-download model at build time (optional — comment out to use network volume instead)
# RUN python3 -c "from diffusers import QwenImageEditPlusPipeline; \
#     p = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2511', torch_dtype='auto'); \
#     print('Model downloaded')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" || exit 1

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
