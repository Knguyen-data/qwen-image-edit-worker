"""
Model Manager — handles loading, caching, and LoRA management for Qwen-Image-Edit.
"""

import os
import time
import torch
from typing import Optional

# Globals
_pipeline = None
_loaded_loras: list[str] = []

# Config from env
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/models")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
MODEL_PRECISION = os.environ.get("MODEL_PRECISION", "bf16")
NSFW_LORAS_ENV = os.environ.get("NSFW_LORAS", "")  # comma-separated LoRA names

# Known NSFW LoRAs (HuggingFace paths)
LORA_REGISTRY = {
    "gnass": {
        "repo": "gnass-org/GNASS-Qwen-Edit-NSFW",
        "filename": "gnass_qwen_edit_nsfw.safetensors",
        "default_weight": 0.7,
    },
    "bfs_face_v5": {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "filename": "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        "default_weight": 1.0,
    },
    "lightning_4step": {
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "default_weight": 1.0,
    },
}


def get_torch_dtype():
    """Get torch dtype from config."""
    if MODEL_PRECISION == "fp8":
        return torch.float8_e4m3fn
    elif MODEL_PRECISION == "fp16":
        return torch.float16
    return torch.bfloat16


def get_pipeline():
    """Get or load the pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline()
    return _pipeline


def _load_pipeline():
    """Load the Qwen-Image-Edit pipeline."""
    from diffusers import QwenImageEditPlusPipeline

    print(f"[MODEL] Loading {MODEL_ID} (precision={MODEL_PRECISION})...")
    start = time.time()

    dtype = get_torch_dtype()

    # Check for cached model on network volume
    local_path = os.path.join(MODEL_CACHE_DIR, "qwen-image-edit-2511")
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"[MODEL] Loading from cache: {local_path}")
        source = local_path
    else:
        print(f"[MODEL] Downloading from HuggingFace: {MODEL_ID}")
        source = MODEL_ID

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        source,
        torch_dtype=dtype,
        cache_dir=MODEL_CACHE_DIR if source == MODEL_ID else None,
    )

    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)

    # Disable any safety filters
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if hasattr(pipeline, "feature_extractor"):
        pipeline.feature_extractor = None
    if hasattr(pipeline, "watermarker"):
        pipeline.watermarker = None

    elapsed = time.time() - start
    print(f"[MODEL] Pipeline loaded in {elapsed:.1f}s")

    # Load default NSFW LoRAs from env
    if NSFW_LORAS_ENV:
        lora_names = [l.strip() for l in NSFW_LORAS_ENV.split(",") if l.strip()]
        for name in lora_names:
            load_lora(pipeline, name)

    return pipeline


def load_lora(pipeline, lora_name: str, weight: Optional[float] = None):
    """Load a LoRA by name from the registry."""
    global _loaded_loras

    if lora_name in _loaded_loras:
        print(f"[LORA] {lora_name} already loaded, skipping")
        return

    if lora_name not in LORA_REGISTRY:
        print(f"[LORA] Unknown LoRA: {lora_name}. Available: {list(LORA_REGISTRY.keys())}")
        return

    info = LORA_REGISTRY[lora_name]
    w = weight or info["default_weight"]

    print(f"[LORA] Loading {lora_name} (weight={w}) from {info['repo']}...")
    start = time.time()

    try:
        # Check network volume cache first
        local_lora = os.path.join(MODEL_CACHE_DIR, "loras", info["filename"])
        if os.path.exists(local_lora):
            pipeline.load_lora_weights(local_lora, adapter_name=lora_name)
        else:
            pipeline.load_lora_weights(
                info["repo"],
                weight_name=info["filename"],
                adapter_name=lora_name,
            )

        pipeline.set_adapters([lora_name], adapter_weights=[w])
        _loaded_loras.append(lora_name)

        elapsed = time.time() - start
        print(f"[LORA] {lora_name} loaded in {elapsed:.1f}s")

    except Exception as e:
        print(f"[LORA] Failed to load {lora_name}: {e}")


def set_runtime_loras(pipeline, loras: list[dict]):
    """
    Set LoRAs for a specific generation.
    loras: [{"name": "gnass", "weight": 0.8}, ...]
    """
    if not loras:
        return

    for lora in loras:
        name = lora.get("name", "")
        weight = lora.get("weight")
        if name and name not in _loaded_loras:
            load_lora(pipeline, name, weight)

    # Set all active adapters
    names = [l["name"] for l in loras if l["name"] in _loaded_loras]
    weights = [l.get("weight", LORA_REGISTRY.get(l["name"], {}).get("default_weight", 1.0)) for l in loras if l["name"] in _loaded_loras]

    if names:
        pipeline.set_adapters(names, adapter_weights=weights)
        print(f"[LORA] Active: {dict(zip(names, weights))}")
