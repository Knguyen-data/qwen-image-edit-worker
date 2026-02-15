"""
Pre-download models to a network volume for faster cold starts.

Usage:
    python scripts/download_model.py --output /runpod-volume/models/
"""

import argparse
import os


def download_model(output_dir: str):
    from diffusers import QwenImageEditPlusPipeline
    from huggingface_hub import hf_hub_download
    import torch

    model_dir = os.path.join(output_dir, "qwen-image-edit-2511")
    lora_dir = os.path.join(output_dir, "loras")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)

    # Download main model
    print(f"[DL] Downloading Qwen-Image-Edit-2511 to {model_dir}...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        cache_dir=model_dir,
    )
    pipeline.save_pretrained(model_dir)
    print(f"[DL] Model saved to {model_dir}")
    del pipeline

    # Download NSFW LoRAs
    loras = [
        {
            "repo": "Alissonerdx/BFS-Best-Face-Swap",
            "filename": "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        },
        {
            "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        },
    ]

    for lora in loras:
        print(f"[DL] Downloading LoRA: {lora['repo']} / {lora['filename']}...")
        try:
            path = hf_hub_download(
                repo_id=lora["repo"],
                filename=lora["filename"],
                local_dir=lora_dir,
            )
            print(f"[DL] Saved: {path}")
        except Exception as e:
            print(f"[DL] Failed: {e}")

    print("[DL] All downloads complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen models to network volume")
    parser.add_argument("--output", type=str, default="/runpod-volume/models/",
                        help="Output directory for models")
    args = parser.parse_args()
    download_model(args.output)
