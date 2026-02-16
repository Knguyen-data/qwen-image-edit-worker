"""
Convert a ComfyUI-format Qwen-Image-Edit checkpoint to diffusers-compatible transformer weights.

ComfyUI bundles transformer + VAE + text encoder in one file with keys like:
  model.diffusion_model.* → transformer
  first_stage_model.*     → VAE
  cond_stage_model.*      → text encoder

We extract only the transformer weights, remap keys, and save in diffusers format.
"""

import sys
import os
import json
import re
from pathlib import Path
from safetensors.torch import load_file, save_file


def convert_comfyui_to_diffusers(
    comfyui_path: str,
    output_dir: str,
    base_config_dir: str = None,
):
    """
    Convert ComfyUI checkpoint to diffusers transformer format.
    
    Args:
        comfyui_path: Path to the ComfyUI .safetensors checkpoint
        output_dir: Directory to save the diffusers transformer
        base_config_dir: Path to base model config (for config.json)
    """
    print(f"[CONVERT] Loading ComfyUI checkpoint: {comfyui_path}")
    state_dict = load_file(comfyui_path)
    
    print(f"[CONVERT] Total keys: {len(state_dict)}")
    
    # Categorize keys
    transformer_keys = {}
    vae_keys = {}
    text_encoder_keys = {}
    other_keys = []
    
    for key, tensor in state_dict.items():
        if key.startswith("model.diffusion_model."):
            # Strip prefix for diffusers format
            new_key = key.replace("model.diffusion_model.", "")
            transformer_keys[new_key] = tensor
        elif key.startswith("first_stage_model."):
            new_key = key.replace("first_stage_model.", "")
            vae_keys[new_key] = tensor
        elif key.startswith("cond_stage_model."):
            new_key = key.replace("cond_stage_model.", "")
            text_encoder_keys[new_key] = tensor
        else:
            other_keys.append(key)
    
    print(f"[CONVERT] Transformer keys: {len(transformer_keys)}")
    print(f"[CONVERT] VAE keys: {len(vae_keys)}")
    print(f"[CONVERT] Text encoder keys: {len(text_encoder_keys)}")
    print(f"[CONVERT] Other keys: {len(other_keys)}")
    
    if other_keys:
        print(f"[CONVERT] Unmapped keys (sample): {other_keys[:5]}")
    
    # Save transformer
    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    
    # Split into shards if needed (diffusers convention)
    total_size = sum(t.numel() * t.element_size() for t in transformer_keys.values())
    print(f"[CONVERT] Transformer size: {total_size / 1e9:.1f} GB")
    
    # Save as single file for simplicity
    output_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors")
    save_file(transformer_keys, output_path)
    print(f"[CONVERT] Saved transformer to: {output_path}")
    
    # Copy config from base model if available
    if base_config_dir:
        base_config = os.path.join(base_config_dir, "transformer", "config.json")
        if os.path.exists(base_config):
            import shutil
            dest_config = os.path.join(transformer_dir, "config.json")
            shutil.copy2(base_config, dest_config)
            print(f"[CONVERT] Copied config from: {base_config}")
    
    # Save VAE if present
    if vae_keys:
        vae_dir = os.path.join(output_dir, "vae")
        os.makedirs(vae_dir, exist_ok=True)
        vae_path = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        save_file(vae_keys, vae_path)
        print(f"[CONVERT] Saved VAE to: {vae_path}")
        
        if base_config_dir:
            base_vae_config = os.path.join(base_config_dir, "vae", "config.json")
            if os.path.exists(base_vae_config):
                import shutil
                shutil.copy2(base_vae_config, os.path.join(vae_dir, "config.json"))
    
    # Save text encoder if present
    if text_encoder_keys:
        te_dir = os.path.join(output_dir, "text_encoder")
        os.makedirs(te_dir, exist_ok=True)
        te_path = os.path.join(te_dir, "model.safetensors")
        save_file(text_encoder_keys, te_path)
        print(f"[CONVERT] Saved text encoder to: {te_path}")
    
    print(f"[CONVERT] Conversion complete → {output_dir}")
    return {
        "transformer_keys": len(transformer_keys),
        "vae_keys": len(vae_keys),
        "text_encoder_keys": len(text_encoder_keys),
        "total_size_gb": total_size / 1e9,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_checkpoint.py <comfyui.safetensors> <output_dir> [base_config_dir]")
        sys.exit(1)
    
    comfyui_path = sys.argv[1]
    output_dir = sys.argv[2]
    base_config = sys.argv[3] if len(sys.argv) > 3 else None
    
    convert_comfyui_to_diffusers(comfyui_path, output_dir, base_config)
