"""
Qwen-Image-Edit NSFW — RunPod Serverless Handler

Modes:
  text2image      — Pure text-to-image (0 input images)
  image_edit      — Edit image with text instruction (1 image)
  face_swap       — Swap face from source to target body (2 images)
  style_transfer  — Change style/outfit, keep identity (1 image)

Optimal settings:
  steps=4, cfg=1.0, true_cfg_scale=4.0
  sampler=euler_ancestral, scheduler=beta
"""

import runpod
import torch
import time
import traceback
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.model_manager import get_pipeline, set_runtime_loras
from src.image_utils import (
    decode_base64_image,
    encode_image_to_base64,
    normalize_dimensions,
    scale_to_megapixels,
)
from src.prompts import build_prompt, get_negative_prompt


# ─── Validation ───

VALID_JOB_TYPES = {"text2image", "image_edit", "face_swap", "style_transfer"}

VALID_SAMPLERS = {
    "euler", "euler_ancestral", "euler_a",
    "dpm++_2m", "dpm++_2m_sde",
    "sa_solver", "lcm", "er_sde",
}

VALID_SCHEDULERS = {
    "beta", "simple", "normal", "sgm_uniform",
    "karras", "exponential",
}


def validate_input(inp: dict) -> str | None:
    """Validate input, return error message or None if valid."""
    job_type = inp.get("job_type", "text2image")
    if job_type not in VALID_JOB_TYPES:
        return f"Invalid job_type: '{job_type}'. Must be one of: {sorted(VALID_JOB_TYPES)}"

    if job_type == "text2image" and not inp.get("prompt"):
        return "text2image requires a 'prompt'"

    if job_type == "image_edit":
        if not inp.get("image_base64"):
            return "image_edit requires 'image_base64'"
        if not inp.get("prompt"):
            return "image_edit requires a 'prompt'"

    if job_type == "face_swap":
        body = inp.get("body_base64") or inp.get("image_base64")
        face = inp.get("face_base64") or inp.get("image2_base64")
        if not body or not face:
            return "face_swap requires 'body_base64' and 'face_base64'"

    if job_type == "style_transfer":
        if not inp.get("image_base64"):
            return "style_transfer requires 'image_base64'"
        if not inp.get("prompt"):
            return "style_transfer requires a 'prompt'"

    return None


# ─── Image Loading ───

def load_images(inp: dict, job_type: str) -> list:
    """Load and prepare input images based on job type."""
    images = []

    if job_type == "face_swap":
        body_b64 = inp.get("body_base64") or inp.get("image_base64")
        face_b64 = inp.get("face_base64") or inp.get("image2_base64")
        body_img = decode_base64_image(body_b64)
        face_img = decode_base64_image(face_b64)
        # Scale down large images to avoid OOM
        body_img = scale_to_megapixels(body_img, 1.5)
        face_img = scale_to_megapixels(face_img, 1.5)
        images = [body_img, face_img]

    elif job_type in ("image_edit", "style_transfer"):
        img = decode_base64_image(inp["image_base64"])
        img = scale_to_megapixels(img, 1.5)
        images = [img]

    # text2image: no images
    return images


# ─── Main Handler ───

def handler(job):
    """RunPod serverless handler."""
    try:
        inp = job["input"]

        # Validate
        error = validate_input(inp)
        if error:
            return {"error": error}

        # Parse settings
        job_type = inp.get("job_type", "text2image")
        user_prompt = inp.get("prompt", "")
        width = inp.get("width", 1024)
        height = inp.get("height", 1024)
        steps = inp.get("steps", 4)
        cfg = inp.get("cfg", 1.0)
        true_cfg = inp.get("true_cfg_scale", 4.0)
        seed = inp.get("seed", -1)
        num_images = min(inp.get("num_images", 1), 4)  # Cap at 4
        nsfw_boost = inp.get("nsfw_boost", True)
        output_format = inp.get("output_format", "WEBP").upper()
        output_quality = inp.get("output_quality", 92)
        loras = inp.get("loras", [])

        # Normalize
        width, height = normalize_dimensions(width, height)
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        # Build prompt
        prompt = build_prompt(job_type, user_prompt, nsfw_boost=nsfw_boost)
        negative = get_negative_prompt(job_type)

        print(f"[JOB] type={job_type} size={width}x{height} steps={steps} cfg={cfg} seed={seed}")
        print(f"[JOB] prompt='{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")

        # Load model
        pipeline = get_pipeline()

        # Apply runtime LoRAs
        if loras:
            set_runtime_loras(pipeline, loras)

        # Load images
        images = load_images(inp, job_type)
        print(f"[JOB] {len(images)} input image(s)")

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative,
            "generator": torch.manual_seed(seed),
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "true_cfg_scale": true_cfg,
            "num_images_per_prompt": num_images,
            "height": height,
            "width": width,
        }

        if images:
            gen_kwargs["image"] = images

        # Generate
        start = time.time()
        with torch.inference_mode():
            output = pipeline(**gen_kwargs)
        elapsed = time.time() - start

        print(f"[JOB] Generated {len(output.images)} image(s) in {elapsed:.1f}s")

        # Encode output
        if num_images == 1:
            result_img = output.images[0]
            b64 = encode_image_to_base64(result_img, format=output_format, quality=output_quality)
            return {
                "generated_image_base64": b64,
                "image_base64": b64,
                "width": result_img.width,
                "height": result_img.height,
                "seed": seed,
                "job_type": job_type,
                "elapsed_seconds": round(elapsed, 2),
            }
        else:
            results = []
            for i, img in enumerate(output.images):
                results.append({
                    "image_base64": encode_image_to_base64(img, format=output_format, quality=output_quality),
                    "width": img.width,
                    "height": img.height,
                    "index": i,
                })
            return {
                "images": results,
                "seed": seed,
                "job_type": job_type,
                "count": len(results),
                "elapsed_seconds": round(elapsed, 2),
            }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try smaller dimensions or fewer images."}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# ─── Entry Point ───
runpod.serverless.start({"handler": handler})
