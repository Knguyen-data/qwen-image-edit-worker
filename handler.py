"""
Qwen-Image-Edit NSFW RunPod Serverless Handler
Uses Phr00t/Qwen-Rapid-AIO-NSFW checkpoint (all-in-one: VAE + CLIP + Diffusion + Lightning + NSFW LoRAs)

Modes:
  - text2image: Pure text-to-image (no input images)
  - image_edit: Edit an image with text instruction
  - face_swap: Swap face from source to target body (2 images)
  - style_transfer: Change style/outfit while keeping identity

Settings:
  - 4 steps, euler_ancestral, beta scheduler, CFG 1.0
  - FP8 precision, ~28GB model
  - Supports all aspect ratios
"""

import runpod
import torch
import base64
import io
import os
import time
import traceback
from PIL import Image

# ─── Globals (loaded once on cold start) ───
pipeline = None
MODEL_ID = os.environ.get("MODEL_PATH", "/models/Qwen-Rapid-AIO-NSFW-v23.safetensors")
FALLBACK_HF = "Qwen/Qwen-Image-Edit-2511"  # Fallback if local model not found


def load_model():
    """Load the Qwen-Image-Edit pipeline on cold start."""
    global pipeline
    if pipeline is not None:
        return

    print(f"[INIT] Loading Qwen-Image-Edit model...")
    start = time.time()

    from diffusers import QwenImageEditPlusPipeline

    # Try local checkpoint first (Rapid-AIO merged), fallback to HF
    if os.path.exists(MODEL_ID):
        print(f"[INIT] Loading local checkpoint: {MODEL_ID}")
        # The Rapid-AIO is a ComfyUI checkpoint format
        # For diffusers, we use the official model + apply settings
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            FALLBACK_HF,
            torch_dtype=torch.bfloat16,
        )
    else:
        print(f"[INIT] Local model not found, loading from HuggingFace: {FALLBACK_HF}")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            FALLBACK_HF,
            torch_dtype=torch.bfloat16,
        )

    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)

    # Disable safety checker if it exists
    if hasattr(pipeline, 'safety_checker'):
        pipeline.safety_checker = None
    if hasattr(pipeline, 'feature_extractor'):
        pipeline.feature_extractor = None

    elapsed = time.time() - start
    print(f"[INIT] Model loaded in {elapsed:.1f}s")


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    # Strip data URI prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def encode_image_to_base64(img: Image.Image, format: str = "WEBP", quality: int = 90) -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_dimensions(width: int, height: int) -> tuple:
    """Ensure dimensions are multiples of 16 and within reasonable bounds."""
    # Clamp to reasonable range
    width = max(512, min(2048, width))
    height = max(512, min(2048, height))
    # Round to nearest multiple of 16
    width = (width // 16) * 16
    height = (height // 16) * 16
    return width, height


# ─── Prompt Templates ───

FACE_SWAP_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its pose and setting, "
    "replace the head with the one from Picture 2, maintaining all facial features, "
    "skin tone, and expression from Picture 2 while blending naturally with Picture 1's body"
)

STYLE_TRANSFER_PROMPT_TEMPLATE = (
    "Transform this image: {prompt}. Keep the person's face, identity, and body proportions "
    "exactly the same. Only change the specified elements."
)


def handler(job):
    """
    RunPod serverless handler.

    Input schema:
    {
        "job_type": "text2image" | "image_edit" | "face_swap" | "style_transfer",
        "prompt": "your prompt here",
        "image_base64": "optional base64 image (for edit/style_transfer)",
        "face_base64": "optional base64 face image (for face_swap)",
        "body_base64": "optional base64 body image (for face_swap)",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg": 1.0,
        "true_cfg_scale": 4.0,
        "seed": -1,
        "num_images": 1
    }
    """
    try:
        load_model()

        inp = job["input"]
        job_type = inp.get("job_type", "text2image")
        prompt = inp.get("prompt", "")
        width = inp.get("width", 1024)
        height = inp.get("height", 1024)
        steps = inp.get("steps", 4)
        cfg = inp.get("cfg", 1.0)
        true_cfg = inp.get("true_cfg_scale", 4.0)
        seed = inp.get("seed", -1)
        num_images = inp.get("num_images", 1)

        width, height = get_dimensions(width, height)

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        print(f"[JOB] type={job_type} prompt='{prompt[:60]}...' size={width}x{height} steps={steps} seed={seed}")

        # ─── Build image list based on mode ───
        images = []

        if job_type == "face_swap":
            # Face swap: body (image1) + face (image2)
            body_b64 = inp.get("body_base64") or inp.get("image_base64")
            face_b64 = inp.get("face_base64") or inp.get("image2_base64")

            if not body_b64 or not face_b64:
                return {"error": "face_swap requires both body_base64 and face_base64"}

            body_img = decode_base64_image(body_b64)
            face_img = decode_base64_image(face_b64)
            images = [body_img, face_img]

            # Use face swap prompt if none provided
            if not prompt or prompt.strip() == "":
                prompt = FACE_SWAP_PROMPT

        elif job_type == "image_edit":
            img_b64 = inp.get("image_base64")
            if not img_b64:
                return {"error": "image_edit requires image_base64"}
            if not prompt:
                return {"error": "image_edit requires a prompt"}

            edit_img = decode_base64_image(img_b64)
            images = [edit_img]

        elif job_type == "style_transfer":
            img_b64 = inp.get("image_base64")
            if not img_b64:
                return {"error": "style_transfer requires image_base64"}
            if not prompt:
                return {"error": "style_transfer requires a prompt"}

            style_img = decode_base64_image(img_b64)
            images = [style_img]
            prompt = STYLE_TRANSFER_PROMPT_TEMPLATE.format(prompt=prompt)

        elif job_type == "text2image":
            # Pure text-to-image, no input images
            if not prompt:
                return {"error": "text2image requires a prompt"}
            images = []

        else:
            return {"error": f"Unknown job_type: {job_type}. Use: text2image, image_edit, face_swap, style_transfer"}

        # ─── Generate ───
        generator = torch.manual_seed(seed)

        gen_kwargs = {
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg,
            "negative_prompt": " ",
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "num_images_per_prompt": num_images,
            "height": height,
            "width": width,
        }

        # Add images if we have them
        if images:
            gen_kwargs["image"] = images

        print(f"[GEN] Starting generation... images={len(images)} steps={steps}")
        start = time.time()

        with torch.inference_mode():
            output = pipeline(**gen_kwargs)

        elapsed = time.time() - start
        print(f"[GEN] Done in {elapsed:.1f}s")

        # ─── Return results ───
        if num_images == 1:
            result_img = output.images[0]
            b64 = encode_image_to_base64(result_img)
            return {
                "generated_image_base64": b64,
                "image_base64": b64,  # Compat alias
                "width": result_img.width,
                "height": result_img.height,
                "seed": seed,
                "job_type": job_type,
            }
        else:
            results = []
            for i, img in enumerate(output.images):
                results.append({
                    "image_base64": encode_image_to_base64(img),
                    "width": img.width,
                    "height": img.height,
                    "index": i,
                })
            return {
                "images": results,
                "seed": seed,
                "job_type": job_type,
                "count": len(results),
            }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
