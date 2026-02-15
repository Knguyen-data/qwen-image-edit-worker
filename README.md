# Qwen-Image-Edit NSFW — RunPod Serverless Worker

NSFW-capable image generation, editing, face swap, and style transfer powered by **Qwen-Image-Edit-2511** with optional NSFW LoRAs.

## Features

| Mode | Description | Images Required |
|------|-------------|-----------------|
| `text2image` | Pure text-to-image generation | 0 |
| `image_edit` | Edit an image with text instructions | 1 |
| `face_swap` | Swap face from source onto target body | 2 (body + face) |
| `style_transfer` | Change style/outfit, keep identity | 1 |

## Architecture

```
qwen-image-edit-worker/
├── handler.py              # RunPod serverless entry point
├── src/
│   ├── model_manager.py    # Model loading, LoRA, caching
│   ├── image_utils.py      # Image encode/decode/resize
│   └── prompts.py          # Prompt templates per mode
├── scripts/
│   └── download_model.py   # Pre-download models to network volume
├── Dockerfile
├── requirements.txt
├── test_input.json         # Test payloads
└── README.md
```

## Model Options

### Option A: Official Qwen-Image-Edit-2511 (via diffusers)
- **Size:** ~20B params, ~40GB in bf16
- **NSFW:** Works (no safety checker), quality varies
- **LoRAs:** Load GNASS, Snofs, BFS at runtime for better NSFW

### Option B: Phr00t Rapid-AIO NSFW v23 (ComfyUI checkpoint)
- **Size:** 28.4 GB (FP8, all-in-one)
- **NSFW:** Full explicit, baked-in LoRAs
- **Use in:** ComfyUI locally or via ComfyUI-based worker
- **Download:** https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO

## Deployment

### RunPod Serverless (from GitHub)
1. Create a new Serverless endpoint on RunPod
2. Point to this repo: `https://github.com/Knguyen-data/qwen-image-edit-worker`
3. Set container disk: **50 GB** (model is ~40GB)
4. Set GPU: **A100 80GB** or **A6000 48GB** recommended
5. Set env vars (optional):
   - `NSFW_LORAS=gnass,snofs` — load NSFW LoRAs at startup
   - `MODEL_PRECISION=bf16` — `bf16` (quality) or `fp8` (speed)

### Docker Build
```bash
docker build -t qwen-nsfw-worker .
docker push <your-registry>/qwen-nsfw-worker
```

### Network Volume (recommended)
Pre-download models to avoid cold start:
```bash
python scripts/download_model.py --output /runpod-volume/models/
```
Then set env: `MODEL_CACHE_DIR=/runpod-volume/models/`

## API

### Request Schema
```json
{
  "input": {
    "job_type": "text2image",
    "prompt": "a woman in a red dress, professional photography",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "cfg": 1.0,
    "true_cfg_scale": 4.0,
    "seed": -1,
    "num_images": 1,
    "sampler": "euler_ancestral",
    "scheduler": "beta",
    "image_base64": "<optional base64>",
    "body_base64": "<for face_swap>",
    "face_base64": "<for face_swap>",
    "loras": [
      {"name": "gnass_nsfw", "weight": 0.8}
    ]
  }
}
```

### Response Schema
```json
{
  "generated_image_base64": "<base64 webp>",
  "image_base64": "<base64 webp>",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "job_type": "text2image",
  "elapsed_seconds": 3.2
}
```

## Optimal Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Steps | 4 | With Lightning acceleration |
| CFG | 1.0 | Higher = more prompt adherence, less quality |
| True CFG Scale | 4.0 | Controls negative prompt strength |
| Sampler | euler_ancestral | Best for this model |
| Scheduler | beta | Required for Qwen architecture |
| Denoise | 1.0 | Full reconstruction |

## Face Swap Input Order

⚠️ **CRITICAL:** BFS LoRA uses **inverted order**:
- `body_base64` = Target body (pose/scene you want)
- `face_base64` = Source face (identity to keep)

## License
Apache 2.0 (same as Qwen-Image-Edit-2511)
