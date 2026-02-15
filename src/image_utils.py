"""
Image utilities — encode, decode, resize, validate.
"""

import base64
import io
from PIL import Image


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode a base64 string to PIL Image. Handles data URI prefix."""
    if not b64_string:
        raise ValueError("Empty base64 string")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def encode_image_to_base64(
    img: Image.Image,
    format: str = "WEBP",
    quality: int = 92,
) -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    save_kwargs = {"format": format}
    if format.upper() in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
    img.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_dimensions(width: int, height: int) -> tuple[int, int]:
    """
    Ensure dimensions are:
    - Multiples of 16 (required by VAE)
    - Within reasonable bounds (512-2048)
    - Total pixels <= ~2MP (for VRAM safety)
    """
    width = max(512, min(2048, width))
    height = max(512, min(2048, height))

    # Round to nearest multiple of 16
    width = (width // 16) * 16
    height = (height // 16) * 16

    # Cap total pixels at ~2MP
    total = width * height
    max_pixels = 2_097_152  # 2MP
    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        width = int(width * scale) // 16 * 16
        height = int(height * scale) // 16 * 16

    return width, height


def resize_to_target(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to target dimensions, maintaining aspect ratio with padding."""
    if img.width == width and img.height == height:
        return img

    # Calculate scale to fit within target
    scale = min(width / img.width, height / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)

    # Resize with high quality
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # If exact match needed, center-crop or pad
    if new_w != width or new_h != height:
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        paste_x = (width - new_w) // 2
        paste_y = (height - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas

    return resized


def scale_to_megapixels(img: Image.Image, megapixels: float = 1.0) -> Image.Image:
    """Scale image to approximately N megapixels while keeping aspect ratio."""
    current_mp = (img.width * img.height) / 1_000_000
    if current_mp <= megapixels * 1.1:  # 10% tolerance
        return img

    scale = (megapixels / current_mp) ** 0.5
    new_w = int(img.width * scale) // 16 * 16
    new_h = int(img.height * scale) // 16 * 16

    return img.resize((new_w, new_h), Image.LANCZOS)
