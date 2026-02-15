"""
Prompt templates and engineering for each job mode.
"""

# ─── Face Swap ───
FACE_SWAP_DEFAULT = (
    "head_swap: start with Picture 1 as the base image, keeping its pose "
    "and setting, replace the head with the one from Picture 2, maintaining "
    "all facial features, skin tone, and expression from Picture 2 while "
    "blending naturally with Picture 1's body"
)

FACE_SWAP_DETAILED = (
    "head swap from Picture 2 to Picture 1, keep all facial details and hair "
    "from Picture 2, blend naturally with Picture 1's body, match skin tone "
    "perfectly, maintain natural lighting and shadows"
)

# ─── Style Transfer ───
STYLE_TRANSFER_TEMPLATE = (
    "Transform this image: {prompt}. Keep the person's face, identity, "
    "and body proportions exactly the same. Only change the specified elements. "
    "Professional digital photography, detailed skin texture, natural lighting."
)

# ─── Image Edit ───
EDIT_NSFW_BOOST = (
    " Professional digital photography, detailed realistic skin texture, "
    "natural lighting, high quality, 8k resolution"
)

# ─── Text to Image ───
T2I_QUALITY_SUFFIX = (
    " Professional digital photography, highly detailed, sharp focus, "
    "natural skin texture, volumetric lighting, 8k resolution"
)


def build_prompt(job_type: str, user_prompt: str, nsfw_boost: bool = True) -> str:
    """Build the final prompt for generation."""

    if job_type == "face_swap":
        # Use user prompt if provided, otherwise default
        if user_prompt and user_prompt.strip():
            return user_prompt
        return FACE_SWAP_DEFAULT

    elif job_type == "style_transfer":
        if not user_prompt:
            return ""
        prompt = STYLE_TRANSFER_TEMPLATE.format(prompt=user_prompt)
        return prompt

    elif job_type == "image_edit":
        prompt = user_prompt or ""
        if nsfw_boost and prompt:
            prompt += EDIT_NSFW_BOOST
        return prompt

    elif job_type == "text2image":
        prompt = user_prompt or ""
        if nsfw_boost and prompt:
            prompt += T2I_QUALITY_SUFFIX
        return prompt

    return user_prompt or ""


def get_negative_prompt(job_type: str) -> str:
    """Get negative prompt for the job type."""
    # Qwen uses empty/space negative prompt with true_cfg_scale for guidance
    return " "
