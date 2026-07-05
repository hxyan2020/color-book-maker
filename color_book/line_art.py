from __future__ import annotations

import base64
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


COLORING_PROMPT = """Convert this photo into a clean, printable coloring book page.

Requirements:
- Pure black line art on a white background
- Bold, smooth outlines with clearly separated regions for coloring
- Preserve the main subject, pose, and composition from the reference photo
- Simplify busy textures into colorable shapes
- No shading, gradients, grayscale fills, hatching, or color
- No text, letters, numbers, logos, or watermarks
- Professional quality suitable for printing

Style: {style}
Detail level: {detail}
"""


def create_line_art_opencv(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    line_art = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )
    return line_art


def create_line_art_gemini(
    image_rgb: np.ndarray,
    style: str = "classic coloring book",
    detail: str = "medium",
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[np.ndarray | None, str]:
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "Gemini API key not set — using OpenCV line art."

    model = model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
    prompt = COLORING_PROMPT.format(style=style, detail=detail)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        image_bytes = _encode_png(image_rgb)

        interaction = client.interactions.create(
            model=model,
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                    "mime_type": "image/png",
                },
            ],
        )

        if not interaction.output_image or not interaction.output_image.data:
            return None, "Nano Banana did not return an image — using OpenCV fallback."

        generated = Image.open(BytesIO(base64.b64decode(interaction.output_image.data))).convert("RGB")
        line_art_rgb = np.array(generated)
        line_art_gray = cv2.cvtColor(line_art_rgb, cv2.COLOR_RGB2GRAY)
        _, cleaned = cv2.threshold(line_art_gray, 200, 255, cv2.THRESH_BINARY)
        return cleaned, f"Line art generated with Nano Banana ({model})."
    except Exception as exc:
        return None, f"Nano Banana unavailable ({exc}). Using OpenCV fallback."


def _encode_png(image_rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()
