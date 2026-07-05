from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


@dataclass
class PaletteColor:
    name: str
    rgb: tuple[int, int, int]
    hex: str
    usage: str
    source: str = "kmeans"


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def extract_palette_kmeans(image_rgb: np.ndarray, num_colors: int = 6) -> list[PaletteColor]:
    pixels = image_rgb.reshape((-1, 3)).astype(np.float32)
    sample_size = min(len(pixels), 50_000)
    if len(pixels) > sample_size:
        indices = np.random.default_rng(42).choice(len(pixels), sample_size, replace=False)
        sample = pixels[indices]
    else:
        sample = pixels

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(sample)
    centers = kmeans.cluster_centers_.astype(int)

    colors: list[PaletteColor] = []
    for idx, center in enumerate(centers):
        rgb = (int(center[0]), int(center[1]), int(center[2]))
        colors.append(
            PaletteColor(
                name=f"Color {idx + 1}",
                rgb=rgb,
                hex=_rgb_to_hex(rgb),
                usage="Suggested fill for a region in the coloring page.",
                source="kmeans",
            )
        )
    return colors


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _merge_gemini_palette(
    kmeans_colors: list[PaletteColor],
    gemini_payload: dict[str, Any],
) -> list[PaletteColor]:
    raw_colors = gemini_payload.get("colors") or gemini_payload.get("palette") or []
    if not isinstance(raw_colors, list) or not raw_colors:
        return kmeans_colors

    merged: list[PaletteColor] = []
    for idx, item in enumerate(raw_colors):
        fallback = kmeans_colors[idx] if idx < len(kmeans_colors) else kmeans_colors[-1]
        if not isinstance(item, dict):
            merged.append(fallback)
            continue

        rgb = item.get("rgb")
        if isinstance(rgb, list) and len(rgb) >= 3:
            rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        else:
            rgb_tuple = fallback.rgb

        hex_value = item.get("hex") or _rgb_to_hex(rgb_tuple)
        merged.append(
            PaletteColor(
                name=str(item.get("name") or fallback.name),
                rgb=rgb_tuple,
                hex=str(hex_value),
                usage=str(item.get("usage") or item.get("where_to_use") or fallback.usage),
                source="gemini",
            )
        )

    if len(merged) < len(kmeans_colors):
        merged.extend(kmeans_colors[len(merged) :])
    return merged[: len(kmeans_colors)]


def analyze_palette_with_gemini(
    image_rgb: np.ndarray,
    kmeans_colors: list[PaletteColor],
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[list[PaletteColor], str]:
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return kmeans_colors, "Gemini API key not set — using K-Means palette only."

    model = model or os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    color_lines = "\n".join(
        f"- {color.hex} rgb{color.rgb}" for color in kmeans_colors
    )

    prompt = f"""You are a professional coloring-book art director.

Analyze the uploaded photo and recommend a {len(kmeans_colors)}-color palette for coloring this scene.
Use the dominant colors below as anchors, but refine them into a harmonious palette suitable for colored pencils or markers.

Dominant colors from the photo:
{color_lines}

Return ONLY valid JSON in this shape:
{{
  "title": "short palette title",
  "description": "1-2 sentence overview of the mood",
  "colors": [
    {{
      "name": "Forest Green",
      "hex": "#2d6a4f",
      "rgb": [45, 106, 79],
      "usage": "Use for leaves and background foliage"
    }}
  ]
}}"""

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        pil_image = Image.fromarray(image_rgb)
        response = client.models.generate_content(
            model=model,
            contents=[pil_image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json",
            ),
        )
        text = response.text or ""
        payload = _parse_json_payload(text)
        if not payload:
            return kmeans_colors, "Gemini returned an unreadable palette response — using K-Means colors."

        merged = _merge_gemini_palette(kmeans_colors, payload)
        summary = payload.get("description") or payload.get("title") or "Gemini-enhanced palette ready."
        return merged, str(summary)
    except Exception as exc:
        return kmeans_colors, f"Gemini palette analysis unavailable ({exc}). Using K-Means colors."


def render_palette_strip(colors: list[PaletteColor], swatch_size: int = 80) -> np.ndarray:
    strip = np.zeros((swatch_size, swatch_size * len(colors), 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        strip[:, index * swatch_size : (index + 1) * swatch_size] = color.rgb
    return strip


def palette_to_markdown(colors: list[PaletteColor], summary: str = "") -> str:
    lines = ["### Recommended color palette"]
    if summary:
        lines.append(summary)
    lines.append("")
    for color in colors:
        lines.append(
            f"- **{color.name}** `{color.hex}` — {color.usage}"
        )
    return "\n".join(lines)
