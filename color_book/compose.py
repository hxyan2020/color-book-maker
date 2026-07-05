from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from color_book.palette import PaletteColor, render_palette_strip


@dataclass
class ColorBookResult:
    original_rgb: np.ndarray
    line_art: np.ndarray
    palette_colors: list[PaletteColor]
    palette_strip: np.ndarray
    color_book_page: np.ndarray
    palette_markdown: str
    line_art_note: str
    palette_note: str
    output_dir: Path


def compose_color_book_page(
    line_art: np.ndarray,
    palette_colors: list[PaletteColor],
    page_width: int = 1200,
) -> np.ndarray:
    if line_art.ndim == 2:
        line_art_rgb = cv2.cvtColor(line_art, cv2.COLOR_GRAY2RGB)
    else:
        line_art_rgb = line_art

    art = _fit_within(line_art_rgb, page_width - 120, max_height=900)
    palette_strip = render_palette_strip(palette_colors, swatch_size=70)

    footer_height = 220
    page = np.full((art.shape[0] + footer_height + 80, page_width, 3), 255, dtype=np.uint8)
    x_offset = (page_width - art.shape[1]) // 2
    page[40 : 40 + art.shape[0], x_offset : x_offset + art.shape[1]] = art

    footer_y = 40 + art.shape[0] + 30
    strip = _fit_within(palette_strip, page_width - 120, max_height=70)
    strip_x = (page_width - strip.shape[1]) // 2
    page[footer_y : footer_y + strip.shape[0], strip_x : strip_x + strip.shape[1]] = strip

    page_pil = Image.fromarray(page)
    draw = ImageDraw.Draw(page_pil)
    font = ImageFont.load_default()
    label_y = footer_y + strip.shape[0] + 12
    slot_width = strip.shape[1] // max(len(palette_colors), 1)

    for index, color in enumerate(palette_colors):
        label = f"{color.name} {color.hex}"
        text_x = strip_x + index * slot_width + 4
        draw.text((text_x, label_y), label, fill=(30, 30, 30), font=font)

    return np.array(page_pil)


def _fit_within(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def save_outputs(result: ColorBookResult, stem: str) -> dict[str, str]:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "line_art": str(result.output_dir / f"{stem}_line_art.png"),
        "palette": str(result.output_dir / f"{stem}_palette.png"),
        "color_book": str(result.output_dir / f"{stem}_color_book.png"),
        "original": str(result.output_dir / f"{stem}_original.png"),
    }

    cv2.imwrite(paths["original"], cv2.cvtColor(result.original_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["line_art"], result.line_art)
    cv2.imwrite(paths["palette"], cv2.cvtColor(result.palette_strip, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["color_book"], cv2.cvtColor(result.color_book_page, cv2.COLOR_RGB2BGR))
    return paths
