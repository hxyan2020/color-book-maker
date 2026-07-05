from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from color_book.compose import ColorBookResult, compose_color_book_page, save_outputs
from color_book.line_art import create_line_art_gemini, create_line_art_opencv
from color_book.palette import (
    analyze_palette_with_gemini,
    extract_palette_kmeans,
    palette_to_markdown,
    render_palette_strip,
)


@dataclass
class ColorBookOptions:
    num_colors: int = 6
    use_gemini_line_art: bool = True
    use_gemini_palette: bool = True
    style: str = "classic coloring book"
    detail: str = "medium"
    output_dir: str = "outputs"


def create_color_book(image_path: str, options: ColorBookOptions | None = None) -> ColorBookResult:
    options = options or ColorBookOptions()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    kmeans_colors = extract_palette_kmeans(image_rgb, num_colors=options.num_colors)

    if options.use_gemini_palette:
        palette_colors, palette_note = analyze_palette_with_gemini(image_rgb, kmeans_colors)
    else:
        palette_colors, palette_note = kmeans_colors, "Local K-Means palette."

    line_art_note = "OpenCV adaptive threshold line art."
    line_art = create_line_art_opencv(image_bgr)

    if options.use_gemini_line_art:
        gemini_line_art, gemini_note = create_line_art_gemini(
            image_rgb,
            style=options.style,
            detail=options.detail,
        )
        line_art_note = gemini_note
        if gemini_line_art is not None:
            line_art = gemini_line_art

    palette_strip = render_palette_strip(palette_colors)
    color_book_page = compose_color_book_page(line_art, palette_colors)
    palette_markdown = palette_to_markdown(palette_colors, summary=palette_note)

    stem = Path(image_path).stem
    output_dir = Path(options.output_dir)
    result = ColorBookResult(
        original_rgb=image_rgb,
        line_art=line_art,
        palette_colors=palette_colors,
        palette_strip=palette_strip,
        color_book_page=color_book_page,
        palette_markdown=palette_markdown,
        line_art_note=line_art_note,
        palette_note=palette_note,
        output_dir=output_dir,
    )
    save_outputs(result, stem)
    return result
