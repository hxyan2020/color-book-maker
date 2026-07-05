from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv

from color_book.pipeline import ColorBookOptions, create_color_book

load_dotenv()

STYLE_CHOICES = [
    "classic coloring book",
    "cute cartoon",
    "detailed botanical",
    "simple kids",
    "elegant mandala-inspired",
]
DETAIL_CHOICES = ["low", "medium", "high"]


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise gr.Error("Please upload an image first.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def generate_color_book(
    uploaded_image: np.ndarray,
    num_colors: int,
    use_gemini_line_art: bool,
    use_gemini_palette: bool,
    style: str,
    detail: str,
):
    image_rgb = _to_rgb(uploaded_image)
    uploads_dir = Path("outputs/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    input_path = uploads_dir / "latest_upload.png"
    cv2.imwrite(str(input_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    options = ColorBookOptions(
        num_colors=int(num_colors),
        use_gemini_line_art=use_gemini_line_art,
        use_gemini_palette=use_gemini_palette,
        style=style,
        detail=detail,
        output_dir="outputs",
    )
    result = create_color_book(str(input_path), options)

    line_art_display = result.line_art
    if line_art_display.ndim == 2:
        line_art_display = cv2.cvtColor(line_art_display, cv2.COLOR_GRAY2RGB)

    status = "\n".join(
        [
            f"**Line art:** {result.line_art_note}",
            f"**Palette:** {result.palette_note}",
            "",
            "Files saved under `outputs/`.",
        ]
    )
    return (
        result.original_rgb,
        line_art_display,
        result.palette_strip,
        result.color_book_page,
        result.palette_markdown,
        status,
    )


def build_app() -> gr.Blocks:
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    key_hint = (
        "Gemini API key detected."
        if has_key
        else "Add `GEMINI_API_KEY` to a `.env` file (see `.env.example`) to enable Nano Banana line art and smart palettes."
    )

    with gr.Blocks(title="Color Book Maker") as demo:
        gr.Markdown(
            """
            # Color Book Maker
            Upload any photo to get a printable coloring page plus a recommended color palette.

            - **Nano Banana 2** (`gemini-3.1-flash-image`) converts your photo into clean line art
            - **Gemini vision** analyzes the scene and names each suggested color
            - **OpenCV + K-Means** provides fast local fallbacks when AI is unavailable
            """
        )
        gr.Markdown(key_hint)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload photo", type="numpy")
                num_colors = gr.Slider(3, 12, value=6, step=1, label="Palette size")
                style = gr.Dropdown(STYLE_CHOICES, value=STYLE_CHOICES[0], label="Line art style")
                detail = gr.Dropdown(DETAIL_CHOICES, value="medium", label="Detail level")
                use_gemini_line_art = gr.Checkbox(value=True, label="Use Nano Banana line art (Gemini)")
                use_gemini_palette = gr.Checkbox(value=True, label="Use Gemini palette recommendations")
                generate_btn = gr.Button("Create color book", variant="primary")
            with gr.Column(scale=2):
                with gr.Tab("Color book page"):
                    color_book_output = gr.Image(label="Printable color book page")
                with gr.Tab("Line art"):
                    line_art_output = gr.Image(label="Coloring page")
                with gr.Tab("Palette"):
                    palette_output = gr.Image(label="Palette strip")
                    palette_text = gr.Markdown()
                with gr.Tab("Original"):
                    original_output = gr.Image(label="Original upload")

        status_output = gr.Markdown()

        generate_btn.click(
            fn=generate_color_book,
            inputs=[
                image_input,
                num_colors,
                use_gemini_line_art,
                use_gemini_palette,
                style,
                detail,
            ],
            outputs=[
                original_output,
                line_art_output,
                palette_output,
                color_book_output,
                palette_text,
                status_output,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
