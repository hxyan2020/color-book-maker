"""CLI entry point for batch processing."""

from __future__ import annotations

import argparse

from color_book.pipeline import ColorBookOptions, create_color_book


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a photo into a color book page.")
    parser.add_argument("image_path", help="Path to the uploaded photo")
    parser.add_argument("--colors", type=int, default=6, help="Number of palette colors")
    parser.add_argument("--no-gemini-art", action="store_true", help="Skip Nano Banana line art")
    parser.add_argument("--no-gemini-palette", action="store_true", help="Skip Gemini palette analysis")
    parser.add_argument("--style", default="classic coloring book")
    parser.add_argument("--detail", default="medium", choices=["low", "medium", "high"])
    args = parser.parse_args()

    options = ColorBookOptions(
        num_colors=args.colors,
        use_gemini_line_art=not args.no_gemini_art,
        use_gemini_palette=not args.no_gemini_palette,
        style=args.style,
        detail=args.detail,
    )
    result = create_color_book(args.image_path, options)
    print(result.line_art_note)
    print(result.palette_note)
    print(result.palette_markdown)


if __name__ == "__main__":
    main()
