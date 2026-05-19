#!/usr/bin/env python3
"""
Render thesis_card.html to a high-resolution PNG (and optionally a PDF).

Install once:
    pip install playwright
    playwright install chromium

Usage:
    python render_thesis_card.py                   # 1600x1000 @ 3x  -> 4800x3000 PNG
    python render_thesis_card.py --scale 4         # 4x DPI          -> 6400x4000 PNG
    python render_thesis_card.py --width 2000      # custom viewport
    python render_thesis_card.py --pdf             # also export PDF
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
HTML = HERE / "thesis_card.html"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale",  type=int, default=3, help="device pixel ratio (default: 3)")
    ap.add_argument("--width",  type=int, default=1600, help="viewport width  in CSS px (default: 1600)")
    ap.add_argument("--height", type=int, default=1000, help="viewport height in CSS px (default: 1000)")
    ap.add_argument("--out",    type=Path, default=HERE / "thesis_card.png", help="PNG output path")
    ap.add_argument("--pdf",    action="store_true", help="also export a PDF next to the PNG")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=args.scale,
        )
        page = ctx.new_page()
        page.goto(HTML.as_uri())
        page.wait_for_load_state("networkidle")

        page.screenshot(path=str(args.out))
        out_w, out_h = args.width * args.scale, args.height * args.scale
        print(f"PNG  saved : {args.out}  ({out_w}x{out_h} px)")

        if args.pdf:
            pdf_path = args.out.with_suffix(".pdf")
            page.pdf(
                path=str(pdf_path),
                width=f"{args.width}px",
                height=f"{args.height}px",
                print_background=True,
            )
            print(f"PDF  saved : {pdf_path}")

        browser.close()


if __name__ == "__main__":
    main()
