"""Converts every generated plot under plots/output/ to JPG, mirroring the same relative
directory structure into a separate plots/output_jpg/ folder (so the original PNG/SVG folders are
left untouched).

SOURCE IS PNG, NOT SVG: every plot script in this suite already saves a PNG alongside its SVG
with the same base name, so rasterizing the PNG (trivial with Pillow, already a dependency here)
covers every figure without needing an SVG-to-raster library (e.g. cairosvg, not installed in
this environment and often finicky to install on Windows since it needs the system Cairo binary,
not just a pip package). If an SVG ever exists with no PNG sibling, it's reported at the end as
skipped rather than silently ignored -- convert that one by hand (e.g. open it and "export as PNG"
in a browser/Inkscape) or install cairosvg and rerun.

JPG has no alpha channel, so any transparent/RGBA PNG is flattened onto a plain white background
first (matplotlib's default savefig background is white anyway, so this matches what everything
already looks like on screen).

Edit SOURCE_DIR / OUT_DIR / JPEG_QUALITY below, then run:
    python plots/convert_to_jpg.py
"""
from __future__ import annotations

import os

from PIL import Image

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================
# EDIT THESE to point at a different source/output location or change JPEG quality.
# ================================================================================================
SOURCE_DIR = os.path.join(PLOTS_DIR, "output")
OUT_DIR = os.path.join(PLOTS_DIR, "output_jpg")
JPEG_QUALITY = 92  # 0-95; Pillow disallows 100 for baseline JPEG. 92 is visually lossless-ish.


def png_to_jpg(src_path: str, dst_path: str) -> None:
    with Image.open(src_path) as im:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[-1])
            im = background
        else:
            im = im.convert("RGB")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        im.save(dst_path, "JPEG", quality=JPEG_QUALITY)


def main() -> None:
    if not os.path.isdir(SOURCE_DIR):
        raise SystemExit(f"SOURCE_DIR does not exist: {SOURCE_DIR!r} -- edit it at the top of this file.")

    converted = 0
    skipped_svgs = []
    for root, _dirs, files in os.walk(SOURCE_DIR):
        for fname in files:
            lower = fname.lower()
            rel_dir = os.path.relpath(root, SOURCE_DIR)
            if lower.endswith(".png"):
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(OUT_DIR, rel_dir, os.path.splitext(fname)[0] + ".jpg")
                png_to_jpg(src_path, dst_path)
                converted += 1
            elif lower.endswith(".svg"):
                base = fname[:-4]
                if not os.path.isfile(os.path.join(root, base + ".png")):
                    skipped_svgs.append(os.path.join(root, fname))

    print(f"Converted {converted} PNG(s) to JPG under {OUT_DIR}")
    if skipped_svgs:
        print(f"\n{len(skipped_svgs)} SVG(s) had no PNG sibling and were skipped (no SVG rasterizer "
              "installed -- see module docstring):")
        for p in skipped_svgs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
