"""Draw bounding boxes on tb0392: one solid, one dashed, both #20c6bb.

Copies the source image first and annotates the copy; the original is
never modified.
"""

import shutil
from pathlib import Path

from PIL import Image, ImageDraw

SRC = Path.home() / "Downloads/tbx11k-simplified/images/tb0392.png"
COPY = SRC.with_name("tb0392_copy.png")
DST = SRC.with_name("tb0392_boxes.png")
COLOR = "#2cf1e4"
WIDTH = 3

# (xmin, ymin, width, height) -> pixel box (x0, y0, x1, y1)
solid = {"xmin": 69.79932403564453, "ymin": 82.55555725097656,
         "width": 145.48397827148438, "height": 289.066650390625}
dashed = {"xmin": 283.27655029296875, "ymin": 72.42222595214844,
          "width": 169.01181030273438, "height": 215.99998474121094}


def to_xyxy(b):
    x0, y0 = b["xmin"], b["ymin"]
    return (x0, y0, x0 + b["width"], y0 + b["height"])


def draw_dashed_rect(draw, box, color, width, dash=10, gap=7):
    x0, y0, x1, y1 = box

    def seg(p0, p1, horizontal):
        a, b = (p0[0], p1[0]) if horizontal else (p0[1], p1[1])
        pos = a
        while pos < b:
            end = min(pos + dash, b)
            if horizontal:
                draw.line([(pos, p0[1]), (end, p0[1])], fill=color, width=width)
            else:
                draw.line([(p0[0], pos), (p0[0], end)], fill=color, width=width)
            pos = end + gap

    seg((x0, y0), (x1, y0), True)   # top
    seg((x0, y1), (x1, y1), True)   # bottom
    seg((x0, y0), (x0, y1), False)  # left
    seg((x1, y0), (x1, y1), False)  # right


shutil.copy2(SRC, COPY)

img = Image.open(COPY).convert("RGB")
draw = ImageDraw.Draw(img)

draw.rectangle(to_xyxy(solid), outline=COLOR, width=WIDTH)
draw_dashed_rect(draw, to_xyxy(dashed), COLOR, WIDTH)

img.save(DST)
print(f"copied source -> {COPY.name}")
print(f"saved {DST.name} ({img.size[0]}x{img.size[1]})")
