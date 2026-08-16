"""Scrub the fake stock-photo watermark flux-depth-pro renders into the frame.

Found during the 2026-08-16 web-UI implementation pass: three separate prompts
(two cityscape, one indoor/no-skyline) all produced a fake caption or URL in
the bottom margin of the render, regardless of prompt wording — a positive-
prompt exclusion clause did not suppress it (see pose_lift/prompt.py). Since
it isn't prompt-controllable, this crops the bottom margin band where the
artifact consistently lands and edge-extends the last clean row back to the
original height, so callers always get back the same image dimensions.
"""

import io

import numpy as np
from PIL import Image

MARGIN_FRACTION = 0.06


def scrub_bottom_margin(png_bytes: bytes, margin_fraction: float = MARGIN_FRACTION) -> bytes:
    """Replace the bottom margin_fraction of the image with the row above it, repeated.

    Returns PNG bytes at the same dimensions as the input.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img)
    h = arr.shape[0]
    margin_px = int(round(h * margin_fraction))
    if margin_px <= 0:
        return png_bytes

    last_clean_row = arr[h - margin_px - 1 : h - margin_px, :, :]
    arr[h - margin_px :, :, :] = np.repeat(last_clean_row, margin_px, axis=0)

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
