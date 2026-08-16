import io

import numpy as np
import pytest
from PIL import Image

from pose_lift.watermark_guard import scrub_bottom_margin


def _png_bytes(h, w, bottom_stripe_value=None):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, :] = 100  # uniform "clean" body
    if bottom_stripe_value is not None:
        stripe = int(round(h * 0.03))
        arr[h - stripe :, :, :] = bottom_stripe_value
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_output_dimensions_match_input():
    src = _png_bytes(400, 300)
    out = scrub_bottom_margin(src)
    img = Image.open(io.BytesIO(out))
    assert img.size == (300, 400)


def test_bottom_margin_no_longer_contains_the_stray_pixels():
    src = _png_bytes(400, 300, bottom_stripe_value=255)
    out = scrub_bottom_margin(src)
    arr = np.array(Image.open(io.BytesIO(out)))
    assert not np.any(arr[-5:, :, :] == 255), "bright artifact pixels must be scrubbed"


def test_body_above_the_margin_is_untouched():
    src = _png_bytes(400, 300, bottom_stripe_value=255)
    out = scrub_bottom_margin(src)
    arr = np.array(Image.open(io.BytesIO(out)))
    assert np.all(arr[0, :, :] == 100), "content well above the margin must be preserved"


def test_zero_margin_is_a_noop():
    src = _png_bytes(100, 100, bottom_stripe_value=255)
    out = scrub_bottom_margin(src, margin_fraction=0)
    assert out == src
