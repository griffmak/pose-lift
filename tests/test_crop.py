import numpy as np
import pytest

from pose_lift.crop import crop_to_9x16

TARGET = 9 / 16


def _ratio(img):
    h, w = img.shape[:2]
    return w / h


def test_wide_image_is_cropped_horizontally():
    img = np.zeros((480, 640), dtype=np.uint8)
    img[100:400, 300:340] = 200  # subject slightly right of center
    out = crop_to_9x16(img)
    assert out.shape[0] == 480, "height must be preserved when cropping width"
    assert _ratio(out) == pytest.approx(TARGET, abs=0.01)


def test_crop_follows_the_subject_not_the_frame_center():
    img = np.zeros((480, 640), dtype=np.uint8)
    img[100:400, 500:560] = 200  # subject far right; a naive center crop loses it
    out = crop_to_9x16(img)
    assert out.max() == 200, "subject must survive the crop"


def test_crop_window_stays_inside_the_frame():
    img = np.zeros((480, 640), dtype=np.uint8)
    img[100:400, 620:640] = 200  # subject jammed against the right edge
    out = crop_to_9x16(img)
    assert out.shape == (480, 270)


def test_tall_image_is_cropped_vertically():
    img = np.zeros((1000, 400), dtype=np.uint8)
    img[400:600, 150:250] = 200
    out = crop_to_9x16(img)
    assert out.shape[1] == 400, "width must be preserved when cropping height"
    assert _ratio(out) == pytest.approx(TARGET, abs=0.01)


def test_already_9x16_is_unchanged():
    img = np.zeros((1600, 900), dtype=np.uint8)
    img[700:900, 400:500] = 200
    out = crop_to_9x16(img)
    assert out.shape == img.shape


def test_empty_conditioning_image_raises():
    with pytest.raises(ValueError):
        crop_to_9x16(np.zeros((480, 640), dtype=np.uint8))
