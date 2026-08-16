"""Center the depth conditioning image on the subject and crop it to 9:16.

flux-depth-pro has no aspect_ratio input — the output inherits the dimensions
of control_image. Cropping here is therefore the only way to get a vertical
render (validated 2026-08-16).
"""

import numpy as np

TARGET_RATIO = 9 / 16  # width / height


def crop_to_9x16(conditioning: np.ndarray) -> np.ndarray:
    """Crop a grayscale conditioning image to 9:16, centered on the subject.

    The crop follows the horizontal/vertical centroid of the non-zero (rendered
    mesh) pixels rather than the frame center, so an off-center subject is not
    sliced in half. Raises ValueError if the image has no subject pixels.
    """
    h, w = conditioning.shape[:2]
    ys, xs = np.nonzero(conditioning)
    if xs.size == 0:
        raise ValueError("conditioning image is empty — nothing to crop around")

    if w / h > TARGET_RATIO:
        new_w = int(round(h * TARGET_RATIO))
        start = _clamped_start(int(xs.mean()), new_w, w)
        return conditioning[:, start:start + new_w]

    new_h = int(round(w / TARGET_RATIO))
    if new_h >= h:
        return conditioning  # already at or narrower than 9:16
    start = _clamped_start(int(ys.mean()), new_h, h)
    return conditioning[start:start + new_h, :]


def _clamped_start(center: int, window: int, extent: int) -> int:
    """Top-left coordinate of a `window`-wide crop centered on `center`."""
    return max(0, min(center - window // 2, extent - window))
