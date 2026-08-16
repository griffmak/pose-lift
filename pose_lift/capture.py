"""Single-frame webcam capture."""

import time

import cv2


def capture_still(camera_id: int = 1) -> "cv2.typing.MatLike":
    """Open the webcam, grab one frame, release the device.

    Raises RuntimeError if the camera can't be opened or a frame can't be read.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera_id}")
    try:
        ok, frame = False, None
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError("failed to read a frame from the camera")
    return frame


def save_still(frame, path: str) -> None:
    cv2.imwrite(path, frame)
