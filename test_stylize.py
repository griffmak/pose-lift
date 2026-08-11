"""Self-check for stylize.py — mocks replicate.run, no real API call."""

import io
from unittest.mock import patch

import numpy as np

from pose_lift.stylize import stylize


def test_stylize_encodes_and_returns_bytes():
    depth_image = np.zeros((10, 10), dtype=np.uint8)
    depth_image[3:7, 3:7] = 200

    class FakeOutput:
        def read(self):
            return b"fake-png-bytes"

    with patch("pose_lift.stylize._client.run", return_value=FakeOutput()) as mock_run:
        result = stylize(depth_image, "test prompt")

    assert result == b"fake-png-bytes"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["input"]["prompt"] == "test prompt"
    assert isinstance(call_kwargs["input"]["control_image"], io.BytesIO)
    print("test_stylize_encodes_and_returns_bytes: OK")


if __name__ == "__main__":
    test_stylize_encodes_and_returns_bytes()
