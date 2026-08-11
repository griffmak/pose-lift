"""Depth conditioning image + prompt -> stylized image via Replicate ControlNet.

Design doc section 4. The Replicate SDK retries 429/503/504 internally
(see replicate-python README), so no custom retry loop is needed here.
"""

import io

import httpx
import numpy as np
from PIL import Image
from replicate.client import Client
from replicate.exceptions import ReplicateException

MODEL = "black-forest-labs/flux-depth-pro"

# flux-depth-pro can take well over httpx's 5s default read timeout to finish.
_client = Client(timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0))


def stylize(depth_image: np.ndarray, prompt: str, *, guidance: float = 30) -> bytes:
    """Send a depth conditioning image + prompt to Replicate, return PNG bytes.

    Raises ReplicateException on failure — callers should still have the
    depth image (passed in) to retry the render step without recapturing.
    """
    buf = io.BytesIO()
    Image.fromarray(depth_image).save(buf, format="PNG")
    buf.seek(0)

    try:
        output = _client.run(
            MODEL,
            input={
                "prompt": prompt,
                "control_image": buf,
                "guidance": guidance,
                "output_format": "png",
            },
        )
    except ReplicateException as e:
        raise ReplicateException(f"Replicate stylize call failed: {e}") from e

    return output.read()
