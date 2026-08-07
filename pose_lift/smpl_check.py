"""Fail-fast guard for the gated SMPL body model file.

Per design doc Error Handling: ROMP's own error is a buried stack trace
mid-pipeline. This checks at startup instead, with a clear message pointing
at the expected path and the registration source.
"""

from pathlib import Path

SMPL_PATH = Path.home() / ".romp" / "SMPL_NEUTRAL.pth"


def require_smpl_model() -> Path:
    if not SMPL_PATH.exists():
        raise RuntimeError(
            f"SMPL body model not found at {SMPL_PATH}.\n"
            "Register and download SMPL_NEUTRAL.pth from https://smpl.is.tue.mpg.de "
            "(non-commercial/research-use license), then place it at that path."
        )
    return SMPL_PATH
