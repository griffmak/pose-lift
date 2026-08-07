"""In-process ROMP wrapper: one model load, reused across captures.

Design doc section 2: avoid the CLI's per-invocation reload cost by driving
romp.ROMP directly. This module also enforces the fail-fast SMPL guard
(smpl_check.require_smpl_model) before constructing ROMP, since ROMP builds
its SMPL_parser unconditionally in __init__ regardless of --calc_smpl.
"""

import numpy as np

from .smpl_check import require_smpl_model


class PoseReconstructor:
    def __init__(self, gpu: int = -1):
        require_smpl_model()

        from romp.main import romp_settings, ROMP

        settings = romp_settings(input_args=[])
        settings.GPU = gpu
        settings.calc_smpl = True
        settings.render_mesh = False
        settings.show_largest = True

        self._romp = ROMP(settings)

    def reconstruct(self, frame: np.ndarray) -> dict:
        """Run ROMP on a single BGR frame (as from cv2). Returns None if no
        person was detected, else a dict with 'verts' (6890x3) and 'joints'
        among other ROMP output keys.
        """
        outputs = self._romp(frame)
        if outputs is None:
            return None
        return outputs
