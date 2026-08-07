"""Render a SMPL mesh (verts+faces) to a depth-map conditioning image.

Depth is the primary ControlNet conditioning signal (design doc section 3):
it captures body volume/shape in the exact pose, not just a 2D skeleton.
"""

import numpy as np
import pyrender
import trimesh


def render_depth_map(
    verts: np.ndarray,
    faces: np.ndarray,
    image_width: int,
    image_height: int,
    yfov: float = np.pi / 3.0,
    camera_distance: float = 3.0,
) -> np.ndarray:
    """Render verts/faces from a camera facing +Z toward the mesh origin.

    Returns a (image_height, image_width) float32 depth buffer, 0.0 where
    no geometry was hit.
    """
    # ROMP outputs camera-space verts with Y pointing down; pyrender is Y-up.
    verts = verts.copy()
    verts[:, 1] *= -1

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(mesh)

    scene = pyrender.Scene()
    scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, camera_distance],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(image_width, image_height)
    try:
        _color, depth = renderer.render(scene)
    finally:
        renderer.delete()

    if np.count_nonzero(depth) == 0:
        raise RuntimeError("depth render produced no geometry — mesh may be empty or off-camera")

    return depth


def depth_to_conditioning_image(depth: np.ndarray) -> np.ndarray:
    """Normalize a raw depth buffer to a uint8 grayscale image for ControlNet input."""
    mask = depth > 0
    if not mask.any():
        raise RuntimeError("depth buffer has no valid pixels to normalize")
    d_min, d_max = depth[mask].min(), depth[mask].max()
    normalized = np.zeros_like(depth, dtype=np.float32)
    if d_max > d_min:
        normalized[mask] = 1.0 - (depth[mask] - d_min) / (d_max - d_min)
    else:
        normalized[mask] = 1.0
    return (normalized * 255).astype(np.uint8)
