import numpy as np
import trimesh
import pyrender

mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
render_mesh = pyrender.Mesh.from_trimesh(mesh)

scene = pyrender.Scene()
scene.add(render_mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

light = pyrender.DirectionalLight(intensity=3.0)
scene.add(light, pose=camera_pose)

renderer = pyrender.OffscreenRenderer(400, 400)
color, depth = renderer.render(scene)
renderer.delete()

nonzero = np.count_nonzero(depth)
print(f"depth.shape={depth.shape} dtype={depth.dtype}")
print(f"nonzero depth pixels: {nonzero} / {depth.size}")
assert nonzero > 0, "depth buffer is empty — offscreen rendering did not produce geometry"
print("PYRENDER OFFSCREEN SMOKE TEST: PASS")
