"""End-to-end CLI: capture/image -> 3D mesh -> depth conditioning -> stylized render.

Usage:
    python run.py --prompt "Spider-Man, Amazing Spider-Man #50 homage cover style" [--image test_person.jpg]

Without --image, grabs a single frame from the webcam.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
from PIL import Image
from replicate.exceptions import ReplicateException

from pose_lift.capture import capture_still
from pose_lift.crop import crop_to_9x16
from pose_lift.prompt import enforce_prompt_hygiene
from pose_lift.reconstruct import PoseReconstructor
from pose_lift.render_depth import depth_to_conditioning_image, render_depth_map
from pose_lift.stylize import stylize

RESULTS_DIR = Path(__file__).parent / "results"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_env_file(Path(__file__).parent / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", help="path to a still image; omit to use the webcam")
    args = parser.parse_args()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN not set (checked .env and environment)", file=sys.stderr)
        return 1

    frame = cv2.imread(args.image) if args.image else capture_still()
    if frame is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1

    reconstructor = PoseReconstructor()
    outputs = reconstructor.reconstruct(frame)
    if outputs is None:
        print("no pose detected, try again", file=sys.stderr)
        return 1

    verts = outputs["verts"][0]
    h, w = frame.shape[:2]
    depth, mesh_render = render_depth_map(verts, reconstructor.faces, w, h)
    conditioning = crop_to_9x16(depth_to_conditioning_image(depth))

    RESULTS_DIR.mkdir(exist_ok=True)
    mesh_path = RESULTS_DIR / "mesh_render.png"
    Image.fromarray(mesh_render).save(mesh_path)
    print(f"3D mesh render saved to {mesh_path}")

    conditioning_path = RESULTS_DIR / "depth_conditioning.png"
    Image.fromarray(conditioning).save(conditioning_path)
    print(f"depth conditioning image saved to {conditioning_path}")

    try:
        image_bytes = stylize(conditioning, enforce_prompt_hygiene(args.prompt))
    except ReplicateException as e:
        print(f"stylize call failed: {e}", file=sys.stderr)
        print("depth conditioning image above is preserved — retry the render step without recapturing", file=sys.stderr)
        return 1

    output_path = RESULTS_DIR / "stylized.png"
    output_path.write_bytes(image_bytes)
    print(f"stylized image saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
