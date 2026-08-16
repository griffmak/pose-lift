"""Pose Lift web backend on Modal.

Browser -> this app -> ROMP mesh -> pyrender depth -> 9:16 crop -> Replicate.
Two endpoints so that editing a prompt never re-pays for reconstruction.
"""

from pathlib import Path

import modal

APP_NAME = "pose-lift"

MINUTES = 60

# Cost control for the public instance. Self-hosters can raise or disable this.
GENERATION_CAP = 3
CAP_WINDOW_SECONDS = 24 * 60 * 60

# pyrender needs a headless GL backend. There is no GPU/EGL on a CPU container,
# so OSMesa (software rasterizer) is the working path — libosmesa6-dev provides it.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libosmesa6-dev",
        "freeglut3-dev",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "wget",
    )
    .env({"PYOPENGL_PLATFORM": "osmesa"})
    .pip_install_from_requirements("requirements.txt")
    # Gated, non-commercial-license file. Never in git; baked from the builder's
    # own registered copy. Path matches pose_lift.smpl_check.SMPL_PATH, which is
    # Path.home()/".romp" -> /root/.romp inside the container.
    .add_local_file(
        Path.home() / ".romp" / "SMPL_NEUTRAL.pth",
        "/root/.romp/SMPL_NEUTRAL.pth",
        copy=True,
    )
    .add_local_file("test_person.jpg", "/root/test_person.jpg", copy=True)
    .add_local_python_source("pose_lift")
)

app = modal.App(APP_NAME, image=image)

# ponytail: a single global Dict is fine at demo traffic. If this ever sees
# real load, the read-modify-write below can lose a concurrent increment —
# swap in an atomic counter then.
rate_limits = modal.Dict.from_name("pose-lift-rate-limits", create_if_missing=True)


@app.cls(
    timeout=10 * MINUTES,          # a slow Replicate call must not kill the request
    scaledown_window=5 * MINUTES,  # keep a warm container between demo takes
    secrets=[
        modal.Secret.from_name("pose-lift-replicate", required_keys=["REPLICATE_API_TOKEN"]),
        modal.Secret.from_name("pose-lift-openai", required_keys=["OPENAI_API_KEY"]),
    ],
)
class PoseLift:
    @modal.enter()
    def load(self):
        """Cold start: fail fast on the gated SMPL file, then load ROMP once."""
        from pose_lift.reconstruct import PoseReconstructor

        # PoseReconstructor.__init__ already calls require_smpl_model() first,
        # so a missing/unbaked SMPL file surfaces here as a clear startup error
        # instead of a buried traceback mid-request.
        self.reconstructor = PoseReconstructor(gpu=-1)

    @modal.method()
    def healthcheck(self) -> dict:
        """Prove the whole native stack works: SMPL loaded, ROMP ran, GL rendered."""
        from pose_lift.render_depth import render_depth_map

        outputs = self.reconstructor.reconstruct(_test_frame())
        if outputs is None:
            return {"ok": False, "stage": "reconstruct", "detail": "no pose detected"}
        depth, _mesh = render_depth_map(outputs["verts"][0], self.reconstructor.faces, 640, 480)
        return {"ok": True, "depth_nonzero": int((depth > 0).sum())}

    @modal.asgi_app()
    def api(self):
        import base64
        import io
        import os
        import time

        import cv2
        import numpy as np
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from openai import OpenAI
        from PIL import Image
        from pydantic import BaseModel
        from replicate.exceptions import ReplicateException

        from pose_lift.crop import crop_to_9x16
        from pose_lift.prompt import OPTIMIZER_SYSTEM_PROMPT, enforce_prompt_hygiene
        from pose_lift.render_depth import depth_to_conditioning_image, render_depth_map
        from pose_lift.stylize import stylize

        web_app = FastAPI(title="Pose Lift")
        # The browser calls this app directly (no Vercel proxy), so Modal sees
        # the real client IP for rate limiting. That requires open CORS.
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["POST", "OPTIONS"],
            allow_headers=["*"],
        )
        openai_client = OpenAI()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        def _png_b64(array) -> str:
            buf = io.BytesIO()
            Image.fromarray(array).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        def _client_key(request: Request, session_id: str) -> str:
            """Rate-limit on IP, since a session id is client-controlled.

            The browser calls Modal directly, so x-forwarded-for's first hop is
            the real client. session_id is only a tiebreaker for shared NATs.
            """
            forwarded = request.headers.get("x-forwarded-for", "")
            ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
            return f"{ip}|{session_id[:64]}"

        def _check_and_count(key: str) -> None:
            now = time.time()
            count, window_start = rate_limits.get(key, (0, now))
            if now - window_start > CAP_WINDOW_SECONDS:
                count, window_start = 0, now
            if count >= GENERATION_CAP:
                raise HTTPException(
                    429,
                    f"Generation limit reached ({GENERATION_CAP} per day). "
                    "This is a public demo — self-host the repo to remove the cap.",
                )
            rate_limits[key] = (count + 1, window_start)

        class OptimizeRequest(BaseModel):
            idea: str

        @web_app.post("/optimize-prompt")
        def optimize_prompt(req: OptimizeRequest):
            idea = req.idea.strip()
            if not idea:
                raise HTTPException(400, "idea is empty")
            if len(idea) > 500:
                raise HTTPException(400, "idea is too long (500 char max)")
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": idea},
                ],
                max_tokens=300,
            )
            drafted = completion.choices[0].message.content or ""
            # The LLM sometimes forgets the exclusion clause; the guard is not
            # optional, so it runs on the way out regardless.
            return {"prompt": enforce_prompt_hygiene(drafted)}

        class GenerateRequest(BaseModel):
            image: str
            prompt: str
            session_id: str = ""

        @web_app.post("/generate")
        def generate(req: GenerateRequest, request: Request):
            _check_and_count(_client_key(request, req.session_id))

            try:
                raw = base64.b64decode(req.image, validate=True)
            except Exception:
                raise HTTPException(400, "image is not valid base64")
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(400, "image could not be decoded")

            outputs = self.reconstructor.reconstruct(frame)
            if outputs is None:
                raise HTTPException(422, "no pose detected — step back and make sure your whole body is in frame")

            h, w = frame.shape[:2]
            depth, mesh_render = render_depth_map(outputs["verts"][0], self.reconstructor.faces, w, h)
            conditioning = crop_to_9x16(depth_to_conditioning_image(depth))

            try:
                stylized = stylize(conditioning, enforce_prompt_hygiene(req.prompt))
            except ReplicateException as e:
                # The mesh and depth survive; the client can show them and let
                # the user retry the render without recapturing.
                raise HTTPException(502, f"stylize failed: {e}")

            return {
                "mesh_render": _png_b64(mesh_render),
                "depth_conditioning": _png_b64(conditioning),
                "stylized": base64.b64encode(stylized).decode(),
            }

        return web_app


def _test_frame():
    """Load the repo's bundled test photo from inside the container."""
    import cv2

    frame = cv2.imread("/root/test_person.jpg")
    if frame is None:
        raise RuntimeError("test_person.jpg missing from image")
    return frame
