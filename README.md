# Pose Lift

Strike a pose on webcam → the pose is reconstructed as a true 3D mesh (SMPL, via ROMP) →
that exact pose drives a depth-conditioned AI render of a superhero in a hand-painted
comic-cover style. The 3D reconstruction is the technical core, not a thin wrapper
around a generative-image API — the depth map that conditions the final render comes
from a real recovered body mesh, not a 2D pose skeleton.

Live demo: https://pose-lift.vercel.app

## Architecture

```
[Webcam frame] → [ROMP: monocular 3D mesh recovery] → [SMPL mesh: verts+joints]
                                                              │
                                        ┌─────────────────────┴─────────────────────┐
                                        ▼                                           ▼
                          [Depth render of mesh]                     [OpenPose-style joint render]
                                        │                                           │
                                        └─────────────────────┬─────────────────────┘
                                                              ▼
                                    [Replicate ControlNet (depth conditioning)
                                     + text prompt: superhero + style] → [final image]
```

Two deployable pieces:
- **`pose_lift/` + `run.py`** — the core Python pipeline (capture → reconstruct → depth
  render → 9:16 crop → stylize), runnable as a CLI.
- **`modal_app.py` + `web/`** — the same pipeline behind a FastAPI backend on
  [Modal](https://modal.com) (GPU-free, scales to zero) with a Next.js frontend on
  [Vercel](https://vercel.com).

## Setup

### 1. SMPL body model (required, gated download)

ROMP needs the SMPL body model file to run at all, even without mesh export flags.
It's gated behind a free registration + non-commercial research license — not
mirrored here.

1. Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and download
   **"Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)"**
   (the model data loads fine from Python 3 despite the label). Rename
   `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`.
2. Download ROMP's auxiliary data (no registration needed):
   https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_model_data.zip
   (`J_regressor_extra.npy`, `J_regressor_h36m.npy`, `smpl_kid_template.npy`).
3. Put all four files in one folder and convert:
   ```
   romp.prepare_smpl -source_dir=<folder>
   ```
   `chumpy` (needed to unpickle the `.pkl`) requires Python <3.11-compatible numpy
   (`np.int`/`np.bool`, `inspect.getargspec`) — if your main environment has a newer
   numpy pinned by torch, do this conversion in a throwaway venv (`numpy<1.24`,
   `chumpy`, `scipy`) and copy the converted `.pth` files over.
4. Place the converted files at `~/.romp/SMPL_NEUTRAL.pth` and
   `~/.romp/SMPLA_NEUTRAL.pth`.

### 2. Python environment (CLI pipeline)

```
python -m venv venv && source venv/bin/activate
pip install --no-build-isolation -r requirements.txt
cp .env.example .env   # fill in REPLICATE_API_TOKEN
python run.py --prompt "Spider-Man, Amazing Spider-Man #50 homage cover style" --image test_person.jpg
```

`--no-build-isolation` is required — `simple-romp`'s `setup.py` imports Cython/numpy
directly at build time, which fails under pip's default isolated build.

### 3. Modal backend (web API)

```
pip install modal
modal secret create pose-lift-replicate REPLICATE_API_TOKEN=<your token>
modal secret create pose-lift-openai OPENAI_API_KEY=<your key>
modal deploy modal_app.py
```

### 4. Next.js frontend

```
cd web
npm install
echo "NEXT_PUBLIC_MODAL_URL=<your Modal app URL>" > .env.local
npm run dev   # or `vercel --prod` to deploy
```

## Privacy / data handling

The webcam frame is captured client-side and sent directly to your own Modal
deployment for processing — it is not stored by this repo's code beyond the
`results/` directory (gitignored) on the CLI, or the base64 response returned
to the browser on the web version. The frame and the resulting depth/mesh images
are forwarded to Replicate (for the stylized render) and OpenAI (for the prompt
rewrite) as third-party API calls under your own API keys — see each provider's
own data-retention policy if that matters for your use case. Nothing is persisted
server-side between requests; Modal's rate-limit `Dict` only stores a request count
per IP, not any image data.

## Cost control

`modal_app.py` rate-limits generations per IP via `GENERATION_CAP` (default: 3 per
24h, keyed on `x-forwarded-for` + session ID). Raise this in `modal_app.py` before a
demo/filming session if you need more headroom, then redeploy.

## Notes

- Depth is the primary conditioning signal for the final render — it captures body
  volume/shape in the exact pose, which is what demonstrates a real 3D reconstruction
  happened (a 2D pose tool could only produce a stick-figure skeleton).
- `flux-depth-pro` (the Replicate model used for stylization) reliably renders a fake
  stock-photo watermark into the bottom image margin regardless of prompt wording —
  `pose_lift/watermark_guard.py` scrubs it deterministically post-render rather than
  fighting it with prompt engineering.
