# Pose Lift Web UI: Design Spec
Date: 2026-08-16 · Status: approved (design), not yet planned/implemented

## Context

The Pose Lift pipeline (webcam capture → ROMP 3D mesh reconstruction → pyrender depth map → Replicate `flux-depth-pro` stylized render, in Griffin's own painting style) is complete and validated end-to-end as a CLI tool (`run.py`) as of 2026-08-11, with a live e2e test session on 2026-08-16 that fixed camera capture bugs, added mesh-render output, and validated a prompt-quality fix (explicit exclusion of text/logos/watermarks) plus a 9:16 vertical crop of the depth-conditioning image.

Goal of this spec: define a public web version of the tool, to be shown live in an upcoming OpenMontage demo video and shared as a public GitHub template repo. This is a design spec only — no implementation yet.

## Decisions

### 1. Deployment split
- **Frontend**: Next.js on Vercel. Chosen over a Python-templated UI (Gradio/Streamlit) because this is a public, portfolio-quality demo sitting alongside Griffin's other Vercel-hosted tools (the-council, competitive-cage-match) — production polish matters here, not just functionality.
- **Backend**: Modal (Python, serverless, GPU-capable, scale-to-zero). Chosen over a persistent Render/Fly box because ROMP + pyrender are heavy/GPU-ish Python dependencies with bursty, unpredictable public traffic (a spike around the video release, then a long tail) — Modal avoids paying for idle compute between spikes and handles the gnarly dependency packaging (including the gated `SMPL_NEUTRAL.pth` requirement) via its image builder.
- Rejected: cramming ROMP/pyrender into Vercel serverless functions — hard time/memory/binary-dependency limits make this infeasible.

### 2. Repo / credentials split
- The public GitHub repo is a **self-hostable template**: no real credentials committed, `.env.example` only, README documents "bring your own Replicate/OpenAI/Modal keys."
- The **live Vercel + Modal deployment** (Griffin's own keys) is the instance demoed in the OpenMontage video and linked from the end card.

### 3. Frontend flow (fixed order — no reordering option)
1. Landing screen → webcam permission → 5-second countdown → capture.
2. Style picker: chips for Griffin's existing painting subjects (Hulk, Spider-Man, etc.) + a custom free-text field for anything else.
3. User's rough idea is sent to the backend; OpenAI rewrites it into a full `flux-depth-pro` prompt, baking in lessons validated this session (explicit exclusion of text/logos/watermarks; concrete aesthetic/setting detail instead of vague style words like "ninja aesthetic"). The optimized prompt is shown to the user in an **editable text box** before generation — not applied silently. Rationale: catches a bad optimization before it burns a paid Replicate call, and is more transparent for a public tool.
4. "Generate" triggers the backend pipeline: mesh reconstruction → depth map → **center-crop to 9:16 by default** (matches the video format and majority-mobile traffic; no landscape toggle in v1) → Replicate `flux-depth-pro` call with the (possibly user-edited) prompt.
5. Reveal: a **build-up animation** is the default reveal style — mesh render fades in, morphs into the depth map, morphs into the final stylized render. Rejected: a plain loading-spinner-then-final-image with mesh/depth as a secondary toggle — the build-up is more compelling, reuses assets already generated for free, and doubles as visual marketing when results are shared.

### 4. Backend API contract
Two separate Modal endpoints, kept apart so editing the prompt doesn't require re-running the expensive mesh step:
- `POST /optimize-prompt {idea: string}` → OpenAI call → returns the drafted `flux-depth-pro` prompt for the editable box.
- `POST /generate {image: base64, prompt: string}` → runs `reconstruct → render_depth → crop 9:16 → stylize` (webcam capture itself happens client-side in the browser via `getUserMedia`; no server-side cv2 webcam code needed) → returns `{mesh_render, depth_conditioning, stylized}` as base64 or signed URLs.

### 5. Rate limiting / cost control
Per-session cap (e.g. 3 generations per session, enforced server-side in the Modal endpoint via a cookie/session id — not just client-side, since that's trivially bypassed) on the **live instance**, to prevent runaway Replicate + Modal spend once the tool is public. The template repo ships the same cap mechanism as a config value self-hosters can change or remove.

## Explicitly out of scope for v1
- Landscape/portrait toggle (9:16 is the only default; revisit only if requested).
- Reordering the flow (style-first vs pose-first) — fixed order only.
- Any auth/login system — sessions are anonymous, rate-limited by session/IP only.
- Multi-pose or multi-render comparisons in one session (matches the CLI tool's single-shot design).

## Carried-forward gotchas (from `memory_smart_search`, verified against the current design, none contradict it)
- ROMP requires the gated `SMPL_NEUTRAL.pth` file unconditionally, even without `--calc_smpl` — must ship inside the Modal image, with a fail-fast check at cold start rather than a buried mid-pipeline traceback.
- Secrets should be loaded via Modal's native secrets mechanism, not a hand-rolled `.env` parser — the CLI's `load_env_file()` had a real gap (module-level `Client()` instantiation in `pose_lift/stylize.py` reads env vars at import time, before ad hoc scripts had a chance to load `.env`), and separately, unquoted `.env` values with trailing comments silently corrupt (seen in the OpenMontage project's dotenv parsing).

## Next steps
1. Next session: run `writing-plans` against this spec to produce a full implementation plan (task breakdown, Modal app scaffolding, Next.js app scaffolding, API contract implementation, rate-limit implementation, template repo README).
2. Session after that: film + assemble the OpenMontage video per the locked shot list in `ai-workspace/brainstorms/2026-08-16-pose-lift-openmontage-shotlist.md` (paintings showcase → pose transition → pipeline reveal montage using the new web UI → closeout), once the web UI is live.
