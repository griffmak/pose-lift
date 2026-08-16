"""Prompt construction for flux-depth-pro.

Two rules validated in the 2026-08-16 e2e session, and a third found during
the 2026-08-16 web-UI implementation pass:
  1. The model happily paints text, logos and watermarks unless told not to.
     flux-depth-pro has no negative-prompt input, so the exclusion has to live
     in the prompt text itself.
  2. Vague style words ("ninja aesthetic") produce mush. Concrete setting,
     lighting and materials produce a usable render.
  3. A plain "no watermarks, no text" negation list is not reliable — a
     dramatic cityscape render (Spider-Man over Manhattan) still produced a
     fake stock-photo watermark URL in the corner. Rephrasing as a positive
     "clean original painting, not a stock photo" description, naming the
     stock-photo watermark pattern specifically, is the fix that held up.
"""

HYGIENE_CLAUSE = (
    "The final image is a clean, original digital painting with a completely "
    "unmarked surface: not a stock photo, with no overlaid caption, no "
    "signature, no logo, no watermark, and no fictitious URL or website name "
    "anywhere in the frame, especially not in the corners."
)

OPTIMIZER_SYSTEM_PROMPT = """You rewrite a user's rough idea into a single \
image-generation prompt for flux-depth-pro, which is rendering a human figure \
whose exact pose is already fixed by a depth map.

Rules:
- Never describe the pose, body position, or limb placement. The depth map \
controls those; describing them fights the conditioning.
- Be concrete. Name the setting, the time of day, the lighting, the materials \
and the color palette. Replace vague style words like "ninja aesthetic" or \
"cool vibes" with specific, visual detail.
- Keep the user's chosen character or subject exactly as they named it.
- Do not write your own "no text/no watermark" disclaimer — one is appended \
automatically after your output.
- Output only the prompt. One paragraph, roughly 40-70 words. No preamble, no \
quotes, no explanation."""


def enforce_prompt_hygiene(prompt: str) -> str:
    """Guarantee the exclusion clause is present, whatever wrote the prompt.

    Applies to the optimizer output, a user's hand-edited prompt, and the CLI —
    the model needs this whether or not the LLM remembered to add it.
    """
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt is empty")
    if "watermark" in prompt.lower():
        return prompt
    return f"{prompt} {HYGIENE_CLAUSE}"
