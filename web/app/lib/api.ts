const BASE = process.env.NEXT_PUBLIC_MODAL_URL;

export type GenerateResult = {
  mesh_render: string;
  depth_conditioning: string;
  stylized: string;
};

export function sessionId(): string {
  let id = localStorage.getItem("pose-lift-session");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("pose-lift-session", id);
  }
  return id;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  if (!BASE) throw new Error("NEXT_PUBLIC_MODAL_URL is not set");
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `request failed (${res.status})`);
  }
  return res.json();
}

export async function optimizePrompt(idea: string): Promise<string> {
  const { prompt } = await post<{ prompt: string }>("/optimize-prompt", { idea });
  return prompt;
}

export function generate(image: string, prompt: string): Promise<GenerateResult> {
  return post<GenerateResult>("/generate", { image, prompt, session_id: sessionId() });
}
