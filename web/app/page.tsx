"use client";

import { useState } from "react";
import Capture from "./components/Capture";
import Reveal from "./components/Reveal";
import StylePicker from "./components/StylePicker";
import { generate, optimizePrompt, type GenerateResult } from "./lib/api";

type Stage = "capturing" | "styling" | "optimizing" | "editing" | "generating" | "revealing";

export default function Home() {
  const [stage, setStage] = useState<Stage>("capturing");
  const [shot, setShot] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleIdea(idea: string) {
    setStage("optimizing");
    setError(null);
    try {
      setPrompt(await optimizePrompt(idea));
      setStage("editing");
    } catch (e) {
      setError((e as Error).message);
      setStage("styling");
    }
  }

  async function handleGenerate() {
    if (!shot) return;
    setStage("generating");
    setError(null);
    try {
      setResult(await generate(shot, prompt));
      setStage("revealing");
    } catch (e) {
      setError((e as Error).message);
      setStage("editing");
    }
  }

  if (stage === "capturing") {
    return (
      <Capture
        onCapture={(b64) => {
          setShot(b64);
          setStage("styling");
        }}
      />
    );
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col items-center gap-6 p-8">
      <h1 className="text-3xl font-bold">Pose Lift</h1>
      {error && <p className="text-red-400">{error}</p>}

      {shot && stage !== "revealing" && (
        <img src={`data:image/jpeg;base64,${shot}`} alt="your pose" className="max-w-[12rem] rounded-lg" />
      )}

      {stage === "styling" && <StylePicker onSubmit={handleIdea} />}
      {stage === "optimizing" && <p className="animate-pulse">Writing your prompt…</p>}

      {stage === "editing" && (
        <div className="flex w-full max-w-sm flex-col gap-3">
          <p className="text-sm opacity-70">Edit this before generating if you want:</p>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={7}
            className="rounded border border-white/30 bg-transparent p-3 text-sm"
          />
          <button onClick={handleGenerate} className="rounded bg-white px-6 py-3 font-semibold text-black">
            Generate
          </button>
        </div>
      )}

      {stage === "generating" && (
        <p className="animate-pulse">Lifting your pose into 3D and painting it… (about a minute)</p>
      )}
      {stage === "revealing" && result && <Reveal result={result} />}

      <footer className="mt-auto pt-8 text-sm opacity-50">
        <a href="https://github.com/griffmak/pose-lift" className="underline">
          Source — self-host it with your own keys
        </a>
      </footer>
    </main>
  );
}
