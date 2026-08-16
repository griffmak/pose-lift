"use client";

import { useState } from "react";

// Subjects from Griffin's actual paintings — see griffin-art.vercel.app.
const SUBJECTS = ["Spider-Man", "Hulk", "Batman", "Venom", "Wolverine", "Iron Man"];

export default function StylePicker({ onSubmit }: { onSubmit: (idea: string) => void }) {
  const [idea, setIdea] = useState("");

  return (
    <div className="flex w-full max-w-sm flex-col gap-4">
      <p className="text-sm opacity-70">Who do you want to be?</p>
      <div className="flex flex-wrap gap-2">
        {SUBJECTS.map((s) => (
          <button
            key={s}
            onClick={() => setIdea(s)}
            className={`rounded-full border px-4 py-2 text-sm ${
              idea === s ? "border-white bg-white text-black" : "border-white/30"
            }`}
          >
            {s}
          </button>
        ))}
      </div>
      <input
        value={idea}
        onChange={(e) => setIdea(e.target.value)}
        placeholder="…or describe anything else"
        className="rounded border border-white/30 bg-transparent px-4 py-3"
      />
      <button
        onClick={() => onSubmit(idea.trim())}
        disabled={!idea.trim()}
        className="rounded bg-white px-6 py-3 font-semibold text-black disabled:opacity-40"
      >
        Write my prompt
      </button>
    </div>
  );
}
