"use client";

import { useEffect, useState } from "react";
import type { GenerateResult } from "../lib/api";

const HOLD_MS = 1400;

export default function Reveal({ result }: { result: GenerateResult }) {
  // Layers stack; each one fades in over the last, so the render appears to
  // build up: 3D mesh -> depth map -> final painting.
  const layers = [result.mesh_render, result.depth_conditioning, result.stylized];
  const [visible, setVisible] = useState(1);

  useEffect(() => {
    if (visible >= layers.length) return;
    const t = setTimeout(() => setVisible(visible + 1), HOLD_MS);
    return () => clearTimeout(t);
  }, [visible, layers.length]);

  return (
    <div className="relative w-full max-w-sm overflow-hidden rounded-lg">
      {layers.map((b64, i) => (
        <img
          key={i}
          src={`data:image/png;base64,${b64}`}
          alt=""
          className={`${i === 0 ? "relative" : "absolute inset-0"} w-full object-cover transition-opacity duration-700 ${
            i < visible ? "opacity-100" : "opacity-0"
          }`}
        />
      ))}
    </div>
  );
}
