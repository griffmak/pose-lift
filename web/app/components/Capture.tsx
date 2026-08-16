"use client";

import { useEffect, useRef, useState } from "react";

const COUNTDOWN_SECONDS = 5;

export default function Capture({ onCapture }: { onCapture: (base64: string) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    navigator.mediaDevices
      // Ask for a portrait stream so the server-side 9:16 crop is near-noop
      // and no limbs get sliced off.
      .getUserMedia({ video: { aspectRatio: { ideal: 9 / 16 }, width: { ideal: 1080 } } })
      .then((s) => {
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch(() => setError("Camera access denied. Allow it in your browser and reload."));
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  useEffect(() => {
    if (count === null) return;
    if (count === 0) {
      const video = videoRef.current;
      if (!video) return;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")!.drawImage(video, 0, 0);
      // Strip the "data:image/jpeg;base64," prefix — the API wants bare base64.
      onCapture(canvas.toDataURL("image/jpeg", 0.92).split(",")[1]);
      setCount(null);
      return;
    }
    const t = setTimeout(() => setCount(count - 1), 1000);
    return () => clearTimeout(t);
  }, [count, onCapture]);

  if (error) return <p className="text-red-400">{error}</p>;

  return (
    <div className="relative flex flex-col items-center gap-4">
      <video ref={videoRef} autoPlay playsInline muted className="w-full max-w-sm rounded-lg" />
      {count !== null && count > 0 && (
        <span className="absolute inset-0 flex items-center justify-center text-8xl font-bold">
          {count}
        </span>
      )}
      <button
        onClick={() => setCount(COUNTDOWN_SECONDS)}
        disabled={count !== null}
        className="rounded bg-white px-6 py-3 font-semibold text-black disabled:opacity-40"
      >
        {count === null ? "Strike a pose" : "Get ready…"}
      </button>
    </div>
  );
}
