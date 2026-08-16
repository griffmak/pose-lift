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
      // No aspectRatio constraint: forcing a portrait ratio at the camera
      // level makes some devices pick a digitally-cropped (zoomed-in) sensor
      // mode. CSS object-cover on the <video> handles the crop-to-fill
      // instead, using the camera's natural field of view.
      .getUserMedia({ video: { width: { ideal: 1080 } } })
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

  if (error)
    return (
      <p className="flex min-h-screen items-center justify-center p-8 text-red-400">{error}</p>
    );

  return (
    <div className="fixed inset-0 flex flex-col items-center justify-end bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 h-full w-full object-cover"
      />
      {count !== null && count > 0 && (
        <span className="absolute inset-0 z-10 flex items-center justify-center text-[8rem] font-bold text-white drop-shadow-lg">
          {count}
        </span>
      )}
      <button
        onClick={() => setCount(COUNTDOWN_SECONDS)}
        disabled={count !== null}
        className="relative z-10 mb-10 rounded bg-white px-6 py-3 font-semibold text-black disabled:opacity-40"
      >
        {count === null ? "Strike a pose" : "Get ready…"}
      </button>
    </div>
  );
}
