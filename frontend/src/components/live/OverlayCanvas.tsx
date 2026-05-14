import { useEffect, useRef } from "react";
import type { OverlayPayload } from "../../api/client";

type Props = {
  overlay: OverlayPayload | null;
  naturalWidth: number;
  naturalHeight: number;
};

export default function OverlayCanvas({ overlay, naturalWidth, naturalHeight }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const w = c.clientWidth;
    const h = c.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.max(1, Math.floor(w * dpr));
    c.height = Math.max(1, Math.floor(h * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!overlay || naturalWidth <= 0 || naturalHeight <= 0) return;

    const sx = w / naturalWidth;
    const sy = h / naturalHeight;

    for (const p of overlay.people) {
      const bb = p.bbox;
      if (!bb || bb.length !== 4) continue;
      const [x1, y1, x2, y2] = bb;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
      ctx.font = "12px system-ui";
      const label = `id ${p.track_id}`;
      ctx.fillText(label, x1 * sx + 4, Math.max(12, y1 * sy - 4));
      const sk = p.skeleton;
      if (sk && sk.length >= 2) {
        ctx.strokeStyle = "rgba(52, 211, 153, 0.9)";
        ctx.lineWidth = 2;
        for (let i = 0; i < sk.length - 1; i++) {
          const a = sk[i];
          const b = sk[i + 1];
          if (a.length >= 2 && b.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(a[0] * sx, a[1] * sy);
            ctx.lineTo(b[0] * sx, b[1] * sy);
            ctx.stroke();
          }
        }
      }
    }
  }, [overlay, naturalWidth, naturalHeight]);

  return <canvas ref={ref} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden />;
}
