import { useState } from "react";
import type { OverlayPayload } from "../../api/client";
import OverlayCanvas from "./OverlayCanvas";

type Props = { mjpegSrc: string | null; overlay: OverlayPayload | null };

export default function LiveVideoPanel({ mjpegSrc, overlay }: Props) {
  const [nw, setNw] = useState(640);
  const [nh, setNh] = useState(360);

  if (!mjpegSrc) {
    return (
      <div className="flex aspect-video w-full items-center justify-center rounded-lg border border-slate-800 bg-slate-950 text-sm text-slate-500">
        Select a camera to preview
      </div>
    );
  }

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-slate-800 bg-black">
      <img
        src={mjpegSrc}
        alt="Live camera"
        className="h-full w-full object-contain"
        onLoad={(e) => {
          const el = e.currentTarget;
          if (el.naturalWidth && el.naturalHeight) {
            setNw(el.naturalWidth);
            setNh(el.naturalHeight);
          }
        }}
      />
      <OverlayCanvas overlay={overlay} naturalWidth={nw} naturalHeight={nh} />
    </div>
  );
}
