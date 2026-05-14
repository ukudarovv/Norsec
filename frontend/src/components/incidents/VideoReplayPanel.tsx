import { incidentMediaUrl } from "../../api/client";

type Props = {
  incidentId: string;
  videoClipUrl: string;
  cameraLabel: string;
  startSec: number;
  endSec: number;
};

export default function VideoReplayPanel({ incidentId, videoClipUrl, cameraLabel, startSec, endSec }: Props) {
  if (!videoClipUrl) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-4">
        <h3 className="mb-2 text-sm font-semibold text-slate-200">Video clip</h3>
        <p className="text-xs text-slate-500">
          No stored clip for this incident. Context: camera <span className="font-mono">{cameraLabel}</span>,{" "}
          {startSec.toFixed(1)}s–{endSec.toFixed(1)}s. Playback is for operator context only — AI output remains a
          risk candidate.
        </p>
      </div>
    );
  }
  const src = incidentMediaUrl(incidentId, "clip");
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Video clip</h3>
      <p className="mb-2 text-xs text-slate-500">
        Camera <span className="font-mono">{cameraLabel}</span> · {startSec.toFixed(1)}s–{endSec.toFixed(1)}s
      </p>
      <video src={src} controls className="max-h-80 w-full rounded bg-black" />
    </div>
  );
}
