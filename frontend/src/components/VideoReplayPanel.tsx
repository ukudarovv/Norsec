type Props = {
  startSec: number;
  endSec: number;
  cameraId: string;
};

export default function VideoReplayPanel({ startSec, endSec, cameraId }: Props) {
  return (
    <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Video replay</h3>
      <p className="text-xs text-slate-500">
        Placeholder: attach stored clip URI / RTSP offset here (camera{" "}
        <span className="font-mono">{cameraId}</span>, {startSec.toFixed(1)}s–{endSec.toFixed(1)}s). Playback is for
        operator context only — AI output remains a risk candidate.
      </p>
    </div>
  );
}
