import type { CameraRow } from "../../api/client";
import { canControlLiveAnalysis } from "../../auth/permissions";

type Props = {
  camera: CameraRow | null;
  role: string | undefined;
  busy: boolean;
  onTest: () => void;
  onStart: () => void;
  onStop: () => void;
  onRestart: () => void;
};

export default function CameraControlPanel({ camera, role, busy, onTest, onStart, onStop, onRestart }: Props) {
  const can = canControlLiveAnalysis(role);
  if (!camera) {
    return <p className="text-sm text-slate-500">Choose a camera from the list.</p>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      <button
        type="button"
        disabled={!can || busy}
        className="rounded border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-700 disabled:opacity-40"
        onClick={onTest}
      >
        Test connection
      </button>
      <button
        type="button"
        disabled={!can || busy}
        className="rounded border border-sky-800 bg-sky-950 px-3 py-1.5 text-sm text-sky-200 hover:bg-sky-900 disabled:opacity-40"
        onClick={onStart}
      >
        Start analysis
      </button>
      <button
        type="button"
        disabled={!can || busy}
        className="rounded border border-amber-900 bg-amber-950 px-3 py-1.5 text-sm text-amber-100 hover:bg-amber-900 disabled:opacity-40"
        onClick={onStop}
      >
        Stop analysis
      </button>
      <button
        type="button"
        disabled={!can || busy}
        className="rounded border border-slate-600 bg-slate-900 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800 disabled:opacity-40"
        onClick={onRestart}
      >
        Restart
      </button>
    </div>
  );
}
