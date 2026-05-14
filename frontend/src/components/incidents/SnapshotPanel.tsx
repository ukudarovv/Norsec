import { incidentMediaUrl } from "../../api/client";

type Props = { incidentId: string; snapshotUrl: string };

export default function SnapshotPanel({ incidentId, snapshotUrl }: Props) {
  if (!snapshotUrl) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-4">
        <h3 className="mb-2 text-sm font-semibold text-slate-200">Snapshot</h3>
        <p className="text-xs text-slate-500">No snapshot_path in evidence for this incident.</p>
      </div>
    );
  }
  const src = incidentMediaUrl(incidentId, "snapshot");
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Snapshot</h3>
      <img src={src} alt="Incident snapshot" className="max-h-64 w-full rounded object-contain" />
    </div>
  );
}
