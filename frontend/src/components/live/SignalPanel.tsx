import type { OverlayPayload } from "../../api/client";

type Props = { overlay: OverlayPayload | null };

export default function SignalPanel({ overlay }: Props) {
  const sigs = overlay?.signals ?? [];
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Active signals</h3>
      {sigs.length === 0 ? (
        <p className="text-xs text-slate-500">No fused signals on this frame.</p>
      ) : (
        <ul className="space-y-1 text-xs text-slate-300">
          {sigs.map((s, i) => (
            <li key={i}>
              <span className="text-slate-500">{(s.type ?? "?") + " · "}</span>
              <span className="text-slate-200">{s.name ?? "—"}</span>
              {typeof s.severity === "number" && (
                <span className="text-slate-500"> ({(s.severity * 100).toFixed(0)}%)</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
