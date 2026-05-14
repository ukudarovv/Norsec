import type { OverlayPayload } from "../../api/client";

type AnalyticsBlock = {
  social?: Array<Record<string, unknown>>;
  pose?: Array<Record<string, unknown>>;
  trajectory_preview?: Array<{ track_id: number; points: number[][] }>;
  suppression?: Record<string, unknown>;
  risk_modifiers?: string[];
};

type Props = {
  overlay: OverlayPayload | null;
  /** Когда WS ещё пуст, но REST ``/analytics/live`` уже отдаёт снимок. */
  analyticsOverride?: OverlayPayload["analytics"] | null;
};

export default function LiveAnalyticsPanel({ overlay, analyticsOverride }: Props) {
  const a = (analyticsOverride ?? overlay?.analytics ?? null) as AnalyticsBlock | null;
  if (!a) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
        <h3 className="mb-2 text-sm font-semibold text-slate-200">Phase 2 analytics</h3>
        <p className="text-xs text-slate-500">No analytics payload yet (start camera worker).</p>
      </div>
    );
  }
  const social = a.social ?? [];
  const pose = a.pose ?? [];
  const mods = a.risk_modifiers ?? [];
  const sup = a.suppression ?? {};

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Phase 2 — live analytics</h3>
      <p className="mb-2 text-xs text-slate-500">Risk signal candidates only; requires review.</p>
      <div className="mb-3">
        <div className="text-xs font-medium text-slate-400">Active social signals</div>
        {social.length === 0 ? (
          <p className="text-xs text-slate-600">—</p>
        ) : (
          <ul className="mt-1 space-y-1 text-xs text-slate-300">
            {social.map((s, i) => (
              <li key={i}>
                <span className="text-sky-400/90">{String(s.signal_type ?? "?")}</span>
                {typeof s.severity === "number" && (
                  <span className="text-slate-500"> · {(Number(s.severity) * 100).toFixed(0)}%</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="mb-3">
        <div className="text-xs font-medium text-slate-400">Active pose signals</div>
        {pose.length === 0 ? (
          <p className="text-xs text-slate-600">—</p>
        ) : (
          <ul className="mt-1 space-y-1 text-xs text-slate-300">
            {pose.map((s, i) => (
              <li key={i}>
                <span className="text-emerald-400/90">{String(s.signal_type ?? "?")}</span>
                {typeof s.severity === "number" && (
                  <span className="text-slate-500"> · {(Number(s.severity) * 100).toFixed(0)}%</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div>
        <div className="text-xs font-medium text-slate-400">Risk modifiers / suppression</div>
        {mods.length === 0 ? (
          <p className="text-xs text-slate-600">—</p>
        ) : (
          <ul className="mt-1 list-inside list-disc space-y-0.5 text-xs text-amber-200/90">
            {mods.map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        )}
        <pre className="mt-2 max-h-24 overflow-auto rounded border border-slate-800/80 bg-slate-950 p-2 text-[10px] text-slate-500">
          {JSON.stringify(sup, null, 2)}
        </pre>
      </div>
    </div>
  );
}
