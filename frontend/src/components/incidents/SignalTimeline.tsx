import type { IncidentDetailPayload } from "../../api/client";

function row(label: string, sub?: string) {
  return (
    <div className="flex gap-2 border-b border-slate-800/80 py-1.5 text-xs last:border-0">
      <div className="w-24 shrink-0 text-slate-500">{label}</div>
      <div className="text-slate-200">{sub ?? "—"}</div>
    </div>
  );
}

export default function SignalTimeline({ detail }: { detail: IncidentDetailPayload | null }) {
  if (!detail) return <p className="text-xs text-slate-500">No detail loaded.</p>;
  const inc = detail.incident;
  const lines: { label: string; text: string }[] = [];
  const t0 = inc.start_sec;
  lines.push({ label: "window", text: `${t0.toFixed(1)}s – ${inc.end_sec.toFixed(1)}s` });
  (detail.analytics.social_signals as Record<string, unknown>[]).forEach((s, i) => {
    lines.push({
      label: `social ${i + 1}`,
      text: `${String(s.signal_type ?? "?")} · sev ${String(s.severity ?? "—")}`,
    });
  });
  (detail.analytics.pose_signals as Record<string, unknown>[]).forEach((s, i) => {
    lines.push({
      label: `pose ${i + 1}`,
      text: `${String(s.signal_type ?? "?")} · tid ${String(s.track_id ?? "—")}`,
    });
  });
  (detail.analytics.action_signals as Record<string, unknown>[]).forEach((s, i) => {
    lines.push({
      label: `action ${i + 1}`,
      text: `${String(s.action_type ?? s.type ?? "?")}`,
    });
  });
  if (lines.length <= 1) {
    return <p className="text-xs text-slate-500">No structured timeline rows in analytics bundle.</p>;
  }
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Signal timeline (summary)</h3>
      <p className="mb-2 text-xs text-slate-500">AI risk candidate — requires human review. Times are window-relative.</p>
      <div>{lines.map((l, i) => row(l.label, l.text))}</div>
    </div>
  );
}
