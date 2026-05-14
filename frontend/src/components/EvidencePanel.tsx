type Props = { evidence: Record<string, unknown> };

export default function EvidencePanel({ evidence }: Props) {
  const keys = ["social_signals", "pose_signals", "action_signals", "audio_signals", "context"] as const;
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-slate-200">Evidence</h3>
      {keys.map((k) => {
        const chunk = evidence[k];
        return (
          <div key={k} className="rounded-lg border border-slate-800 bg-slate-900/40">
            <div className="border-b border-slate-800 px-3 py-2 text-xs font-mono text-slate-400">{k}</div>
            <pre className="max-h-64 overflow-auto p-3 text-xs text-slate-300">
              {JSON.stringify(chunk ?? [], null, 2)}
            </pre>
          </div>
        );
      })}
    </div>
  );
}
