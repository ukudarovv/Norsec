type Props = { level: string; score: number };

export default function RiskIndicator({ level, score }: Props) {
  const cls =
    level === "red"
      ? "border-red-500 bg-red-950/60 text-red-200"
      : level === "orange"
        ? "border-amber-500 bg-amber-950/60 text-amber-200"
        : level === "yellow"
          ? "border-yellow-500 bg-yellow-950/50 text-yellow-100"
          : "border-emerald-600 bg-emerald-950/40 text-emerald-200";
  return (
    <div className={`rounded-lg border px-3 py-2 ${cls}`}>
      <div className="text-xs uppercase tracking-wide text-slate-400">Risk</div>
      <div className="text-lg font-semibold">{level}</div>
      <div className="text-sm text-slate-300">score {(score * 100).toFixed(0)}%</div>
    </div>
  );
}
