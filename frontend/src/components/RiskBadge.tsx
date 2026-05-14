type Props = { level: string };

const colors: Record<string, string> = {
  green: "bg-emerald-900/60 text-emerald-200 ring-emerald-700",
  yellow: "bg-amber-900/50 text-amber-100 ring-amber-700",
  orange: "bg-orange-900/50 text-orange-100 ring-orange-700",
  red: "bg-red-900/60 text-red-100 ring-red-700",
};

export default function RiskBadge({ level }: Props) {
  const cls = colors[level] ?? "bg-slate-800 text-slate-200 ring-slate-600";
  return (
    <span className={`inline-flex rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${cls}`}>
      {level}
    </span>
  );
}
