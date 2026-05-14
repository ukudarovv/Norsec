type Props = { status: string };

const colors: Record<string, string> = {
  new: "bg-slate-700 text-slate-100",
  needs_review: "bg-amber-900/60 text-amber-100",
  confirmed: "bg-rose-900/50 text-rose-100",
  false_positive: "bg-emerald-900/40 text-emerald-100",
  training_candidate: "bg-sky-900/50 text-sky-100",
  archived: "bg-slate-800 text-slate-400",
};

export default function ReviewStatusBadge({ status }: Props) {
  const cls = colors[status] ?? "bg-slate-800 text-slate-200";
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>{status}</span>;
}
