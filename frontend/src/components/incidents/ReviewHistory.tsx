import type { ReviewRow } from "../../api/client";

export default function ReviewHistory({ reviews }: { reviews: ReviewRow[] }) {
  if (reviews.length === 0) {
    return <p className="text-xs text-slate-500">No reviews yet.</p>;
  }
  return (
    <ul className="space-y-2 text-xs text-slate-300">
      {reviews.map((r) => (
        <li key={r.id} className="rounded border border-slate-800 bg-slate-950/40 p-2">
          <div className="font-medium text-slate-200">{r.status}</div>
          <div className="text-slate-500">{r.created_at}</div>
          {r.comment && <div className="mt-1 text-slate-400">{r.comment}</div>}
          {r.tags && r.tags.length > 0 && (
            <div className="mt-1 text-[10px] text-sky-300/90">tags: {r.tags.join(", ")}</div>
          )}
        </li>
      ))}
    </ul>
  );
}
