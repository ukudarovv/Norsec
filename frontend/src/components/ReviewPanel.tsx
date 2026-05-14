import { useState } from "react";
import { submitReview } from "../api/client";

type Props = {
  incidentId: string;
  onUpdated: () => void;
  canReview: boolean;
};

const buttons: { label: string; status: string }[] = [
  { label: "Confirm incident", status: "confirmed" },
  { label: "False positive", status: "false_positive" },
  { label: "Send to training", status: "training_candidate" },
  { label: "Needs review", status: "needs_review" },
];

export default function ReviewPanel({ incidentId, onUpdated, canReview }: Props) {
  const [comment, setComment] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function send(status: string) {
    setBusy(true);
    setMsg(null);
    try {
      await submitReview(incidentId, {
        status,
        comment: comment.trim() || undefined,
      });
      setMsg("Review saved.");
      onUpdated();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Operator review</h3>
      <p className="mb-3 text-xs text-slate-500">
        Human decision only — labels are not «bullying confirmed by AI».
      </p>
      {!canReview && <p className="mb-2 text-xs text-amber-600">Your role cannot submit reviews.</p>}
      <div className="mb-3 grid gap-2 sm:grid-cols-1">
        <input
          className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
          placeholder="Comment (optional)"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
      </div>
      <div className="flex flex-wrap gap-2">
        {buttons.map((b) => (
          <button
            key={b.status}
            type="button"
            disabled={busy || !canReview}
            className="rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-100 hover:bg-slate-700 disabled:opacity-50"
            onClick={() => void send(b.status)}
          >
            {b.label}
          </button>
        ))}
      </div>
      {msg && <p className="mt-2 text-xs text-slate-400">{msg}</p>}
    </div>
  );
}
