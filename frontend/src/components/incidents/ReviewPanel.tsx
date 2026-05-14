import { useMemo, useState } from "react";
import { postOperatorNote, submitReview } from "../../api/client";

type Props = {
  incidentId: string;
  onUpdated: () => void;
  canReview: boolean;
  canAddNotes: boolean;
};

const TAG_OPTIONS = [
  "rough_play",
  "sports",
  "real_conflict",
  "audio_unclear",
  "bad_camera_angle",
  "false_positive",
  "use_for_training",
];

const ACTIONS: { label: string; status: string }[] = [
  { label: "Confirm", status: "confirmed" },
  { label: "False positive", status: "false_positive" },
  { label: "Needs review", status: "needs_review" },
  { label: "Send to training", status: "training_candidate" },
  { label: "Archive", status: "archived" },
];

export default function ReviewPanel({ incidentId, onUpdated, canReview, canAddNotes }: Props) {
  const [comment, setComment] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const needsComment = useMemo(() => new Set(["confirmed", "false_positive", "training_candidate"]), []);

  async function send(status: string) {
    if (needsComment.has(status) && !comment.trim()) {
      setMsg("Comment is required for this status.");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      await submitReview(incidentId, {
        status,
        comment: comment.trim() || undefined,
        tags: tags.length ? tags : undefined,
      });
      setMsg(status === "confirmed" ? "Confirmed by reviewer." : "Review saved.");
      onUpdated();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-200">Review actions</h3>
      <p className="mb-3 text-xs text-slate-500">
        AI risk candidate — requires human review. Labels are not «bullying confirmed by AI».
      </p>
      {!canReview && <p className="mb-2 text-xs text-amber-600">Your role cannot submit status changes.</p>}
      <div className="mb-3">
        <div className="mb-1 text-xs text-slate-500">Tags</div>
        <div className="flex flex-wrap gap-2">
          {TAG_OPTIONS.map((t) => (
            <label key={t} className="flex items-center gap-1 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={tags.includes(t)}
                disabled={!canReview}
                onChange={() =>
                  setTags((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]))
                }
              />
              {t}
            </label>
          ))}
        </div>
      </div>
      <div className="mb-3 grid gap-2 sm:grid-cols-1">
        <textarea
          className="min-h-[72px] rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
          placeholder="Comment (required for Confirm / False positive / Send to training)"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          disabled={!canReview}
        />
      </div>
      <div className="flex flex-wrap gap-2">
        {ACTIONS.map((b) => (
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
      {canAddNotes && (
        <OperatorNoteBlock incidentId={incidentId} onDone={() => void onUpdated()} />
      )}
      {msg && <p className="mt-2 text-xs text-slate-400">{msg}</p>}
    </div>
  );
}

function OperatorNoteBlock({ incidentId, onDone }: { incidentId: string; onDone: () => void }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  return (
    <div className="mt-4 border-t border-slate-800 pt-3">
      <h4 className="mb-1 text-xs font-semibold text-slate-400">Operator note</h4>
      <textarea
        className="mb-2 min-h-[56px] w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
        placeholder="Add a note (does not change review status)"
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />
      <button
        type="button"
        disabled={busy || !note.trim()}
        className="rounded bg-slate-700 px-3 py-1 text-xs text-white hover:bg-slate-600 disabled:opacity-50"
        onClick={() => {
          setBusy(true);
          setMsg(null);
          void (async () => {
            try {
              await postOperatorNote(incidentId, note.trim());
              setNote("");
              setMsg("Note saved.");
              onDone();
            } catch (e) {
              setMsg(e instanceof Error ? e.message : "Failed");
            } finally {
              setBusy(false);
            }
          })();
        }}
      >
        Save note
      </button>
      {msg && <p className="mt-1 text-xs text-slate-500">{msg}</p>}
    </div>
  );
}
