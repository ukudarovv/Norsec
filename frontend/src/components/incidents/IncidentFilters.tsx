type Props = {
  cameraId: string;
  riskLevel: string;
  reviewStatus: string;
  signalType: string;
  createdAfter: string;
  createdBefore: string;
  onChange: (patch: Partial<Record<string, string>>) => void;
  onApply: () => void;
};

export default function IncidentFilters({
  cameraId,
  riskLevel,
  reviewStatus,
  signalType,
  createdAfter,
  createdBefore,
  onChange,
  onApply,
}: Props) {
  return (
    <div className="grid gap-3 rounded-lg border border-slate-800 bg-slate-900/30 p-4 sm:grid-cols-2 lg:grid-cols-3">
      <label className="text-xs text-slate-400">
        camera_id
        <input
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
          value={cameraId}
          onChange={(e) => onChange({ camera_id: e.target.value })}
        />
      </label>
      <label className="text-xs text-slate-400">
        risk_level
        <select
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
          value={riskLevel}
          onChange={(e) => onChange({ risk_level: e.target.value })}
        >
          <option value="">any</option>
          <option value="green">green</option>
          <option value="yellow">yellow</option>
          <option value="orange">orange</option>
          <option value="red">red</option>
        </select>
      </label>
      <label className="text-xs text-slate-400">
        review_status
        <select
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
          value={reviewStatus}
          onChange={(e) => onChange({ review_status: e.target.value })}
        >
          <option value="">any</option>
          <option value="new">new</option>
          <option value="needs_review">needs_review</option>
          <option value="confirmed">confirmed</option>
          <option value="false_positive">false_positive</option>
          <option value="training_candidate">training_candidate</option>
          <option value="archived">archived</option>
        </select>
      </label>
      <label className="text-xs text-slate-400">
        signal_type
        <input
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
          placeholder="e.g. crowding"
          value={signalType}
          onChange={(e) => onChange({ signal_type: e.target.value })}
        />
      </label>
      <label className="text-xs text-slate-400">
        date_from (ISO)
        <input
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-sm"
          value={createdAfter}
          onChange={(e) => onChange({ created_after: e.target.value })}
        />
      </label>
      <label className="text-xs text-slate-400">
        date_to (ISO)
        <input
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-sm"
          value={createdBefore}
          onChange={(e) => onChange({ created_before: e.target.value })}
        />
      </label>
      <div className="flex items-end">
        <button
          type="button"
          className="rounded bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-600"
          onClick={() => onApply()}
        >
          Apply filters
        </button>
      </div>
    </div>
  );
}
