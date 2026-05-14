import { Link } from "react-router-dom";
import type { Incident } from "../../api/client";
import ReviewStatusBadge from "./ReviewStatusBadge";
import RiskBadge from "./RiskBadge";

type Props = {
  incidents: Incident[];
};

export default function IncidentTable({ incidents }: Props) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800">
      <table className="min-w-full text-left text-sm">
        <thead className="bg-slate-900 text-slate-400">
          <tr>
            <th className="px-3 py-2">Time</th>
            <th className="px-3 py-2">Camera</th>
            <th className="px-3 py-2">Risk</th>
            <th className="px-3 py-2">Level</th>
            <th className="px-3 py-2">Status</th>
            <th className="px-3 py-2">Signals</th>
            <th className="px-3 py-2">Reviewer</th>
            <th className="px-3 py-2">Action</th>
          </tr>
        </thead>
        <tbody>
          {incidents.map((row) => (
            <tr key={row.id} className="border-t border-slate-800 hover:bg-slate-900/50">
              <td className="px-3 py-2 whitespace-nowrap text-slate-300">{row.created_at || "—"}</td>
              <td className="px-3 py-2 font-mono text-xs">{row.camera_external_key || row.camera_id}</td>
              <td className="px-3 py-2">{row.risk_score.toFixed(2)}</td>
              <td className="px-3 py-2">
                <RiskBadge level={row.risk_level} />
              </td>
              <td className="px-3 py-2">
                <ReviewStatusBadge status={row.review_status} />
              </td>
              <td className="max-w-xs truncate px-3 py-2 text-xs text-slate-400">
                {row.signal_types.join(", ") || "—"}
              </td>
              <td className="px-3 py-2 text-xs text-slate-500">{row.last_reviewer_email || "—"}</td>
              <td className="px-3 py-2">
                <Link className="text-sky-400 hover:underline" to={`/incidents/${row.id}`}>
                  Open
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {incidents.length === 0 && (
        <p className="px-3 py-6 text-center text-slate-500">No incidents match filters.</p>
      )}
    </div>
  );
}
