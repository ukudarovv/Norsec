import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { DashboardStats } from "../api/client";
import { fetchDashboardStats, fetchIncidents } from "../api/client";
import type { Incident } from "../api/client";
import IncidentTable from "../components/incidents/IncidentTable";

function BarRow({ label, value, max }: { label: string; value: number; max: number }) {
  const w = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div className="h-2 overflow-hidden rounded bg-slate-800">
        <div className="h-full bg-sky-600" style={{ width: `${w}%` }} />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recent, setRecent] = useState<Incident[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const [s, rows] = await Promise.all([fetchDashboardStats(), fetchIncidents({})]);
      setStats(s);
      setRecent(rows.slice(0, 8));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Load failed");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const maxStatus = stats ? Math.max(...Object.values(stats.by_status), 1) : 1;
  const maxLevel = stats ? Math.max(...Object.values(stats.by_risk_level), 1) : 1;
  const maxDay = stats ? Math.max(...Object.values(stats.by_day), 1) : 1;
  const maxCam = stats ? Math.max(...Object.values(stats.by_camera), 1) : 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Dashboard</h1>
        <p className="text-sm text-slate-400">
          AI risk candidates — requires human review. Use Incidents for full table and filters.
        </p>
      </div>
      {err && <p className="text-sm text-red-400">{err}</p>}
      {stats && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">Totals</h2>
            <ul className="space-y-2 text-sm text-slate-300">
              <li>Total risk candidates: {stats.totals.risk_candidates}</li>
              <li>Needs review (new + needs_review): {stats.totals.needs_review}</li>
              <li>Confirmed: {stats.totals.confirmed}</li>
              <li>False positives: {stats.totals.false_positives}</li>
              <li>Average risk score: {stats.totals.average_risk_score.toFixed(3)}</li>
              <li>False positive rate: {(stats.totals.false_positive_rate * 100).toFixed(1)}%</li>
            </ul>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">Camera health</h2>
            <p className="text-sm text-slate-300">Cameras: {stats.camera_health.total_cameras}</p>
            <p className="text-sm text-slate-300">Online / active: {stats.camera_health.online_or_active}</p>
            <Link className="mt-3 inline-block text-sm text-sky-400 hover:underline" to="/cameras-ui">
              Manage cameras
            </Link>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">Quick links</h2>
            <ul className="space-y-2 text-sm">
              <li>
                <Link className="text-sky-400 hover:underline" to="/incidents">
                  All incidents
                </Link>
              </li>
              <li>
                <Link className="text-sky-400 hover:underline" to="/review-queue">
                  Review queue
                </Link>
              </li>
              <li>
                <Link className="text-sky-400 hover:underline" to="/analytics">
                  Analytics
                </Link>
              </li>
            </ul>
          </div>
        </div>
      )}
      {stats && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">Review status distribution</h2>
            {Object.entries(stats.by_status).map(([k, v]) => (
              <BarRow key={k} label={k} value={v} max={maxStatus} />
            ))}
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">Risk level distribution</h2>
            {Object.entries(stats.by_risk_level).map(([k, v]) => (
              <BarRow key={k} label={k} value={v} max={maxLevel} />
            ))}
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">Incidents by day</h2>
            {Object.entries(stats.by_day).map(([k, v]) => (
              <BarRow key={k} label={k} value={v} max={maxDay} />
            ))}
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">Incidents by camera (top)</h2>
            {Object.entries(stats.by_camera).map(([k, v]) => (
              <BarRow key={k} label={k} value={v} max={maxCam} />
            ))}
          </div>
        </div>
      )}
      <div>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Recent incidents</h2>
        <IncidentTable incidents={recent} />
      </div>
    </div>
  );
}
