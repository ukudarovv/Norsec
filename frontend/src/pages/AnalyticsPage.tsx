import { useCallback, useEffect, useState } from "react";
import type { DashboardStats } from "../api/client";
import { fetchAnalyticsSignals, fetchDashboardStats } from "../api/client";

export default function AnalyticsPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [cat, setCat] = useState<{ social_signals: string[]; pose_signals: string[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const [s, c] = await Promise.all([fetchDashboardStats(), fetchAnalyticsSignals()]);
      setStats(s);
      setCat({ social_signals: c.social_signals, pose_signals: c.pose_signals });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Load failed");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-white">Analytics</h1>
      <p className="text-sm text-slate-400">Operational overview and signal catalog.</p>
      {err && <p className="text-sm text-red-400">{err}</p>}
      {stats && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-300">
          <p>Total risk candidates: {stats.totals.risk_candidates}</p>
          <p>False positive rate (reviewed subset): {(stats.totals.false_positive_rate * 100).toFixed(1)}%</p>
        </div>
      )}
      {cat && (
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
            <h2 className="mb-2 text-sm font-semibold text-slate-200">Social signal types</h2>
            <p className="text-xs text-slate-500">{cat.social_signals.join(", ")}</p>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
            <h2 className="mb-2 text-sm font-semibold text-slate-200">Pose signal types</h2>
            <p className="text-xs text-slate-500">{cat.pose_signals.join(", ")}</p>
          </div>
        </div>
      )}
    </div>
  );
}
