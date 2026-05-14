import { useCallback, useEffect, useState } from "react";
import type { Incident } from "../api/client";
import { fetchReviewQueue } from "../api/client";
import IncidentTable from "../components/incidents/IncidentTable";

export default function ReviewQueuePage() {
  const [rows, setRows] = useState<Incident[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      setRows(await fetchReviewQueue());
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Load failed");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-white">Review queue</h1>
      <p className="text-sm text-slate-400">Incidents in status new or needs_review (oldest first).</p>
      {err && <p className="text-sm text-red-400">{err}</p>}
      <IncidentTable incidents={rows} />
    </div>
  );
}
