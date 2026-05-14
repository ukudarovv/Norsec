import { useCallback, useEffect, useState } from "react";
import type { Incident } from "../api/client";
import { fetchIncidents } from "../api/client";
import IncidentFilters from "../components/incidents/IncidentFilters";
import IncidentTable from "../components/incidents/IncidentTable";

type Filters = {
  camera_id: string;
  risk_level: string;
  review_status: string;
  signal_type: string;
  created_after: string;
  created_before: string;
};

const emptyFilters: Filters = {
  camera_id: "",
  risk_level: "",
  review_status: "",
  signal_type: "",
  created_after: "",
  created_before: "",
};

export default function IncidentsPage() {
  const [rows, setRows] = useState<Incident[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [draft, setDraft] = useState<Filters>({ ...emptyFilters });
  const [applied, setApplied] = useState<Filters>({ ...emptyFilters });

  const load = useCallback(async () => {
    setErr(null);
    try {
      const data = await fetchIncidents({
        camera_id: applied.camera_id || undefined,
        risk_level: applied.risk_level || undefined,
        review_status: applied.review_status || undefined,
        signal_type: applied.signal_type || undefined,
        created_after: applied.created_after || undefined,
        created_before: applied.created_before || undefined,
      });
      setRows(data);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Load failed");
    }
  }, [applied]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-white">Incidents</h1>
      <p className="text-sm text-slate-400">AI risk candidates — filter and open for review.</p>
      <IncidentFilters
        cameraId={draft.camera_id}
        riskLevel={draft.risk_level}
        reviewStatus={draft.review_status}
        signalType={draft.signal_type}
        createdAfter={draft.created_after}
        createdBefore={draft.created_before}
        onChange={(patch) => setDraft((prev) => ({ ...prev, ...patch }))}
        onApply={() => {
          setApplied({ ...draft });
        }}
      />
      {err && <p className="text-sm text-red-400">{err}</p>}
      <IncidentTable incidents={rows} />
    </div>
  );
}
