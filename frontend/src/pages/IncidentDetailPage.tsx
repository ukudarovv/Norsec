import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { IncidentDetailPayload } from "../api/client";
import { fetchIncidentDetail } from "../api/client";
import { useAuth } from "../auth/AuthProvider";
import { canAddOperatorNotes, canReview } from "../auth/permissions";
import EvidencePanel from "../components/incidents/EvidencePanel";
import ReviewHistory from "../components/incidents/ReviewHistory";
import ReviewPanel from "../components/incidents/ReviewPanel";
import ReviewStatusBadge from "../components/incidents/ReviewStatusBadge";
import RiskBadge from "../components/incidents/RiskBadge";
import SignalTimeline from "../components/incidents/SignalTimeline";
import SnapshotPanel from "../components/incidents/SnapshotPanel";
import VideoReplayPanel from "../components/incidents/VideoReplayPanel";

function TrajectoryMiniView({ summary }: { summary: Record<string, unknown> }) {
  const tracks = summary.tracks as Array<{ track_id?: number; points?: number[][] }> | undefined;
  if (!tracks || !Array.isArray(tracks) || tracks.length === 0) {
    return (
      <p className="text-xs text-slate-600">
        No trajectory polyline in stored summary. When the pipeline stores a &apos;tracks&apos; field in evidence,
        paths appear here.
      </p>
    );
  }
  return (
    <div className="flex flex-wrap gap-3">
      {tracks.slice(0, 4).map((tr, idx) => {
        const pts = tr.points ?? [];
        if (pts.length < 2) return null;
        let minX = Infinity,
          minY = Infinity,
          maxX = -Infinity,
          maxY = -Infinity;
        for (const p of pts) {
          if (p.length < 2) continue;
          minX = Math.min(minX, p[0]);
          minY = Math.min(minY, p[1]);
          maxX = Math.max(maxX, p[0]);
          maxY = Math.max(maxY, p[1]);
        }
        const w = Math.max(1, maxX - minX);
        const h = Math.max(1, maxY - minY);
        const scale = 80 / Math.max(w, h);
        const d = pts
          .map((p) => `${(p[0] - minX) * scale + 4},${(p[1] - minY) * scale + 4}`)
          .join(" ");
        return (
          <div key={idx} className="rounded border border-slate-800 bg-slate-950/60 p-2">
            <div className="mb-1 text-[10px] text-slate-500">track {String(tr.track_id ?? idx)}</div>
            <svg width={96} height={96} className="text-sky-400">
              <polyline fill="none" stroke="currentColor" strokeWidth="1.5" points={d} />
            </svg>
          </div>
        );
      })}
    </div>
  );
}

export default function IncidentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { user } = useAuth();
  const [detail, setDetail] = useState<IncidentDetailPayload | null | undefined>(undefined);

  const load = useCallback(async () => {
    if (!id) {
      setDetail(null);
      return;
    }
    try {
      const d = await fetchIncidentDetail(id);
      setDetail(d);
    } catch {
      setDetail(null);
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  if (detail === undefined) {
    return <p className="text-slate-400">Loading…</p>;
  }
  if (detail === null) {
    return (
      <p className="text-slate-400">
        Not found. <Link to="/incidents">Back to incidents</Link>
      </p>
    );
  }

  const row = detail.incident;
  const traj = (detail.evidence.trajectory || detail.evidence.trajectory_summary || {}) as Record<string, unknown>;

  return (
    <div className="space-y-6">
      <div>
        <Link to="/incidents" className="text-sm text-sky-400 hover:underline">
          ← Incidents
        </Link>
        <h1 className="mt-2 text-xl font-semibold text-white">Incident {row.id}</h1>
        <p className="text-sm text-slate-500">AI risk candidate — requires human review.</p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <div className="text-xs text-slate-500">Risk score</div>
          <div className="text-2xl font-semibold text-white">{row.risk_score.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500">Level</div>
          <RiskBadge level={row.risk_level} />
        </div>
        <div>
          <div className="text-xs text-slate-500">Review status</div>
          <ReviewStatusBadge status={row.review_status} />
        </div>
        {row.review_status === "confirmed" && (
          <p className="text-xs text-rose-200/90">Confirmed by reviewer (human).</p>
        )}
      </div>
      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Explanation</h2>
        <ul className="list-inside list-disc space-y-1 text-sm text-slate-300">
          {row.explanation.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      </section>
      <VideoReplayPanel
        incidentId={row.id}
        videoClipUrl={detail.video_clip_url}
        cameraLabel={row.camera_external_key || row.camera_id}
        startSec={row.start_sec}
        endSec={row.end_sec}
      />
      <SnapshotPanel incidentId={row.id} snapshotUrl={detail.snapshot_url} />
      <SignalTimeline detail={detail} />
      <EvidencePanel evidence={detail.evidence} />
      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Suppression reasons</h2>
        {detail.analytics.suppression_reasons.length > 0 ? (
          <ul className="list-inside list-disc text-sm text-amber-200/90">
            {detail.analytics.suppression_reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-slate-600">—</p>
        )}
      </section>
      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Trajectory summary</h2>
        <TrajectoryMiniView summary={traj} />
      </section>
      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Review history</h2>
        <ReviewHistory reviews={detail.reviews} />
      </section>
      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Raw evidence JSON</h2>
        <pre className="max-h-80 overflow-auto rounded border border-slate-800 bg-slate-950 p-3 text-xs text-slate-300">
          {JSON.stringify(detail.evidence, null, 2)}
        </pre>
      </section>
      <section>
        <h2 className="mb-2 text-sm font-semibold text-slate-200">Signals (types)</h2>
        <p className="text-sm text-slate-400">{row.signal_types.join(", ") || "—"}</p>
      </section>
      <ReviewPanel
        incidentId={row.id}
        onUpdated={() => void load()}
        canReview={canReview(user?.role)}
        canAddNotes={canAddOperatorNotes(user?.role)}
      />
    </div>
  );
}
