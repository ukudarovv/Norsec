import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  fetchCameraAnalyticsLive,
  fetchCameraStatus,
  fetchIncidents,
  listCamerasDetailed,
  mjpegUrl,
  overlayWsUrl,
  restartCameraAnalysis,
  startCameraAnalysis,
  stopCameraAnalysis,
  testCameraConnection,
} from "../api/client";
import type { CameraRow, Incident, LiveCameraAnalytics, OverlayPayload } from "../api/client";
import { useAuth } from "../auth/AuthProvider";
import CameraControlPanel from "../components/live/CameraControlPanel";
import LiveAnalyticsPanel from "../components/live/LiveAnalyticsPanel";
import LiveVideoPanel from "../components/live/LiveVideoPanel";
import RiskIndicator from "../components/live/RiskIndicator";
import SignalPanel from "../components/live/SignalPanel";

export default function LiveMonitoringPage() {
  const { user } = useAuth();
  const [cams, setCams] = useState<CameraRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<OverlayPayload | null>(null);
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [liveAx, setLiveAx] = useState<LiveCameraAnalytics | null>(null);

  const load = useCallback(async () => {
    const rows = await listCamerasDetailed();
    setCams(rows);
    setSelected((prev) => prev ?? rows[0]?.id ?? null);
  }, []);

  useEffect(() => {
    void load().catch((e: Error) => setErr(e.message));
  }, [load]);

  const selectedCam = cams.find((c) => c.id === selected) ?? null;

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    void (async () => {
      try {
        const st = await fetchCameraStatus(selected);
        if (!cancelled) setStatus(st);
        const inc = await fetchIncidents({ camera_id: selected });
        if (!cancelled) setIncidents(inc.slice(0, 10));
      } catch {
        if (!cancelled) setStatus(null);
      }
    })();
    const t = window.setInterval(() => {
      void fetchCameraStatus(selected)
        .then((st) => {
          if (!cancelled) setStatus(st);
        })
        .catch(() => {});
    }, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [selected]);

  useEffect(() => {
    if (!selected) {
      setLiveAx(null);
      return;
    }
    let cancelled = false;
    const tick = () => {
      void fetchCameraAnalyticsLive(selected)
        .then((x) => {
          if (!cancelled) setLiveAx(x);
        })
        .catch(() => {
          if (!cancelled) setLiveAx(null);
        });
    };
    tick();
    const id = window.setInterval(tick, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [selected]);

  useEffect(() => {
    if (!selected) return;
    setOverlay(null);
    const url = overlayWsUrl(selected);
    const ws = new WebSocket(url);
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data as string) as OverlayPayload;
        setOverlay(data);
      } catch {
        /* ignore */
      }
    };
    return () => {
      ws.close();
    };
  }, [selected]);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setErr(null);
    try {
      await fn();
      await load();
      if (selected) {
        setStatus((await fetchCameraStatus(selected)) as Record<string, unknown>);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const live = (status?.live as Record<string, unknown> | undefined) ?? {};

  const analyticsBlock: OverlayPayload["analytics"] | null =
    overlay?.analytics ??
    (liveAx
      ? {
          social: (liveAx.active_social_signals as Record<string, unknown>[]) ?? [],
          pose: (liveAx.active_pose_signals as Record<string, unknown>[]) ?? [],
          trajectory_preview:
            (liveAx.trajectory_preview as Array<{ track_id: number; points: number[][] }>) ?? [],
          suppression: liveAx.suppression,
          risk_modifiers: (liveAx.risk_modifiers as string[]) ?? [],
        }
      : null);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Live monitoring</h1>
        <p className="text-sm text-slate-500">MJPEG preview + WebSocket overlay (MVP). Not a bullying verdict.</p>
      </div>
      {err && <p className="text-sm text-red-400">{err}</p>}
      <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
        <aside className="space-y-2">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Cameras</h2>
          <ul className="space-y-1">
            {cams.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  onClick={() => setSelected(c.id)}
                  className={`w-full rounded px-2 py-1.5 text-left text-sm ${
                    c.id === selected ? "bg-sky-900/50 text-white" : "text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {c.name}
                  <span className="block text-xs text-slate-500">{c.status}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>
        <div className="space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-medium text-slate-100">{selectedCam?.name ?? "—"}</h2>
              <p className="text-xs text-slate-500">
                Runtime: <span className="text-slate-300">{String(live.camera_status ?? "—")}</span> · worker{" "}
                {live.worker_running ? "on" : "off"} · seq {String(live.overlay_seq ?? 0)} · fps{" "}
                {String(live.fps_estimate ?? "—")} · dropped {String(live.dropped_frames ?? 0)}
              </p>
              {typeof live.last_error === "string" && live.last_error.length > 0 && (
                <p className="text-xs text-amber-400/90">Last error: {live.last_error}</p>
              )}
            </div>
            <RiskIndicator
              level={(overlay?.risk.level as string | undefined) ?? (liveAx?.risk?.level as string | undefined) ?? "green"}
              score={Number(overlay?.risk.score ?? liveAx?.risk?.score ?? 0)}
            />
          </div>
          <CameraControlPanel
            camera={selectedCam}
            role={user?.role}
            busy={busy}
            onTest={() =>
              void run(async () => {
                if (!selected) return;
                await testCameraConnection(selected);
              })
            }
            onStart={() =>
              void run(async () => {
                if (!selected) return;
                await startCameraAnalysis(selected);
              })
            }
            onStop={() =>
              void run(async () => {
                if (!selected) return;
                await stopCameraAnalysis(selected);
              })
            }
            onRestart={() =>
              void run(async () => {
                if (!selected) return;
                await restartCameraAnalysis(selected);
              })
            }
          />
          <LiveVideoPanel mjpegSrc={selected ? mjpegUrl(selected) : null} overlay={overlay} />
          <div className="grid gap-4 md:grid-cols-2">
            <SignalPanel overlay={overlay} />
            <LiveAnalyticsPanel overlay={overlay} analyticsOverride={analyticsBlock} />
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
            <h3 className="mb-2 text-sm font-semibold text-slate-200">Recent incidents</h3>
            <ul className="space-y-1 text-xs text-slate-400">
              {incidents.map((i) => (
                <li key={i.id}>
                  <Link className="text-sky-400 hover:underline" to={`/incidents/${i.id}`}>
                    {i.id.slice(0, 8)}…
                  </Link>{" "}
                  <span className="text-slate-500">{i.risk_level}</span> · {i.review_status}
                </li>
              ))}
              {incidents.length === 0 && <li>No incidents yet for this camera.</li>}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
